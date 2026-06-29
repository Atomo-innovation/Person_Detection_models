#!/usr/bin/env python3
"""
YOLO26s person detection on Khadas Electron (asnn) — RTSP + Video File pipeline.

Preprocess:  RGB → adaptive enhance (dark/normal/overexposed/noisy) → letterbox 640×640 → NPU tensor
Postprocess: 3-scale decode → refine → desk-aware NMS → map to frame → conf filter
Display:     green boxes only at or above --conf (default 0.34)
Live JSON:   ``person_live.json`` (atomic write each loop)

NEW:
  - Live person count printed prominently in terminal (large ASCII banner + inline)
  - --video  : accept a local video file instead of (or alongside) --rtsp
  - --output : save annotated output video to file (MP4/AVI)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2 as cv
import numpy as np
from asnn.api import asnn
from asnn.types import output_format

log = logging.getLogger("yolo26_detect")

# ── Model geometry (YOLO26s, reg_max=1, 84 ch / scale) ─────────────────────
GRID_SIZES = (20, 40, 80)
STRIDES = (32, 16, 8)
LISTSIZE = 5
NUM_CLS = 1
SCALE_CONF_MUL = (1.0, 0.92, 0.78)

INV_255 = 1.0 / 255.0
LETTERBOX_PAD = 114

# ── Production defaults ─────────────────────────────────────────────────────
DEFAULT_CONF = 0.34
DEFAULT_NMS = 0.56
DEFAULT_IMGSZ = (640, 640)
PERSON_LIVE_JSON = "person_live.json"

DECODE_MARGIN_DARK = 0.06
DECODE_FLOOR = 0.26

MIN_BOX_AREA = 0.00008
MIN_BOX_W = 0.004
MIN_BOX_H = 0.006
MIN_ASPECT = 0.35
MAX_ASPECT = 5.5
VERT_NMS_SEP = 0.055

# ── Brightness thresholds ────────────────────────────────────────────────────
DARK_THRESH        = 60
MILD_DARK_THRESH   = 90
BRIGHT_THRESH      = 170
MILD_BRIGHT_THRESH = 140

cv.setNumThreads(2)


# ── Live count display (NEW) ─────────────────────────────────────────────────

# Small digit templates for an ASCII-art person counter (0–9)
_DIGITS = {
    "0": ["███", "█ █", "█ █", "█ █", "███"],
    "1": [" █ ", "██ ", " █ ", " █ ", "███"],
    "2": ["███", "  █", "███", "█  ", "███"],
    "3": ["███", "  █", "███", "  █", "███"],
    "4": ["█ █", "█ █", "███", "  █", "  █"],
    "5": ["███", "█  ", "███", "  █", "███"],
    "6": ["███", "█  ", "███", "█ █", "███"],
    "7": ["███", "  █", "  █", "  █", "  █"],
    "8": ["███", "█ █", "███", "█ █", "███"],
    "9": ["███", "█ █", "███", "  █", "███"],
}

def _render_count_banner(n: int) -> str:
    """Return a 5-line ASCII-art banner for person count ``n``."""
    digits = [_DIGITS[c] for c in str(n)]
    lines = []
    for row in range(5):
        lines.append("  ".join(d[row] for d in digits))
    return "\n".join(lines)


_last_printed_count: int = -1   # track changes to avoid redundant redraws

def print_live_count(n_person: int, force: bool = False) -> None:
    """
    Print a large, clearly visible person count to stdout.
    Reprints only when the count changes (or ``force=True``).
    Uses ANSI codes when stdout is a TTY (terminal); plain text otherwise.
    """
    global _last_printed_count
    if n_person == _last_printed_count and not force:
        return
    _last_printed_count = n_person

    banner = _render_count_banner(n_person)

    if sys.stdout.isatty():
        # ANSI: move cursor up 8 lines to overwrite previous banner if already printed
        # On first print there's nothing to overwrite — that's fine.
        GREEN  = "\033[92m"
        YELLOW = "\033[93m"
        RED    = "\033[91m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"
        color  = GREEN if n_person == 0 else (YELLOW if n_person <= 3 else RED)
        label  = "PERSONS DETECTED"
        sep    = "─" * 24
        output = (
            f"\n{BOLD}{color}{sep}{RESET}\n"
            f"{BOLD}{color}  👤 {label}{RESET}\n"
            f"{BOLD}{color}\n"
        )
        for line in banner.splitlines():
            output += f"  {BOLD}{color}{line}{RESET}\n"
        output += f"{BOLD}{color}{sep}{RESET}\n"
        print(output, end="", flush=True)
    else:
        # Plain output for pipes / log files
        print("\n=== PERSONS DETECTED ===", flush=True)
        print(banner, flush=True)
        print("========================", flush=True)


# ── Runtime / config state ───────────────────────────────────────────────────

@dataclass
class RuntimeState:
    img_w: int = 640
    img_h: int = 640
    prebuf: np.ndarray | None = None
    grid_caches: tuple = ()
    gamma_luts: dict = field(default_factory=dict)


@dataclass
class PreprocessConfig:
    clahe_clip: float = 2.4
    clahe_grid: int = 8
    gamma: float = 1.42
    luma_scale: float = 1.06
    dark_threshold: int = 100
    brightness_beta: int = 16
    max_boost: int = 48
    sharpen: bool = True
    denoise: bool = True
    denoise_d: int = 5
    denoise_sigma: float = 30.0


@dataclass
class LetterboxMeta:
    ratio: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int


RT = RuntimeState()
PP = PreprocessConfig()


class GridCache:
    __slots__ = (
        "ax", "ay", "inv_w", "inv_h", "grid_h",
        "scale_conf_mul", "y_center_norm", "spatial_thresh",
    )

    def __init__(self, grid_h: int, grid_w: int, stride: int, img_w: int, img_h: int, scale_conf_mul: float):
        col = np.arange(grid_w, dtype=np.float32)
        row = np.arange(grid_h, dtype=np.float32)
        self.ax = (col + 0.5).reshape(1, grid_w)
        self.ay = (row + 0.5).reshape(grid_h, 1)
        self.inv_w = stride / float(img_w)
        self.inv_h = stride / float(img_h)
        self.grid_h = grid_h
        self.scale_conf_mul = float(scale_conf_mul)
        y = ((row + 0.5) * stride / float(img_h)).reshape(grid_h, 1)
        self.y_center_norm = y.astype(np.float32)
        self.spatial_thresh = self._build_spatial_scale(grid_h, y)

    @staticmethod
    def _build_spatial_scale(grid_h: int, y: np.ndarray) -> np.ndarray:
        t = np.ones((grid_h, 1), dtype=np.float32)
        t = np.where(y >= 0.46, t * 0.82, t)
        if grid_h >= 40:
            t = np.where(y <= 0.34, t * 0.88, t)
            mid = (y >= 0.18) & (y <= 0.42)
            t = np.where(mid, t * 0.90, t)
        return t


def init_runtime(img_w: int, img_h: int) -> None:
    RT.img_w, RT.img_h = int(img_w), int(img_h)
    RT.prebuf = np.empty((3, RT.img_h, RT.img_w), dtype=np.float32)
    RT.grid_caches = tuple(
        GridCache(g, g, s, RT.img_w, RT.img_h, m)
        for g, s, m in zip(GRID_SIZES, STRIDES, SCALE_CONF_MUL)
    )


# ── Utilities ────────────────────────────────────────────────────────────────

class FpsMeter:
    __slots__ = ("interval", "count", "fps", "t0")

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.count = 0
        self.fps = 0.0
        self.t0 = time.perf_counter()

    def tick(self) -> float:
        self.count += 1
        elapsed = time.perf_counter() - self.t0
        if elapsed >= self.interval:
            self.fps = self.count / elapsed
            self.count = 0
            self.t0 = time.perf_counter()
        return self.fps


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def measure_brightness(rgb: np.ndarray) -> float:
    return float(np.mean(cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)))


def to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("empty frame")
    if frame.ndim == 2:
        return cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    if frame.shape[2] == 3:
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if frame.shape[2] == 4:
        return cv.cvtColor(frame, cv.COLOR_BGRA2RGB)
    raise ValueError("unsupported channels: {}".format(frame.shape[2]))


def decode_confidence(display_conf: float, mean_l_in: float) -> float:
    if mean_l_in >= 100:
        return display_conf
    t = max(0.0, min(1.0, (100.0 - mean_l_in) / 50.0))
    return max(DECODE_FLOOR, display_conf - DECODE_MARGIN_DARK * t)


# ── Preprocess ──────────────────────────────────────────────────────────────

def _make_clahe(clip: float, grid: int) -> cv.CLAHE:
    return cv.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))


def _gamma_lut(gamma: float) -> np.ndarray:
    key = round(gamma, 3)
    if key not in RT.gamma_luts:
        inv = 1.0 / gamma
        RT.gamma_luts[key] = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return RT.gamma_luts[key]


def _cap_luma_rgb(rgb: np.ndarray, target: float) -> np.ndarray:
    mean_l = measure_brightness(rgb)
    if mean_l <= target + 2:
        return rgb
    lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab)
    l = np.clip(l.astype(np.float32) * (target / mean_l), 0, 255).astype(np.uint8)
    return cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2RGB)


def enhance_rgb(rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean_in = measure_brightness(rgb)

    if PP.denoise and mean_in < MILD_DARK_THRESH:
        rgb = cv.bilateralFilter(rgb, d=PP.denoise_d,
                                 sigmaColor=PP.denoise_sigma,
                                 sigmaSpace=PP.denoise_sigma)

    lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab)

    if mean_in < DARK_THRESH:
        clahe = _make_clahe(clip=3.0, grid=PP.clahe_grid)
        l = clahe.apply(l)
        boost = int(min(PP.max_boost, (DARK_THRESH - mean_in) * 0.5 + PP.brightness_beta))
        if boost > 0:
            l = cv.add(l, boost)
        l = np.clip(l.astype(np.float32) * PP.luma_scale, 0, 255).astype(np.uint8)
    elif mean_in < MILD_DARK_THRESH:
        clahe = _make_clahe(clip=2.2, grid=PP.clahe_grid)
        l = clahe.apply(l)
        boost = int((MILD_DARK_THRESH - mean_in) * 0.2)
        if boost > 0:
            l = cv.add(l, boost)
        luma_scale = min(PP.luma_scale, 1.06)
        l = np.clip(l.astype(np.float32) * luma_scale, 0, 255).astype(np.uint8)
    elif mean_in > BRIGHT_THRESH:
        scale = 130.0 / float(mean_in)
        l = np.clip(l.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        clahe = _make_clahe(clip=1.5, grid=PP.clahe_grid)
        l = clahe.apply(l)
    elif mean_in > MILD_BRIGHT_THRESH:
        clahe = _make_clahe(clip=1.8, grid=PP.clahe_grid)
        l = clahe.apply(l)
    else:
        clahe = _make_clahe(clip=2.0, grid=PP.clahe_grid)
        l = clahe.apply(l)

    out = cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2RGB)

    if mean_in < MILD_DARK_THRESH:
        gamma = PP.gamma + max(0.0, (MILD_DARK_THRESH - mean_in) * 0.003)
        out = cv.LUT(out, _gamma_lut(gamma))

    if PP.sharpen and mean_in < BRIGHT_THRESH:
        blur = cv.GaussianBlur(out, (0, 0), sigmaX=1.2)
        out = cv.addWeighted(out, 1.4, blur, -0.4, 0)

    cap = 92.0 if mean_in < DARK_THRESH else 96.0
    out = _cap_luma_rgb(out, cap)

    return out, mean_in, measure_brightness(out)


def letterbox(rgb: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
    h, w = rgb.shape[:2]
    tw, th = RT.img_w, RT.img_h
    r = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    resized = cv.resize(rgb, (nw, nh), interpolation=cv.INTER_LINEAR)
    pad_x, pad_y = (tw - nw) // 2, (th - nh) // 2
    out = np.full((th, tw, 3), LETTERBOX_PAD, dtype=np.uint8)
    out[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return out, LetterboxMeta(r, pad_x, pad_y, w, h)


def prepare_frame(bgr: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, LetterboxMeta]:
    rgb = to_rgb(bgr)
    rgb, mean_in, mean_out = enhance_rgb(rgb)
    lettered, meta = letterbox(rgb)
    np.copyto(RT.prebuf, lettered.astype(np.float32).transpose(2, 0, 1) * INV_255)
    return RT.prebuf, mean_in, mean_out, cv.cvtColor(rgb, cv.COLOR_RGB2BGR), meta


# ── Postprocess ─────────────────────────────────────────────────────────────

def decode_scale(raw: np.ndarray, cache: GridCache, conf_decode: float) -> tuple[np.ndarray, np.ndarray]:
    probs = sigmoid(raw[0])
    thresh = conf_decode * cache.scale_conf_mul * cache.spatial_thresh
    mask = probs >= thresh
    if not np.any(mask):
        return np.empty((0, 4), np.float32), np.empty(0, np.float32)

    l, t, r, b = raw[NUM_CLS : NUM_CLS + 4]
    x1 = (cache.ax - l) * cache.inv_w
    y1 = (cache.ay - t) * cache.inv_h
    x2 = (cache.ax + r) * cache.inv_w
    y2 = (cache.ay + b) * cache.inv_h
    boxes = np.stack((x1[mask], y1[mask], x2[mask], y2[mask]), axis=-1).astype(np.float32)
    return boxes, probs[mask].astype(np.float32)


def refine_boxes(boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes, scores
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    ar = h / (w + 1e-6)
    ok = (
        (w > MIN_BOX_W)
        & (h > MIN_BOX_H)
        & (w * h > MIN_BOX_AREA)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (ar > MIN_ASPECT)
        & (ar < MAX_ASPECT)
    )
    return boxes[ok], scores[ok]


def nms_desk_aware(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    cy = (y1 + y2) * 0.5
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[rest] - inter + 1e-6)
        dup = (ovr > iou_thresh) & (np.abs(cy[i] - cy[rest]) < VERT_NMS_SEP)
        order = rest[~dup]
    return np.array(keep, dtype=np.int64)


def suppress_nested(boxes: np.ndarray, scores: np.ndarray, min_score: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(scores)
    if n < 2:
        return boxes, scores
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    order = scores.argsort()[::-1]
    keep = np.ones(n, dtype=bool)
    for ii, i in enumerate(order):
        if not keep[i] or scores[i] < min_score:
            continue
        for j in order[ii + 1 :]:
            if not keep[j] or scores[j] >= min_score:
                continue
            if abs(cy[i] - cy[j]) > 0.045:
                continue
            xx1, yy1 = max(boxes[i, 0], boxes[j, 0]), max(boxes[i, 1], boxes[j, 1])
            xx2, yy2 = min(boxes[i, 2], boxes[j, 2]), min(boxes[i, 3], boxes[j, 3])
            if xx2 <= xx1 or yy2 <= yy1:
                continue
            inter = (xx2 - xx1) * (yy2 - yy1)
            aj = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            if inter / (ai + aj - inter + 1e-6) > 0.38:
                keep[j] = False
    return boxes[keep], scores[keep]


def map_boxes_to_frame(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    tw, th = float(RT.img_w), float(RT.img_h)
    ow, oh = float(meta.orig_w), float(meta.orig_h)
    r, px, py = meta.ratio, float(meta.pad_x), float(meta.pad_y)
    out = boxes.copy()
    out[:, 0] = (boxes[:, 0] * tw - px) / r / ow
    out[:, 2] = (boxes[:, 2] * tw - px) / r / ow
    out[:, 1] = (boxes[:, 1] * th - py) / r / oh
    out[:, 3] = (boxes[:, 3] * th - py) / r / oh
    np.clip(out, 0.0, 1.0, out=out)
    return out


def postprocess(
    outputs: list[np.ndarray],
    conf_decode: float,
    conf_display: float,
    nms_thresh: float,
    meta: LetterboxMeta,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    boxes_list, scores_list = [], []
    for raw, cache in zip(outputs, RT.grid_caches):
        b, s = decode_scale(raw, cache, conf_decode)
        if b.size:
            boxes_list.append(b)
            scores_list.append(s)
    if not boxes_list:
        return None, None

    boxes = np.concatenate(boxes_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    boxes, scores = refine_boxes(boxes, scores)
    if boxes.size == 0:
        return None, None

    keep = nms_desk_aware(boxes, scores, nms_thresh)
    boxes, scores = boxes[keep], scores[keep]
    boxes, scores = suppress_nested(boxes, scores, conf_display)
    boxes = map_boxes_to_frame(boxes, meta)

    ok = scores >= conf_display
    boxes, scores = boxes[ok], scores[ok]
    if boxes.size == 0:
        return None, None
    return boxes, scores


def parse_npu_output(arr: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    expected = LISTSIZE * grid_h * grid_w
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == LISTSIZE:
            return arr
        if arr.shape[-1] == LISTSIZE:
            return arr.transpose(2, 0, 1)
    if arr.size == expected:
        return arr.reshape(LISTSIZE, grid_h, grid_w)
    raise ValueError("unexpected asnn shape {} for {}x{}".format(arr.shape, grid_h, grid_w))


def run_inference(net: asnn, tensor: np.ndarray) -> list[np.ndarray]:
    data = net.nn_inference(
        [tensor],
        platform="ONNX",
        reorder="2 1 0",
        output_tensor=3,
        output_format=output_format.OUT_FORMAT_FLOAT32,
    )
    return [
        parse_npu_output(data[2], GRID_SIZES[0], GRID_SIZES[0]),
        parse_npu_output(data[1], GRID_SIZES[1], GRID_SIZES[1]),
        parse_npu_output(data[0], GRID_SIZES[2], GRID_SIZES[2]),
    ]


# ── RTSP capture ─────────────────────────────────────────────────────────────

class LatestFrameReader:
    """Background thread that always holds the newest RTSP frame."""

    def __init__(self, url: str, transport: str = "tcp"):
        self.url = url
        self.transport = transport
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._stamp = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv.VideoCapture | None = None

    def _open(self) -> cv.VideoCapture:
        proto = "tcp" if self.transport == "tcp" else "udp"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;{}|fflags;nobuffer|flags;low_delay|max_delay;0".format(proto)
        )
        cap = cv.VideoCapture(self.url, cv.CAP_FFMPEG)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> bool:
        self._cap = self._open()
        if not self._cap.isOpened():
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rtsp-reader")
        self._thread.start()
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.5)
                self._cap = self._open()
                continue
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.2)
                if self._cap is not None:
                    self._cap.release()
                self._cap = self._open()
                continue
            with self._lock:
                self._frame = frame
                self._ok = True
                self._stamp += 1

    def get_copy(self) -> tuple[bool, np.ndarray | None, int]:
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None, 0
            return True, self._frame.copy(), self._stamp

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()


# ── Video file source (NEW) ──────────────────────────────────────────────────

class VideoFileReader:
    """
    Sequential frame reader for a local video file.
    Provides the same ``get_copy()`` API as ``LatestFrameReader`` so the
    main loop can treat both sources identically.
    """

    def __init__(self, path: str):
        self.path = path
        self._cap: cv.VideoCapture | None = None
        self._stamp = 0
        self._total_frames: int = 0

    def start(self) -> bool:
        self._cap = cv.VideoCapture(self.path)
        if not self._cap.isOpened():
            return False
        self._total_frames = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))
        log.info(
            "Video file opened: %s  (%d frames, %.1f fps, %dx%d)",
            self.path,
            self._total_frames,
            self._cap.get(cv.CAP_PROP_FPS),
            int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        return True

    def get_copy(self) -> tuple[bool, np.ndarray | None, int]:
        """Read the next frame. Returns (False, None, stamp) at end-of-file."""
        if self._cap is None or not self._cap.isOpened():
            return False, None, self._stamp
        ret, frame = self._cap.read()
        if not ret:
            return False, None, self._stamp   # EOF
        self._stamp += 1
        return True, frame, self._stamp

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv.CAP_PROP_FPS)) if self._cap else 0.0

    @property
    def frame_size(self) -> tuple[int, int]:
        """Returns (width, height)."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()


# ── Video writer helper (NEW) ────────────────────────────────────────────────

def open_video_writer(
    path: str,
    fps: float,
    width: int,
    height: int,
) -> cv.VideoWriter:
    """
    Open a VideoWriter for the given path.
    Codec is chosen from the file extension:
      .mp4 / .m4v  →  mp4v
      .avi         →  XVID
      anything else → mp4v (renamed to .mp4)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".m4v"):
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv.VideoWriter_fourcc(*"XVID")
    else:
        log.warning("Unknown extension '%s' — forcing mp4v codec, renaming output to .mp4", ext)
        path = os.path.splitext(path)[0] + ".mp4"
        fourcc = cv.VideoWriter_fourcc(*"mp4v")

    fps = max(1.0, fps)
    writer = cv.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for path: {}".format(path))
    log.info("Output video: %s  (%dx%d @ %.1f fps)", path, width, height, fps)
    return writer


# ── UI / logging ─────────────────────────────────────────────────────────────

def _brightness_label(mean_in: float) -> str:
    if mean_in < DARK_THRESH:
        return "DARK"
    if mean_in < MILD_DARK_THRESH:
        return "DIM"
    if mean_in > BRIGHT_THRESH:
        return "OVEREXP"
    if mean_in > MILD_BRIGHT_THRESH:
        return "BRIGHT"
    return "NORMAL"


def draw_overlay(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> None:
    h, w = frame.shape[:2]
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        left, top = max(0, int(x1 * w)), max(0, int(y1 * h))
        right, bottom = min(w, int(x2 * w)), min(h, int(y2 * h))
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv.putText(
            frame,
            "person {:.2f}".format(float(score)),
            (left, max(0, top - 4)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    n_person: int,
    cam_fps: float,
    npu_fps: float,
    latency_ms: float,
    conf: float,
    mean_in: float,
    mean_out: float,
) -> None:
    label = _brightness_label(mean_in)
    # ── Large person count in top-right corner (NEW) ──────────────────────────
    count_text = "PERSONS: {}".format(n_person)
    h, w = frame.shape[:2]
    (tw, th), _ = cv.getTextSize(count_text, cv.FONT_HERSHEY_DUPLEX, 1.2, 2)
    cx = w - tw - 12
    cy = 36
    # semi-transparent background rectangle
    overlay = frame.copy()
    cv.rectangle(overlay, (cx - 8, cy - th - 6), (cx + tw + 8, cy + 6), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    color = (0, 255, 0) if n_person == 0 else ((0, 255, 255) if n_person <= 3 else (0, 80, 255))
    cv.putText(frame, count_text, (cx, cy), cv.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv.LINE_AA)

    # ── Standard HUD line at top-left ─────────────────────────────────────────
    cv.putText(
        frame,
        "pers {} | cam {:.0f} | npu {:.0f} | {:.0f}ms | conf {:.2f} | in~{:.0f} out~{:.0f} [{}]".format(
            n_person, cam_fps, npu_fps, latency_ms, conf, mean_in, mean_out, label
        ),
        (8, 24),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )


def print_status(
    n_person: int,
    cam_fps: float,
    npu_fps: float,
    latency_ms: float,
    conf: float,
    mean_in: float,
    mean_out: float,
    headless: bool,
    frame_no: int = 0,
    total_frames: int = 0,
) -> None:
    label = _brightness_label(mean_in)
    progress = ""
    if total_frames > 0:
        pct = 100.0 * frame_no / total_frames
        progress = " | frame {}/{} ({:.1f}%)".format(frame_no, total_frames, pct)
    line = (
        "[{mode}] *** PERSONS={n} *** | fps={cam:.1f} | npu={npu:.1f} | {ms:.0f}ms | "
        "conf={cf:.2f} | luma {i:.0f}→{o:.0f} [{lbl}]{prog}"
    ).format(
        mode="HEADLESS" if headless else "DISPLAY",
        n=n_person,
        cam=cam_fps,
        npu=npu_fps,
        ms=latency_ms,
        cf=conf,
        i=mean_in,
        o=mean_out,
        lbl=label,
        prog=progress,
    )
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K" + line)
    else:
        print(line)
    sys.stdout.flush()


# ── Live JSON ─────────────────────────────────────────────────────────────────

def _persons_list_json(
    boxes: np.ndarray | None,
    scores: np.ndarray | None,
    orig_w: int,
    orig_h: int,
) -> list[dict[str, Any]]:
    if boxes is None or scores is None or len(scores) == 0:
        return []
    ow, oh = float(max(orig_w, 1)), float(max(orig_h, 1))
    out: list[dict[str, Any]] = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = (float(boxes[i, 0]), float(boxes[i, 1]),
                           float(boxes[i, 2]), float(boxes[i, 3]))
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        out.append({
            "index": i,
            "score": float(scores[i]),
            "x1": x1 * ow, "y1": y1 * oh, "x2": x2 * ow, "y2": y2 * oh,
            "x1_norm": x1, "y1_norm": y1, "x2_norm": x2, "y2_norm": y2,
            "center_x": cx * ow, "center_y": cy * oh,
            "center_x_norm": cx, "center_y_norm": cy,
        })
    return out


def build_live_status_payload(
    *,
    source: str,
    seq: int,
    frame: np.ndarray,
    boxes: np.ndarray | None,
    scores: np.ndarray | None,
    conf_threshold: float,
    mean_in: float,
    mean_out: float,
) -> dict[str, Any]:
    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    persons = _persons_list_json(boxes, scores, fw, fh)
    n = len(persons)
    any_person = n > 0
    if any_person:
        best_i = int(np.argmax(scores))
        bx = boxes[best_i]
        best_x1, best_y1 = float(bx[0]), float(bx[1])
        best_x2, best_y2 = float(bx[2]), float(bx[3])
        best_sc = float(scores[best_i])
    else:
        best_x1 = best_y1 = best_x2 = best_y2 = best_sc = 0.0
    best_cx = (best_x1 + best_x2) * 0.5 if any_person else 0.0
    best_cy = (best_y1 + best_y2) * 0.5 if any_person else 0.0
    now = time.time()
    return {
        "updated_unix": now,
        "updated_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "any_person": any_person,
        "person_count": n,
        "conf_threshold": float(conf_threshold),
        "brightness_in": round(mean_in, 1),
        "brightness_out": round(mean_out, 1),
        "scene_condition": _brightness_label(mean_in),
        "cameras": [{
            "camera": 0,
            "source": source,
            "seq": int(seq),
            "frame_width": fw,
            "frame_height": fh,
            "person": any_person,
            "person_count": n,
            "x1_norm": best_x1, "y1_norm": best_y1,
            "x2_norm": best_x2, "y2_norm": best_y2,
            "center_x_norm": best_cx, "center_y_norm": best_cy,
            "score": best_sc,
            "persons": persons,
        }],
    }


def atomic_write_json(path: str, obj: dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = "{}.tmp.{}.{}".format(path, os.getpid(), threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    sys.exit("--imgsz: one value (square) or two: W H")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO26s asnn person detector — RTSP + Video File",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--library", required=True, help="libnn_yolo26s.so path")
    p.add_argument("--model",   required=True, help="yolo26s.nb path")

    # ── Input source (mutually exclusive) ────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--rtsp",  metavar="URL",  help="RTSP stream URL")
    src.add_argument("--video", metavar="FILE", help="Local video file (mp4, avi, …)")

    p.add_argument("--transport", default="tcp", choices=["tcp", "udp"],
                   help="RTSP transport (ignored for --video)")

    # ── Output video (NEW) ───────────────────────────────────────────────────
    p.add_argument("--output", metavar="FILE", default=None,
                   help="Save annotated output video to this file (e.g. out.mp4)")

    p.add_argument("--level",  default="0",          help="asnn performance level")
    p.add_argument("--conf",   type=float,            default=DEFAULT_CONF,
                   help="Person confidence threshold")
    p.add_argument("--nms",    type=float,            default=DEFAULT_NMS,
                   help="NMS IoU threshold")
    p.add_argument("--imgsz",  type=int, nargs="+",   default=list(DEFAULT_IMGSZ), metavar="N")
    p.add_argument("--width",  type=int,              default=960,
                   help="Preview/output width (0 = native)")
    p.add_argument("--headless",    action="store_true")
    p.add_argument("--no-display",  action="store_true")
    p.add_argument("--no-sharpen",  action="store_true")
    p.add_argument("--no-denoise",  action="store_true")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    if not os.path.isfile(args.model):
        sys.exit("model not found: {}".format(args.model))
    if not os.path.isfile(args.library):
        sys.exit("library not found: {}".format(args.library))

    conf = max(0.05, min(0.95, args.conf))
    nms  = max(0.3,  min(0.9,  args.nms))
    img_w, img_h = parse_imgsz(args.imgsz)
    init_runtime(img_w, img_h)

    PP.sharpen = not args.no_sharpen
    PP.denoise = not args.no_denoise

    level   = int(args.level) if args.level in ("1", "2") else 0
    headless = args.headless or args.no_display
    is_video = args.video is not None
    source   = args.video if is_video else args.rtsp

    log.info(
        "Init  source=%s  conf=%.2f  nms=%.2f  size=%dx%d  sharpen=%s  denoise=%s  live_json=%s",
        source, conf, nms, img_w, img_h, PP.sharpen, PP.denoise, PERSON_LIVE_JSON,
    )

    net = asnn("Electron")
    net.nn_init(library=args.library, model=args.model, level=level)

    # ── Open source ───────────────────────────────────────────────────────────
    if is_video:
        reader = VideoFileReader(source)
    else:
        reader = LatestFrameReader(source, args.transport)

    if not reader.start():
        sys.exit("Cannot open source: {}".format(source))

    # ── Determine output frame size for writer / display ──────────────────────
    if is_video:
        src_w, src_h = reader.frame_size          # type: ignore[union-attr]
        src_fps      = reader.fps                  # type: ignore[union-attr]
        total_frames = reader.total_frames         # type: ignore[union-attr]
    else:
        src_w, src_h = 960, 540   # unknown until first frame; use fallback
        src_fps      = 25.0
        total_frames = 0

    # Resolve display/output width
    out_w = args.width if args.width > 0 else src_w
    out_h = int(src_h * out_w / src_w) if src_w > 0 else src_h

    # ── Open video writer (optional) ──────────────────────────────────────────
    writer: cv.VideoWriter | None = None
    if args.output:
        writer = open_video_writer(args.output, src_fps, out_w, out_h)

    # ── Display window ────────────────────────────────────────────────────────
    win = "YOLO26 Person Detection"
    if not headless:
        cv.namedWindow(win, cv.WINDOW_NORMAL)

    cam_fps_m, npu_fps_m = FpsMeter(), FpsMeter()
    cam_fps = npu_fps = 0.0
    last_stamp = -1
    frame_no   = 0

    # Print initial count banner
    print_live_count(0, force=True)

    try:
        while True:
            ok, frame, stamp = reader.get_copy()

            # EOF for video files
            if not ok:
                if is_video:
                    log.info("End of video file reached (%d frames processed).", frame_no)
                    break
                # RTSP: wait for next frame
                time.sleep(0.005)
                continue

            if frame is None:
                time.sleep(0.005)
                continue

            frame_no += 1

            if stamp != last_stamp:
                cam_fps    = cam_fps_m.tick()
                last_stamp = stamp

            tensor, mean_in, mean_out, preview, meta = prepare_frame(frame)
            conf_decode = decode_confidence(conf, mean_in)

            t0 = time.perf_counter()
            outputs = run_inference(net, tensor)
            boxes, scores = postprocess(outputs, conf_decode, conf, nms, meta)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            npu_fps    = npu_fps_m.tick()

            n_person = 0 if boxes is None else len(boxes)

            # ── Live count banner ─────────────────────────────────────────────
            print_live_count(n_person)

            # ── Inline status line ────────────────────────────────────────────
            print_status(
                n_person, cam_fps, npu_fps, latency_ms, conf, mean_in, mean_out,
                headless, frame_no, total_frames,
            )

            # ── Live JSON ─────────────────────────────────────────────────────
            try:
                payload = build_live_status_payload(
                    source=source, seq=stamp, frame=frame,
                    boxes=boxes, scores=scores,
                    conf_threshold=conf, mean_in=mean_in, mean_out=mean_out,
                )
                atomic_write_json(PERSON_LIVE_JSON, payload)
            except Exception:
                log.exception("Failed writing %s", PERSON_LIVE_JSON)

            # ── Build annotated frame ─────────────────────────────────────────
            if not headless or writer is not None:
                show = preview.copy()
                if boxes is not None:
                    draw_overlay(show, boxes, scores)
                draw_hud(show, n_person, cam_fps, npu_fps, latency_ms, conf, mean_in, mean_out)

                # Resize to target display/output resolution
                if args.width > 0:
                    fh, fw = show.shape[:2]
                    if fw != args.width:
                        show = cv.resize(
                            show,
                            (args.width, int(fh * args.width / fw)),
                            interpolation=cv.INTER_LINEAR,
                        )

                # ── Write to output video ─────────────────────────────────────
                if writer is not None:
                    # Re-check size on first real frame (RTSP size may differ from fallback)
                    if not is_video and frame_no == 1:
                        fh2, fw2 = show.shape[:2]
                        if fw2 != out_w or fh2 != out_h:
                            log.info(
                                "Re-opening writer for actual size %dx%d", fw2, fh2
                            )
                            writer.release()
                            writer = open_video_writer(args.output, src_fps, fw2, fh2)
                    writer.write(show)

                # ── Display window ────────────────────────────────────────────
                if not headless:
                    cv.imshow(win, show)
                    key = cv.waitKey(1 if not is_video else max(1, int(1000 / src_fps)))
                    if key & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        print("", flush=True)
        reader.stop()
        if writer is not None:
            writer.release()
            log.info("Output video saved: %s", args.output)
        if not headless:
            cv.destroyAllWindows()
        log.info("Done. Frames processed: %d", frame_no)


if __name__ == "__main__":
    init_runtime(*DEFAULT_IMGSZ)
    main()
