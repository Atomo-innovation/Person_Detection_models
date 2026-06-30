#!/usr/bin/env python3
"""
YOLO26s-family multi-model comparison on Khadas Electron (asnn).

Runs the SAME video (or the SAME live RTSP stream, captured for N seconds
per model) through up to 3 models (shared output geometry: grid 20/40/80,
strides 32/16/8, 640x640 letterbox) back-to-back, and for each model saves:
  - an annotated output video (boxes + live HUD)
  - per-model metrics (avg/median FPS, avg/p95 latency, detection stats)

At the end, prints + saves a comparison table/report so you can decide
which model is best for person detection.

── Usage A: video file (full length, all 3 models) ─────────────────────────

  python yolo26_compare.py \\
      --video input.mp4 \\
      --model-a libA.so modelA.nb "ModelA" \\
      --model-b libB.so modelB.nb "ModelB" \\
      --model-c libC.so modelC.nb "ModelC" \\
      --outdir results/ \\
      --conf 0.34

── Usage B: RTSP stream, 20 seconds captured per model ──────────────────────

  python yolo26_compare.py \\
      --rtsp rtsp://camera-ip/stream \\
      --rtsp-duration 20 \\
      --model-a libA.so modelA.nb "ModelA" \\
      --model-b libB.so modelB.nb "ModelB" \\
      --model-c libC.so modelC.nb "ModelC" \\
      --outdir results/ \\
      --conf 0.34

  NOTE: each model captures its OWN 20-second window from the live stream
  (back-to-back, not simultaneous) since only one model runs on the NPU at
  a time. The windows are therefore consecutive, not identical, in wall-clock
  time. For a true apples-to-apples comparison, record once with --video-only
  capture first (e.g. ffmpeg) and pass that file via --video to all 3 models.

Each --model-X takes exactly 3 values: LIBRARY_PATH  MODEL_PATH  NAME

Outputs (under --outdir):
  results/ModelA_annotated.mp4
  results/ModelB_annotated.mp4
  results/ModelC_annotated.mp4
  results/ModelA_metrics.json
  results/ModelB_metrics.json
  results/ModelC_metrics.json
  results/comparison_report.json
  results/comparison_report.txt   <- human-readable, read this first
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2 as cv
import numpy as np
from asnn.api import asnn
from asnn.types import output_format

log = logging.getLogger("yolo26_compare")

# ── Model geometry (shared across all compared models) ─────────────────────
GRID_SIZES = (20, 40, 80)
STRIDES = (32, 16, 8)
LISTSIZE = 5
NUM_CLS = 1
SCALE_CONF_MUL = (1.0, 0.92, 0.78)

INV_255 = 1.0 / 255.0
LETTERBOX_PAD = 114

DEFAULT_CONF = 0.34
DEFAULT_NMS = 0.56
DEFAULT_IMGSZ = (640, 640)

DECODE_MARGIN_DARK = 0.06
DECODE_FLOOR = 0.26

MIN_BOX_AREA = 0.00008
MIN_BOX_W = 0.004
MIN_BOX_H = 0.006
MIN_ASPECT = 0.35
MAX_ASPECT = 5.5
VERT_NMS_SEP = 0.055

DARK_THRESH        = 60
MILD_DARK_THRESH   = 90
BRIGHT_THRESH      = 170
MILD_BRIGHT_THRESH = 140

cv.setNumThreads(2)


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


# ── Drawing ──────────────────────────────────────────────────────────────────

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
            frame, "person {:.2f}".format(float(score)),
            (left, max(0, top - 4)),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA,
        )


def draw_hud(frame: np.ndarray, model_name: str, n_person: int, fps: float,
             latency_ms: float, conf: float, mean_in: float, mean_out: float,
             frame_no: int, total_frames: int) -> None:
    h, w = frame.shape[:2]
    label = _brightness_label(mean_in)

    # Model name banner (top-left, distinct color band)
    cv.rectangle(frame, (0, 0), (w, 30), (40, 40, 40), -1)
    cv.putText(frame, "MODEL: {}".format(model_name), (8, 21),
               cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Large person count, top-right
    count_text = "PERSONS: {}".format(n_person)
    (tw, th), _ = cv.getTextSize(count_text, cv.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cx, cy = w - tw - 12, 60
    overlay = frame.copy()
    cv.rectangle(overlay, (cx - 8, cy - th - 6), (cx + tw + 8, cy + 6), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    color = (0, 255, 0) if n_person == 0 else ((0, 255, 255) if n_person <= 3 else (0, 80, 255))
    cv.putText(frame, count_text, (cx, cy), cv.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv.LINE_AA)

    # Stats line
    progress = " {}/{}".format(frame_no, total_frames) if total_frames else ""
    cv.putText(
        frame,
        "fps {:.1f} | {:.0f}ms | conf {:.2f} | in~{:.0f} out~{:.0f} [{}]{}".format(
            fps, latency_ms, conf, mean_in, mean_out, label, progress
        ),
        (8, h - 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA,
    )


# ── Video I/O ────────────────────────────────────────────────────────────────

class VideoFileReader:
    def __init__(self, path: str):
        self.path = path
        self._cap: cv.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv.VideoCapture(self.path)
        return self._cap.isOpened()

    def read(self):
        return self._cap.read()

    @property
    def fps(self) -> float:
        f = float(self._cap.get(cv.CAP_PROP_FPS)) if self._cap else 0.0
        return f if f > 1 else 25.0

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class RTSPDurationReader:
    """
    Reads frames directly (no background thread) from an RTSP stream for use
    in the per-model comparison loop. Frames are consumed live — there's no
    "total_frames" upfront, so the caller stops after a wall-clock duration.
    """

    def __init__(self, url: str, transport: str = "tcp"):
        self.url = url
        self.transport = transport
        self._cap: cv.VideoCapture | None = None

    def open(self) -> bool:
        proto = "tcp" if self.transport == "tcp" else "udp"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;{}|fflags;nobuffer|flags;low_delay|max_delay;0".format(proto)
        )
        self._cap = cv.VideoCapture(self.url, cv.CAP_FFMPEG)
        self._cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        return self._cap.isOpened()

    def read(self):
        return self._cap.read()

    @property
    def fps(self) -> float:
        # RTSP-reported FPS is often wrong/0; caller should prefer measured FPS.
        f = float(self._cap.get(cv.CAP_PROP_FPS)) if self._cap else 0.0
        return f if f > 1 else 25.0

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


def open_video_writer(path: str, fps: float, width: int, height: int) -> cv.VideoWriter:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".m4v"):
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv.VideoWriter_fourcc(*"XVID")
    else:
        path = os.path.splitext(path)[0] + ".mp4"
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(path, fourcc, max(1.0, fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for: {}".format(path))
    return writer


# ── Per-model evaluation ──────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    name: str
    library: str
    model_path: str
    frames_processed: int = 0
    frames_with_person: int = 0
    total_person_detections: int = 0     # sum over all frames
    max_persons_single_frame: int = 0
    person_counts: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)   # all detection confidences
    latencies_ms: list[float] = field(default_factory=list)
    fps_samples: list[float] = field(default_factory=list)
    init_time_s: float = 0.0
    total_runtime_s: float = 0.0
    output_video: str = ""

    def summary(self) -> dict[str, Any]:
        n = max(1, self.frames_processed)
        det_rate = self.frames_with_person / n
        avg_lat = statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0
        p95_lat = (sorted(self.latencies_ms)[int(0.95 * len(self.latencies_ms)) - 1]
                   if self.latencies_ms else 0.0)
        avg_fps = statistics.mean(self.fps_samples) if self.fps_samples else 0.0
        avg_score = statistics.mean(self.scores) if self.scores else 0.0
        med_score = statistics.median(self.scores) if self.scores else 0.0
        avg_persons = statistics.mean(self.person_counts) if self.person_counts else 0.0
        # Stability: how much person-count flickers frame to frame (lower = steadier)
        flicker = 0.0
        if len(self.person_counts) > 1:
            diffs = [abs(self.person_counts[i] - self.person_counts[i - 1])
                     for i in range(1, len(self.person_counts))]
            flicker = statistics.mean(diffs)
        return {
            "name": self.name,
            "library": self.library,
            "model_path": self.model_path,
            "output_video": self.output_video,
            "frames_processed": self.frames_processed,
            "frames_with_person": self.frames_with_person,
            "detection_rate_pct": round(det_rate * 100, 2),
            "total_person_detections": self.total_person_detections,
            "avg_persons_per_frame": round(avg_persons, 3),
            "max_persons_single_frame": self.max_persons_single_frame,
            "avg_confidence": round(avg_score, 4),
            "median_confidence": round(med_score, 4),
            "avg_inference_latency_ms": round(avg_lat, 2),
            "p95_inference_latency_ms": round(p95_lat, 2),
            "avg_fps": round(avg_fps, 2),
            "person_count_flicker": round(flicker, 3),
            "init_time_s": round(self.init_time_s, 2),
            "total_runtime_s": round(self.total_runtime_s, 2),
        }


def run_one_model(
    *,
    library: str,
    model_path: str,
    name: str,
    video_path: str | None,
    rtsp_url: str | None,
    rtsp_transport: str,
    rtsp_duration_s: float,
    outdir: str,
    conf: float,
    nms: float,
    img_w: int,
    img_h: int,
    display_width: int,
    show_window: bool,
) -> ModelMetrics:
    """
    Run a single model over either:
      - a video file (full length), or
      - an RTSP stream for exactly ``rtsp_duration_s`` seconds (wall clock,
        starting from when the stream is opened for THIS model).
    Returns its metrics.
    """
    is_rtsp = rtsp_url is not None
    source_desc = rtsp_url if is_rtsp else video_path

    log.info("=" * 70)
    log.info("Running model: %s", name)
    log.info("  library = %s", library)
    log.info("  model   = %s", model_path)
    log.info("  source  = %s%s", source_desc,
              "  (RTSP, {:.0f}s capture)".format(rtsp_duration_s) if is_rtsp else "")
    log.info("=" * 70)

    metrics = ModelMetrics(name=name, library=library, model_path=model_path)

    # Fresh runtime geometry (safe to reuse across models since geometry is identical,
    # but re-init guarantees a clean prebuf/grid cache state per run).
    init_runtime(img_w, img_h)

    t_init0 = time.perf_counter()
    net = asnn("Electron")
    net.nn_init(library=library, model=model_path, level=0)
    metrics.init_time_s = time.perf_counter() - t_init0

    # ── Open source ───────────────────────────────────────────────────────────
    reader: VideoFileReader | RTSPDurationReader
    if is_rtsp:
        reader = RTSPDurationReader(rtsp_url, rtsp_transport)
        if not reader.open():
            raise RuntimeError("Cannot open RTSP stream: {}".format(rtsp_url))
        # Drain a couple of frames — first frames after connect are often stale/garbage
        for _ in range(2):
            reader.read()
    else:
        reader = VideoFileReader(video_path)
        if not reader.open():
            raise RuntimeError("Cannot open video: {}".format(video_path))

    src_w, src_h = reader.frame_size
    total_frames = 0 if is_rtsp else reader.total_frames  # type: ignore[union-attr]

    out_w = display_width if display_width > 0 else src_w
    out_h = int(src_h * out_w / src_w) if src_w > 0 else src_h

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    out_path = os.path.join(outdir, "{}_annotated.mp4".format(safe_name))

    # For RTSP we don't know true FPS until we've measured it; open the writer with
    # a reasonable placeholder, then rebuild it once measured FPS stabilizes.
    writer_fps = reader.fps  # placeholder (file: real value; rtsp: often inaccurate)
    writer = open_video_writer(out_path, writer_fps, out_w, out_h)
    metrics.output_video = out_path

    win = "Comparing: {}".format(name)
    if show_window:
        cv.namedWindow(win, cv.WINDOW_NORMAL)

    frame_no = 0
    t_run0 = time.perf_counter()
    last_tick = time.perf_counter()
    capture_deadline = (t_run0 + rtsp_duration_s) if is_rtsp else None

    # Buffer frames if we need to rebuild the writer with corrected RTSP fps
    rtsp_fps_locked = not is_rtsp  # file sources: fps already correct, no relock needed
    pending_frames_for_relock: list[np.ndarray] = []

    try:
        while True:
            if is_rtsp and time.perf_counter() >= capture_deadline:
                log.info("RTSP %.0fs capture window elapsed for %s.", rtsp_duration_s, name)
                break

            ret, frame = reader.read()
            if not ret:
                if is_rtsp:
                    # Transient RTSP read failure — brief retry within the time window
                    time.sleep(0.01)
                    continue
                break  # video file EOF

            frame_no += 1

            tensor, mean_in, mean_out, preview, meta = prepare_frame(frame)
            conf_decode = decode_confidence(conf, mean_in)

            t0 = time.perf_counter()
            outputs = run_inference(net, tensor)
            boxes, scores = postprocess(outputs, conf_decode, conf, nms, meta)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, now - last_tick)
            last_tick = now

            n_person = 0 if boxes is None else len(boxes)

            metrics.frames_processed += 1
            metrics.person_counts.append(n_person)
            metrics.latencies_ms.append(latency_ms)
            metrics.fps_samples.append(inst_fps)
            if n_person > 0:
                metrics.frames_with_person += 1
                metrics.total_person_detections += n_person
                metrics.max_persons_single_frame = max(metrics.max_persons_single_frame, n_person)
                metrics.scores.extend(float(s) for s in scores)

            show = preview.copy()
            if boxes is not None:
                draw_overlay(show, boxes, scores)
            draw_hud(show, name, n_person, inst_fps, latency_ms, conf,
                     mean_in, mean_out, frame_no, total_frames)

            if display_width > 0:
                fh, fw = show.shape[:2]
                if fw != display_width:
                    show = cv.resize(show, (display_width, int(fh * display_width / fw)),
                                      interpolation=cv.INTER_LINEAR)

            # ── RTSP fps relock: after ~15 frames we have a stable measured fps.
            # Rebuild the writer once with the correct fps so the saved clip's
            # playback speed matches real time, replaying buffered frames first.
            if is_rtsp and not rtsp_fps_locked:
                pending_frames_for_relock.append(show)
                if len(metrics.fps_samples) >= 15:
                    measured_fps = statistics.median(metrics.fps_samples[-15:])
                    measured_fps = max(1.0, min(60.0, measured_fps))
                    writer.release()
                    writer = open_video_writer(out_path, measured_fps, out_w, out_h)
                    for buffered in pending_frames_for_relock:
                        writer.write(buffered)
                    pending_frames_for_relock = []
                    rtsp_fps_locked = True
                    log.info("RTSP measured fps locked at %.2f for %s", measured_fps, name)
            else:
                writer.write(show)

            if show_window:
                cv.imshow(win, show)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    log.info("User requested skip to next model.")
                    break

            if frame_no % 25 == 0 or (not is_rtsp and frame_no == total_frames):
                if is_rtsp:
                    remaining = max(0.0, capture_deadline - time.perf_counter())
                    sys.stdout.write(
                        "\r\033[K[{}] RTSP capturing | {:.1f}s left | frames={} | persons now={} | "
                        "fps={:.1f} | {:.0f}ms".format(
                            name, remaining, frame_no, n_person, inst_fps, latency_ms
                        )
                    )
                else:
                    pct = (100.0 * frame_no / total_frames) if total_frames else 0.0
                    sys.stdout.write(
                        "\r\033[K[{}] frame {}/{} ({:.1f}%) | persons now={} | fps={:.1f} | {:.0f}ms".format(
                            name, frame_no, total_frames, pct, n_person, inst_fps, latency_ms
                        )
                    )
                sys.stdout.flush()

    finally:
        print("")
        metrics.total_runtime_s = time.perf_counter() - t_run0
        reader.release()
        # Flush any still-pending buffered frames if relock never triggered
        # (e.g. very short/sparse capture) — write them at best-guess fps.
        if is_rtsp and not rtsp_fps_locked and pending_frames_for_relock:
            for buffered in pending_frames_for_relock:
                writer.write(buffered)
        writer.release()
        if show_window:
            cv.destroyWindow(win)

    log.info(
        "Done %-12s | frames=%d | det_rate=%.1f%% | avg_fps=%.2f | avg_latency=%.1fms | output=%s",
        name, metrics.frames_processed,
        100.0 * metrics.frames_with_person / max(1, metrics.frames_processed),
        statistics.mean(metrics.fps_samples) if metrics.fps_samples else 0.0,
        statistics.mean(metrics.latencies_ms) if metrics.latencies_ms else 0.0,
        out_path,
    )
    return metrics


# ── Comparison / verdict ──────────────────────────────────────────────────────

def score_model(summary: dict[str, Any]) -> float:
    """
    Composite score for ranking models on PERSON DETECTION QUALITY (not just speed).
    Weighted toward detection reliability and confidence, with FPS as a tiebreaker.

      +  detection_rate_pct        (higher = catches people more consistently)
      +  avg_confidence  * 100     (higher = more certain detections)
      -  person_count_flicker * 20 (lower = more stable/consistent across frames)
      +  avg_fps * 0.5             (higher = faster, minor weight)
    """
    return (
        summary["detection_rate_pct"] * 1.0
        + summary["avg_confidence"] * 100.0
        - summary["person_count_flicker"] * 20.0
        + summary["avg_fps"] * 0.5
    )


def build_comparison_report(all_metrics: list[ModelMetrics]) -> dict[str, Any]:
    summaries = [m.summary() for m in all_metrics]
    for s in summaries:
        s["composite_score"] = round(score_model(s), 3)
    ranked = sorted(summaries, key=lambda s: s["composite_score"], reverse=True)
    best = ranked[0] if ranked else None
    return {
        "generated_unix": time.time(),
        "generated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": summaries,
        "ranking_best_to_worst": [s["name"] for s in ranked],
        "recommended_model": best["name"] if best else None,
        "scoring_formula": (
            "detection_rate_pct*1.0 + avg_confidence*100 "
            "- person_count_flicker*20 + avg_fps*0.5"
        ),
    }


def format_text_report(report: dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("YOLO MODEL COMPARISON REPORT — PERSON DETECTION")
    lines.append("Generated: {}".format(report["generated_local"]))
    lines.append("=" * 78)
    lines.append("")

    header = "{:<14}{:>10}{:>10}{:>10}{:>10}{:>10}{:>12}".format(
        "Model", "DetRate%", "AvgConf", "Flicker", "AvgFPS", "AvgMs", "Score"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in sorted(report["models"], key=lambda x: x["composite_score"], reverse=True):
        lines.append("{:<14}{:>10.2f}{:>10.4f}{:>10.3f}{:>10.2f}{:>10.2f}{:>12.3f}".format(
            s["name"][:13], s["detection_rate_pct"], s["avg_confidence"],
            s["person_count_flicker"], s["avg_fps"],
            s["avg_inference_latency_ms"], s["composite_score"],
        ))
    lines.append("")
    lines.append("Detail per model:")
    lines.append("-" * 78)
    for s in report["models"]:
        lines.append("")
        lines.append("[{}]".format(s["name"]))
        lines.append("  library              : {}".format(s["library"]))
        lines.append("  model file           : {}".format(s["model_path"]))
        lines.append("  output video         : {}".format(s["output_video"]))
        lines.append("  frames processed     : {}".format(s["frames_processed"]))
        lines.append("  frames with person   : {} ({:.2f}% detection rate)".format(
            s["frames_with_person"], s["detection_rate_pct"]))
        lines.append("  total detections     : {}".format(s["total_person_detections"]))
        lines.append("  avg persons/frame    : {}".format(s["avg_persons_per_frame"]))
        lines.append("  max persons (1 frame): {}".format(s["max_persons_single_frame"]))
        lines.append("  avg confidence       : {}".format(s["avg_confidence"]))
        lines.append("  median confidence    : {}".format(s["median_confidence"]))
        lines.append("  count flicker (lower better): {}".format(s["person_count_flicker"]))
        lines.append("  avg inference latency: {} ms".format(s["avg_inference_latency_ms"]))
        lines.append("  p95 inference latency: {} ms".format(s["p95_inference_latency_ms"]))
        lines.append("  avg fps              : {}".format(s["avg_fps"]))
        lines.append("  model init time      : {} s".format(s["init_time_s"]))
        lines.append("  total runtime        : {} s".format(s["total_runtime_s"]))
        lines.append("  composite score      : {}".format(s["composite_score"]))

    lines.append("")
    lines.append("=" * 78)
    lines.append("RANKING (best → worst): {}".format(" > ".join(report["ranking_best_to_worst"])))
    lines.append("")
    lines.append(">>> RECOMMENDED MODEL FOR PERSON DETECTION: {} <<<".format(report["recommended_model"]))
    lines.append("=" * 78)
    lines.append("")
    lines.append("How to read this:")
    lines.append("  - DetRate%% : % of frames where at least one person was detected.")
    lines.append("                Higher is better IF your video actually has people in most frames.")
    lines.append("  - AvgConf  : average confidence of detections. Higher = more certain model.")
    lines.append("  - Flicker  : avg frame-to-frame change in person count. Lower = more stable")
    lines.append("                (fewer false on/off flickers on the same people).")
    lines.append("  - AvgFPS / AvgMs : speed on this hardware. Useful as a tiebreaker.")
    lines.append("  - Score    : weighted composite — higher is better overall.")
    lines.append("")
    lines.append("IMPORTANT: open the annotated output videos and visually confirm —")
    lines.append("metrics can be skewed by false positives. The recommendation above")
    lines.append("is a strong starting point, not a substitute for a quick visual check.")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def model_triplet(values: list[str]) -> tuple[str, str, str]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "expected exactly 3 values: LIBRARY_PATH MODEL_PATH NAME"
        )
    return values[0], values[1], values[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare up to 3 YOLO26-family models on the same video/RTSP feed for person detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Local video file to test all models on")
    src.add_argument("--rtsp", metavar="URL", help="RTSP stream URL — captures N seconds per model")

    p.add_argument("--rtsp-duration", type=float, default=20.0,
                   help="Seconds of RTSP stream to capture PER MODEL (only used with --rtsp)")
    p.add_argument("--rtsp-transport", default="tcp", choices=["tcp", "udp"])

    p.add_argument("--model-a", nargs=3, metavar=("LIBRARY", "MODEL", "NAME"), required=True)
    p.add_argument("--model-b", nargs=3, metavar=("LIBRARY", "MODEL", "NAME"), required=False)
    p.add_argument("--model-c", nargs=3, metavar=("LIBRARY", "MODEL", "NAME"), required=False)
    p.add_argument("--outdir", default="comparison_results", help="Directory for all outputs")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--nms", type=float, default=DEFAULT_NMS)
    p.add_argument("--imgsz", type=int, nargs="+", default=list(DEFAULT_IMGSZ), metavar="N")
    p.add_argument("--width", type=int, default=960, help="Output video width (0 = native)")
    p.add_argument("--show", action="store_true", help="Show a live preview window while running")
    p.add_argument("--no-sharpen", action="store_true")
    p.add_argument("--no-denoise", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def parse_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    sys.exit("--imgsz: one value (square) or two: W H")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    is_rtsp = args.rtsp is not None
    if not is_rtsp and not os.path.isfile(args.video):
        sys.exit("video not found: {}".format(args.video))

    model_specs = [args.model_a]
    if args.model_b:
        model_specs.append(args.model_b)
    if args.model_c:
        model_specs.append(args.model_c)

    for lib, mdl, name in model_specs:
        if not os.path.isfile(lib):
            sys.exit("library not found for {}: {}".format(name, lib))
        if not os.path.isfile(mdl):
            sys.exit("model file not found for {}: {}".format(name, mdl))

    names = [s[2] for s in model_specs]
    if len(set(names)) != len(names):
        sys.exit("Model names must be unique (--model-a/-b/-c third value).")

    os.makedirs(args.outdir, exist_ok=True)

    conf = max(0.05, min(0.95, args.conf))
    nms = max(0.3, min(0.9, args.nms))
    img_w, img_h = parse_imgsz(args.imgsz)

    PP.sharpen = not args.no_sharpen
    PP.denoise = not args.no_denoise

    log.info("Comparing %d model(s) on %s", len(model_specs),
              "RTSP {} ({:.0f}s per model)".format(args.rtsp, args.rtsp_duration)
              if is_rtsp else "video: {}".format(args.video))
    log.info("Output directory: %s", os.path.abspath(args.outdir))

    all_metrics: list[ModelMetrics] = []
    for lib, mdl, name in model_specs:
        m = run_one_model(
            library=lib, model_path=mdl, name=name,
            video_path=None if is_rtsp else args.video,
            rtsp_url=args.rtsp if is_rtsp else None,
            rtsp_transport=args.rtsp_transport,
            rtsp_duration_s=args.rtsp_duration,
            outdir=args.outdir,
            conf=conf, nms=nms, img_w=img_w, img_h=img_h,
            display_width=args.width, show_window=args.show,
        )
        all_metrics.append(m)

        # Save per-model metrics immediately (so partial runs aren't lost)
        per_model_path = os.path.join(
            args.outdir,
            "{}_metrics.json".format("".join(c if c.isalnum() or c in "-_" else "_" for c in name)),
        )
        with open(per_model_path, "w", encoding="utf-8") as f:
            json.dump(m.summary(), f, indent=2)

    if args.show:
        cv.destroyAllWindows()

    report = build_comparison_report(all_metrics)
    report_json_path = os.path.join(args.outdir, "comparison_report.json")
    report_txt_path = os.path.join(args.outdir, "comparison_report.txt")

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    text_report = format_text_report(report)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)

    print("\n" + text_report)
    log.info("Comparison complete. Reports saved to: %s", args.outdir)


if __name__ == "__main__":
    main()
