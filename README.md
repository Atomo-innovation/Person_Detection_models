# RTSP, headless, save output
python detection.py --library libnn.so --model yolo26s.nb \
    --rtsp rtsp://cam/stream --headless --output out.mp4

# Local video file, display + save
python detection.py --library libnn.so --model yolo26s.nb \
    --video input.mp4 --output detected.mp4 --conf 0.40
