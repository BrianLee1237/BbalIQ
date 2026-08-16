import sys
import cv2
from ultralytics import YOLO

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
model = YOLO("models/court_keypoints.pt")

capture = cv2.VideoCapture(video_path)
ok, frame = capture.read()
capture.release()
if not ok:
    raise SystemExit(f"Could not read a frame from {video_path}")

result = model(frame, verbose=False)[0]
print("num boxes:", 0 if result.boxes is None else len(result.boxes))
if result.boxes is not None and len(result.boxes) > 0:
    print("box confidences:", result.boxes.conf.tolist())

if result.keypoints is not None:
    print("keypoints xy shape:", result.keypoints.xy.shape)
    print("keypoints xy:", result.keypoints.xy.tolist())
    if result.keypoints.conf is not None:
        print("keypoints conf:", result.keypoints.conf.tolist())
    else:
        print("keypoints conf: None (model has no per-keypoint confidence output)")
else:
    print("keypoints: None")
