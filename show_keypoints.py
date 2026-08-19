"""Draws the raw detected court keypoints on a single frame -- no
homography, no on-court filtering, just what the model itself sees.
Each point is numbered (0-17, matching KEYPOINT_MODEL_COURT_POINTS_FT's
order) and colored by confidence: green >= 0.5, yellow >= 0.25, red below.

Usage: python3 show_keypoints.py <video>
Output: keypoint_detection.png
"""
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
out = frame.copy()

if result.boxes is not None and len(result.boxes) > 0:
    box_conf = float(result.boxes.conf[0])
    x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].tolist())
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(out, f"detected instance conf={box_conf:.2f}", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

if result.keypoints is not None:
    xy = result.keypoints.xy[0].tolist()
    conf = result.keypoints.conf[0].tolist() if result.keypoints.conf is not None else [1.0] * len(xy)
    for i, ((x, y), c) in enumerate(zip(xy, conf)):
        color = (0, 255, 0) if c >= 0.5 else (0, 255, 255) if c >= 0.25 else (0, 0, 255)
        cv2.circle(out, (int(x), int(y)), 8, color, -1)
        cv2.putText(out, f"{i}:{c:.2f}", (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    print(f"Detected {len(xy)} keypoints.")
    for i, ((x, y), c) in enumerate(zip(xy, conf)):
        print(f"  {i}: pixel=({x:.0f},{y:.0f}) conf={c:.2f}")
else:
    print("No keypoints detected at all.")

cv2.imwrite("keypoint_detection.png", out)
print("\nWrote keypoint_detection.png -- green=confident, yellow=medium, red=low confidence.")
