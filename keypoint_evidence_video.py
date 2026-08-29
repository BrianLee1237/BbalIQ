"""Runs the trained court-keypoint model on every frame of a video (not
just frame 0) and renders what it actually detects -- box + keypoints if
found, a red "NO DETECTION" banner if not -- so detection rate over a real
clip can be seen directly instead of inferred from a single frame.

No changes to the model or the input frames -- this only visualizes what
the model already outputs, at whatever confidence it outputs.

Usage: python3 keypoint_evidence_video.py <video> [num_frames] [conf]
Output: keypoint_evidence.mp4
"""
import sys
import cv2
from ultralytics import YOLO

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300
conf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

model = YOLO("models/court_keypoints.pt")

capture = cv2.VideoCapture(video_path)
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("keypoint_evidence.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_idx = 0
frames_with_detection = 0
frames_with_full_18 = 0
best_frame_conf = 0.0
best_frame_idx = -1

while frame_idx < num_frames:
    ok, frame = capture.read()
    if not ok:
        break

    result = model(frame, conf=conf_threshold, verbose=False)[0]
    out = frame.copy()

    has_detection = result.keypoints is not None and len(result.keypoints.xy) > 0
    if has_detection:
        frames_with_detection += 1
        xy = result.keypoints.xy[0].tolist()
        conf = result.keypoints.conf[0].tolist() if result.keypoints.conf is not None else [1.0] * len(xy)
        if len(xy) == 18:
            frames_with_full_18 += 1
        mean_conf = sum(conf) / len(conf) if conf else 0.0
        if mean_conf > best_frame_conf:
            best_frame_conf = mean_conf
            best_frame_idx = frame_idx

        box_conf = float(result.boxes.conf[0]) if result.boxes is not None and len(result.boxes) > 0 else 0.0
        if result.boxes is not None and len(result.boxes) > 0:
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].tolist())
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)

        for (x, y), c in zip(xy, conf):
            color = (0, 255, 0) if c >= 0.5 else (0, 255, 255) if c >= 0.25 else (0, 0, 255)
            cv2.circle(out, (int(x), int(y)), 7, color, -1)

        cv2.putText(out, f"frame {frame_idx}: {len(xy)} keypoints, box_conf={box_conf:.2f}, mean_kp_conf={mean_conf:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(out, f"frame {frame_idx}: NO DETECTION", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    writer.write(out)
    frame_idx += 1

capture.release()
writer.release()

print(f"Processed {frame_idx} frames.")
print(f"Frames with ANY detection (conf >= {conf_threshold}): {frames_with_detection}/{frame_idx} "
      f"({100.0 * frames_with_detection / frame_idx:.1f}%)")
print(f"Frames with the full 18 keypoints detected: {frames_with_full_18}/{frame_idx}")
if best_frame_idx >= 0:
    print(f"Best single frame: #{best_frame_idx}, mean keypoint confidence {best_frame_conf:.2f}")
else:
    print("No frame in this range produced any detection at all.")
print("\nWrote keypoint_evidence.mp4 -- green=frame had a detection (dots colored by per-keypoint "
      "confidence), red text = zero detections that frame.")
