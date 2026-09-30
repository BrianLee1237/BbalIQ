"""Runs the ball detector across a video with NO court/homography
filtering at all -- pure raw detection rate and confidence distribution,
to tell apart "the model isn't finding the ball" from "it's finding it
but getting filtered out downstream."

Usage: python3 ball_detection_diagnostic.py <video> [num_frames] [conf]
"""
import sys
import cv2
import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300
conf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

ball_model = c._load_model("models/basketball_ball.pt")
ball_class = c.resolve_ball_class(ball_model)
print(f"Ball class resolved to: {ball_class} (model names: {ball_model.names})")

capture = cv2.VideoCapture(video_path)
frame_idx = 0
detected_count = 0
confidences = []

while frame_idx < num_frames:
    ok, frame = capture.read()
    if not ok:
        break
    result = ball_model(frame, classes=[ball_class], conf=conf_threshold, verbose=False)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        best_conf = float(max(result.boxes.conf).item())
        confidences.append(best_conf)
        detected_count += 1
    frame_idx += 1

capture.release()

print(f"\nProcessed {frame_idx} frames at conf_threshold={conf_threshold}")
print(f"Frames with ANY ball detection: {detected_count}/{frame_idx} ({100.0*detected_count/frame_idx:.1f}%)")
if confidences:
    confidences.sort()
    n = len(confidences)
    print(f"Confidence distribution: min={confidences[0]:.2f} "
          f"median={confidences[n//2]:.2f} max={confidences[-1]:.2f}")
    print(f"  >= 0.25: {sum(1 for c_ in confidences if c_ >= 0.25)}")
    print(f"  >= 0.5:  {sum(1 for c_ in confidences if c_ >= 0.5)}")
else:
    print("No detections at all, even at this low threshold.")
