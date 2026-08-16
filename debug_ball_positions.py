import sys
import cv2
import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
num_frames = 60

H = c.keypoint_model_homography(video_path, "models/court_keypoints.pt", conf_threshold=conf)
ball_model = c._load_model("models/basketball_ball.pt")
ball_class = c.resolve_ball_class(ball_model)
print(f"ball_class index: {ball_class}")

capture = cv2.VideoCapture(video_path)
on_court_count = 0
total_count = 0
for i in range(num_frames):
    ok, frame = capture.read()
    if not ok:
        break
    result = ball_model(frame, classes=[ball_class], conf=0.25, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        continue
    box = max(result.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    x_ft, y_ft = c.project_point(H, center)
    on_court = c.is_on_court(x_ft, y_ft)
    total_count += 1
    on_court_count += on_court
    print(f"frame {i}: ball_px=({center[0]:.0f},{center[1]:.0f}) -> court_ft=({x_ft:.1f},{y_ft:.1f}) on_court={on_court}")
capture.release()

print(f"\n{on_court_count}/{total_count} ball detections landed on-court ({on_court_count/total_count:.1%})" if total_count else "No ball detections found in these frames.")
