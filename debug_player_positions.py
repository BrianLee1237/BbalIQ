import sys
import cv2
import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
num_frames = 15

H = c.keypoint_model_homography(video_path, "models/court_keypoints.pt", conf_threshold=conf)
player_model = c._load_model("models/basketball_players.pt")
person_class = c.resolve_person_class(player_model)
print(f"person_class index: {person_class}")

capture = cv2.VideoCapture(video_path)
on_court_count = 0
total_count = 0
for i in range(num_frames):
    ok, frame = capture.read()
    if not ok:
        break
    result = player_model(frame, classes=[person_class], conf=0.4, verbose=False)[0]
    if result.boxes is None:
        continue
    for box in result.boxes.xyxy.tolist():
        x1, y1, x2, y2 = box
        foot = ((x1 + x2) / 2, y2)
        x_ft, y_ft = c.project_point(H, foot)
        on_court = c.is_on_court(x_ft, y_ft)
        total_count += 1
        on_court_count += on_court
        print(f"frame {i}: foot_px=({foot[0]:.0f},{foot[1]:.0f}) -> court_ft=({x_ft:.1f},{y_ft:.1f}) on_court={on_court}")
capture.release()

print(f"\n{on_court_count}/{total_count} player detections landed on-court ({on_court_count/total_count:.1%})" if total_count else "No player detections found in these frames.")
