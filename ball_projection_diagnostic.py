"""For each frame's ball detection, prints the projected court coordinate
and whether is_on_court() accepts or rejects it, and by how much -- to
tell apart "homography is basically right, just needs a bigger margin"
from "homography has real, large error."

Usage: python3 ball_projection_diagnostic.py <video> [num_frames]
"""
import sys
import cv2
import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300

H = c.auto_homography(video_path)
ball_model = c._load_model("models/basketball_ball.pt")
ball_class = c.resolve_ball_class(ball_model)

capture = cv2.VideoCapture(video_path)
frame_idx = 0
accepted = 0
rejected_margins = []

while frame_idx < num_frames:
    ok, frame = capture.read()
    if not ok:
        break
    result = ball_model(frame, classes=[ball_class], conf=0.25, verbose=False)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        box = max(result.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        x_ft, y_ft = c.project_point(H, ((x1 + x2) / 2, (y1 + y2) / 2))
        on_court = c.is_on_court(x_ft, y_ft)
        if on_court:
            accepted += 1
        else:
            # How far outside the court rectangle (plus margin) is this point?
            over_x = max(0.0 - x_ft, x_ft - c.COURT_WIDTH_FT, 0.0)
            over_y = max(0.0 - y_ft, y_ft - c.COURT_LENGTH_FT, 0.0)
            rejected_margins.append(max(over_x, over_y))
            if frame_idx % 20 == 0:
                print(f"frame {frame_idx}: ball at ft=({x_ft:.1f},{y_ft:.1f}) REJECTED "
                      f"(court is 0-{c.COURT_WIDTH_FT:.1f} x 0-{c.COURT_LENGTH_FT:.1f}, margin={c.ON_COURT_MARGIN_FT})")
    frame_idx += 1

capture.release()

total = accepted + len(rejected_margins)
print(f"\nAccepted: {accepted}/{total} ({100.0*accepted/total:.1f}%)" if total else "No ball detections.")
if rejected_margins:
    rejected_margins.sort()
    n = len(rejected_margins)
    print(f"Rejected margin distribution (ft beyond court+margin): "
          f"min={rejected_margins[0]:.1f} median={rejected_margins[n//2]:.1f} max={rejected_margins[-1]:.1f}")
