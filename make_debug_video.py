"""Renders an annotated debug video: player boxes (green=on-court,
red=off-court), the ball (same color coding), and the assumed court
boundary + all 18 calibration keypoints projected BACK onto the real
footage via the inverse homography -- so you can see by eye whether the
computed court boundary actually lines up with the real floor, instead of
reading coordinate dumps.

Uses auto_homography() (classical-CV floor detection), same as the
default CLI/web pipeline -- NOT the keypoint model, which requires a
full end-to-end court view (both baselines in frame) that half-court
broadcast angles don't provide.

Usage: python3 make_debug_video.py <video> [num_frames]
Output: court_analysis_debug.mp4
"""
import sys
import cv2
import numpy as np
import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300

H = c.auto_homography(video_path)
H_inv = np.linalg.inv(H)

player_model = c._load_model("models/basketball_players.pt")
person_class = c.resolve_person_class(player_model)
ball_model = c._load_model("models/basketball_ball.pt")
ball_class = c.resolve_ball_class(ball_model)


def project_inverse(pt):
    src = np.array([[pt]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_inv)
    return tuple(map(int, dst[0][0]))


capture = cv2.VideoCapture(video_path)
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter("court_analysis_debug.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Court boundary rectangle, inverse-projected into image space -- this is
# what the code THINKS the court outline is, drawn directly on the real footage.
boundary_ft = [(0, 0), (49.21, 0), (49.21, 91.86), (0, 91.86)]
boundary_px = [project_inverse(p) for p in boundary_ft]

# All 18 calibration keypoints too, so you can see each individual anchor point.
keypoint_px = [project_inverse(p) for p in c.KEYPOINT_MODEL_COURT_POINTS_FT]

frame_idx = 0
while frame_idx < num_frames:
    ok, frame = capture.read()
    if not ok:
        break

    # Court boundary outline (cyan) + individual keypoints (yellow dots)
    for i in range(4):
        cv2.line(frame, boundary_px[i], boundary_px[(i + 1) % 4], (255, 255, 0), 2)
    for px in keypoint_px:
        cv2.circle(frame, px, 5, (0, 255, 255), -1)

    # Players
    player_result = player_model(frame, classes=[person_class], conf=0.4, verbose=False)[0]
    if player_result.boxes is not None:
        for box in player_result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = map(int, box)
            foot = ((x1 + x2) / 2, y2)
            x_ft, y_ft = c.project_point(H, foot)
            on_court = c.is_on_court(x_ft, y_ft)
            color = (0, 255, 0) if on_court else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"({x_ft:.0f},{y_ft:.0f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Ball
    ball_result = ball_model(frame, classes=[ball_class], conf=0.25, verbose=False)[0]
    if ball_result.boxes is not None and len(ball_result.boxes) > 0:
        box = max(ball_result.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        x_ft, y_ft = c.project_point(H, center)
        on_court = c.is_on_court(x_ft, y_ft)
        color = (0, 165, 255) if on_court else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"BALL ({x_ft:.0f},{y_ft:.0f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    writer.write(frame)
    frame_idx += 1

capture.release()
writer.release()
print(f"Wrote court_analysis_debug.mp4 ({frame_idx} frames)")
print("Green = on-court, Red = off-court. Cyan outline = assumed court boundary.")
print("Yellow dots = the 18 calibration keypoints, projected back onto the frame.")
