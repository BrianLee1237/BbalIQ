import sys

import cv2

import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"

capture = cv2.VideoCapture(video_path)
ok, frame = capture.read()
capture.release()
if not ok:
    raise SystemExit(f"Could not read a frame from {video_path}")

quad = c.detect_court_quad(frame)
out = frame.copy()

if quad is None:
    print("detect_court_quad() found nothing -- the full-frame fallback homography is being used instead.")
    print("That means the ENTIRE frame (including any bench/crowd visible) is being treated as on-court.")
else:
    print("Detected court quad (image pixel coordinates):", quad)
    pts = [tuple(map(int, p)) for p in quad]
    for i in range(4):
        cv2.line(out, pts[i], pts[(i + 1) % 4], (0, 0, 255), 4)
    for label, p in zip(["TL", "TR", "BR", "BL"], pts):
        cv2.putText(out, label, p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

cv2.imwrite("court_detection_debug.png", out)
print("Wrote court_detection_debug.png -- open it and check whether the red quad actually")
print("traces the court floor, or whether it's including the bench/sideline/crowd area too.")
