"""Verifies a computed homography by inverse-projecting RECOGNIZABLE court
landmarks back onto the real frame -- the painted key, the free-throw
line, the rim, the three-point corners -- not just the court outline.

The outline alone can't tell you the homography is right: it traces
whatever region was detected, so it looks plausible even when the court
coordinate axes are transposed or the scale is off. The painted key is
unambiguous -- if the yellow box doesn't sit exactly on the real key under
the hoop, the homography is wrong no matter how good the outline looks.

Usage: python3 verify_court_calibration.py <video> [frame_number]
Output: court_calibration_check.png
"""
import sys
import cv2
import numpy as np
import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
frame_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0

H = c.auto_homography(video_path)
H_inv = np.linalg.inv(H)

capture = cv2.VideoCapture(video_path)
if frame_number > 0:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = capture.read()
capture.release()
if not ok:
    raise SystemExit(f"Could not read frame {frame_number} from {video_path}")


def to_px(pt_ft):
    src = np.array([[list(pt_ft)]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_inv)
    return tuple(map(int, dst[0][0]))


def draw_poly(points_ft, color, label, thickness=3):
    pts = [to_px(p) for p in points_ft]
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], color, thickness)
    cv2.putText(frame, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# Court outline (near half): baseline y=0 to halfcourt y=47, width x=0..50
draw_poly([(0, 0), (50, 0), (50, 47), (0, 47)], (255, 255, 0), "COURT OUTLINE")

# The painted key/paint: 16ft wide, 19ft deep from the baseline. This is the
# unambiguous check -- it should land exactly on the real painted key.
draw_poly([(17, 0), (33, 0), (33, 19), (17, 19)], (0, 255, 255), "KEY (should sit on real paint)")

# The rim, and the two three-point corners on the baseline.
cv2.circle(frame, to_px((25, 5.25)), 12, (0, 0, 255), 3)
cv2.putText(frame, "HOOP", to_px((25, 5.25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
for pt, label in [((3, 0), "3PT-L"), ((47, 0), "3PT-R")]:
    cv2.circle(frame, to_px(pt), 9, (255, 0, 255), -1)
    cv2.putText(frame, label, to_px(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

# Where the hoop was actually DETECTED in the image (as opposed to the red
# circle above, which is where the homography THINKS a hoop should be).
# The gap between the two is the calibration error, made visible.
hoop_px = c.detect_hoop(video_path)
if hoop_px is None:
    print("NOTE: no hoop detected -- court orientation could not be anchored.")
else:
    px = tuple(map(int, hoop_px))
    cv2.circle(frame, px, 16, (0, 255, 0), 4)
    cv2.putText(frame, "DETECTED HOOP", (px[0] + 20, px[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    hoop_ft = c.project_point(H, hoop_px)
    print(f"Detected hoop at image {px}, which projects to court ({hoop_ft[0]:.1f}, {hoop_ft[1]:.1f}) ft.")
    print(f"A real hoop sits at {c.COURT_LANDMARKS['hoop']} ft.")

cv2.imwrite("court_calibration_check.png", frame)
print("Wrote court_calibration_check.png")
print()
print("What to look for:")
print("  CYAN key box  -> should sit exactly on the real painted key under the hoop")
print("  RED circle    -> where the homography THINKS the hoop is")
print("  GREEN circle  -> where the hoop was actually DETECTED")
print("  (red and green far apart = calibration error, made visible)")
print("  MAGENTA dots  -> should sit on the baseline three-point corners")
print("If these land rotated, mirrored, or somewhere unrelated, the homography's")
print("court-axis mapping is wrong even if the outline looked correct.")
