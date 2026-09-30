"""Ground-truth tests for court calibration.

Renders synthetic gym frames through KNOWN homographies -- several camera
placements, wood floor plus out-of-bounds apron, a painted key, crowd --
then checks the detector recovers the court coordinates we started from.
Real footage can't test this: there you can only eyeball whether an overlay
looks plausible, which is exactly how a ~40ft calibration error survived
unnoticed (the court OUTLINE looked fine while the mapping was transposed).

Run: python3 test_court_calibration.py
"""
import math
import sys

import cv2
import numpy as np

import courtiq_core as c

IMG_W, IMG_H = 1858, 1030
WOOD_BGR, KEY_BGR, CROWD_BGR = (150, 190, 215), (90, 50, 40), (70, 70, 70)

CAMERAS = {
    "sideline A": [(1500, 300), (500, 300), (1750, 950), (200, 950)],
    "sideline B (opposite side)": [(358, 300), (1358, 300), (108, 950), (1658, 950)],
    "baseline-end": [(700, 300), (500, 900), (1150, 300), (1350, 900)],
    "corner / diagonal": [(1250, 250), (350, 480), (1700, 700), (450, 980)],
}

# Kept near the key: calibrating off a 16x19ft rectangle is accurate around
# it and degrades as you extrapolate away, so these are the positions that
# matter for possession and shot-range work.
CHECKS = [(25, 5.25), (25, 19), (17, 0), (33, 0), (5, 25), (45, 25), (25, 30)]
PAIRS = [((5, 25), (45, 25)), ((25, 5.25), (25, 19)), ((17, 0), (33, 19))]
TOLERANCE_FT = 1.0


def build(img_points):
    return cv2.getPerspectiveTransform(
        np.array([(0, 0), (0, 47), (50, 0), (50, 47)], dtype=np.float32),
        np.array(img_points, dtype=np.float32),
    )


def render(court_to_img):
    def to_img(pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        return cv2.perspectiveTransform(src, court_to_img)[0][0]

    frame = np.full((IMG_H, IMG_W, 3), CROWD_BGR, dtype=np.uint8)
    for poly, color in [
        ([(-4, -4), (54, -4), (54, 51), (-4, 51)], WOOD_BGR),
        (c.KEY_CORNERS_FT, KEY_BGR),
    ]:
        cv2.fillPoly(frame, [np.array([to_img(p) for p in poly], dtype=np.int32)], color)
    return frame, to_img


def calibrate(frame, to_img):
    # The rim is 10ft up, so its image position sits well off its floor
    # position -- model that, since it's what defeats rim-geometry shortcuts.
    hoop_floor = to_img((25, 5.25))
    hoop_px = (float(hoop_floor[0]), float(hoop_floor[1]) - 120.0)
    key_quad = c.detect_key_quad(frame, hoop_px)
    if key_quad is None:
        return None
    floor_mask = c.court_quad_debug(frame).get("morphed_color_mask")
    if floor_mask is None:
        return None
    return c.key_anchored_homography(key_quad, floor_mask)


def check_camera(name, img_points):
    court_to_img = build(img_points)
    frame, to_img = render(court_to_img)
    H = calibrate(frame, to_img)
    if H is None:
        print(f"  {name}: FAIL -- no calibration produced")
        return False

    # Position accuracy, allowing a whole-court left/right mirror. The key is
    # symmetric about x=25 and the rim sits on that axis, so nothing in the
    # image distinguishes left from right -- the information simply isn't
    # there. That's acceptable because mirroring is an isometry: it cannot
    # change any distance, and distances are all the pipeline consumes.
    direct = mirrored = 0.0
    for pt in CHECKS:
        back = c.project_point(H, tuple(map(float, to_img(pt))))
        direct = max(direct, math.hypot(back[0] - pt[0], back[1] - pt[1]))
        mirrored = max(mirrored, math.hypot(back[0] - (50 - pt[0]), back[1] - pt[1]))
    position_error = min(direct, mirrored)

    # Distance preservation -- the property that actually has to hold.
    distance_error = 0.0
    for a, b in PAIRS:
        ra = c.project_point(H, tuple(map(float, to_img(a))))
        rb = c.project_point(H, tuple(map(float, to_img(b))))
        got = math.hypot(ra[0] - rb[0], ra[1] - rb[1])
        expected = math.hypot(a[0] - b[0], a[1] - b[1])
        distance_error = max(distance_error, abs(got - expected))

    ok = position_error < TOLERANCE_FT and distance_error < TOLERANCE_FT
    flip = " (mirrored)" if mirrored < direct else ""
    print(f"  {name}: position {position_error:.2f} ft{flip}, "
          f"distance {distance_error:.2f} ft -- {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(f"Court calibration vs ground truth (tolerance {TOLERANCE_FT} ft):")
    results = [check_camera(name, pts) for name, pts in CAMERAS.items()]
    failures = results.count(False)
    print(f"\n{len(results) - failures}/{len(results)} cameras pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
