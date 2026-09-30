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


LINE_BGR = (70, 70, 70)


def draw_court_markings(frame, to_img, line_bgr=LINE_BGR):
    """Paint the lines a real court is covered in.

    These matter because they form a connected network: sidelines meeting the
    baseline, the arc meeting the lane, the centre circle. On real footage
    that network fused every painted area into one region covering 46% of the
    floor, so the lane couldn't be isolated. Synthetic courts without lines
    can't catch that -- their paint sits in splendid isolation, which no real
    gym does.
    """
    def line(a, b):
        cv2.line(frame, tuple(map(int, to_img(a))), tuple(map(int, to_img(b))), line_bgr, 5)

    # Boundary, lane lines, free-throw line, and a centre line.
    for a, b in [
        ((0, 0), (50, 0)), ((0, 0), (0, 47)), ((50, 0), (50, 47)), ((0, 47), (50, 47)),
        ((17, 0), (17, 19)), ((33, 0), (33, 19)), ((17, 19), (33, 19)),
        ((0, 28), (50, 28)),
    ]:
        line(a, b)

    # Three-point arc, which runs from the lane out to the sidelines.
    arc = [(25 + 22 * math.cos(th), 5.25 + 22 * math.sin(th))
           for th in [math.pi * i / 40 for i in range(41)]]
    for a, b in zip(arc, arc[1:]):
        line(a, b)

    # Centre circle, touching the centre line.
    circle = [(25 + 6 * math.cos(th), 28 + 6 * math.sin(th))
              for th in [2 * math.pi * i / 40 for i in range(41)]]
    for a, b in zip(circle, circle[1:]):
        line(a, b)


def render(court_to_img, floor_bgr=None, paint_bgr=None, with_markings=True):
    def to_img(pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        return cv2.perspectiveTransform(src, court_to_img)[0][0]

    frame = np.full((IMG_H, IMG_W, 3), CROWD_BGR, dtype=np.uint8)
    for poly, color in [
        ([(-4, -4), (54, -4), (54, 51), (-4, 51)], floor_bgr or WOOD_BGR),
        (c.KEY_CORNERS_FT, paint_bgr or KEY_BGR),
    ]:
        cv2.fillPoly(frame, [np.array([to_img(p) for p in poly], dtype=np.int32)], color)
    if with_markings:
        draw_court_markings(frame, to_img)
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


def check_occlusion(tmp_path="/tmp/courtiq_occlusion_test.mp4"):
    """Players stand in the paint constantly, so most frames show a lane
    with a chunk bitten out of it. Build a clip that's mostly occluded and
    check the multi-frame search still finds a clean view to calibrate off.
    """
    court_to_img = build(CAMERAS["sideline A"])
    _, to_img = render(court_to_img)

    def frame_with(occluded):
        frame = np.full((IMG_H, IMG_W, 3), CROWD_BGR, dtype=np.uint8)
        for poly, color in [
            ([(-4, -4), (54, -4), (54, 51), (-4, 51)], WOOD_BGR),
            (c.KEY_CORNERS_FT, KEY_BGR),
        ]:
            cv2.fillPoly(frame, [np.array([to_img(p) for p in poly], dtype=np.int32)], color)
        draw_court_markings(frame, to_img)
        if occluded:
            for pos in [(20, 4), (30, 8), (25, 14), (21, 17)]:
                base = to_img(pos)
                cv2.rectangle(frame, (int(base[0] - 45), int(base[1] - 190)),
                              (int(base[0] + 45), int(base[1])), (40, 40, 160), -1)
        return frame

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (IMG_W, IMG_H))
    for i in range(40):
        writer.write(frame_with(occluded=(i % 13 != 0)))
    writer.release()

    hoop_floor = to_img((25, 5.25))
    hoop_px = (float(hoop_floor[0]), float(hoop_floor[1]) - 120.0)

    occluded_score = c.detect_key_with_score(frame_with(True), hoop_px)
    clear_score = c.detect_key_with_score(frame_with(False), hoop_px)
    if occluded_score is None or clear_score is None:
        print("  occlusion: FAIL -- key not detected")
        return False
    # The score has to actually separate the two cases. Comparing a fitted
    # quad against itself yields exactly 1.0 for everything, which looks
    # like it's working while ranking frames arbitrarily.
    if not occluded_score[1] < clear_score[1] - 0.05:
        print(f"  occlusion: FAIL -- score does not distinguish occluded "
              f"({occluded_score[1]:.2f}) from clear ({clear_score[1]:.2f})")
        return False

    found = c.detect_key_quad_from_video(tmp_path, hoop_px)
    if found is None:
        print("  occlusion: FAIL -- no key found across the clip")
        return False
    H = c.key_anchored_homography(found[0], found[1])
    if H is None:
        print("  occlusion: FAIL -- calibration failed")
        return False

    direct = mirrored = 0.0
    for pt in CHECKS:
        back = c.project_point(H, tuple(map(float, to_img(pt))))
        direct = max(direct, math.hypot(back[0] - pt[0], back[1] - pt[1]))
        mirrored = max(mirrored, math.hypot(back[0] - (50 - pt[0]), back[1] - pt[1]))
    error = min(direct, mirrored)
    ok = error < TOLERANCE_FT
    print(f"  occlusion (36/40 frames blocked): score {occluded_score[1]:.2f} occluded vs "
          f"{clear_score[1]:.2f} clear, error {error:.2f} ft -- {'PASS' if ok else 'FAIL'}")
    return ok


def check_false_hoop():
    """Reproduces the real-footage failure: the rim detector also fires on
    the crowd, and a hoop in the stands makes the key search grab whatever
    paint is nearest it -- a centre-court logo -- which then defines the
    whole coordinate system. The bad candidate has to be REJECTED, not
    silently used.
    """
    court_to_img = build(CAMERAS["sideline A"])
    _, to_img = render(court_to_img)

    frame = np.full((IMG_H, IMG_W, 3), CROWD_BGR, dtype=np.uint8)
    for poly, color in [
        ([(-4, -4), (54, -4), (54, 51), (-4, 51)], WOOD_BGR),
        (c.KEY_CORNERS_FT, KEY_BGR),
        # A painted logo out in open floor, with court running away on BOTH
        # sides of it -- that's what makes it distinguishable from the lane,
        # which is pinned against the baseline.
        ([(18, 22), (32, 22), (32, 32), (18, 32)], KEY_BGR),
    ]:
        cv2.fillPoly(frame, [np.array([to_img(p) for p in poly], dtype=np.int32)], color)
    draw_court_markings(frame, to_img)

    floor_mask = c.court_quad_debug(frame).get("morphed_color_mask")
    if floor_mask is None:
        print("  false hoop: FAIL -- no floor mask")
        return False

    # A hoop "detected" in the crowd, near the logo rather than the rim.
    logo_img = to_img((25, 27))
    false_hoop = (float(logo_img[0]), float(logo_img[1]) - 250.0)
    bad = c.detect_key_with_score(frame, false_hoop)
    if bad is None:
        print("  false hoop: PASS (no region found near the false hoop)")
        return True
    rejected = c.key_anchored_homography(bad[0], floor_mask) is None

    # ...and the real rim must still calibrate correctly on the same frame.
    hoop_floor = to_img((25, 5.25))
    real_hoop = (float(hoop_floor[0]), float(hoop_floor[1]) - 120.0)
    good = c.detect_key_with_score(frame, real_hoop)
    H = c.key_anchored_homography(good[0], floor_mask) if good else None
    if H is None:
        print("  false hoop: FAIL -- real rim no longer calibrates")
        return False
    direct = mirrored = 0.0
    for pt in CHECKS:
        back = c.project_point(H, tuple(map(float, to_img(pt))))
        direct = max(direct, math.hypot(back[0] - pt[0], back[1] - pt[1]))
        mirrored = max(mirrored, math.hypot(back[0] - (50 - pt[0]), back[1] - pt[1]))
    error = min(direct, mirrored)

    ok = rejected and error < TOLERANCE_FT
    print(f"  false hoop (crowd -> logo): bad candidate "
          f"{'rejected' if rejected else 'ACCEPTED (bad)'}, "
          f"real rim error {error:.2f} ft -- {'PASS' if ok else 'FAIL'}")
    return ok


def check_crowd_bridge():
    """Reproduces the real-footage failure where no key is found at all.

    The lane is identified as paint ENCLOSED by wood. Players and referees
    are non-wood, so anyone standing between the lane and the sideline forms
    a continuous non-wood channel from the frame border into the paint --
    the lane stops being enclosed and vanishes. On real footage somebody is
    nearly always standing there, so this is the common case; an earlier
    occlusion test missed it by putting players strictly INSIDE the lane,
    where they never bridge out to the crowd.
    """
    court_to_img = build(CAMERAS["sideline A"])
    _, to_img = render(court_to_img)

    frame = np.full((IMG_H, IMG_W, 3), CROWD_BGR, dtype=np.uint8)
    for poly, color in [
        ([(-4, -4), (54, -4), (54, 51), (-4, 51)], WOOD_BGR),
        (c.KEY_CORNERS_FT, KEY_BGR),
    ]:
        cv2.fillPoly(frame, [np.array([to_img(p) for p in poly], dtype=np.int32)], color)
    draw_court_markings(frame, to_img)

    def frame_with_bridge(bridged):
        out = frame.copy()
        if bridged:
            # A chain of bodies from the lane edge out past the sideline.
            for pos in [(17, 10), (10, 10), (4, 10), (-2, 10)]:
                base = to_img(pos)
                cv2.rectangle(out, (int(base[0] - 40), int(base[1] - 200)),
                              (int(base[0] + 40), int(base[1])), (40, 40, 160), -1)
        return out

    # Most frames bridged, a few clear -- which is how real footage behaves,
    # and why the pipeline searches frames instead of trusting one.
    tmp_path = "/tmp/courtiq_bridge_test.mp4"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (IMG_W, IMG_H))
    for i in range(40):
        writer.write(frame_with_bridge(bridged=(i % 13 != 0)))
    writer.release()

    hoop_floor = to_img((25, 5.25))
    hoop_px = (float(hoop_floor[0]), float(hoop_floor[1]) - 120.0)

    # A bridged frame must score WORSE than a clear one, or frame selection
    # has nothing to go on.
    bridged_score = c.detect_key_with_score(frame_with_bridge(True), hoop_px)
    clear_score = c.detect_key_with_score(frame_with_bridge(False), hoop_px)
    if bridged_score is None or clear_score is None:
        print("  crowd bridge: FAIL -- lane not found at all")
        return False
    if not bridged_score[1] < clear_score[1] - 0.05:
        print(f"  crowd bridge: FAIL -- bridged frame scores {bridged_score[1]:.2f}, "
              f"clear {clear_score[1]:.2f}; selection can't tell them apart")
        return False

    found = c.detect_key_quad_from_video(tmp_path, hoop_px)
    if found is None:
        print("  crowd bridge: FAIL -- no lane found across the clip")
        return False
    H = c.key_anchored_homography(found[0], found[1])
    if H is None:
        print("  crowd bridge: FAIL -- lane found but calibration rejected")
        return False

    direct = mirrored = 0.0
    for pt in CHECKS:
        back = c.project_point(H, tuple(map(float, to_img(pt))))
        direct = max(direct, math.hypot(back[0] - pt[0], back[1] - pt[1]))
        mirrored = max(mirrored, math.hypot(back[0] - (50 - pt[0]), back[1] - pt[1]))
    error = min(direct, mirrored)
    ok = error < TOLERANCE_FT
    print(f"  crowd bridge (players linking lane to stands): score "
          f"{bridged_score[1]:.2f} bridged vs {clear_score[1]:.2f} clear, "
          f"error {error:.2f} ft -- {'PASS' if ok else 'FAIL'}")
    return ok


FLOOR_SCHEMES = {
    "pale maple / navy paint": ((150, 190, 215), (90, 50, 40)),
    "dark stained wood / red paint": ((45, 70, 110), (50, 50, 170)),
    "grey sports floor / green paint": ((140, 140, 140), (70, 130, 60)),
    "blue-tinted floor / cream paint": ((170, 140, 110), (190, 220, 235)),
    "warm tungsten lighting": ((110, 165, 205), (105, 70, 45)),
}


def check_floor_colors():
    """The floor colour must be LEARNED, not assumed.

    Every other test here uses one floor colour, so they'd all still pass
    with a hardcoded range -- they can't tell the difference. These vary the
    floor and paint colours instead. A hardcoded tan range fails all but the
    first; a learned one shouldn't care, since it only needs paint to differ
    from floor, not to be any particular colour.
    """
    court_to_img = build(CAMERAS["sideline A"])
    _, to_img = render(court_to_img)
    all_ok = True

    for name, (floor_bgr, paint_bgr) in FLOOR_SCHEMES.items():
        frame, _ = render(court_to_img, floor_bgr=floor_bgr, paint_bgr=paint_bgr)

        hoop_floor = to_img((25, 5.25))
        hoop_px = (float(hoop_floor[0]), float(hoop_floor[1]) - 120.0)
        scored = c.detect_key_with_score(frame, hoop_px)
        floor_mask = c.court_quad_debug(frame).get("morphed_color_mask")
        H = (c.key_anchored_homography(scored[0], floor_mask)
             if scored is not None and floor_mask is not None else None)
        if H is None:
            print(f"    {name}: FAIL -- no calibration")
            all_ok = False
            continue

        direct = mirrored = 0.0
        for pt in CHECKS:
            back = c.project_point(H, tuple(map(float, to_img(pt))))
            direct = max(direct, math.hypot(back[0] - pt[0], back[1] - pt[1]))
            mirrored = max(mirrored, math.hypot(back[0] - (50 - pt[0]), back[1] - pt[1]))
        error = min(direct, mirrored)
        ok = error < TOLERANCE_FT
        all_ok = all_ok and ok
        print(f"    {name}: {error:.2f} ft -- {'PASS' if ok else 'FAIL'}")

    print(f"  adaptive floor colour -- {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def check_distractors():
    """Small non-floor things must not be mistaken for the lane.

    Real footage is full of them: a referee, a scoreboard overlay, a
    centre-court logo, a bench of substitutes, a spectator leaning in. Each
    reads as a painted region, and on real footage a REFEREE was chosen as
    the lane -- so the filter has to know a lane's real size relative to the
    floor rather than just "not tiny".
    """
    court_to_img = build(CAMERAS["sideline A"])
    frame, to_img = render(court_to_img)

    # Referee, mid-court logo, scoreboard overlay, bench along the sideline.
    ref = to_img((8, 25))
    cv2.rectangle(frame, (int(ref[0] - 35), int(ref[1] - 190)),
                  (int(ref[0] + 35), int(ref[1])), (55, 55, 55), -1)
    cv2.fillPoly(frame, [np.array([to_img(p) for p in
                                   [(20, 26), (30, 26), (30, 33), (20, 33)]], dtype=np.int32)],
                 (60, 45, 90))
    cv2.rectangle(frame, (30, IMG_H - 190), (330, IMG_H - 20), (35, 35, 35), -1)
    for i in range(6):
        seat = to_img((52, 8 + i * 5))
        cv2.rectangle(frame, (int(seat[0] - 30), int(seat[1] - 150)),
                      (int(seat[0] + 30), int(seat[1])), (50, 40, 90), -1)

    hoop_floor = to_img((25, 5.25))
    hoop_px = (float(hoop_floor[0]), float(hoop_floor[1]) - 120.0)

    regions = c.detect_painted_regions(frame)
    scored = c.detect_key_with_score(frame, hoop_px)
    floor_mask = c.court_quad_debug(frame).get("morphed_color_mask")
    H = (c.key_anchored_homography(scored[0], floor_mask)
         if scored is not None and floor_mask is not None else None)
    if H is None:
        print(f"  distractors: FAIL -- no calibration ({len(regions)} regions)")
        return False

    direct = mirrored = 0.0
    for pt in CHECKS:
        back = c.project_point(H, tuple(map(float, to_img(pt))))
        direct = max(direct, math.hypot(back[0] - pt[0], back[1] - pt[1]))
        mirrored = max(mirrored, math.hypot(back[0] - (50 - pt[0]), back[1] - pt[1]))
    error = min(direct, mirrored)
    ok = error < TOLERANCE_FT
    print(f"  distractors (ref/logo/scoreboard/bench): {len(regions)} regions kept, "
          f"error {error:.2f} ft -- {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(f"Court calibration vs ground truth (tolerance {TOLERANCE_FT} ft):")
    results = [check_camera(name, pts) for name, pts in CAMERAS.items()]
    results.append(check_occlusion())
    results.append(check_false_hoop())
    results.append(check_crowd_bridge())
    results.append(check_floor_colors())
    results.append(check_distractors())
    failures = results.count(False)
    print(f"\n{len(results) - failures}/{len(results)} checks pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
