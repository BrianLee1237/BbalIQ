"""Ground-truth tests for line-based court fitting.

Renders synthetic courts -- markings drawn through a KNOWN homography, at
high-school and college dimensions -- and checks the fit recovers the court
coordinates, the right court size, and the right orientation.

Court size matters on its own: a high-school lane is 12ft wide against 16ft
for college/NBA, so fitting the wrong model distorts every width by a third
even when the alignment looks fine.

Run: python3 test_court_fit.py
"""
import math
import sys

import cv2
import numpy as np

import court_fit
import courtiq_core as c
from court_model import COURT_MODELS

IMG_W, IMG_H = 1858, 1030
FLOOR_BGR, LINE_BGR, CROWD_BGR = (150, 190, 215), (95, 105, 120), (70, 70, 70)
PAINT_BGR = (90, 50, 40)

CAMERAS = {
    "sideline": [(1500, 300), (500, 300), (1750, 950), (200, 950)],
    "baseline-end": [(700, 300), (500, 900), (1150, 300), (1350, 900)],
    "corner / diagonal": [(1250, 250), (350, 480), (1700, 700), (450, 980)],
}
TOLERANCE_FT = 2.0


def build(model, img_points):
    """Map this model's half-court corners onto the given image quad."""
    return cv2.getPerspectiveTransform(
        np.array([(0, 0), (0, model.half_length),
                  (model.width, 0), (model.width, model.half_length)], dtype=np.float32),
        np.array(img_points, dtype=np.float32),
    )


def render(model, court_to_img, paint_lane=True, colours=None):
    def to_img(pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        return cv2.perspectiveTransform(src, court_to_img)[0][0]

    floor_bgr, paint_bgr, line_bgr = colours or (FLOOR_BGR, PAINT_BGR, LINE_BGR)
    frame = np.full((IMG_H, IMG_W, 3), CROWD_BGR, dtype=np.uint8)
    apron = 4.0
    cv2.fillPoly(frame, [np.array([to_img(p) for p in [
        (-apron, -apron), (model.width + apron, -apron),
        (model.width + apron, model.half_length + apron), (-apron, model.half_length + apron),
    ]], dtype=np.int32)], floor_bgr)

    if paint_lane:
        lane_left = (model.width - model.lane_width) / 2.0
        lane_right = (model.width + model.lane_width) / 2.0
        cv2.fillPoly(frame, [np.array([to_img(p) for p in [
            (lane_left, 0), (lane_right, 0),
            (lane_right, model.lane_length), (lane_left, model.lane_length),
        ]], dtype=np.int32)], paint_bgr)

    for polyline in model.polylines():
        pts = [tuple(map(int, to_img(p))) for p in polyline]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, line_bgr, 4)
    return frame, to_img


def evaluate(model, frame, to_img, label, hoop_px=None):
    floor_mask = c.court_quad_debug(frame).get("morphed_color_mask")
    if floor_mask is None:
        print(f"  {label}: FAIL -- no floor mask")
        return False
    floor_region = c.floor_hull_region(floor_mask)
    if floor_region is None:
        print(f"  {label}: FAIL -- no floor region")
        return False

    result = court_fit.fit_court(frame, floor_region, floor_mask,
                                 hoop_px=hoop_px, verbose=False)
    if result is None:
        print(f"  {label}: FAIL -- no fit")
        return False
    H, info = result

    checks = [(model.width / 2, 5.25), (model.width / 2, model.lane_length),
              (5, 25), (model.width - 5, 25), (model.width / 2, model.half_length - 5)]
    direct = mirrored = 0.0
    for pt in checks:
        back = c.project_point(H, tuple(map(float, to_img(pt))))
        direct = max(direct, math.hypot(back[0] - pt[0], back[1] - pt[1]))
        mirrored = max(mirrored, math.hypot(back[0] - (model.width - pt[0]), back[1] - pt[1]))
    error = min(direct, mirrored)

    right_model = info["model"].name == model.name
    ok = error < TOLERANCE_FT and right_model
    note = "" if right_model else f", WRONG MODEL ({info['model'].name})"
    print(f"  {label}: {error:.2f} ft, {info['cost']:.1f}px marking error{note} "
          f"-- {'PASS' if ok else 'FAIL'}")
    return ok


COLOUR_SCHEMES = {
    "pale maple": ((150, 190, 215), (90, 50, 40), (95, 105, 120)),
    "dark stained wood": ((45, 70, 110), (50, 50, 170), (150, 160, 170)),
    "grey sports floor": ((140, 140, 140), (70, 130, 60), (230, 230, 230)),
    "warm tungsten": ((110, 165, 205), (105, 70, 45), (90, 110, 130)),
}


def add_other_sport_lines(frame, to_img, model, colour=(120, 90, 90)):
    """Lines belonging to OTHER sports, as school gyms are always marked.

    A gym floor typically carries volleyball, badminton and often a second,
    cross-wise basketball court on top of the main one. These are the same
    kind of painted line, so they cannot be told apart by appearance -- and
    they let a wrong alignment land its model lines on somebody else's
    markings. Measured on real footage: a 3.0px marking error with the lane
    sitting in open floor and the arc curving the wrong way.
    """
    w, hl = model.width, model.half_length

    def line(a, b):
        cv2.line(frame, tuple(map(int, to_img(a))), tuple(map(int, to_img(b))), colour, 4)

    # Volleyball court, inset and rotated relative to the basketball court.
    for a, b in [((6, 6), (w - 6, 6)), ((6, hl - 8), (w - 6, hl - 8)),
                 ((6, 6), (6, hl - 8)), ((w - 6, 6), (w - 6, hl - 8)),
                 ((6, (hl - 2) / 2), (w - 6, (hl - 2) / 2))]:
        line(a, b)
    # A cross-court practice basketball key, off to one side.
    for a, b in [((3, 12), (3, 24)), ((15, 12), (15, 24)), ((3, 24), (15, 24))]:
        line(a, b)
    # Badminton tramlines.
    for offset in (11, 14):
        line((offset, 4), (offset, hl - 6))


def add_players(frame, to_img, model, count=10, seed=7):
    """Bodies on the court, including some standing over the markings."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        x = rng.uniform(2, model.width - 2)
        y = rng.uniform(1, model.half_length - 2)
        base = to_img((x, y))
        # Size the player by the local scale, so a body is ~2ft wide and
        # ~6.5ft tall wherever it stands. Drawing a fixed pixel size instead
        # makes players enormous on any camera where the court is small in
        # frame, covering it more than half the time -- which defeats a
        # median that would work perfectly well on real footage.
        near = to_img((min(x + 1.0, model.width), y))
        px_per_ft = max(2.0, math.hypot(near[0] - base[0], near[1] - base[1]))
        half_w = int(px_per_ft * 1.0)
        height = int(px_per_ft * 6.5)
        jersey = tuple(int(v) for v in rng.integers(20, 210, size=3))
        cv2.rectangle(frame, (int(base[0] - half_w), int(base[1] - height)),
                      (int(base[0] + half_w), int(base[1])), jersey, -1)


def check_realistic(model, camera, label, colours=None, players=True):
    floor_bgr, paint_bgr, line_bgr = colours or COLOUR_SCHEMES["pale maple"]
    court_to_img = build(model, CAMERAS[camera])

    def to_img(pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        return cv2.perspectiveTransform(src, court_to_img)[0][0]

    frame = np.full((IMG_H, IMG_W, 3), CROWD_BGR, dtype=np.uint8)
    apron = 4.0
    cv2.fillPoly(frame, [np.array([to_img(p) for p in [
        (-apron, -apron), (model.width + apron, -apron),
        (model.width + apron, model.half_length + apron), (-apron, model.half_length + apron),
    ]], dtype=np.int32)], floor_bgr)
    lane_left = (model.width - model.lane_width) / 2.0
    lane_right = (model.width + model.lane_width) / 2.0
    cv2.fillPoly(frame, [np.array([to_img(p) for p in [
        (lane_left, 0), (lane_right, 0),
        (lane_right, model.lane_length), (lane_left, model.lane_length),
    ]], dtype=np.int32)], paint_bgr)
    for polyline in model.polylines():
        pts = [tuple(map(int, to_img(p))) for p in polyline]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, line_bgr, 4)
    if players:
        add_players(frame, to_img, model)

    return evaluate(model, frame, to_img, label)


def check_video_median(model, camera, label, colours=None, clutter=False,
                       use_hoop=True):
    """The real code path: players move between frames, so fit the median.

    Fitting a single frame has to cope with ten bodies covering the markings.
    Across a clip they're never in the same place twice, so the median frame
    is essentially the empty court.
    """
    import court_fit as cf

    court_to_img = build(model, CAMERAS[camera])

    def to_img(pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        return cv2.perspectiveTransform(src, court_to_img)[0][0]

    base, _ = render(model, court_to_img, colours=colours)
    if clutter:
        add_other_sport_lines(base, to_img, model)
    path = "/tmp/courtiq_fit_video.mp4"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (IMG_W, IMG_H))
    for frame_index in range(40):
        frame = base.copy()
        add_players(frame, to_img, model, seed=frame_index)
        writer.write(frame)
    writer.release()

    median = cf.median_frame(path)
    if median is None:
        print(f"  {label}: FAIL -- no median frame")
        return False
    hoop_px = None
    if use_hoop:
        # The rim as the detector would report it: above its floor point,
        # because it's 10ft up.
        floor_point = to_img(model.hoop)
        hoop_px = (float(floor_point[0]), float(floor_point[1]) - 130.0)
    return evaluate(model, median, to_img, label, hoop_px=hoop_px)


def main():
    print(f"Line-based court fitting vs ground truth (tolerance {TOLERANCE_FT} ft):")
    results = []
    for model in COURT_MODELS:
        for camera, img_points in CAMERAS.items():
            court_to_img = build(model, img_points)
            frame, to_img = render(model, court_to_img)
            results.append(evaluate(model, frame, to_img, f"{model.name:14s} / {camera}"))

    # No painted lane at all -- the case that breaks lane-based calibration.
    model = COURT_MODELS[0]
    court_to_img = build(model, CAMERAS["sideline"])
    frame, to_img = render(model, court_to_img, paint_lane=False)
    results.append(evaluate(model, frame, to_img, "unpainted lane "))

    # Real conditions: players on the court, varied floor/paint/line colours.
    print("  -- video, players moving, fit on median frame --")
    for camera in CAMERAS:
        results.append(check_video_median(COURT_MODELS[0], camera,
                                          f"hs median / {camera:18s}"))
    print("  -- multi-sport line clutter (as real gyms are marked) --")
    for camera in CAMERAS:
        results.append(check_video_median(COURT_MODELS[0], camera,
                                          f"hs clutter / {camera:18s}", clutter=True))
    print("  -- varied colours, players moving, median frame --")
    for name, colours in COLOUR_SCHEMES.items():
        results.append(check_video_median(COURT_MODELS[0], "sideline",
                                          f"{name:18s} / sideline", colours=colours))

    failures = results.count(False)
    print(f"\n{len(results) - failures}/{len(results)} checks pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
