"""Fit a court model to the line markings visible in a frame.

Scores a whole hypothesised court against every line pixel at once (a chamfer
distance), rather than trying to identify individual lines. Identifying lines
first is what defeated the earlier attempt: under perspective, court lines
that are parallel on the floor are not parallel in the image, so grouping
them by angle produces families that don't correspond to anything real.

Search shape: the visible floor gives a starting quad, each court model and
each way round that quad gives a hypothesis, and each hypothesis is then
refined by nudging its four corners until the model's markings line up with
the painted ones. The best-scoring hypothesis wins, which also settles the
court's orientation and which level's dimensions it is -- all from the
markings themselves, with nothing hardcoded about colour.
"""
from __future__ import annotations

import math
from typing import Optional

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None

from court_model import COURT_MODELS

LINE_KERNEL_FRACTION = 0.006     # wider than a painted line, narrower than filled paint
PAINT_MIN_FLOOR_FRACTION = 0.04  # a thick region this big is paint; smaller is a player
REFINE_ROUNDS = 200              # corner-nudging iterations per hypothesis
REFINE_START_FRACTION = 0.05     # first nudge, as a fraction of frame width
REFINE_SHRINK = 0.7              # step decay once a pass stops improving
REFINE_MIN_STEP_PX = 0.25        # keep refining until nudges are sub-pixel
REFINE_RESTARTS = 3              # re-widen the step afterwards to escape shallow minima
INITIAL_SHRINKS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.72)  # court sits inside the floor by the apron
POLISH_ATTEMPTS = 6              # descents from the winner, first un-jittered
POLISH_JITTER_FRACTION = 0.012   # how far to displace corners between attempts
POLISH_STEP_FRACTION = 0.015     # polish starts finer than the coarse sweep
# Distances are capped before averaging. Without a cap a few points stranded
# far from any marking dominate the mean, so the search optimises for those
# outliers instead of the alignment of everything else, and the landscape it
# has to descend is much rougher.
DISTANCE_CAP_PX = 40.0
# One-way chamfer (every model line near a real line) is exploitable: shrink
# the model onto a dense patch of markings and it scores beautifully while
# being completely wrong -- measured at 1.0px alongside 51ft of court error.
# So also require the reverse, that real markings are EXPLAINED by the model.
# Together they pin the fit: the model can neither wander off the markings nor
# collapse onto a subset of them.
# Players hide parts of the markings, so a fraction of the model will always
# have nothing to match. Averaging over everything makes those hidden segments
# dominate -- measured, a known-perfect alignment scored 1.6px with an empty
# court and 15.3px with ten players on it, close enough to the reject
# threshold that wrong answers started winning. Averaging over the best-
# matching share instead ignores what's occluded without letting the model
# wander, since the rest still has to line up.
TRIM_KEEP_FRACTION = 0.7
COVERAGE_CELL_PX = 14.0
# Coverage must stay gentle. A school gym floor carries several sports' lines
# -- volleyball, badminton, a second basketball court -- and a basketball
# model can never explain those, so demanding full coverage penalises the
# CORRECT alignment hardest. It's here to stop the model shrinking onto a
# dense patch, not to insist every line belongs to basketball.
COVERAGE_WEIGHT = 25.0
# The rim is a far stronger anchor than any line, because there's exactly one
# and we detect it directly. Its floor point sits at the middle of the court's
# width, just off the baseline, so a fit that puts the court elsewhere is
# wrong no matter how neatly its lines happen to land on somebody else's
# sport. Measured on real footage: a 3.0px marking error with the lane sitting
# in open floor and the arc curving the wrong way.
HOOP_WEIGHT = 90.0
HOOP_HEIGHT_FT = 10.0            # a basketball rim, by the rules of the game
# Turns the rim miss in feet into a penalty: at this many feet out, the rim
# term costs HOOP_WEIGHT. Set to about the lane's width, so landing the hoop
# a lane away from the rim is already a decisive objection.
HOOP_ERROR_SCALE_FT = 12.0

# The filled painted key, matched by overlap. Lines alone cannot say WHERE on
# a multi-sport floor the basketball court is, because volleyball, badminton
# and cross-court keys are painted in the same way and a wrong alignment can
# land its lines on theirs. A large filled area of paint is different: no
# other sport has one, so the key's position, the court's front-to-back
# direction and the lane's width -- which is what separates a high-school
# court from a college one -- are all pinned by it. Weighted above the
# markings term because it is the more trustworthy evidence of the two.
PAINT_WEIGHT = 120.0
KEY_MIN_FLOOR_FRACTION = 0.01   # measured: the key runs 6.6-8.8% of the floor
KEY_MAX_FLOOR_FRACTION = 0.15   # crowd wedges inside the hull ran 21-23%
KEY_MIN_RECTANGULARITY = 0.75   # key fills 0.90-0.96 of its minAreaRect; wedges 0.51
KEY_MIN_ENCLOSURE = 0.90        # fraction of the ring around it that must be floor
KEY_EDGE_MARGIN = 0.008         # of frame width; a key touching the floor's edge isn't one
KEY_SAMPLE_POINTS = 800
MAX_COST_PX = 18.0               # mean line-to-model distance we'll still believe
# The modelled half-court can't be a sliver of the visible floor, nor vastly
# bigger than it. Bounds are loose -- they exist to rule out collapse, not to
# tune the fit.
MIN_COURT_AREA_RATIO = 0.25
MAX_COURT_AREA_RATIO = 3.0


def median_frame(video_path, samples=31):
    """A pixel-wise median across frames -- the court without the players.

    The court never moves and the players never stop, so at any given pixel
    the floor is what's there most of the time and a player is a brief
    excursion. Taking the median keeps the former and discards the latter.

    This matters because players occlude the markings the fit depends on.
    Trimming the cost tolerates some of that, but occlusion also roughens the
    search landscape badly: with ten players the search reached 9.2px where
    4.7px was available and the answer was ~28ft out, despite the objective
    itself still preferring the truth. Removing the players is a better
    answer than tolerating them.
    """
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required.")
    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        capture.release()
        return None
    step = max(1, total // samples)
    frames = []
    for index in range(0, total, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
        if len(frames) >= samples:
            break
    capture.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def detect_line_pixels(frame, floor_region, floor_mask=None):
    """Thin painted structures inside the floor -- the court's markings.

    Markings are thin; filled paint, players and shadows are not. Subtracting
    an opened copy keeps only what the opening destroys, which is exactly the
    thin things.

    Computes its own floor mask rather than accepting the pipeline's. That
    one has been morphologically closed with a kernel tens of pixels wide to
    consolidate the floor, which swallows every painted line into it -- the
    markings this function exists to find are gone before it starts. Measured:
    of ~103k non-floor pixels, the closed mask left only 5k thin ones, and
    the true court alignment scored 304px because the lines it should have
    matched weren't there.
    """
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required.")
    from courtiq_core import floor_color_mask

    height, width = frame.shape[:2]
    raw_floor = floor_color_mask(frame)
    non_floor = ((raw_floor == 0) & (floor_region > 0)).astype(np.uint8) * 255
    size = max(3, int(width * LINE_KERNEL_FRACTION) | 1)
    kernel = np.ones((size, size), np.uint8)
    opened = cv2.morphologyEx(non_floor, cv2.MORPH_OPEN, kernel)
    thin = cv2.subtract(non_floor, opened)

    # Include the OUTLINE of filled paint. Where a lane is painted, its
    # boundary lines sit against that paint, so opening removes them along
    # with it and the model's lane lines have nothing left to match -- which
    # measured as 16.6px of error on a known-perfect alignment. The edge of a
    # painted lane is itself a court line, so adding those outlines restores
    # exactly what was lost.
    #
    # Only for regions big enough to BE paint, though. Players are thick too,
    # so outlining everything traces ten silhouettes as if they were markings;
    # measured, that wrecked the fit outright (34-87ft). A painted lane is far
    # larger than a player, so area separates them cleanly.
    floor_area = float(cv2.countNonZero(floor_region))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    paint = np.zeros_like(opened)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= floor_area * PAINT_MIN_FLOOR_FRACTION:
            paint[labels == label] = 255
    outlines = cv2.morphologyEx(paint, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return cv2.bitwise_or(thin, outlines)


def _distance_to_lines(line_pixels):
    """Distance from every pixel to the nearest marking pixel."""
    return cv2.distanceTransform((line_pixels == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _quad_area(corners):
    total = 0.0
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _is_convex(corners):
    signs = []
    for i in range(4):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % 4]
        cx, cy = corners[(i + 2) % 4]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def _cost(img_corners, court_corners, model_points, distance, shape,
          floor_area=None, line_cells=None, grid_shape=None,
          hoop_px=None, hoop_court=None,
          paint_img=None, paint_mask=None, lane_grid=None, lane_rect=None):
    """Mean distance from the model's markings to the nearest real marking.

    Rejects degenerate hypotheses first. Without that, the search has a
    trivial winner: squash the whole court onto a single image line and every
    model point lands on a real marking, scoring near zero. Measured, that
    minimum was found every time -- 0.8px of marking error alongside 60ft of
    actual court error.
    """
    if not _is_convex(img_corners):
        return float("inf")
    if floor_area:
        area = _quad_area(img_corners)
        if not (floor_area * MIN_COURT_AREA_RATIO <= area <= floor_area * MAX_COURT_AREA_RATIO):
            return float("inf")
    try:
        transform = cv2.getPerspectiveTransform(
            np.array(court_corners, dtype=np.float32),
            np.array(img_corners, dtype=np.float32),
        )
    except cv2.error:
        return float("inf")
    projected = cv2.perspectiveTransform(model_points.reshape(-1, 1, 2), transform).reshape(-1, 2)

    height, width = shape
    xs = projected[:, 0]
    ys = projected[:, 1]
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if inside.sum() < len(projected) * 0.35:
        # Mostly off-screen: the court can't be mainly outside the picture.
        return float("inf")
    sampled = distance[ys[inside].astype(np.int32), xs[inside].astype(np.int32)]
    sampled = np.minimum(sampled, DISTANCE_CAP_PX)
    # Charge off-screen points the worst on-screen distance, so a hypothesis
    # can't score well by pushing most of the court out of frame.
    if len(sampled) == 0:
        return float("inf")
    keep = max(1, int(len(sampled) * TRIM_KEEP_FRACTION))
    trimmed = float(np.mean(np.sort(sampled)[:keep]))
    # Charge for the court leaving the frame separately, so trimming can
    # forgive occlusion without also forgiving a model that drifts off-screen.
    off_screen_fraction = 1.0 - (inside.sum() / len(projected))
    model_to_line = trimmed + DISTANCE_CAP_PX * off_screen_fraction
    if line_cells is None:
        return model_to_line

    # Coverage: what fraction of the real markings has a model line near it.
    # Computed on a coarse grid so it costs almost nothing per evaluation.
    rows, cols = grid_shape
    gx = np.clip((xs[inside] / COVERAGE_CELL_PX).astype(np.int32), 0, cols - 1)
    gy = np.clip((ys[inside] / COVERAGE_CELL_PX).astype(np.int32), 0, rows - 1)
    occupied = np.zeros((rows, cols), dtype=bool)
    occupied[gy, gx] = True
    # Spread each occupied cell to its neighbours. Model points are sampled
    # along the markings in FEET, so in the image they land further apart than
    # a cell wherever the court is near the camera, skipping cells that the
    # model genuinely covers. Measured, that made even a perfect alignment
    # score 31% coverage, so the penalty was punishing the correct answer.
    spread = occupied.copy()
    spread[1:, :] |= occupied[:-1, :]
    spread[:-1, :] |= occupied[1:, :]
    spread[:, 1:] |= occupied[:, :-1]
    spread[:, :-1] |= occupied[:, 1:]
    coverage = float(spread.ravel()[line_cells].mean()) if len(line_cells) else 0.0
    total = model_to_line + COVERAGE_WEIGHT * (1.0 - coverage)

    if hoop_px is not None and hoop_court is not None:
        projected_hoop = cv2.perspectiveTransform(
            np.array([[[hoop_court[0], hoop_court[1]]]], dtype=np.float32), transform
        )[0][0]
        # The rim is 10ft above its floor point, so in the image it sits
        # above that point by roughly ten times whatever a foot is worth
        # there. Measure the gap against that, in feet, rather than only
        # checking the rim is somewhere above: a court laid at right angles
        # to the true one put its hoop 600px below the rim at the same x,
        # which scored zero on both an across-frame test and an ordering
        # test and so was never penalised at all.
        #
        # The scale comes from the hypothesis itself -- how far apart a foot
        # of court lands near the hoop -- so this holds at any resolution or
        # camera distance.
        neighbour = cv2.perspectiveTransform(
            np.array([[[hoop_court[0], hoop_court[1] + 1.0]]], dtype=np.float32), transform
        )[0][0]
        px_per_ft = max(1e-6, float(np.hypot(neighbour[0] - projected_hoop[0],
                                             neighbour[1] - projected_hoop[1])))
        gap_x = (float(projected_hoop[0]) - hoop_px[0]) / px_per_ft
        gap_y = (float(projected_hoop[1]) - hoop_px[1]) / px_per_ft
        # Sideways the rim should be right over its floor point; vertically
        # it should be about a rim's height above it.
        error_ft = math.hypot(gap_x, gap_y - HOOP_HEIGHT_FT)
        total += HOOP_WEIGHT * (error_ft / HOOP_ERROR_SCALE_FT)

    if paint_img is not None and lane_grid is not None:
        total += PAINT_WEIGHT * (1.0 - _paint_agreement(
            transform, court_corners, img_corners,
            paint_img, paint_mask, lane_grid, lane_rect, shape))
    return total


def _paint_agreement(transform, court_corners, img_corners,
                     paint_img, paint_mask, lane_grid, lane_rect, shape):
    """How well the model's lane and the floor's painted area coincide, 0..1.

    Scored both ways round, because either direction alone is satisfiable by
    a degenerate answer: a lane shrunk to a point sits entirely inside the
    paint, and a lane blown up to the whole court contains all of it.

    The paint's pixels are carried back into court coordinates rather than the
    lane being rasterised into the image, which turns the test into "is this
    point inside an axis-aligned rectangle" -- cheap enough to run inside the
    search loop.
    """
    try:
        inverse = cv2.getPerspectiveTransform(
            np.array(img_corners, dtype=np.float32),
            np.array(court_corners, dtype=np.float32),
        )
    except cv2.error:
        return 0.0
    left, right, near, far = lane_rect
    back = cv2.perspectiveTransform(paint_img.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    precision = float(np.mean(
        (back[:, 0] >= left) & (back[:, 0] <= right)
        & (back[:, 1] >= near) & (back[:, 1] <= far)))

    height, width = shape
    forward = cv2.perspectiveTransform(lane_grid.reshape(-1, 1, 2), transform).reshape(-1, 2)
    xs = np.clip(forward[:, 0].astype(np.int32), 0, width - 1)
    ys = np.clip(forward[:, 1].astype(np.int32), 0, height - 1)
    on_screen = ((forward[:, 0] >= 0) & (forward[:, 0] < width)
                 & (forward[:, 1] >= 0) & (forward[:, 1] < height))
    recall = float(np.mean((paint_mask[ys, xs] > 0) & on_screen))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _hull_quad(hull):
    """Four corners approximating a floor hull.

    Sweeps the approximation tolerance looking for a genuine 4-corner fit
    rather than taking one tolerance and falling back to a bounding rectangle.
    When the floor runs off the side of the frame the hull picks up extra
    corners where the image crops it -- measured, a 5-corner hull on a
    diagonal camera fell back to minAreaRect, whose axis-aligned box is a poor
    starting guess for an angled court, and the search never recovered (30ft
    of error against 0.2ft for the same camera at other court dimensions).
    """
    peri = cv2.arcLength(hull, True)
    for epsilon in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12):
        approx = cv2.approxPolyDP(hull, epsilon * peri, True)
        if len(approx) == 4:
            return [tuple(map(float, p)) for p in approx.reshape(-1, 2)]
    return [tuple(map(float, p)) for p in cv2.boxPoints(cv2.minAreaRect(hull))]


def _global_moves(corners, step):
    """Whole-quad moves: translate, scale, rotate.

    Nudging one corner at a time can only reach a better fit by walking
    through worse ones whenever the whole court needs to shift or resize, so
    the search stalls in shallow minima. These moves change the quad
    coherently and converge far faster on exactly those cases.
    """
    cx = sum(p[0] for p in corners) / 4.0
    cy = sum(p[1] for p in corners) / 4.0
    moves = []
    for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
        moves.append([[x + dx, y + dy] for x, y in corners])
    size = max(1.0, math.hypot(corners[0][0] - cx, corners[0][1] - cy))
    for factor in (1 + step / size, 1 - step / size):
        moves.append([[cx + (x - cx) * factor, cy + (y - cy) * factor] for x, y in corners])
    for angle in (step / size, -step / size):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        moves.append([[cx + (x - cx) * cos_a - (y - cy) * sin_a,
                       cy + (x - cx) * sin_a + (y - cy) * cos_a] for x, y in corners])
    return moves


def _refine(img_corners, court_corners, model_points, distance, shape, step,
            floor_area, line_cells, grid_shape, hoop_px=None, hoop_court=None,
            paint=None):
    """Search for the corner positions that best align the model's markings."""
    paint = paint or {}

    def cost_of(candidate):
        return _cost(candidate, court_corners, model_points, distance, shape,
                     floor_area, line_cells, grid_shape, hoop_px, hoop_court,
                     **paint)

    corners = [list(pt) for pt in img_corners]
    best = cost_of(corners)
    initial_step = step
    for restart in range(REFINE_RESTARTS + 1):
        step = initial_step if restart == 0 else initial_step * (0.3 ** restart)
        for _ in range(REFINE_ROUNDS):
            improved = False
            candidates = _global_moves(corners, step)
            for index in range(4):
                for axis in (0, 1):
                    for direction in (1, -1):
                        trial = [list(pt) for pt in corners]
                        trial[index][axis] += direction * step
                        candidates.append(trial)
            for trial in candidates:
                cost = cost_of(trial)
                if cost < best:
                    best, corners, improved = cost, [list(p) for p in trial], True
            if not improved:
                step *= REFINE_SHRINK
                if step < REFINE_MIN_STEP_PX:
                    break
    return corners, best


def detect_key_paint(frame, floor_region):
    """Pixels of the painted key, or None when this view doesn't show one.

    Returning None is a normal outcome, not a failure. The key anchor exists
    to break the ambiguity of a multi-sport floor, so it is only worth having
    when we are sure which region the key is; a wrongly chosen one points the
    search away from the answer, which is worse than not anchoring at all.
    Measured: taking the biggest painted area scored the TRUE alignment at 0%
    agreement on a diagonal camera, because the floor's convex hull spans
    wedges of crowd past the court's corners and the largest of those dwarfs
    the key.

    Three properties identify it, and a region must have all three:

    - Size. The key is a few percent of the floor. Those crowd wedges ran to
      21-23%, well outside the range a lane can occupy.
    - Rectangularity, measured as how much of its minimum-area rectangle it
      fills. A lane is a rectangle in perspective, so it fills 0.90-0.96; the
      wedges fill 0.51. Note this is not convexity -- the wedges are convex,
      and testing them against their convex hull passed them at 1.00.
    - Enclosure. The key lies within the floor, with floor all around it.

    On a camera looking down the court the key fails enclosure honestly: the
    apron beyond the far baseline is only a few pixels deep in perspective,
    so the key merges with the crowd and becomes a notch in the floor's edge
    rather than a region inside it. There is no key to find in that view, and
    this returns None rather than settling for the nearest thing.
    """
    from courtiq_core import floor_color_mask

    wood = floor_color_mask(frame)
    contours, _ = cv2.findContours(floor_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    boundary = np.zeros(floor_region.shape, np.uint8)
    cv2.drawContours(boundary, [max(contours, key=cv2.contourArea)], -1, 255,
                     max(3, int(frame.shape[1] * KEY_EDGE_MARGIN)))

    non_floor = ((wood == 0) & (floor_region > 0)).astype(np.uint8) * 255
    size = max(3, int(frame.shape[1] * LINE_KERNEL_FRACTION) | 1)
    thick = cv2.morphologyEx(non_floor, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    floor_area = float(cv2.countNonZero(floor_region))
    best = None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (floor_area * KEY_MIN_FLOOR_FRACTION <= area
                <= floor_area * KEY_MAX_FLOOR_FRACTION):
            continue
        component = (labels == label).astype(np.uint8) * 255
        if cv2.countNonZero(cv2.bitwise_and(component, boundary)):
            continue  # runs into the edge of the floor, so it is not enclosed
        outline, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not outline:
            continue
        rect = cv2.minAreaRect(max(outline, key=cv2.contourArea))
        rect_area = rect[1][0] * rect[1][1]
        if rect_area <= 0 or area / rect_area < KEY_MIN_RECTANGULARITY:
            continue
        ring = cv2.subtract(cv2.dilate(component, np.ones((15, 15), np.uint8)), component)
        ring_px = max(1, cv2.countNonZero(ring))
        if cv2.countNonZero(cv2.bitwise_and(ring, wood)) / ring_px < KEY_MIN_ENCLOSURE:
            continue
        if best is None or area > best[0]:
            best = (area, label)
    if best is None:
        return None
    ys, xs = np.nonzero(labels == best[1])
    if len(xs) > KEY_SAMPLE_POINTS:  # enough to locate an area, cheap in the search loop
        pick = np.linspace(0, len(xs) - 1, KEY_SAMPLE_POINTS).astype(np.int32)
        ys, xs = ys[pick], xs[pick]
    return np.stack([xs, ys], axis=1).astype(np.float32), thick


def _paint_terms(model, paint_img, paint_mask):
    """The paint-anchor arguments for this court model, or none if no paint."""
    if paint_img is None:
        return None
    left = (model.width - model.lane_width) / 2.0
    right = (model.width + model.lane_width) / 2.0
    grid_x, grid_y = np.meshgrid(np.linspace(left, right, 10),
                                np.linspace(0.0, model.lane_length, 18))
    lane_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    return {"paint_img": paint_img, "paint_mask": paint_mask,
            "lane_grid": lane_grid, "lane_rect": (left, right, 0.0, model.lane_length)}


def _fit_on_axes(frame, floor_region, line_pixels, shape, distance, floor_area,
                 line_cells, grid_shape, hoop_px, paint_img, paint_mask, verbose):
    """Fit with the court's orientation taken from its vanishing points.

    Returns the same (homography, info) as fit_court, or None when the axes
    can't be found or the result doesn't line up with the markings -- in
    which case the caller falls back to the free-corner search.
    """
    import court_axes
    import court_axes_fit

    segments, line_width = court_axes.detect_segments(line_pixels, floor_region)
    axes = court_axes.find_axes(segments, line_width)
    if axes is None:
        if verbose:
            print("[court_fit] could not find two court axes; "
                  "falling back to searching the corners.")
        return None
    if verbose:
        print(f"[court_fit] court axes from {len(axes[0]['segments'])}"
              f"+{len(axes[1]['segments'])} of {len(segments)} line segments.")

    def cost_of(corners, model):
        paint = _paint_terms(model, paint_img, paint_mask) or {}
        return _cost(corners, model.corners, model.sample_points(), distance,
                     shape, floor_area, line_cells, grid_shape,
                     hoop_px, model.hoop, **paint)

    result = court_axes_fit.fit_on_axes(COURT_MODELS, axes, floor_region, shape,
                                        cost_of, verbose=verbose)
    if result is None:
        return None

    model, corners = result["model"], result["corners"]
    # Judge acceptance on the marking distance alone, as the free-corner
    # search does: the combined score carries search-only penalties.
    cost = _cost(corners, model.corners, model.sample_points(), distance,
                 shape, floor_area)
    if verbose:
        print(f"[court_fit] axis-locked fit: {model.name} court, "
              f"mean marking error {cost:.1f}px")
    if cost > MAX_COST_PX:
        if verbose:
            print(f"[court_fit] axis-locked fit rejected ({cost:.1f}px > "
                  f"{MAX_COST_PX}px); falling back to searching the corners.")
        return None

    court_to_image = cv2.getPerspectiveTransform(
        np.array(model.corners, dtype=np.float32),
        np.array(corners, dtype=np.float32),
    )
    agreement = None
    paint = _paint_terms(model, paint_img, paint_mask)
    if paint is not None:
        agreement = _paint_agreement(
            court_to_image, model.corners, corners, paint["paint_img"],
            paint["paint_mask"], paint["lane_grid"], paint["lane_rect"], shape)
    return np.linalg.inv(court_to_image), {
        "model": model, "cost": cost, "corners": corners, "mirrored": False,
        "line_pixels": line_pixels, "paint_agreement": agreement,
        "from_axes": True,
    }


def fit_court(frame, floor_region, floor_mask, hoop_px=None, verbose=True):
    """Best-fitting court model for this frame.

    Returns (homography_image_to_court, info) or None. The homography maps
    image pixels to court feet for the model that fits best.
    """
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required.")
    line_pixels = detect_line_pixels(frame, floor_region, floor_mask)
    if cv2.countNonZero(line_pixels) < 200:
        return None
    found = detect_key_paint(frame, floor_region)
    paint_img, paint_mask = found if found else (None, None)
    if verbose:
        print("[court_fit] painted key found, anchoring on it." if found
              else "[court_fit] no painted key in this view; fitting on markings alone.")
    distance = _distance_to_lines(line_pixels)
    shape = frame.shape[:2]

    contours, _ = cv2.findContours(floor_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    quad = _hull_quad(hull)

    floor_area = float(cv2.countNonZero(floor_region))
    rows = int(shape[0] / COVERAGE_CELL_PX) + 1
    cols = int(shape[1] / COVERAGE_CELL_PX) + 1
    grid_shape = (rows, cols)
    line_ys, line_xs = np.nonzero(line_pixels)
    line_cells = np.unique(
        (line_ys / COVERAGE_CELL_PX).astype(np.int32) * cols
        + (line_xs / COVERAGE_CELL_PX).astype(np.int32)
    )
    # Prefer a fit locked to the court's own axes. Those are fixed by every
    # long line in the picture, and they leave four parameters to settle
    # instead of eight -- which matters most on exactly the footage the
    # free-corner search cannot handle, where the court's corners are outside
    # the frame and there is nothing to start those eight from.
    axed = _fit_on_axes(frame, floor_region, line_pixels, shape, distance,
                        floor_area, line_cells, grid_shape, hoop_px,
                        paint_img, paint_mask, verbose)
    if axed is not None:
        return axed

    step = shape[1] * REFINE_START_FRACTION
    best = None
    for model in COURT_MODELS:
        model_points = model.sample_points()
        court_corners = model.corners
        paint = _paint_terms(model, paint_img, paint_mask)
        for mirrored in (False, True):
            base = quad[::-1] if mirrored else quad
            for rotation in range(4):
                rotated = [base[(rotation + k) % 4] for k in range(4)]
                # The visible floor includes the out-of-bounds apron, so the
                # court itself sits inside it. Starting from a few shrunken
                # versions as well gives the search a start near the answer
                # instead of one the apron's width away.
                for shrink in INITIAL_SHRINKS:
                    qcx = sum(p[0] for p in rotated) / 4.0
                    qcy = sum(p[1] for p in rotated) / 4.0
                    start = [[qcx + (x - qcx) * shrink, qcy + (y - qcy) * shrink]
                             for x, y in rotated]
                    corners, cost = _refine(start, court_corners, model_points,
                                            distance, shape, step, floor_area,
                                            line_cells, grid_shape, hoop_px, model.hoop,
                                            paint)
                    if best is None or cost < best[0]:
                        best = (cost, corners, model, mirrored, rotation)

    if best is None:
        return None

    # Polish the winner: descend again from it, and from a few jittered
    # copies. The coarse sweep gets close but can settle just short of the
    # optimum -- measured, a diagonal camera stopped at 6.4px where 2.8px was
    # reachable, which is the difference between 2.4ft and sub-foot accuracy.
    # Jittering gives the descent a way out of a shallow basin.
    cost, corners, model, mirrored, rotation = best
    model_points = model.sample_points()
    paint = _paint_terms(model, paint_img, paint_mask)
    rng = np.random.default_rng(0)
    for attempt in range(POLISH_ATTEMPTS):
        start = [list(pt) for pt in corners]
        if attempt > 0:
            jitter = shape[1] * POLISH_JITTER_FRACTION
            start = [[x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)]
                     for x, y in start]
        polished, polished_cost = _refine(
            start, model.corners, model_points, distance, shape,
            shape[1] * POLISH_STEP_FRACTION, floor_area, line_cells, grid_shape,
            hoop_px, model.hoop, paint)
        if polished_cost < cost:
            cost, corners = polished_cost, polished
    # Judge acceptance on the marking distance alone. The combined score
    # includes the coverage penalty, which is a search signal rather than a
    # measure of alignment quality, and mixing them makes the threshold
    # meaningless.
    cost = _cost(corners, model.corners, model_points, distance, shape, floor_area)
    if verbose:
        print(f"[court_fit] best fit: {model.name} court, mean marking error {cost:.1f}px "
              f"(rotation {rotation}{', mirrored' if mirrored else ''})")
    if cost > MAX_COST_PX:
        if verbose:
            print(f"[court_fit] rejecting: {cost:.1f}px exceeds the {MAX_COST_PX}px limit, "
                  f"so the model never actually lined up with the markings.")
        return None

    court_to_image = cv2.getPerspectiveTransform(
        np.array(model.corners, dtype=np.float32),
        np.array(corners, dtype=np.float32),
    )
    homography = np.linalg.inv(court_to_image)
    # Report the paint agreement alongside the marking error. The marking
    # error on its own can look excellent on a wrong answer -- on real
    # multi-sport footage it read 3.0px while the lane sat in open floor --
    # so a number that says whether the key landed on the key belongs in the
    # result rather than only inside the search.
    agreement = None
    if paint is not None:
        agreement = _paint_agreement(
            court_to_image, model.corners, corners, paint["paint_img"],
            paint["paint_mask"], paint["lane_grid"], paint["lane_rect"], shape)
        if verbose:
            print(f"[court_fit] painted key agreement: {agreement:.0%}")
    return homography, {"model": model, "cost": cost, "corners": corners,
                        "mirrored": mirrored, "line_pixels": line_pixels,
                        "paint_agreement": agreement}
