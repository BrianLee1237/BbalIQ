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
COVERAGE_WEIGHT = 60.0
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
          floor_area=None, line_cells=None, grid_shape=None):
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
    return model_to_line + COVERAGE_WEIGHT * (1.0 - coverage)


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
            floor_area, line_cells, grid_shape):
    """Search for the corner positions that best align the model's markings."""
    def cost_of(candidate):
        return _cost(candidate, court_corners, model_points, distance, shape,
                     floor_area, line_cells, grid_shape)

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
    step = shape[1] * REFINE_START_FRACTION
    best = None
    for model in COURT_MODELS:
        model_points = model.sample_points()
        court_corners = model.corners
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
                                            line_cells, grid_shape)
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
    rng = np.random.default_rng(0)
    for attempt in range(POLISH_ATTEMPTS):
        start = [list(pt) for pt in corners]
        if attempt > 0:
            jitter = shape[1] * POLISH_JITTER_FRACTION
            start = [[x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)]
                     for x, y in start]
        polished, polished_cost = _refine(
            start, model.corners, model_points, distance, shape,
            shape[1] * POLISH_STEP_FRACTION, floor_area, line_cells, grid_shape)
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
    return homography, {"model": model, "cost": cost, "corners": corners,
                        "mirrored": mirrored, "line_pixels": line_pixels}
