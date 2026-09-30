"""The court's two axes, recovered from where its lines converge.

Fitting a court by moving its four corners assumes the corners are in the
picture. On a zoomed-in game camera they usually are not: on the footage this
was built against, exactly one of the four court edges is visible, so three
of the "corners" the search starts from are frame edges. The search then has
eight free parameters and almost no evidence pinning them, and it settles on
whatever lands its lines on some marking -- a gym floor has several sports'
worth of those, so it settles wrongly and still scores well.

What a court gives you regardless of framing is direction. Every line painted
on it runs along one of two perpendicular axes, and so does every plank seam,
and so does every other sport's markings, since those courts are laid out
square to the room. Under perspective each family converges on a vanishing
point. Those two points fix the court's orientation without a single corner
being visible, and they leave only four unknowns -- the scale along each axis
and where the origin sits -- for the markings to settle.

Everything here is measured against the frame and the floor rather than set
in pixels, so the same code works on a phone clip and a 4K camera.
"""
from __future__ import annotations

import math

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - reported by the caller
    cv2 = None
    np = None

# Segments shorter than this fraction of the floor's own width are too short
# to say anything reliable about direction: a 2-degree error over a short
# segment is indistinguishable from noise, and letters and logo strokes live
# at that length. Expressed against the floor, not the frame, so zooming in
# or out doesn't change which structures qualify.
MIN_SEGMENT_FLOOR_FRACTION = 0.075
# How far inside the floor a segment must lie. The floor's boundary is where
# the benches, the crowd and the frame's own edge are, and tracing where the
# floor ends is not a marking. Measured on real footage, leaving this out let
# the second pencil lock onto the bench edge instead of the court.
INTERIOR_MARGIN_FLOOR_FRACTION = 0.03
# A detected line may be interrupted -- by a player, by worn paint, by
# another line crossing it -- and still be one line. How long an interruption
# is forgivable scales with the scene, not with how thick the paint is, so
# this is a fraction of the length a segment has to reach. Deriving it from
# line thickness instead gave a four-pixel tolerance that broke every real
# marking into fragments: 2 segments survived where 25 should have.
MAX_GAP_SEGMENT_FRACTION = 0.10
# Endpoint noise, in multiples of the detected line thickness, which becomes
# the angular tolerance below: a short segment of thick paint says less about
# direction than a long one of thin paint. Swept against real footage, 3, 5
# and 8 line widths all explain 24-25 of 25 segments and agree on the axes to
# within a degree, so this sits in the middle of a broad plateau rather than
# on a value that had to be found.
ENDPOINT_NOISE_LINE_WIDTHS = 5.0
# The two axes must actually be different directions; below this they are the
# same pencil found twice.
MIN_AXIS_SEPARATION_DEG = 12.0
RANSAC_ITERATIONS = 6000
HORIZON_SAMPLES = 4000


def _line_width_px(line_pixels):
    """Typical half-width of the detected markings, in pixels.

    Taken from the markings themselves with a distance transform rather than
    assumed, so it follows the footage's resolution and how far away the
    court is.
    """
    inside = cv2.distanceTransform(line_pixels, cv2.DIST_L2, 3)
    values = inside[line_pixels > 0]
    if values.size == 0:
        return 1.0
    return max(1.0, float(np.median(values)))


def _floor_span_px(floor_region):
    """A single length standing for how big the floor is in this frame."""
    contours, _ = cv2.findContours(floor_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float(min(floor_region.shape))
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return max(float(rect[1][0]), float(rect[1][1]))


def detect_segments(line_pixels, floor_region):
    """Long straight runs of marking, well inside the floor.

    Connectivity is deliberately not used. Court lines cross one another, so
    they form one tangled component whose shape says nothing, while lettering
    forms its own tidy elongated strips -- filtering components by shape
    measured exactly backwards, keeping the word "SCRIPPS RANCH" and dropping
    the court. A Hough transform asks the question that actually separates
    them: is there a long straight run of pixels here?
    """
    span = _floor_span_px(floor_region)
    width = _line_width_px(line_pixels)

    margin = int(span * INTERIOR_MARGIN_FLOOR_FRACTION) | 1
    interior = cv2.erode(floor_region, np.ones((max(3, margin), max(3, margin)), np.uint8))
    candidates = cv2.bitwise_and(line_pixels, interior)

    min_length = max(20.0, span * MIN_SEGMENT_FLOOR_FRACTION)
    raw = cv2.HoughLinesP(
        candidates, 1, np.pi / 360,
        threshold=int(min_length * 0.45),
        minLineLength=int(min_length),
        maxLineGap=int(max(3.0, min_length * MAX_GAP_SEGMENT_FRACTION)),
    )
    if raw is None:
        return np.empty((0, 4)), width
    return raw.reshape(-1, 4).astype(np.float64), width


def _as_lines(segments):
    return np.array([
        np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0]) for s in segments
    ])


def find_axes(segments, line_width_px, horizon=None):
    """The two vanishing points the segments converge on.

    Scored by angle, not by distance to the vanishing point. A distant
    vanishing point turns a fraction of a degree into hundreds of pixels, so
    a pixel threshold throws away the very pencils worth having -- measured,
    it explained 17 of 54 segments where the angular test explained 41.

    The tolerance comes from the data: a segment's direction is only as
    certain as its endpoints, so a short segment made of thick paint is
    allowed more angular slack than a long one made of thin paint.
    """
    if len(segments) < 4:
        return None
    mids = np.stack([(segments[:, 0] + segments[:, 2]) / 2,
                     (segments[:, 1] + segments[:, 3]) / 2], axis=1)
    directions = np.stack([segments[:, 2] - segments[:, 0],
                           segments[:, 3] - segments[:, 1]], axis=1)
    lengths = np.hypot(directions[:, 0], directions[:, 1])
    directions = directions / np.maximum(lengths, 1e-9)[:, None]
    lines = _as_lines(segments)

    noise = line_width_px * ENDPOINT_NOISE_LINE_WIDTHS
    tolerances = np.degrees(np.arctan2(noise, np.maximum(lengths, 1.0)))

    def inliers_of(vp, pool):
        if abs(vp[2]) < 1e-12:
            toward = np.tile([vp[0], vp[1]], (len(mids), 1))
        else:
            toward = np.array([vp[0] / vp[2], vp[1] / vp[2]]) - mids
        norm = np.hypot(toward[:, 0], toward[:, 1])
        cosang = np.abs(np.sum(toward * directions, axis=1)) / np.maximum(norm, 1e-9)
        angle = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
        ok = (norm > 1e-9) & (angle < tolerances)
        return [k for k in np.where(ok)[0] if k in pool]

    rng = np.random.default_rng(0)
    available = set(range(len(segments)))
    axes = []
    for _ in range(2):
        pool = sorted(available)
        if len(pool) < 2:
            break
        best = None
        if horizon is not None:
            # With the horizon known there is nothing to sample: a ground
            # plane's vanishing point lies on it, and each segment's line
            # meets it in exactly one place. So every segment proposes one
            # candidate, and we simply take the best-supported.
            #
            # This replaces projecting a freely-sampled vanishing point onto
            # the horizon afterwards, which quietly failed: the projection
            # moved the point far enough that its own segments stopped
            # counting as inliers, so the code fell back to the unconstrained
            # estimate it was meant to correct, leaving it 285px off.
            for i in pool:
                vp = np.cross(lines[i], horizon)
                if not np.isfinite(vp).all() or np.allclose(vp, 0.0):
                    continue
                inl = inliers_of(vp, available)
                if len(inl) < 2:
                    continue
                weight = float(lengths[inl].sum())
                if best is None or weight > best[0]:
                    best = (weight, vp, inl)
        else:
            for _ in range(RANSAC_ITERATIONS):
                i, j = rng.choice(pool, 2, replace=False)
                vp = np.cross(lines[i], lines[j])
                if not np.isfinite(vp).all() or np.allclose(vp, 0.0):
                    continue
                inl = inliers_of(vp, available)
                if len(inl) < 3:
                    continue
                weight = float(lengths[inl].sum())
                if best is None or weight > best[0]:
                    best = (weight, vp, inl)
        if best is None:
            break
        weight, vp, inl = best
        # Refit over every inlier rather than keeping the two-segment estimate
        # the sample happened to give. Two nearly parallel segments put their
        # intersection almost anywhere -- measured, an axis carrying only five
        # segments landed its vanishing point on the frame's corner, which
        # says the court recedes to nothing within the picture and collapsed
        # the fit to a sliver. Every inlier together is far better posed.
        for _ in range(3):
            if horizon is not None:
                refined = _vp_on_horizon(lines[inl], lengths[inl], horizon)
            else:
                refined = _least_squares_vp(lines[inl], lengths[inl])
            if refined is None:
                break
            grown = inliers_of(refined, available | set(inl))
            if len(grown) < 2:
                # Too few to refit again, but the refined point is still the
                # better estimate -- keep it rather than reverting.
                vp = refined
                break
            vp, inl = refined, grown
        weight = float(lengths[inl].sum())
        axes.append({"vp": vp / np.linalg.norm(vp), "segments": inl, "weight": weight})
        available -= set(inl)

    if len(axes) < 2:
        return None
    if _axis_separation_deg(axes[0], axes[1], mids) < MIN_AXIS_SEPARATION_DEG:
        return None
    return axes


def horizon_from_uprights(boxes, min_separation_px=None):
    """The ground plane's horizon, from people standing on it.

    Two upright objects of the same height give a point on the horizon: the
    line through their heads and the line through their feet meet there.
    Players are near enough the same height as each other for this, and a
    game frame has ten of them, so there are plenty of pairs.

    This is worth having because a pencil of markings can leave its vanishing
    point badly determined -- five short, nearly parallel segments put it
    almost anywhere, and measured on real footage one landed 218px off the
    horizon while the well-supported axis sat 49px from it. The horizon is
    the constraint that says so, and it comes from the players rather than
    from the markings, so it is independent evidence.

    `boxes` are (x1, y1, x2, y2) player boxes. Returns a homogeneous line, or
    None if there aren't enough usable pairs.
    """
    if len(boxes) < 4:
        return None
    feet, heads, heights = [], [], []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0
        feet.append(np.array([cx, y2, 1.0]))
        heads.append(np.array([cx, y1, 1.0]))
        heights.append(y2 - y1)
    heights = np.array(heights)
    # Drop boxes whose height is out of keeping with the rest: those are two
    # players merged into one box, or someone cut off by the frame, and
    # either breaks the equal-height assumption this rests on.
    low, high = np.percentile(heights, 15), np.percentile(heights, 95)
    usable = [i for i in range(len(boxes)) if low <= heights[i] <= high]
    if len(usable) < 4:
        return None

    if min_separation_px is None:
        # Two people standing nearly in line give a direction dominated by
        # box noise. Require them to be apart by a player's own height,
        # which scales with the footage instead of being a pixel count.
        min_separation_px = float(np.median(heights[usable]))

    rng = np.random.default_rng(0)
    points = []
    for _ in range(HORIZON_SAMPLES):
        i, j = rng.choice(usable, 2, replace=False)
        if abs(feet[i][0] - feet[j][0]) < min_separation_px:
            continue
        meet = np.cross(np.cross(feet[i], feet[j]), np.cross(heads[i], heads[j]))
        if abs(meet[2]) < 1e-9:
            continue
        points.append((meet / meet[2])[:2])
    if len(points) < 8:
        return None

    points = np.array(points)
    centre = np.median(points, axis=0)
    spread = np.median(np.abs(points - centre), axis=0)
    kept = points[np.abs(points[:, 1] - centre[1]) < 5.0 * max(1.0, spread[1])]
    if len(kept) < 8:
        return None
    vx, vy, x0, y0 = cv2.fitLine(kept.astype(np.float32), cv2.DIST_HUBER,
                                 0, 0.01, 0.01).ravel()
    return np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])


def _vp_on_horizon(inlier_lines, weights, horizon):
    """Best vanishing point for this pencil, restricted to the horizon.

    A vanishing point of the ground plane lies on its horizon, so there is
    one degree of freedom here, not two. Writing the point as a combination
    of two points spanning the horizon turns the fit into a two-unknown
    least-squares problem, which a handful of segments can support where an
    unconstrained fit cannot.
    """
    a, b, c = horizon
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return None
    # Any point on the line, plus its direction (a point at infinity on it).
    if abs(a) >= abs(b):
        anchor = np.array([-c / a, 0.0, 1.0])
    else:
        anchor = np.array([0.0, -c / b, 1.0])
    along = np.array([b, -a, 0.0])

    scale = np.maximum(np.hypot(inlier_lines[:, 0], inlier_lines[:, 1]), 1e-9)
    normalised = inlier_lines / scale[:, None]
    weighting = np.sqrt(np.maximum(weights, 1e-9))[:, None]
    design = np.stack([normalised @ anchor, normalised @ along], axis=1) * weighting
    try:
        _, _, vt = np.linalg.svd(design)
    except np.linalg.LinAlgError:
        return None
    alpha, beta = vt[-1]
    vp = alpha * anchor + beta * along
    if not np.isfinite(vp).all() or np.allclose(vp, 0.0):
        return None
    return vp


def _least_squares_vp(inlier_lines, weights):
    """The point lying closest to all of these lines at once.

    A vanishing point is on every line of its pencil, so it is the null
    vector of their stacked coefficients. Longer segments locate their line
    better, so they are weighted accordingly. Taken as the smallest singular
    vector, which is the least-squares answer when the lines don't meet
    exactly -- and they never do.
    """
    if len(inlier_lines) < 2:
        return None
    scale = np.maximum(np.hypot(inlier_lines[:, 0], inlier_lines[:, 1]), 1e-9)
    normalised = inlier_lines / scale[:, None]
    weighted = normalised * np.sqrt(np.maximum(weights, 1e-9))[:, None]
    try:
        _, _, vt = np.linalg.svd(weighted)
    except np.linalg.LinAlgError:
        return None
    vp = vt[-1]
    if not np.isfinite(vp).all() or np.allclose(vp, 0.0):
        return None
    return vp


def _axis_separation_deg(first, second, mids):
    """Angle between the two pencils, at the middle of the picture."""
    centre = mids.mean(axis=0)

    def direction(axis):
        vp = axis["vp"]
        if abs(vp[2]) < 1e-12:
            v = np.array([vp[0], vp[1]])
        else:
            v = np.array([vp[0] / vp[2], vp[1] / vp[2]]) - centre
        return v / max(1e-9, np.linalg.norm(v))

    cosang = abs(float(np.dot(direction(first), direction(second))))
    return math.degrees(math.acos(min(1.0, cosang)))


def homography_from_axes(vp_x, vp_y, origin_px, scale_x, scale_y):
    """Court feet -> image pixels, given the two axes and where the court starts.

    With the vanishing points known, a court point (X, Y) maps through the
    matrix whose first two columns are the axis directions and whose third is
    the origin. That leaves only the two scales and the origin to find, four
    numbers instead of the eight a free-corner search has to guess.
    """
    column_x = np.asarray(vp_x, dtype=np.float64) * scale_x
    column_y = np.asarray(vp_y, dtype=np.float64) * scale_y
    column_o = np.array([origin_px[0], origin_px[1], 1.0], dtype=np.float64)
    return np.stack([column_x, column_y, column_o], axis=1)
