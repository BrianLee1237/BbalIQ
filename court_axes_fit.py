"""Fit a court model that is locked to the two axes found in the image.

The free-corner search has eight parameters. Once the court's two vanishing
points are known, only four are left: how many pixels a foot covers along
each axis, and where on the image the court's origin sits. Everything else
-- the orientation, and the way both shrink with distance -- is already
determined.

That is the whole point. Eight free parameters over a floor carrying several
sports' markings has enough freedom to land the model's lines on somebody
else's and score well anywhere; four, with the directions fixed by every long
line in the picture, does not. It also stops depending on the court's corners
being visible, which on a game camera they usually are not.

The cost function is the one the free-corner search already uses, so
whatever a hypothesis is worth is measured the same way as before; only the
set of hypotheses reachable has changed.
"""
from __future__ import annotations

import math

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - reported by the caller
    cv2 = None
    np = None

# Scales to try, as pixels-per-foot relative to a first guess taken from the
# floor's size in frame. Wide enough to cover seeing a third of the court or
# all of it, since how much is in shot is exactly what we don't know.
SCALE_MULTIPLIERS = (0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.6)
# Where the court's origin might sit, as a fraction of the frame, swept
# across and beyond it: the origin corner is very often out of shot.
ORIGIN_SWEEP = (-0.6, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2)
# The court has to explain the floor we can see. Area alone does not say
# that: a long thin sliver laid diagonally across the picture can have
# exactly the right area while covering almost none of the actual floor, and
# that is precisely what an ill-determined axis produces. Overlap catches it.
# Set low because the floor includes the out-of-bounds apron and is clipped
# by the frame, so even a perfect court explains only part of it -- this is
# a guard against nonsense, not a measure of quality.
MIN_FLOOR_OVERLAP = 0.35
REFINE_ROUNDS = 90
REFINE_SHRINK = 0.72
# Refinement stops when a nudge moves the origin less than this fraction of
# the frame, and the scales by less than this in relative terms.
REFINE_MIN_RELATIVE_STEP = 2.5e-4


def _axis_direction(vp, at_point):
    """Unit image direction of an axis, seen from a point in the picture."""
    if abs(vp[2]) < 1e-12:
        v = np.array([vp[0], vp[1]], dtype=np.float64)
    else:
        v = np.array([vp[0] / vp[2], vp[1] / vp[2]], dtype=np.float64) - at_point
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return None
    return v / norm


def court_to_image(vp_x, vp_y, origin, scale_x, scale_y):
    """The 3x3 taking court feet to image pixels for these four parameters.

    With the vanishing points known, a court point (X, Y) maps through the
    matrix whose first two columns are the axis points and whose third is the
    origin; the scales say how many pixels a foot buys along each.
    """
    cx = np.asarray(vp_x, dtype=np.float64) * scale_x
    cy = np.asarray(vp_y, dtype=np.float64) * scale_y
    co = np.array([origin[0], origin[1], 1.0], dtype=np.float64)
    return np.stack([cx, cy, co], axis=1)


def corners_of(model, transform):
    """Where this model's four court corners land in the image."""
    pts = np.array([[c] for c in model.corners], dtype=np.float64)
    out = []
    for (x, y), in pts:
        v = transform @ np.array([x, y, 1.0])
        if abs(v[2]) < 1e-9:
            return None
        out.append([float(v[0] / v[2]), float(v[1] / v[2])])
    return out


def _initial_scale(floor_region, vp, model_extent_ft):
    """First guess at pixels-per-foot along one axis.

    Assumes the visible floor spans roughly the court's extent along that
    axis. It only has to be within the multiplier sweep above, which is
    generous, because the sweep is what actually finds the scale.
    """
    contours, _ = cv2.findContours(floor_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    span = max(float(rect[1][0]), float(rect[1][1]))
    return max(1e-6, span / max(1.0, model_extent_ft))


def _floor_overlap(corners, floor_region):
    """Fraction of the visible floor the projected court covers."""
    quad = np.array([[int(round(x)), int(round(y))] for x, y in corners],
                    dtype=np.int32)
    painted = np.zeros(floor_region.shape, np.uint8)
    cv2.fillConvexPoly(painted, quad, 255)
    floor_px = cv2.countNonZero(floor_region)
    if floor_px == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(painted, floor_region)) / float(floor_px)


def fit_on_axes(models, axes, floor_region, shape, cost_of, verbose=False):
    """Search the four remaining parameters for every model and axis pairing.

    `cost_of(img_corners, model)` scores a hypothesis; it is the caller's, so
    this shares the free-corner search's notion of a good fit.

    Both ways of assigning the two axes to the court's width and length are
    tried, since nothing so far says which pencil is which, and both
    directions along each axis, since a vanishing point gives a line, not an
    arrow.
    """
    height, width = shape
    centre = np.array([width / 2.0, height / 2.0])
    best = None

    for model in models:
        for swap in (False, True):
            vp_a = axes[1]["vp"] if swap else axes[0]["vp"]
            vp_b = axes[0]["vp"] if swap else axes[1]["vp"]
            if _axis_direction(vp_a, centre) is None:
                continue
            if _axis_direction(vp_b, centre) is None:
                continue
            base_x = _initial_scale(floor_region, vp_a, model.width)
            base_y = _initial_scale(floor_region, vp_b, model.half_length)

            for sign_x in (1.0, -1.0):
                for sign_y in (1.0, -1.0):
                    for mx in SCALE_MULTIPLIERS:
                        for my in SCALE_MULTIPLIERS:
                            sx = base_x * mx * sign_x
                            sy = base_y * my * sign_y
                            for fx in ORIGIN_SWEEP:
                                for fy in ORIGIN_SWEEP:
                                    origin = (fx * width, fy * height)
                                    t = court_to_image(vp_a, vp_b, origin, sx, sy)
                                    corners = corners_of(model, t)
                                    if corners is None:
                                        continue
                                    if _floor_overlap(corners, floor_region) < MIN_FLOOR_OVERLAP:
                                        continue
                                    cost = cost_of(corners, model)
                                    if best is None or cost < best[0]:
                                        best = (cost, model, vp_a, vp_b,
                                                origin, sx, sy)
    if best is None:
        return None
    if verbose:
        print(f"[court_axes_fit] coarse sweep: {best[0]:.1f} on {best[1].name}")
    return _refine(best + (floor_region,), shape, cost_of, verbose)


def _refine(start, shape, cost_of, verbose=False):
    """Coordinate descent on the four parameters, from the sweep's winner."""
    height, width = shape
    cost, model, vp_a, vp_b, origin, sx, sy, floor_region = start
    ox, oy = origin
    origin_step = width * 0.05
    scale_step = 0.12  # relative, so it works whatever a foot is worth here

    for _ in range(REFINE_ROUNDS):
        improved = False
        trials = []
        for d in (1.0, -1.0):
            trials.append((ox + d * origin_step, oy, sx, sy))
            trials.append((ox, oy + d * origin_step, sx, sy))
            trials.append((ox, oy, sx * (1.0 + d * scale_step), sy))
            trials.append((ox, oy, sx, sy * (1.0 + d * scale_step)))
            # Move both scales together: the court's overall size in frame is
            # a likelier error than its shape, so this direction is worth a
            # step of its own rather than two that each look worse alone.
            trials.append((ox, oy, sx * (1.0 + d * scale_step),
                           sy * (1.0 + d * scale_step)))
        for tx, ty, tsx, tsy in trials:
            t = court_to_image(vp_a, vp_b, (tx, ty), tsx, tsy)
            corners = corners_of(model, t)
            if corners is None:
                continue
            if (floor_region is not None
                    and _floor_overlap(corners, floor_region) < MIN_FLOOR_OVERLAP):
                continue
            trial_cost = cost_of(corners, model)
            if trial_cost < cost:
                cost, ox, oy, sx, sy = trial_cost, tx, ty, tsx, tsy
                improved = True
        if not improved:
            origin_step *= REFINE_SHRINK
            scale_step *= REFINE_SHRINK
            if (origin_step / max(1.0, width) < REFINE_MIN_RELATIVE_STEP
                    and scale_step < REFINE_MIN_RELATIVE_STEP):
                break

    transform = court_to_image(vp_a, vp_b, (ox, oy), sx, sy)
    corners = corners_of(model, transform)
    if corners is None:
        return None
    if verbose:
        print(f"[court_axes_fit] refined to {cost:.1f} on {model.name}")
    return {"cost": cost, "model": model, "corners": corners,
            "origin": (ox, oy), "scale": (sx, sy)}
