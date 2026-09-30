"""Court calibration by fitting the court's LINE MARKINGS.

Why this rather than finding the painted lane: on a real high-school court the
lane is not reliably a clean painted rectangle. Measured on real footage, the
largest painted region scored 0.38 rectangularity -- an irregular decorative
navy area merging the lane with a wordmark -- so anything keyed on "the lane
is a filled rectangle" cannot work there.

Line markings are the one thing every court has, in standardised positions:
baseline, sidelines, lane, free-throw line, centre line, three-point arc. So
fit the whole court model to all of them at once instead of trusting any
single feature.

The earlier line attempt failed by grouping segments by their 2D angle. That
is wrong under perspective: lines parallel on the floor converge toward a
vanishing point, so their image angles differ and no angle cluster
corresponds to a real family. This does no grouping at all -- it scores a
whole hypothesised court against all the line pixels at once, which sidesteps
the problem entirely.

Court dimensions differ by level (a high-school court is 84x50 with a 12ft
lane; NBA/college is 94x50 with a 16ft lane), so both are tried and whichever
fits better wins -- rather than assuming, which silently cost ~33% of lane
width on high-school footage.
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


class CourtModel:
    """A court's standard markings, in feet, for the half containing the hoop.

    Court x runs across the width (0..width), y from the baseline (0) toward
    half-court. Only the near half is modelled, since that's what a camera
    covering one basket actually sees.
    """

    def __init__(self, name, length, width, lane_width, lane_length,
                 three_point_radius, three_point_corner_x, hoop_from_baseline=5.25):
        self.name = name
        self.length = length
        self.width = width
        self.lane_width = lane_width
        self.lane_length = lane_length
        self.three_point_radius = three_point_radius
        self.three_point_corner_x = three_point_corner_x
        self.hoop_from_baseline = hoop_from_baseline

    @property
    def half_length(self):
        return self.length / 2.0

    @property
    def corners(self):
        """The four corners of the modelled half, in cyclic order."""
        return [(0.0, 0.0), (self.width, 0.0),
                (self.width, self.half_length), (0.0, self.half_length)]

    @property
    def hoop(self):
        return (self.width / 2.0, self.hoop_from_baseline)

    def polylines(self):
        """Every marking as a polyline of court-space points."""
        w, hl = self.width, self.half_length
        lane_left = (w - self.lane_width) / 2.0
        lane_right = (w + self.lane_width) / 2.0
        lines = [
            [(0.0, 0.0), (w, 0.0)],                    # baseline
            [(0.0, 0.0), (0.0, hl)],                   # sideline
            [(w, 0.0), (w, hl)],                       # sideline
            [(0.0, hl), (w, hl)],                      # half-court line
            [(lane_left, 0.0), (lane_left, self.lane_length)],
            [(lane_right, 0.0), (lane_right, self.lane_length)],
            [(lane_left, self.lane_length), (lane_right, self.lane_length)],
        ]

        # Three-point line: straight run out from the baseline, then the arc.
        cx, cy = self.hoop
        radius = self.three_point_radius
        corner_x = self.three_point_corner_x
        straight_y = math.sqrt(max(radius ** 2 - (cx - corner_x) ** 2, 0.0)) + cy
        lines.append([(corner_x, 0.0), (corner_x, straight_y)])
        lines.append([(w - corner_x, 0.0), (w - corner_x, straight_y)])

        start = math.atan2(straight_y - cy, corner_x - cx)
        end = math.atan2(straight_y - cy, (w - corner_x) - cx)
        arc = []
        steps = 48
        for i in range(steps + 1):
            theta = start + (end - start) * i / steps
            arc.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
        lines.append(arc)

        # Centre circle, bisected by the half-court line.
        circle = [(w / 2.0 + 6.0 * math.cos(2 * math.pi * i / 48),
                   hl + 6.0 * math.sin(2 * math.pi * i / 48))
                  for i in range(49)]
        lines.append(circle)
        return lines

    def sample_points(self, spacing_ft=1.0):
        """Markings as a dense point cloud, for scoring an alignment."""
        points = []
        for polyline in self.polylines():
            for (x1, y1), (x2, y2) in zip(polyline, polyline[1:]):
                length = math.hypot(x2 - x1, y2 - y1)
                steps = max(1, int(length / spacing_ft))
                for i in range(steps + 1):
                    t = i / steps
                    points.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
        return np.array(points, dtype=np.float32)


COURT_MODELS = [
    CourtModel("high school", length=84.0, width=50.0, lane_width=12.0,
               lane_length=19.0, three_point_radius=19.75, three_point_corner_x=5.25),
    CourtModel("college / NBA", length=94.0, width=50.0, lane_width=16.0,
               lane_length=19.0, three_point_radius=22.15, three_point_corner_x=3.0),
]
