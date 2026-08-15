#!/usr/bin/env python3
"""CourtIQ Stage 1 + Stage 2: real-footage tracking, homography, possession,
and decision-rule grading.

This module is standalone (run it directly to produce tracks.json for a
video) and is also imported by app.py to run the same pipeline as a
background job.

Pipeline:
  1. Detect + track players (YOLO + ByteTrack) and the ball (YOLO).
  2. Project each player's foot position (bottom-center of bbox, NOT the
     bbox center) through a homography into real court feet.
  3. Drop off-court detections (bench, crowd, refs) using the projected
     court bounds.
  4. Assign each track to a team via KMeans(k=2) on jersey color sampled
     from a torso crop.
  5. Assign ball possession to the nearest on-court player, with a
     hysteresis hold so a single missed-ball frame doesn't flip possession.
  6. Segment possessions and label how each one ends (pass / turnover /
     lost) from team-identity continuity.
  7. (Stage 2) Grade decisions during each possession using rules that are
     directly computable from the tracked data above -- no invented labels.

BALL MODEL: by default this still falls back to COCO's generic "sports
ball" class via BALL_MODEL_PATH=yolo11n.pt, which is expected to perform
poorly on basketball footage (COCO was never trained on a basketball
specifically). Run `python3 setup_ball_model.py` once to download a
basketball-domain-trained model (MIT-licensed, from
github.com/abdullahtarek/basketball_analysis) to models/basketball_ball.pt,
then set COURTIQ_BALL_MODEL=models/basketball_ball.pt (or pass
--ball-model on the CLI) to use it. That download could not be validated
against this project's own footage from the environment this code was
written in (no network path to fetch it, no test video available there) --
run `python3 courtiq_core.py <video> --ball-model models/basketball_ball.pt`
on your own footage and check the printed "Ball detection rate" before
trusting downstream possession/decision output, per the CourtIQ build
brief's Stage 1.5 requirement to validate on real footage before relying
on it. The ball class index is resolved by name (see resolve_ball_class),
not hardcoded, so this works regardless of which class index the swapped
model uses -- override with COURTIQ_BALL_CLASS if name-matching ever picks
the wrong one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERSON_CLASS = 0
BALL_CLASS_FALLBACK = 32  # COCO "sports ball" index, used only if a model's
                           # class names don't contain a ball-like name below.
# Basketball-trained models (e.g. Roboflow Universe exports, or
# github.com/abdullahtarek/basketball_analysis) name their ball class
# something like "Ball" or "basketball" rather than reusing COCO's index 32.
# resolve_ball_class() below looks the class index up by name so swapping
# BALL_MODEL_PATH doesn't also require guessing a new hardcoded index.
BALL_CLASS_NAMES = {"ball", "basketball", "sports ball"}

DETECT_CONF = float(os.environ.get("COURTIQ_DETECT_CONF", "0.25"))
BALL_MODEL_PATH = os.environ.get("COURTIQ_BALL_MODEL", "yolo11n.pt")
PLAYER_MODEL_PATH = os.environ.get("COURTIQ_PLAYER_MODEL", "yolo11n.pt")

# Only ball detection is strided (run every Nth frame), not player
# tracking. Player tracking runs on every single frame -- ByteTrack
# matches players between calls by position/overlap, and skipping frames
# there multiplies apparent player displacement between calls, breaking
# that matching and fragmenting track IDs (measured: ~141 tracks for what
# should be ~10-12 players+refs, when both were strided together).
# Skipping only ball detection still cuts total inference cost
# meaningfully, since the ball doesn't need frame-to-frame continuity the
# same way -- possession logic already tolerates gaps via
# POSSESSION_HOLD_FRAMES.
FRAME_STRIDE = max(1, int(os.environ.get("COURTIQ_FRAME_STRIDE", "3")))

# Standard NBA half-court reference dimensions, in feet, used as defaults
# for the landmark picker. Full court is 94x50; teams generally care about
# the half being played on.
COURT_LANDMARKS = {
    "baseline_left": (0.0, 0.0),
    "baseline_right": (50.0, 0.0),
    "halfcourt_left": (0.0, 47.0),
    "halfcourt_right": (50.0, 47.0),
    "ft_line_left": (17.0, 19.0),
    "ft_line_right": (33.0, 19.0),
    "paint_left_baseline": (17.0, 0.0),
    "paint_right_baseline": (33.0, 0.0),
    "hoop": (25.0, 5.25),
    "three_pt_left_corner": (3.0, 0.0),
    "three_pt_right_corner": (47.0, 0.0),
}

COURT_WIDTH_FT = 50.0
COURT_LENGTH_FT = 94.0
ON_COURT_MARGIN_FT = 4.0  # tolerance beyond the painted lines for player feet

POSSESSION_HOLD_FRAMES = 6  # frames the ball can go undetected before we
                             # decide possession has actually changed
MAX_POSSESSION_DIST_FT = 6.0  # ball must be this close to a player's feet
                               # to be considered "in that player's hands"

# Stage 2 thresholds -- deliberately simple and documented, not hidden.
OPEN_TEAMMATE_SEPARATION_FT = 6.0     # nearest defender must be this far away
PASSING_LANE_CLEARANCE_FT = 3.0       # no defender within this of the pass line
OPEN_TEAMMATE_HOLD_SECONDS = 1.5      # carrier must hold this long, ignoring it
CONTESTED_SHOT_DEFENDER_FT = 4.0      # defender closer than this = contested
BAD_SPACING_TEAMMATE_FT = 8.0         # same-team players closer than this = crowded

# A possession ending "lost" (ball no longer tracked near any player) is
# used as a proxy for a shot attempt, since tracks.json has no dedicated
# shot-detection signal. But that proxy is only meaningful if the carrier
# was actually within shooting range of the hoop when it happened --
# otherwise "lost" usually just means the ball detector dropped the ball
# mid-court, which is a tracking failure, not a shot.
SHOT_RANGE_FT = 30.0  # beyond the 3-pt line (23.75 ft) plus margin

# Below this on-court ball detection rate, the ball's position data is too
# sparse/noisy to trust for possession- or ball-dependent grading (see the
# BALL_MODEL_PATH limitation in the module docstring). Ball-dependent
# decision types are skipped entirely rather than emitted with false
# confidence; a single low_confidence decision explains why.
MIN_RELIABLE_BALL_DETECTION_RATE = 0.15


# ---------------------------------------------------------------------------
# Data schema (Stage 1 output contract -- do not change without a reason)
# ---------------------------------------------------------------------------

@dataclass
class PlayerSample:
    frame: int
    t: float
    track_id: int
    team: Optional[int]
    x_ft: float
    y_ft: float
    bbox: list


@dataclass
class BallSample:
    frame: int
    t: float
    x_ft: Optional[float]
    y_ft: Optional[float]
    detected: bool


@dataclass
class Possession:
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    team: Optional[int]
    carrier_track_ids: list
    end_reason: str  # "pass" | "turnover" | "lost" | "end_of_clip"


@dataclass
class TrackMeta:
    track_id: int
    team: Optional[int]
    first_frame: int
    last_frame: int
    sample_count: int


@dataclass
class TracksData:
    fps: float
    frame_count: int
    duration: float
    homography: list
    ball_detection_rate: float
    tracks: dict = field(default_factory=dict)  # track_id -> TrackMeta
    players: list = field(default_factory=list)  # list[PlayerSample]
    ball: list = field(default_factory=list)      # list[BallSample]
    possessions: list = field(default_factory=list)  # list[Possession]
    decisions: list = field(default_factory=list)     # Stage 2 output
    iq_scores: dict = field(default_factory=dict)      # track_id -> score


# ---------------------------------------------------------------------------
# Homography / landmark picking
# ---------------------------------------------------------------------------

def compute_homography(image_points: list, landmark_names: list) -> "np.ndarray":
    """Fit a homography from clicked image points to real court feet.

    Requires at least 4 point correspondences; more (and spatially spread
    out) landmarks improve accuracy. Any subset of COURT_LANDMARKS works --
    this is what lets partial-court camera angles be calibrated.
    """
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for homography.")
    if len(image_points) < 4:
        raise ValueError("Need at least 4 landmark correspondences for a homography.")
    world_points = [COURT_LANDMARKS[name] for name in landmark_names]
    src = np.array(image_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise RuntimeError("Could not compute a homography from the given points.")
    return homography


def pick_corners_interactive(video_path: str) -> "np.ndarray":
    """Let the user click 4+ visible court landmarks on a paused frame.

    This intentionally does NOT require a full-court view with all 4
    baseline corners visible -- broadcast and sideline footage frequently
    only shows part of the court. The user clicks whichever named
    landmarks (see COURT_LANDMARKS) are visible, in order, and confirms
    each click by typing its landmark name at the prompt.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for the landmark picker.")
    capture = cv2.VideoCapture(video_path)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}.")

    clicked = []
    window = "CourtIQ - click a landmark, then press its name in the terminal"

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window, frame)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    cv2.imshow(window, frame)
    print("Available landmarks:", ", ".join(COURT_LANDMARKS.keys()))
    print("Click a point on the frame window, then type its landmark name here and press Enter.")
    print("Type 'done' once you have clicked at least 4 landmarks.")

    names = []
    while True:
        cv2.waitKey(1)
        name = input(f"Landmark for click #{len(names) + 1} (or 'done'): ").strip()
        if name == "done":
            break
        if name not in COURT_LANDMARKS:
            print(f"Unknown landmark '{name}'. Choose from: {', '.join(COURT_LANDMARKS.keys())}")
            continue
        if len(names) >= len(clicked):
            print("Click a point on the image window first.")
            continue
        names.append(name)
    cv2.destroyAllWindows()

    return compute_homography(clicked[: len(names)], names)


def default_full_frame_homography(width: int, height: int) -> "np.ndarray":
    """Automatic homography with no clicking required: assumes the video
    frame is a straight-on, full-court view and maps its four corners
    directly to the court's four corners.

    This is a real accuracy trade-off, not a substitute for calibration --
    it's wrong for angled, sideline, partial-court, or zoomed-in footage,
    which is most real game footage. Court positions and everything
    downstream of them (possession, decisions) will be skewed accordingly.
    Use pick_corners_interactive() instead whenever accuracy matters and a
    display is available to click landmarks on.
    """
    image_points = [(0, height), (width, height), (0, 0), (width, 0)]
    landmark_names = ["baseline_left", "baseline_right", "halfcourt_left", "halfcourt_right"]
    return compute_homography(image_points, landmark_names)


def order_quad_points(points) -> list:
    """Order 4 arbitrary points as [top-left, top-right, bottom-right,
    bottom-left] in image space, via the standard sum/difference trick
    (top-left has the smallest x+y, bottom-right the largest; top-right
    has the largest x-y, bottom-left the smallest).
    """
    pts = [tuple(p) for p in points]
    sums = [p[0] + p[1] for p in pts]
    diffs = [p[0] - p[1] for p in pts]
    top_left = pts[sums.index(min(sums))]
    bottom_right = pts[sums.index(max(sums))]
    top_right = pts[diffs.index(max(diffs))]
    bottom_left = pts[diffs.index(min(diffs))]
    return [top_left, top_right, bottom_right, bottom_left]


def detect_court_quad(frame) -> Optional[list]:
    """Best-effort automatic detection of the visible court floor boundary
    -- no manual clicking, no trained model, no downloaded weights. Finds
    the largest contiguous region matching typical hardwood-court color
    (warm tan/orange) and approximates its outline to 4 corners.

    Returns [top-left, top-right, bottom-right, bottom-left] image points,
    or None if no confident quadrilateral was found (unusual lighting, a
    non-wood floor, or a view where the floor doesn't form a clean
    quadrilateral -- e.g. heavily obstructed by players/crowd).

    This is a classical-CV heuristic, not a guarantee -- it can be wrong.
    See auto_homography() for how the result is used and what it falls
    back to when detection fails.
    """
    if cv2 is None or np is None:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Broad band to tolerate lighting variation across gyms/arenas -- warm
    # hardwood tones, not overly saturated (avoids team jerseys), not too
    # dark/bright (avoids shadows and blown-out highlights).
    lower = np.array([5, 30, 80])
    upper = np.array([30, 200, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    frame_area = frame.shape[0] * frame.shape[1]
    if cv2.contourArea(largest) < 0.15 * frame_area:
        return None  # too small to plausibly be the court floor

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        # Contour outline isn't cleanly a quadrilateral (occlusion, odd
        # floor markings) -- fall back to its minimum-area bounding
        # rectangle, which is still a reasonable court-boundary estimate.
        rect = cv2.minAreaRect(largest)
        approx = cv2.boxPoints(rect).reshape(-1, 1, 2)

    points = approx.reshape(-1, 2).astype(float)
    return order_quad_points(points)


def auto_homography(video_path: str) -> "np.ndarray":
    """Fully automatic homography -- no manual clicking. Tries
    detect_court_quad() on the video's first frame; falls back to
    default_full_frame_homography() if no confident court boundary was
    found. This is the default for both the CLI and the web app. Pass
    --interactive (CLI) for accurate manual calibration instead, which
    will outperform this heuristic on angled or partial-court footage.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for automatic homography.")
    capture = cv2.VideoCapture(video_path)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}.")
    height, width = frame.shape[:2]

    quad = detect_court_quad(frame)
    if quad is None:
        print(
            "[courtiq_core] Automatic court-floor detection found no confident boundary; "
            "falling back to a full-frame homography approximation. Pass --interactive "
            "for accurate manual calibration."
        )
        return default_full_frame_homography(width, height)

    print(
        "[courtiq_core] Automatically detected a court floor boundary from color -- "
        "using it for homography without manual clicking. This is a heuristic, not a "
        "guarantee; pass --interactive if results look off."
    )
    # Assumes the detected floor region roughly spans a full-court view,
    # with the near baseline at the bottom of the frame (closer to camera)
    # and the far end (half-court line, in a full-court shot) at the top.
    # That assumption is approximate for partial-court or heavily angled
    # views -- same caveat as default_full_frame_homography().
    landmark_names = ["halfcourt_left", "halfcourt_right", "baseline_right", "baseline_left"]
    return compute_homography(quad, landmark_names)


def project_point(homography: "np.ndarray", point) -> tuple:
    if np is None:
        raise RuntimeError("numpy is required to project points.")
    src = np.array([[point]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)
    return float(dst[0][0][0]), float(dst[0][0][1])


def is_on_court(x_ft: float, y_ft: float, margin: float = ON_COURT_MARGIN_FT) -> bool:
    return (
        -margin <= x_ft <= COURT_WIDTH_FT + margin
        and -margin <= y_ft <= COURT_LENGTH_FT + margin
    )


# ---------------------------------------------------------------------------
# Team assignment
# ---------------------------------------------------------------------------

def torso_crop(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    torso_top = y1 + int(h * 0.15)
    torso_bottom = y1 + int(h * 0.55)
    crop = frame[max(0, torso_top):max(0, torso_bottom), max(0, x1):max(0, x2)]
    return crop


def mean_jersey_color(frame, bbox):
    crop = torso_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    return crop.reshape(-1, 3).mean(axis=0)


def assign_teams(track_colors: dict) -> dict:
    """KMeans(k=2) over each track's mean jersey color -> {track_id: 0|1}."""
    if not track_colors:
        return {}
    if np is None:
        raise RuntimeError("numpy is required for team assignment.")
    ids = list(track_colors.keys())
    data = np.array([track_colors[i] for i in ids], dtype=np.float32)
    if len(ids) < 2:
        return {ids[0]: 0} if ids else {}
    if cv2 is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten().tolist()
    else:
        # Minimal fallback so this module still degrades gracefully without cv2.kmeans.
        centers = [data[0], data[-1]]
        labels = [0 if np.linalg.norm(d - centers[0]) <= np.linalg.norm(d - centers[1]) else 1 for d in data]
    return {track_id: int(label) for track_id, label in zip(ids, labels)}


# ---------------------------------------------------------------------------
# Stage 1: detection + tracking + possession
# ---------------------------------------------------------------------------

def _load_model(path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics is required. Install it with: pip install ultralytics")
    weights = Path(path)
    if not weights.exists():
        raise RuntimeError(f"Model weights not found: {path}")
    return YOLO(str(weights))


def resolve_ball_class(model) -> int:
    """Find the ball class index by name rather than assuming COCO's index
    32. Custom basketball-trained models (Roboflow exports, or e.g.
    github.com/abdullahtarek/basketball_analysis) typically name their ball
    class "Ball"/"basketball" at whatever index their training data used.
    An explicit COURTIQ_BALL_CLASS env var always wins if set.
    """
    override = os.environ.get("COURTIQ_BALL_CLASS")
    if override is not None:
        return int(override)
    names = getattr(model, "names", None) or {}
    for idx, name in names.items():
        if str(name).strip().lower() in BALL_CLASS_NAMES:
            return int(idx)
    return BALL_CLASS_FALLBACK


def run_pipeline(
    video_path: str,
    homography: "np.ndarray",
    ball_model_path: str = BALL_MODEL_PATH,
    player_model_path: str = PLAYER_MODEL_PATH,
    max_frames: Optional[int] = None,
    progress_cb=None,
) -> TracksData:
    """Run Stage 1 tracking + possession segmentation on a video.

    progress_cb, if given, is called with (frames_done, frames_total) so
    a caller (e.g. app.py's background job) can report progress without
    this function knowing anything about HTTP.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required. Install it with: pip install opencv-python")

    player_model = _load_model(player_model_path)
    ball_model = player_model if ball_model_path == player_model_path else _load_model(ball_model_path)
    ball_class = resolve_ball_class(ball_model)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total = min(frame_count, max_frames) if max_frames else frame_count

    track_first_seen = {}
    track_last_seen = {}
    track_colors = defaultdict(list)
    player_samples = []
    ball_samples = []
    ball_detected_count = 0

    frame_idx = 0
    while True:
        if max_frames and frame_idx >= max_frames:
            break
        # Every frame is decoded and given to the player tracker: ByteTrack
        # matches players between consecutive calls by position/overlap, and
        # skipping frames there (as this loop used to do for both models)
        # multiplies how far players appear to move between calls, breaking
        # that matching and fragmenting track IDs. Only ball detection --
        # which doesn't need frame-to-frame continuity the same way, since
        # possession logic already tolerates gaps via POSSESSION_HOLD_FRAMES
        # -- is strided, to still cut overall inference cost.
        ok, frame = capture.read()
        if not ok:
            break
        t = frame_idx / fps

        # Player detection + tracking (ByteTrack keeps IDs stable across frames).
        track_result = player_model.track(
            frame, classes=[PERSON_CLASS], conf=DETECT_CONF, tracker="bytetrack.yaml",
            persist=True, verbose=False,
        )[0]

        if track_result.boxes is not None and track_result.boxes.id is not None:
            boxes = track_result.boxes.xyxy.tolist()
            ids = track_result.boxes.id.int().tolist()
            for bbox, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = bbox
                foot_point = ((x1 + x2) / 2, y2)  # bottom-center, NOT bbox center
                x_ft, y_ft = project_point(homography, foot_point)
                if not is_on_court(x_ft, y_ft):
                    continue
                color = mean_jersey_color(frame, bbox)
                if color is not None:
                    track_colors[track_id].append(color)
                track_first_seen.setdefault(track_id, frame_idx)
                track_last_seen[track_id] = frame_idx
                player_samples.append(PlayerSample(
                    frame=frame_idx, t=t, track_id=track_id, team=None,
                    x_ft=x_ft, y_ft=y_ft, bbox=[x1, y1, x2, y2],
                ))

        # Ball detection (separate model call so a swapped-in basketball
        # model can be used here without touching player tracking above).
        # Strided: only every FRAME_STRIDE'th frame runs ball inference.
        if frame_idx % FRAME_STRIDE == 0:
            ball_result = ball_model(frame, classes=[ball_class], conf=DETECT_CONF, verbose=False)[0]
            ball_x_ft = ball_y_ft = None
            detected = False
            if ball_result.boxes is not None and len(ball_result.boxes) > 0:
                box = max(ball_result.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                ball_x_ft, ball_y_ft = project_point(homography, ((x1 + x2) / 2, (y1 + y2) / 2))
                if is_on_court(ball_x_ft, ball_y_ft):
                    detected = True
                    ball_detected_count += 1
                else:
                    ball_x_ft = ball_y_ft = None
            ball_samples.append(BallSample(frame=frame_idx, t=t, x_ft=ball_x_ft, y_ft=ball_y_ft, detected=detected))

        frame_idx += 1
        if progress_cb and total:
            progress_cb(frame_idx, total)

    capture.release()

    team_by_track = assign_teams({
        track_id: sum(colors) / len(colors) for track_id, colors in track_colors.items() if colors
    })
    for sample in player_samples:
        sample.team = team_by_track.get(sample.track_id)

    tracks = {
        track_id: TrackMeta(
            track_id=track_id,
            team=team_by_track.get(track_id),
            first_frame=track_first_seen[track_id],
            last_frame=track_last_seen[track_id],
            sample_count=sum(1 for s in player_samples if s.track_id == track_id),
        )
        for track_id in track_first_seen
    }

    possessions = segment_possessions(player_samples, ball_samples, fps)

    ball_detection_rate = ball_detected_count / len(ball_samples) if ball_samples else 0.0
    if ball_detection_rate < 0.15:
        print(
            f"[courtiq_core] WARNING: on-court ball detection rate is only "
            f"{ball_detection_rate:.1%}. Possession/decision output built on "
            f"this run will be unreliable. See the BALL_MODEL_PATH limitation "
            f"documented at the top of this file -- swap in a basketball-"
            f"trained model (Stage 1.5) before trusting downstream results."
        )

    return TracksData(
        fps=fps,
        frame_count=frame_count,
        duration=frame_count / fps if fps else 0.0,
        homography=homography.tolist() if hasattr(homography, "tolist") else list(homography),
        ball_detection_rate=ball_detection_rate,
        tracks=tracks,
        players=player_samples,
        ball=ball_samples,
        possessions=possessions,
    )


def _nearest_player(ball_x, ball_y, players_in_frame):
    best, best_dist = None, None
    for sample in players_in_frame:
        dist = math.hypot(sample.x_ft - ball_x, sample.y_ft - ball_y)
        if best_dist is None or dist < best_dist:
            best, best_dist = sample, dist
    return best, best_dist


def segment_possessions(player_samples: list, ball_samples: list, fps: float) -> list:
    """Assign ball possession per frame (nearest on-court player, with a
    hysteresis hold so a single missed-ball frame doesn't flip possession),
    then collapse consecutive frames into Possession segments.
    """
    players_by_frame = defaultdict(list)
    for sample in player_samples:
        players_by_frame[sample.frame].append(sample)

    current_holder = None
    frames_since_seen = 0
    holder_by_frame = {}

    for ball in ball_samples:
        candidates = players_by_frame.get(ball.frame, [])
        if ball.detected and candidates:
            nearest, dist = _nearest_player(ball.x_ft, ball.y_ft, candidates)
            if nearest is not None and dist <= MAX_POSSESSION_DIST_FT:
                current_holder = nearest.track_id
                frames_since_seen = 0
            else:
                frames_since_seen += 1
        else:
            frames_since_seen += 1
        if frames_since_seen > POSSESSION_HOLD_FRAMES:
            current_holder = None
        holder_by_frame[ball.frame] = current_holder

    team_by_track = {}
    for sample in player_samples:
        if sample.track_id not in team_by_track and sample.team is not None:
            team_by_track[sample.track_id] = sample.team

    possessions = []
    open_poss = None
    frames_sorted = sorted(holder_by_frame.keys())
    for frame in frames_sorted:
        holder = holder_by_frame[frame]
        t = frame / fps if fps else 0.0
        if holder is None:
            if open_poss is not None:
                open_poss["end_frame"] = frame
                open_poss["end_t"] = t
                open_poss["end_reason"] = "lost"
                possessions.append(_close_possession(open_poss))
                open_poss = None
            continue
        holder_team = team_by_track.get(holder)
        if open_poss is None:
            open_poss = {
                "start_frame": frame, "end_frame": frame, "start_t": t, "end_t": t,
                "team": holder_team, "carrier_track_ids": [holder],
            }
            continue
        if holder == open_poss["carrier_track_ids"][-1]:
            open_poss["end_frame"] = frame
            open_poss["end_t"] = t
            continue
        # Possession changed hands.
        same_team = holder_team is not None and holder_team == open_poss["team"]
        open_poss["end_frame"] = frame
        open_poss["end_t"] = t
        open_poss["end_reason"] = "pass" if same_team else "turnover"
        possessions.append(_close_possession(open_poss))
        open_poss = {
            "start_frame": frame, "end_frame": frame, "start_t": t, "end_t": t,
            "team": holder_team, "carrier_track_ids": [holder],
        }

    if open_poss is not None:
        open_poss["end_reason"] = "end_of_clip"
        possessions.append(_close_possession(open_poss))

    return possessions


def _close_possession(fields: dict) -> Possession:
    return Possession(
        start_frame=fields["start_frame"], end_frame=fields["end_frame"],
        start_t=fields["start_t"], end_t=fields["end_t"], team=fields["team"],
        carrier_track_ids=fields["carrier_track_ids"], end_reason=fields["end_reason"],
    )


# ---------------------------------------------------------------------------
# Stage 2: decision-rule grading
#
# Every rule below reads only fields already present in TracksData. Nothing
# here infers a play name or intent that isn't directly measurable.
# ---------------------------------------------------------------------------

def _teammates_and_defenders(sample, players_in_frame, team_by_track):
    team = team_by_track.get(sample.track_id)
    teammates, defenders = [], []
    for other in players_in_frame:
        if other.track_id == sample.track_id:
            continue
        other_team = team_by_track.get(other.track_id)
        if other_team is None or team is None:
            continue
        (teammates if other_team == team else defenders).append(other)
    return teammates, defenders


def _nearest_defender_dist(point, defenders) -> Optional[float]:
    if not defenders:
        return None
    return min(math.hypot(point[0] - d.x_ft, point[1] - d.y_ft) for d in defenders)


def _point_to_segment_dist(p, a, b) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def grade_decisions(data: TracksData) -> list:
    team_by_track = {tid: meta.team for tid, meta in data.tracks.items()}
    players_by_frame = defaultdict(list)
    for sample in data.players:
        players_by_frame[sample.frame].append(sample)

    decisions = []

    # Every possession-based rule below (open_teammate_ignored, shot_selection)
    # depends on knowing who's holding the ball, which depends on the ball
    # actually being detected. Below MIN_RELIABLE_BALL_DETECTION_RATE, the
    # possession segmentation this run produced is mostly detector dropout,
    # not real possession changes -- grading on it would just relabel noise
    # with false confidence. Skip those rules and say so explicitly, rather
    # than emit decisions built on data known to be unreliable.
    if data.ball_detection_rate < MIN_RELIABLE_BALL_DETECTION_RATE:
        decisions.append({
            "type": "low_confidence",
            "verdict": "uncertain",
            "frame": 0,
            "t": 0.0,
            "reason": (
                f"On-court ball detection rate for this run was "
                f"{data.ball_detection_rate:.1%}, below the "
                f"{MIN_RELIABLE_BALL_DETECTION_RATE:.0%} reliability threshold. "
                f"Ball-dependent grading (open teammate ignored, shot selection) "
                f"was skipped for this video -- see the BALL_MODEL_PATH "
                f"limitation in courtiq_core.py. Only same-team spacing, which "
                f"doesn't need the ball, was graded below."
            ),
        })
        decisions.extend(grade_spacing(data, players_by_frame, team_by_track))
        return decisions

    # Rule: open teammate ignored. For each possession, check whether the
    # carrier held the ball long enough while a teammate was both
    # (a) more open than their nearest defender by OPEN_TEAMMATE_SEPARATION_FT
    # and (b) reachable by a passing lane clear of defenders.
    for poss in data.possessions:
        duration = poss.end_t - poss.start_t
        if duration < OPEN_TEAMMATE_HOLD_SECONDS or not poss.carrier_track_ids:
            continue
        carrier_id = poss.carrier_track_ids[-1]
        mid_frame = (poss.start_frame + poss.end_frame) // 2
        frame_players = players_by_frame.get(mid_frame, [])
        carrier = next((p for p in frame_players if p.track_id == carrier_id), None)
        if carrier is None:
            continue
        teammates, defenders = _teammates_and_defenders(carrier, frame_players, team_by_track)
        for mate in teammates:
            mate_point = (mate.x_ft, mate.y_ft)
            mate_defender_dist = _nearest_defender_dist(mate_point, defenders)
            if mate_defender_dist is None or mate_defender_dist < OPEN_TEAMMATE_SEPARATION_FT:
                continue
            lane_clear = all(
                _point_to_segment_dist((d.x_ft, d.y_ft), (carrier.x_ft, carrier.y_ft), mate_point)
                >= PASSING_LANE_CLEARANCE_FT
                for d in defenders
            )
            if lane_clear:
                decisions.append({
                    "type": "open_teammate_ignored",
                    "verdict": "bad",
                    "frame": mid_frame,
                    "t": mid_frame / data.fps if data.fps else 0.0,
                    "carrier_track_id": carrier_id,
                    "open_teammate_track_id": mate.track_id,
                    "defender_separation_ft": round(mate_defender_dist, 1),
                    "reason": (
                        f"Held the ball {duration:.1f}s while track {mate.track_id} was "
                        f"{mate_defender_dist:.1f} ft from the nearest defender with a clear lane."
                    ),
                })
                break  # one flag per possession is enough signal

    # Rule: contested vs. open shot, evaluated at possessions that end
    # "lost" (ball possession disappears -- our best proxy for a shot,
    # since we have no separate shot-detection signal in tracks.json).
    # Only count it as a shot if the carrier was actually within shooting
    # range of the hoop when possession was lost -- a "lost" possession at
    # half-court is a tracking dropout, not a shot attempt, and grading it
    # as one would be exactly the kind of invented label this pipeline is
    # supposed to avoid.
    hoop_x, hoop_y = COURT_LANDMARKS["hoop"]
    for poss in data.possessions:
        if poss.end_reason != "lost" or not poss.carrier_track_ids:
            continue
        carrier_id = poss.carrier_track_ids[-1]
        frame_players = players_by_frame.get(poss.end_frame, [])
        carrier = next((p for p in frame_players if p.track_id == carrier_id), None)
        if carrier is None:
            continue
        dist_to_hoop = math.hypot(carrier.x_ft - hoop_x, carrier.y_ft - hoop_y)
        if dist_to_hoop > SHOT_RANGE_FT:
            continue  # too far from the hoop to plausibly be a shot; skip rather than mislabel
        _, defenders = _teammates_and_defenders(carrier, frame_players, team_by_track)
        nearest = _nearest_defender_dist((carrier.x_ft, carrier.y_ft), defenders)
        if nearest is None:
            continue
        contested = nearest < CONTESTED_SHOT_DEFENDER_FT
        decisions.append({
            "type": "shot_selection",
            "verdict": "bad" if contested else "good",
            "frame": poss.end_frame,
            "t": poss.end_t,
            "carrier_track_id": carrier_id,
            "nearest_defender_ft": round(nearest, 1),
            "distance_to_hoop_ft": round(dist_to_hoop, 1),
            "reason": (
                f"Possession ended {dist_to_hoop:.1f} ft from the hoop with the nearest "
                f"defender {nearest:.1f} ft away ({'contested' if contested else 'open'})."
            ),
        })

    decisions.extend(grade_spacing(data, players_by_frame, team_by_track))
    return decisions


def grade_spacing(data: TracksData, players_by_frame: dict, team_by_track: dict) -> list:
    """Same-team pairwise distance, sampled every ~1s. Doesn't depend on
    ball detection at all, so this still runs even when ball tracking is
    too unreliable to trust for the other rules.
    """
    decisions = []
    sample_stride = max(1, int(round(data.fps))) if data.fps else 30
    for frame in sorted(players_by_frame.keys())[::sample_stride]:
        frame_players = players_by_frame[frame]
        by_team = defaultdict(list)
        for p in frame_players:
            team = team_by_track.get(p.track_id)
            if team is not None:
                by_team[team].append(p)
        for team, members in by_team.items():
            if len(members) < 2:
                continue
            dists = [
                math.hypot(a.x_ft - b.x_ft, a.y_ft - b.y_ft)
                for i, a in enumerate(members) for b in members[i + 1:]
            ]
            mean_dist = sum(dists) / len(dists)
            if mean_dist < BAD_SPACING_TEAMMATE_FT:
                decisions.append({
                    "type": "spacing",
                    "verdict": "bad",
                    "frame": frame,
                    "t": frame / data.fps if data.fps else 0.0,
                    "team": team,
                    "mean_teammate_distance_ft": round(mean_dist, 1),
                    "reason": f"Team {team} averaged {mean_dist:.1f} ft of separation (< {BAD_SPACING_TEAMMATE_FT} ft threshold).",
                })
    return decisions


def compute_iq_scores(data: TracksData) -> dict:
    """Per-player IQ score = good decisions / total flagged decisions
    attributed to that player, scaled 0-100. Only decisions with an
    identifiable track_id contribute; spacing flags (team-level) are
    split across that team's on-court players in the sampled frame.
    """
    tallies = defaultdict(lambda: {"good": 0, "bad": 0})
    players_by_frame = defaultdict(list)
    for sample in data.players:
        players_by_frame[sample.frame].append(sample)

    for decision in data.decisions:
        verdict = decision["verdict"]
        if decision["type"] == "spacing":
            team = decision.get("team")
            frame_players = players_by_frame.get(decision["frame"], [])
            for p in frame_players:
                if data.tracks.get(p.track_id) and data.tracks[p.track_id].team == team:
                    tallies[p.track_id][verdict] += 1
            continue
        track_id = decision.get("carrier_track_id")
        if track_id is not None:
            tallies[track_id][verdict] += 1
        if decision["type"] == "open_teammate_ignored":
            # The carrier made the bad decision; the open teammate isn't graded.
            continue

    scores = {}
    for track_id, tally in tallies.items():
        total = tally["good"] + tally["bad"]
        scores[track_id] = round(100 * tally["good"] / total) if total else None
    return scores


def to_json_dict(data: TracksData) -> dict:
    return {
        "fps": data.fps,
        "frame_count": data.frame_count,
        "duration": data.duration,
        "homography": data.homography,
        "ball_detection_rate": data.ball_detection_rate,
        "tracks": {str(k): asdict(v) for k, v in data.tracks.items()},
        "players": [asdict(p) for p in data.players],
        "ball": [asdict(b) for b in data.ball],
        "possessions": [asdict(p) for p in data.possessions],
        "decisions": data.decisions,
        "iq_scores": {str(k): v for k, v in data.iq_scores.items()},
    }


def analyze_video(
    video_path: str,
    homography: "np.ndarray",
    ball_model_path: str = BALL_MODEL_PATH,
    player_model_path: str = PLAYER_MODEL_PATH,
    max_frames: Optional[int] = None,
    progress_cb=None,
) -> TracksData:
    """Run the full Stage 1 + Stage 2 pipeline and return TracksData."""
    data = run_pipeline(
        video_path, homography,
        ball_model_path=ball_model_path, player_model_path=player_model_path,
        max_frames=max_frames, progress_cb=progress_cb,
    )
    data.decisions = grade_decisions(data)
    data.iq_scores = compute_iq_scores(data)
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CourtIQ Stage 1+2 pipeline")
    parser.add_argument("video", help="Path to game footage")
    parser.add_argument("--out", default="tracks.json", help="Output path for tracks.json")
    parser.add_argument("--ball-model", default=BALL_MODEL_PATH, help="Path to ball detector weights")
    parser.add_argument("--player-model", default=PLAYER_MODEL_PATH, help="Path to player detector weights")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed (for quick tests)")
    parser.add_argument(
        "--interactive", action="store_true",
        help="Click 4+ court landmarks on a paused frame for an accurate homography "
             "(needs a display). Default is fully automatic (no clicking): tries "
             "classical-CV court-floor detection first, falling back to a full-frame "
             "guess if that fails. See auto_homography() for the accuracy trade-offs.",
    )
    args = parser.parse_args()

    if args.interactive:
        homography = pick_corners_interactive(args.video)
    else:
        homography = auto_homography(args.video)

    data = analyze_video(
        args.video, homography,
        ball_model_path=args.ball_model, player_model_path=args.player_model,
        max_frames=args.max_frames,
    )
    Path(args.out).write_text(json.dumps(to_json_dict(data), indent=2))
    print(f"Wrote {args.out}")
    print(f"Ball detection rate: {data.ball_detection_rate:.1%}")
    print(f"Tracks: {len(data.tracks)}  Possessions: {len(data.possessions)}  Decisions flagged: {len(data.decisions)}")


if __name__ == "__main__":
    main()
