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
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

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

PERSON_CLASS_FALLBACK = 0  # COCO "person" index, used only if a model's
                            # class names don't contain a person-like name below.
BALL_CLASS_FALLBACK = 32  # COCO "sports ball" index, used only if a model's
                           # class names don't contain a ball-like name below.
# Basketball-trained models (e.g. Roboflow Universe exports, or
# github.com/abdullahtarek/basketball_analysis) name their ball class
# something like "Ball" or "basketball" rather than reusing COCO's index 32.
# resolve_ball_class() below looks the class index up by name so swapping
# BALL_MODEL_PATH doesn't also require guessing a new hardcoded index.
BALL_CLASS_NAMES = {"ball", "basketball", "sports ball"}
# Same idea for the player detector -- a basketball-specific dataset (e.g.
# universe.roboflow.com/workspace-5ujvu/basketball-players-fy4c2-vfsuv)
# names its player class "Player" at whatever index its training data
# used, not COCO's "person" at index 0. This also lets that dataset's
# separate "Ref" class stay excluded automatically, since we only match
# person-like names, not referee-like ones.
PERSON_CLASS_NAMES = {"person", "player"}

# Separate confidence thresholds for player vs. ball detection -- they were
# previously shared (COURTIQ_DETECT_CONF), but a low threshold that's
# needed to catch a small, fast-moving ball also lets through a lot of
# noisy low-confidence person detections (partial occlusions, motion blur),
# which spawn spurious short-lived tracks and inflate fragmentation.
PLAYER_DETECT_CONF = float(os.environ.get("COURTIQ_PLAYER_CONF", os.environ.get("COURTIQ_DETECT_CONF", "0.4")))
BALL_DETECT_CONF = float(os.environ.get("COURTIQ_BALL_CONF", os.environ.get("COURTIQ_DETECT_CONF", "0.25")))
BALL_MODEL_PATH = os.environ.get("COURTIQ_BALL_MODEL", "models/basketball_ball.pt")
PLAYER_MODEL_PATH = os.environ.get("COURTIQ_PLAYER_MODEL", "models/basketball_players.pt")

# Bundled BoT-SORT config with camera motion compensation (see
# courtiq_botsort.yaml for what changed from ultralytics' defaults and
# why -- real footage has enough camera pan/zoom that ByteTrack's
# static-camera assumption fragments tracks regardless of tuning).
# Override with COURTIQ_TRACKER_CONFIG to point at a different tracker
# yaml entirely (courtiq_bytetrack.yaml is still bundled as an option for
# genuinely static-camera footage, where it's cheaper to run).
TRACKER_CONFIG = os.environ.get(
    "COURTIQ_TRACKER_CONFIG", str(Path(__file__).resolve().parent / "courtiq_botsort.yaml")
)

# courtiq_botsort.yaml / courtiq_bytetrack.yaml set track_buffer assuming
# ~30fps video. This ultralytics version does NOT scale that by the
# video's actual frame rate (confirmed by reading byte_tracker.py:
# max_frames_lost = args.track_buffer, used as a raw frame count) -- on
# footage shot at a materially different fps, the real-world occlusion
# tolerance ends up wrong (e.g. measured on real ~53fps footage: the
# intended ~3s buffer was actually only ~1.7s). scaled_tracker_config()
# rewrites track_buffer to match each video's real fps at analysis time.
TARGET_OCCLUSION_SECONDS = float(os.environ.get("COURTIQ_OCCLUSION_SECONDS", "3.0"))

# Tracks shorter than this are dropped as noise (misdetections, motion-blur
# false positives, ID-switch fragments) rather than counted as real
# players -- see the filter in run_pipeline() for why.
MIN_TRACK_DURATION_SECONDS = float(os.environ.get("COURTIQ_MIN_TRACK_SECONDS", "1.0"))

# Only ball detection is strided (run every Nth frame), not player
# tracking. Player tracking runs on every single frame -- ByteTrack
# matches players between calls by position/overlap, and skipping frames
# there multiplies apparent player displacement between calls, breaking
# that matching and fragmenting track IDs (measured: ~141 tracks for what
# should be ~10-12 players+refs, when both were strided together).
# Skipping only ball detection still cuts total inference cost
# meaningfully, since the ball doesn't need frame-to-frame continuity the
# same way -- possession logic already tolerates gaps via
# TARGET_POSSESSION_HOLD_SECONDS.
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

TARGET_POSSESSION_HOLD_SECONDS = float(os.environ.get("COURTIQ_POSSESSION_HOLD_SECONDS", "1.0"))
# ^ how long (real time) the ball can go undetected before we decide
# possession has actually changed. A FIXED frame count here (the previous
# POSSESSION_HOLD_FRAMES=6) meant this tolerance shrank as video fps grew --
# on a 58fps clip it was only ~0.1s of raw video time, and since
# frames_since_seen in segment_possessions() actually counts ball SAMPLES
# (which only occur every FRAME_STRIDE'th frame, not every frame), it was
# really ~0.3s. Either way, real ball-detection gaps (occlusion, motion
# blur, a contested rebound) routinely exceed that, which was silently
# fragmenting one real possession into dozens of spurious ones -- measured
# on real footage: 91 "possessions" in a 21-second clip, physically
# impossible. Scaling by fps/FRAME_STRIDE keeps the tolerance in real
# seconds regardless of the video's frame rate.
MAX_POSSESSION_DIST_FT = 6.0  # ball must be this close to a player's feet
                               # to be considered "in that player's hands"
TARGET_POSSESSION_SWITCH_CONFIRM_SECONDS = float(os.environ.get("COURTIQ_POSSESSION_SWITCH_CONFIRM_SECONDS", "0.3"))
# ^ how long a different player must be the nearest-to-ball candidate,
# consecutively, before segment_possessions() actually switches who holds
# it -- prevents single-sample "nearest player" jitter (contested rebounds,
# screens, a crowd near the ball) from fragmenting one real possession into
# many spurious ones.

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


CALIBRATION_CACHE_DIR = Path(__file__).resolve().parent / "calibrations"


def save_camera_calibration(camera_name: str, homography: "np.ndarray") -> Path:
    """Save a homography to disk, keyed by camera_name, so a one-time manual
    calibration (see calibrate_camera()) can be reused for every future video
    from the same fixed camera setup instead of re-clicking every time.
    """
    CALIBRATION_CACHE_DIR.mkdir(exist_ok=True)
    path = CALIBRATION_CACHE_DIR / f"{camera_name}.json"
    path.write_text(json.dumps(homography.tolist()))
    return path


def load_camera_calibration(camera_name: str) -> Optional["np.ndarray"]:
    """Load a previously saved calibration for camera_name, or None if none exists yet."""
    path = CALIBRATION_CACHE_DIR / f"{camera_name}.json"
    if not path.exists():
        return None
    return np.array(json.loads(path.read_text()), dtype=np.float64)


def calibrate_camera(video_path: str, camera_name: Optional[str] = None, force_manual: bool = False) -> "np.ndarray":
    """The practical, ship-it calibration strategy: automatic detection
    first (detect_court_quad(), no manual work), and ONLY when that fails
    (no confident floor boundary -- see the max/min-area sanity checks in
    court_quad_debug()), fall back to a one-time manual click
    (pick_corners_interactive()). If camera_name is given, that manual
    result is cached to disk and reused automatically for every future
    video passed with the same camera_name -- so the manual step happens
    at most once per fixed camera setup, not once per video.

    This exists because no automatic method (classical-CV heuristic, or
    either of the two trained keypoint models evaluated against real
    footage -- see keypoint_evidence_video.py and the KaliCalib comparison)
    reliably generalizes to an arbitrary, previously-unseen camera angle.
    See default_full_frame_homography()'s docstring and detect_court_quad()'s
    docstring for why silently guessing wrong is worse than asking once.
    """
    if camera_name and not force_manual:
        cached = load_camera_calibration(camera_name)
        if cached is not None:
            print(f"[courtiq_core] Using cached calibration for camera '{camera_name}'.")
            return cached

    if cv2 is None:
        raise RuntimeError("opencv-python is required.")
    capture = cv2.VideoCapture(video_path)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}.")

    quad = None if force_manual else detect_court_quad(frame)
    if quad is not None:
        print("[courtiq_core] Automatic court-floor detection succeeded -- no manual calibration needed.")
        landmark_names = ["halfcourt_left", "halfcourt_right", "baseline_right", "baseline_left"]
        return compute_homography(quad, landmark_names)

    print(
        "[courtiq_core] Automatic detection found no confident court boundary for this camera angle. "
        "Falling back to one-time manual calibration -- click the landmarks you can see."
    )
    homography = pick_corners_interactive(video_path)
    if camera_name:
        path = save_camera_calibration(camera_name, homography)
        print(f"[courtiq_core] Saved calibration to {path} -- future videos with --camera {camera_name} "
              f"will reuse it automatically, no clicking needed.")
    return homography


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

    Color alone is not enough: measured on real footage, wooden bleachers
    and warm gym-wall tones fall in the same broad hardwood color band as
    the court floor, so a color-only mask picked almost the entire frame
    (crowd, bleachers, scoreboard included) as "the court." Floor and
    crowd look similar in color but very different in texture -- the
    floor is visually smooth (sparse painted lines), the crowd is
    visually busy (hundreds of edges from people/clothes/faces) -- so
    this also requires low local edge density, which excludes most of the
    crowd even where its color matches.

    That alone still isn't enough where the sideline is directly
    connected to the court with no color/texture break -- courtside
    officials and bench players standing right at the boundary (measured
    on real footage: still leaking in after the edge-density filter).
    Their dark ref stripes/suits differ from hardwood in saturation, so a
    saturation floor is added too, and a large morphological opening
    erodes away the narrow "bridge" a standing person forms at the
    sideline (severing the connection) before dilating the main floor
    blob back to size.
    """
    debug = court_quad_debug(frame)
    return debug["quad"]


# Tunable as module-level constants (not buried in the function) so they
# can be adjusted and re-tested without hunting through the function body.
# Kernel sizes are FRACTIONS of frame width, not fixed pixel counts -- a
# fixed "35px" kernel is a very different relative size on a 640px test
# image vs. a real ~1850px video frame, which silently broke earlier tuning.
COURT_HSV_LOWER = (5, 45, 80)
COURT_HSV_UPPER = (30, 200, 255)
COURT_CLOSE_KERNEL_FRACTION = 0.03
COURT_OPEN_KERNEL_FRACTION = 0.03
COURT_MIN_AREA_FRACTION = 0.15
# A real court floor, viewed from an elevated broadcast-style angle, never
# fills nearly the whole frame -- crowd/bleachers/ceiling take up a
# meaningful share above it. A contour approaching the full frame area is
# the signature of crowd bleeding into the floor mask (measured on real
# footage: a bad detection hit 79.5%), not a legitimate detection.
COURT_MAX_AREA_FRACTION = 0.65
# A floor row is mostly one continuous wood-colored span (interrupted only
# by players/painted lines); a crowd row is chopped into many short
# wood-matching fragments (individual people's clothing) with gaps.
# Longest-unbroken-run-per-row is still a color signal (unlike per-pixel
# edge density, which measured on real compressed video as ~83% "smooth"
# even over the crowd -- video compression flattens exactly the
# fine-texture detail edge density depends on, so it doesn't reliably
# survive real footage). This should be more robust to that.
COURT_ROW_RUN_MIN_FRACTION = 0.25  # longest run must span this much of the row's width
COURT_ROW_SUSTAIN_COUNT = 10       # ...for this many consecutive rows, to count as "floor starts here"
COURT_ROW_CLOSE_KERNEL_FRACTION = 0.02  # horizontal close width -- bridges thin painted lines only;
                                         # a court logo is handled separately by _fill_enclosed_holes()


def _odd_kernel(size_px: float) -> int:
    size = max(3, round(size_px))
    return size if size % 2 == 1 else size + 1


def _longest_run_fraction_per_row(mask) -> "np.ndarray":
    """For each row, the longest unbroken run of nonzero pixels, as a
    fraction of the row's width. Vectorized: find run boundaries via
    diff() of the binarized row instead of a per-pixel Python loop.
    """
    height, width = mask.shape
    binary = (mask > 0).astype(np.int8)
    fractions = np.zeros(height, dtype=np.float32)
    for y in range(height):
        row = binary[y]
        if not row.any():
            continue
        padded = np.concatenate(([0], row, [0]))
        diffs = np.diff(padded)
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)
        fractions[y] = float((ends - starts).max()) / width
    return fractions


def _fill_enclosed_holes(mask) -> "np.ndarray":
    """Fill regions of the mask that are 0 (not floor-colored) but fully
    ENCLOSED by 1 (floor-colored) pixels on all sides -- e.g. a dark court
    logo painted on the floor, which doesn't match the wood-color range
    but sits entirely inside the floor region.

    This is a real topological distinction, not a size guess: a court
    logo is surrounded by floor and never touches the frame border, while
    gaps between crowd members connect outward to the rest of the crowd
    and ultimately the frame border. Flood-filling the INVERTED mask from
    the border finds exactly the "reachable from outside" 0-regions (real
    non-floor -- crowd, background); whatever 0-regions are NOT reached
    are enclosed holes, which get filled back to floor-colored.

    This fixes the logo problem without needing a bridging-kernel size
    that (as measured on real footage) can't distinguish a large logo gap
    from equally-large gaps between crowd members -- any kernel wide
    enough to bridge one bridges the other too.
    """
    height, width = mask.shape
    inverted = (mask == 0).astype(np.uint8) * 255
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    reachable = inverted.copy()
    cv2.floodFill(reachable, flood_mask, (0, 0), 128)
    # Anything still 255 in `reachable` is a 0-region that flood fill from
    # the border never reached -- i.e. fully enclosed by floor-colored pixels.
    enclosed = reachable == 255
    filled = mask.copy()
    filled[enclosed] = 255
    return filled


def find_court_top_row(color_mask) -> Optional[int]:
    """Scan down from the top of the frame for the first row where the
    longest unbroken wood-colored run is wide AND stays wide for
    COURT_ROW_SUSTAIN_COUNT consecutive rows -- a single wide row could
    just be a lucky bleacher-color alignment; a sustained run of them is
    the actual floor starting. Returns the row index, or None if no such
    transition was found (the color signal never looked floor-like).

    A horizontal morphological close is applied first, sized to bridge
    small internal gaps (painted lines, a court logo, a player standing on
    the floor) without bridging the much larger gap between the floor and
    the crowd/bleachers above it (measured on real footage: that gap is a
    differently-colored band -- a sponsor banner or railing -- that
    doesn't match the wood-color range in the first place, so a modest
    close kernel doesn't risk merging floor and crowd together). Without
    this, a single logo or a row of players interrupting the run was
    enough to make an otherwise-clean floor row fail the width check.
    """
    filled = _fill_enclosed_holes(color_mask)
    close_k = _odd_kernel(color_mask.shape[1] * COURT_ROW_CLOSE_KERNEL_FRACTION)
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((1, close_k), np.uint8))
    run_fraction = _longest_run_fraction_per_row(closed)
    height = len(run_fraction)
    wide = run_fraction >= COURT_ROW_RUN_MIN_FRACTION
    run_length = 0
    for y in range(height):
        if wide[y]:
            run_length += 1
            if run_length >= COURT_ROW_SUSTAIN_COUNT:
                return y - COURT_ROW_SUSTAIN_COUNT + 1
        else:
            run_length = 0
    return None


def court_quad_debug(frame) -> dict:
    """Same detection as detect_court_quad(), but returns every
    intermediate mask and stat instead of just the final answer -- so
    tuning the thresholds above can be based on actual numbers (mask
    coverage %, contour area %) instead of guessing blind between runs.
    See visualize_court.py for how this gets displayed.

    Design note: two earlier versions relied on per-pixel edge density
    ("is this pixel visually busy") to separate crowd from floor. Measured
    on real, compressed footage, edge density was nearly useless -- 83.5%
    of the ENTIRE frame (crowd included) registered as "smooth," because
    video compression flattens exactly the fine high-frequency texture
    detail that signal depended on. It worked on clean synthetic test
    images and failed on the real thing, which is exactly the "looked
    right in code, failed on real data" trap this project has hit twice
    before with other components.

    This version uses a color-only signal instead: the longest unbroken
    horizontal run of wood-colored pixels per image row. A real floor row
    is mostly one continuous span (interrupted only by players/painted
    lines); a crowd row is chopped into many short wood-matching fragments
    (individual people's clothing) with gaps between them. Still
    fundamentally a color measurement, so it should survive compression
    far better than fine per-pixel texture did. find_court_top_row() finds
    where a sustained run of "wide" rows begins, and everything above that
    row is zeroed out of the mask before contour detection -- explicitly
    excluding the crowd/bleachers by geometry-of-the-color-signal rather
    than by texture.
    """
    result = {
        "quad": None,
        "color_mask": None,            # raw HSV color threshold (per-pixel)
        "morphed_color_mask": None,    # color mask, crowd rows zeroed out, after close+open
        "color_coverage": 0.0, "morphed_coverage": 0.0,
        "largest_contour_fraction": 0.0, "court_top_row": None, "reason": None,
    }
    if cv2 is None or np is None:
        result["reason"] = "opencv/numpy not available"
        return result

    height, width = frame.shape[:2]
    frame_area = height * width
    close_k = _odd_kernel(width * COURT_CLOSE_KERNEL_FRACTION)
    open_k = _odd_kernel(width * COURT_OPEN_KERNEL_FRACTION)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array(COURT_HSV_LOWER), np.array(COURT_HSV_UPPER))
    result["color_mask"] = color_mask
    result["color_coverage"] = float((color_mask > 0).sum()) / frame_area

    top_row = find_court_top_row(color_mask)
    result["court_top_row"] = top_row
    if top_row is None:
        # find_court_top_row() exists specifically to exclude the crowd/
        # bleachers from the floor mask before contour detection. If it
        # couldn't find that transition, running contour detection on the
        # unfiltered mask anyway silently lets the crowd back in -- this
        # previously produced a "quad" that was actually floor+bleachers
        # merged into one ~80%-of-frame blob, since nothing was ever
        # zeroed out. Fail loudly instead of returning a wrong quad.
        result["reason"] = "could not find a floor/crowd row transition -- refusing to trust the unfiltered mask"
        return result
    mask = _fill_enclosed_holes(color_mask)
    if top_row > 0:
        mask[:top_row, :] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    result["morphed_color_mask"] = mask
    result["morphed_coverage"] = float((mask > 0).sum()) / frame_area

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        result["reason"] = "no contours survived color masking + morphology"
        return result

    accepted = max(contours, key=cv2.contourArea)
    area_fraction = cv2.contourArea(accepted) / frame_area
    result["largest_contour_fraction"] = area_fraction
    if area_fraction < COURT_MIN_AREA_FRACTION:
        result["reason"] = f"largest surviving contour is only {area_fraction:.1%} of the frame (< {COURT_MIN_AREA_FRACTION:.0%} minimum)"
        return result
    if area_fraction > COURT_MAX_AREA_FRACTION:
        result["reason"] = (
            f"largest surviving contour is {area_fraction:.1%} of the frame (> {COURT_MAX_AREA_FRACTION:.0%} "
            f"maximum) -- this is the signature of crowd/bleachers merging into the floor mask, not a real detection"
        )
        return result

    peri = cv2.arcLength(accepted, True)
    approx = cv2.approxPolyDP(accepted, 0.02 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(accepted)
        approx = cv2.boxPoints(rect).reshape(-1, 1, 2)

    points = approx.reshape(-1, 2).astype(float)
    color_quad = order_quad_points(points)
    result["color_quad"] = color_quad

    # A painted-line-detection refinement (refine_quad_to_painted_lines())
    # was attempted here, to snap the color-region quad to the court's
    # actual boundary lines instead of "where does the wood color stop."
    # Tested against real footage, it locked onto the wrong lines (the
    # arc/logo, not the boundary) and produced a clearly worse result than
    # the plain color-region quad -- see the git history for the attempt.
    # NOT wired in until it's actually validated; using the color-region
    # quad directly is the working, tested behavior.
    result["quad"] = color_quad
    result["line_refined"] = False
    return result


LINE_WHITE_V_MIN = 170       # painted court lines are bright; value channel floor
LINE_WHITE_S_MAX = 90        # ...and low-saturation (white/near-white paint)
LINE_MIN_LENGTH_FRACTION = 0.12   # a real boundary line segment must span at least this much of the frame diagonal
LINE_MAX_GAP_FRACTION = 0.01      # HoughLinesP gap tolerance, as a fraction of frame diagonal
LINE_ANGLE_CLUSTER_TOLERANCE_DEG = 15  # segments within this many degrees of each other are "the same orientation"


def _detect_line_segments(frame, floor_mask):
    """Find long, straight, bright/low-saturation line segments confined
    to the floor region -- candidates for the court's actual painted
    boundary lines (sideline, baseline, etc), as opposed to the crowd,
    scoreboard text, or anything else bright elsewhere in the frame.
    """
    height, width = frame.shape[:2]
    diag = math.hypot(height, width)
    dilate_k = _odd_kernel(width * 0.01)
    roi = cv2.dilate(floor_mask, np.ones((dilate_k, dilate_k), np.uint8))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv, np.array([0, 0, LINE_WHITE_V_MIN]), np.array([180, LINE_WHITE_S_MAX, 255])
    )
    white_mask = cv2.bitwise_and(white_mask, roi)

    edges = cv2.Canny(white_mask, 50, 150)
    min_length = int(diag * LINE_MIN_LENGTH_FRACTION)
    max_gap = int(diag * LINE_MAX_GAP_FRACTION)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                             minLineLength=min_length, maxLineGap=max_gap)
    if lines is None:
        return []
    return [tuple(line) for line in lines.reshape(-1, 4)]


def _line_angle_deg(segment) -> float:
    x1, y1, x2, y2 = segment
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _line_to_general_form(segment):
    """(x1,y1,x2,y2) -> (a, b, c) for the line ax + by + c = 0, normalized."""
    x1, y1, x2, y2 = segment
    a, b = y2 - y1, x1 - x2
    c = -(a * x1 + b * y1)
    norm = math.hypot(a, b) or 1.0
    return a / norm, b / norm, c / norm


def _intersect_lines(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return (x, y)


def refine_quad_to_painted_lines(frame, floor_mask, fallback_quad):
    """Refine a color-region quad to the court's actual painted boundary
    lines. Detects candidate line segments (see _detect_line_segments()),
    groups them into two roughly-perpendicular orientation clusters
    (the two "sideline-like" directions and the two "baseline-like"
    directions a real court boundary has, even when the whole thing is
    diagonal in-frame), keeps the longest segment representing each of the
    4 sides, and intersects adjacent sides to get the 4 true corners.

    Returns None (caller should keep the color-region quad instead) if
    fewer than 4 well-separated boundary lines were found -- a partial or
    unclear line-detection result is worse than the honest color-region
    approximation, not better.
    """
    if cv2 is None or np is None:
        return None
    segments = _detect_line_segments(frame, floor_mask)
    if len(segments) < 4:
        return None

    # Cluster by angle (mod 180) into 2 perpendicular-ish groups using the
    # angle relative to the first segment, wrapped to [-90, 90).
    base_angle = _line_angle_deg(segments[0])
    group_a, group_b = [], []
    for seg in segments:
        angle = _line_angle_deg(seg)
        diff = (angle - base_angle + 90) % 180 - 90
        (group_a if abs(diff) < LINE_ANGLE_CLUSTER_TOLERANCE_DEG else group_b).append(seg)
    if len(group_a) < 2 or len(group_b) < 2:
        return None

    def two_most_separated(group):
        # Within one orientation, the two real boundary lines (e.g. both
        # sidelines) are the ones farthest apart perpendicular to their
        # own direction -- not just the two longest segments, which could
        # both be the same line detected twice.
        lines = [_line_to_general_form(s) for s in group]
        best_pair, best_dist = None, -1.0
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b, c1 = lines[i]
                _, _, c2 = lines[j]
                dist = abs(c1 - c2)
                if dist > best_dist:
                    best_dist, best_pair = dist, (i, j)
        i, j = best_pair
        return lines[i], lines[j]

    line_a1, line_a2 = two_most_separated(group_a)
    line_b1, line_b2 = two_most_separated(group_b)

    corners = []
    for la in (line_a1, line_a2):
        for lb in (line_b1, line_b2):
            pt = _intersect_lines(la, lb)
            if pt is None:
                return None
            corners.append(pt)

    height, width = frame.shape[:2]
    margin = max(width, height) * 0.5
    for x, y in corners:
        if not (-margin <= x <= width + margin and -margin <= y <= height + margin):
            # A detected "corner" wildly outside the frame means the line
            # pairing was wrong (near-parallel lines intersecting far away) --
            # don't trust it.
            return None

    return order_quad_points(corners)


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


# ---------------------------------------------------------------------------
# Third calibration option: a trained YOLOv8-pose court-keypoint model,
# instead of classical-CV floor detection or manual clicking.
#
# Source: github.com/HanaFEKI/AI_BasketBall_Analysis_v1, whose
# tactical_view/tactical_view.py computes a homography via
# cv2.findHomography(image_points, court_points) with 18 positionally-
# ordered court landmarks -- meaning the keypoint model's output indices
# must match this exact order by construction (their own pipeline depends
# on it). This list is that order, converted from their meters (a 28m x
# 15m FIBA court) to feet and with x/y swapped to match this file's
# convention (x = width axis 0-50ft, y = length axis 0-94ft; see
# COURT_WIDTH_FT/COURT_LENGTH_FT -- a FIBA court's 49.2x91.9ft is close
# enough to the NBA 50x94ft this file otherwise assumes that the existing
# ON_COURT_MARGIN_FT tolerance absorbs the difference).
#
# NOT VALIDATED: the trained weights (a Google Drive link in that repo's
# training/README.md, ambiguous -- only one link is given for three
# different models) could not be downloaded or verified from the
# environment this was written in (drive.google.com is blocked there).
# Download it yourself, confirm it's actually the court-keypoint model
# (not the player or ball detector), and validate against your own
# footage with keypoint_model_homography() before trusting it -- same
# rule as every other swapped-in model in this file.
KEYPOINT_MODEL_COURT_POINTS_FT = [
    (0.00, 0.00), (2.99, 0.00), (16.99, 0.00), (32.81, 0.00), (46.26, 0.00), (49.21, 0.00),
    (49.21, 45.93), (0.00, 45.93),
    (16.99, 19.00), (32.81, 19.00),
    (49.21, 91.86), (46.26, 91.86), (32.81, 91.86), (16.99, 91.86), (2.99, 91.86), (0.00, 91.86),
    (16.99, 72.87), (32.81, 72.87),
]
KEYPOINT_MODEL_CONF_THRESHOLD = 0.6
# Reject the computed homography outright if its average reprojection
# error (in court feet, over the RANSAC inlier keypoints) exceeds this --
# better to fail loudly than silently hand back a badly warped homography
# that quietly corrupts every downstream position.
KEYPOINT_MODEL_REPROJ_ERROR_MAX_FT = 5.0


def keypoint_model_homography(video_path: str, model_path: str, conf_threshold: float = KEYPOINT_MODEL_CONF_THRESHOLD) -> "np.ndarray":
    """Compute a homography from a trained court-keypoint model's output on
    the video's first frame, instead of classical-CV floor detection or
    manual clicking. See the KEYPOINT_MODEL_COURT_POINTS_FT comment above
    for the model this expects and what has (and hasn't) been verified.
    """
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required.")
    model = _load_model(model_path)
    capture = cv2.VideoCapture(video_path)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}.")

    result = model(frame, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        raise RuntimeError(
            "The model detected no instance in this frame at all -- confirm model_path is "
            "actually the court-keypoint model, not the player or ball detector, or try a "
            "different frame."
        )
    xy = result.keypoints.xy[0].tolist()
    conf = result.keypoints.conf[0].tolist() if result.keypoints.conf is not None else [1.0] * len(xy)

    if len(xy) != len(KEYPOINT_MODEL_COURT_POINTS_FT):
        raise RuntimeError(
            f"Model produced {len(xy)} keypoints, expected {len(KEYPOINT_MODEL_COURT_POINTS_FT)}. "
            f"This model's keypoint schema doesn't match what this function assumes -- "
            f"do not trust the resulting homography."
        )

    image_points = [pt for pt, c in zip(xy, conf) if c >= conf_threshold]
    court_points = [pt for pt, c in zip(KEYPOINT_MODEL_COURT_POINTS_FT, conf) if c >= conf_threshold]
    if len(image_points) < 4:
        raise RuntimeError(
            f"Only {len(image_points)} keypoints were detected above conf_threshold={conf_threshold} "
            f"(need >= 4). Lower conf_threshold, or this frame/model isn't confident enough."
        )

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(court_points, dtype=np.float32)
    homography, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if homography is None:
        raise RuntimeError("cv2.findHomography could not compute a homography from the detected keypoints.")

    homography = fix_homography_flip(homography)

    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), dtype=bool)
    error = reprojection_error(homography, src[inlier_mask], dst[inlier_mask])
    print(f"[courtiq_core] Keypoint homography reprojection error: {error:.2f} ft "
          f"({int(inlier_mask.sum())}/{len(src)} keypoints used as RANSAC inliers)")
    if error > KEYPOINT_MODEL_REPROJ_ERROR_MAX_FT:
        raise RuntimeError(
            f"Reprojection error {error:.1f} ft exceeds the {KEYPOINT_MODEL_REPROJ_ERROR_MAX_FT} ft "
            f"sanity threshold -- this homography is not trustworthy. Try a different frame, "
            f"a lower/higher --keypoint-conf, or fall back to --interactive."
        )
    return homography


# The four real-world corners of the court, in this file's feet convention
# (see KEYPOINT_MODEL_COURT_POINTS_FT) -- used by hybrid_homography() to
# match classical-CV floor-boundary corners to the nearest true court
# corner by projected position, not by assumed image position (near/far,
# top/bottom). That's what lets this work for a diagonally-framed court:
# nothing here assumes any particular orientation in the image.
TRUE_COURT_CORNERS_FT = [(0.0, 0.0), (COURT_WIDTH_FT, 0.0), (COURT_WIDTH_FT, COURT_LENGTH_FT), (0.0, COURT_LENGTH_FT)]
HYBRID_LENIENT_CONF = 0.05          # use nearly every keypoint the model produced; RANSAC sorts good from bad
HYBRID_CORNER_MATCH_MAX_FT = 15.0   # a classical-CV corner projecting farther than this from every true
                                     # corner isn't trustworthy enough to add as a supplemental point


def hybrid_homography(video_path: str, keypoint_model_path: str, lenient_conf: float = HYBRID_LENIENT_CONF) -> "np.ndarray":
    """Combines the trained keypoint model (better at far/small landmarks,
    even at lower confidence, per visual inspection on real footage --
    it's finding the right general area, just not confidently) with
    detect_court_quad()'s classical-CV floor-boundary detection (better at
    the near/close part of the court, where it's not fighting
    crowd/occlusion or scale the way the trained model was) instead of
    relying on either alone.

    Two-pass approach, deliberately NOT assuming any fixed image
    layout (so a diagonally-framed court still works):
      1. Fit a rough homography from the keypoint model's detections alone.
      2. Use THAT rough homography to project detect_court_quad()'s 4
         floor-boundary corners into real-world court coordinates, then
         match each one to whichever of the 4 TRUE court corners it landed
         closest to. This correspondence is found by projected position,
         not raw image position (top/bottom, left/right) -- a diagonal
         court works the same as an axis-aligned one here.
      3. Add those matched corners as extra correspondence points and
         refit. The classical corners tend to be more confidently placed
         at the boundary than the trained model alone, which should pull
         the final homography's accuracy up specifically at the edges --
         exactly where the keypoint-model-only version was weakest.
    """
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required.")
    model = _load_model(keypoint_model_path)
    capture = cv2.VideoCapture(video_path)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}.")

    result = hybrid_homography_for_frame(frame, model, lenient_conf)
    if result is None:
        raise RuntimeError("Could not compute a homography from this frame -- see printed diagnostics above.")
    homography, inliers, total, error = result
    print(f"[courtiq_core] Hybrid homography reprojection error: {error:.2f} ft ({inliers}/{total} points used as RANSAC inliers)")
    return homography


def hybrid_homography_for_frame(frame, model, lenient_conf: float = HYBRID_LENIENT_CONF):
    """Core logic factored out of hybrid_homography() so it can run against
    any single frame array, not just video_path's frame 0 -- used both by
    hybrid_homography() itself and by best_hybrid_homography()'s
    multi-frame search below. Returns (homography, inlier_count,
    total_point_count, reprojection_error_ft), or None if this frame
    couldn't produce a usable homography at all (too few confident
    keypoints, or findHomography failed outright).
    """
    result = model(frame, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None
    xy = result.keypoints.xy[0].tolist()
    conf = result.keypoints.conf[0].tolist() if result.keypoints.conf is not None else [1.0] * len(xy)
    if len(xy) != len(KEYPOINT_MODEL_COURT_POINTS_FT):
        return None

    image_points = [pt for pt, c in zip(xy, conf) if c >= lenient_conf]
    court_points = [pt for pt, c in zip(KEYPOINT_MODEL_COURT_POINTS_FT, conf) if c >= lenient_conf]
    if len(image_points) < 4:
        return None

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(court_points, dtype=np.float32)
    H1, mask1_raw = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H1 is None:
        return None
    H1 = fix_homography_flip(H1)
    # Real RANSAC inlier mask, NOT a fabricated all-ones one -- using a fake
    # mask here previously caused a frame with a genuinely bad fit (127ft
    # reprojection error) to be reported as "18/18 inliers," which won the
    # multi-frame search's inlier-count comparison despite being garbage.
    mask1 = mask1_raw.ravel().astype(bool) if mask1_raw is not None else np.ones(len(src), dtype=bool)
    if not mask1.any():
        return None

    quad = detect_court_quad(frame)
    extra_src, extra_dst = [], []
    if quad is not None:
        for image_pt in quad:
            rough_x_ft, rough_y_ft = project_point(H1, image_pt)
            distances = [math.hypot(rough_x_ft - cx, rough_y_ft - cy) for cx, cy in TRUE_COURT_CORNERS_FT]
            best_idx = distances.index(min(distances))
            if distances[best_idx] <= HYBRID_CORNER_MATCH_MAX_FT:
                extra_src.append(image_pt)
                extra_dst.append(TRUE_COURT_CORNERS_FT[best_idx])

    combined_src = image_points + extra_src
    combined_dst = court_points + extra_dst
    src2 = np.array(combined_src, dtype=np.float32)
    dst2 = np.array(combined_dst, dtype=np.float32)
    H2, mask2 = cv2.findHomography(src2, dst2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H2 is None:
        error1 = reprojection_error(H1, src[mask1], dst[mask1])
        return H1, int(mask1.sum()), len(src), error1
    H2 = fix_homography_flip(H2)

    inlier_mask = mask2.ravel().astype(bool) if mask2 is not None else np.ones(len(src2), dtype=bool)
    error = reprojection_error(H2, src2[inlier_mask], dst2[inlier_mask])
    if error > KEYPOINT_MODEL_REPROJ_ERROR_MAX_FT:
        inlier_mask1 = mask1
        error1 = reprojection_error(H1, src[inlier_mask1], dst[inlier_mask1])
        return H1, int(inlier_mask1.sum()), len(src), error1
    return H2, int(inlier_mask.sum()), len(src2), error


BEST_FRAME_CANDIDATE_COUNT = 20  # how many frames to sample when searching for the best calibration frame


def best_hybrid_homography(video_path: str, keypoint_model_path: str, lenient_conf: float = HYBRID_LENIENT_CONF,
                            num_candidates: int = BEST_FRAME_CANDIDATE_COUNT) -> "np.ndarray":
    """Calibrating off frame 0 alone assumes that specific frame happens to
    have clear sightlines to every landmark -- there's no reason to expect
    that (players/refs can be standing right on top of a far keypoint in
    any given frame). This scans num_candidates frames spread across the
    video instead, scores each with hybrid_homography_for_frame() (more
    RANSAC inliers, and among ties lower reprojection error, wins), and
    returns whichever homography actually fit best -- using the same
    diagnostic numbers already being computed, just to choose instead of
    only to report.
    """
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required.")
    model = _load_model(keypoint_model_path)
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Could not read frame count from {video_path}.")

    sample_indices = sorted(set(int(i * frame_count / num_candidates) for i in range(num_candidates)))
    best = None  # (inliers, -error, homography, frame_idx, total)
    for idx in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = capture.read()
        if not ok:
            continue
        result = hybrid_homography_for_frame(frame, model, lenient_conf)
        if result is None:
            continue
        homography, inliers, total, error = result
        score = (inliers, -error)
        if best is None or score > best[0]:
            best = (score, homography, idx, inliers, total, error)
    capture.release()

    if best is None:
        raise RuntimeError(
            f"None of the {len(sample_indices)} sampled frames produced a usable homography -- "
            f"try a lower lenient_conf, more candidate frames, or --interactive."
        )
    _, homography, frame_idx, inliers, total, error = best
    print(f"[courtiq_core] Best calibration frame: {frame_idx}/{frame_count} "
          f"(reprojection error {error:.2f} ft, {inliers}/{total} points used as RANSAC inliers, "
          f"out of {len(sample_indices)} candidates tried)")
    return homography


def fix_homography_flip(homography: "np.ndarray") -> "np.ndarray":
    """Detect and correct a horizontal flip in the homography's
    rotation/scaling block (checked via its determinant sign). Ported from
    tactical_view/homography.py in the source repo -- point-correspondence
    homographies can come out mirrored under some point configurations,
    and this was silently missing from the first version of this function.
    """
    rotation_scale = homography[:2, :2]
    if np.linalg.det(rotation_scale) < 0:
        homography = homography.copy()
        homography[:, 0] *= -1
    return homography


def reprojection_error(homography: "np.ndarray", image_points, court_points) -> float:
    """Mean distance (in court feet) between where image_points actually
    land after projection and where court_points says they should be --
    a direct measure of how good the fitted homography actually is,
    instead of trusting whatever cv2.findHomography returns unchecked.
    """
    img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(img_pts, homography).reshape(-1, 2)
    court_pts = np.array(court_points, dtype=np.float32)
    return float(np.mean(np.linalg.norm(projected - court_pts, axis=1)))


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


MIN_COLOR_SAMPLES_FOR_FIT = int(os.environ.get("COURTIQ_MIN_COLOR_SAMPLES", "10"))


def assign_teams(track_color_samples: dict) -> dict:
    """KMeans(k=2) over jersey color -> {track_id: 0|1}.

    track_color_samples maps track_id -> list of per-frame color samples
    (not a pre-averaged single color). Only tracks with at least
    MIN_COLOR_SAMPLES_FOR_FIT samples are used to *fit* the two cluster
    centers -- a track that was only seen for a second or two has a noisy
    mean color (motion blur, partial crops, lighting), and letting it
    influence the cluster boundaries equally with a well-sampled, minutes
    -long track skews both team assignments. Every track, including short
    ones, still gets assigned to whichever fitted center is nearest.
    """
    if not track_color_samples:
        return {}
    if np is None:
        raise RuntimeError("numpy is required for team assignment.")

    means = {tid: sum(colors) / len(colors) for tid, colors in track_color_samples.items() if colors}
    if not means:
        return {}

    fit_ids = [tid for tid, colors in track_color_samples.items() if len(colors) >= MIN_COLOR_SAMPLES_FOR_FIT]
    if len(fit_ids) < 2:
        fit_ids = list(means.keys())  # not enough well-sampled tracks; fit on everything available
    fit_data = np.array([means[tid] for tid in fit_ids], dtype=np.float32)

    if len(fit_ids) < 2:
        only_id = fit_ids[0] if fit_ids else next(iter(means))
        return {only_id: 0}

    if cv2 is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, fit_labels, centers = cv2.kmeans(fit_data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        centers = [centers[0], centers[1]]
    else:
        # Minimal fallback so this module still degrades gracefully without cv2.kmeans.
        centers = [fit_data[0], fit_data[-1]]

    return {
        tid: int(np.argmin([np.linalg.norm(mean - c) for c in centers]))
        for tid, mean in means.items()
    }


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


def scaled_tracker_config(base_path: str, fps: float) -> str:
    """Rewrite track_buffer in the tracker config to TARGET_OCCLUSION_SECONDS
    worth of *this video's actual frame rate*, not a fixed frame count
    tuned for an assumed ~30fps. See the TARGET_OCCLUSION_SECONDS comment
    for why this matters -- this ultralytics version doesn't do this
    scaling itself. Writes a temp file (ultralytics' tracker= arg expects
    a path) and returns its path; the base config on disk is untouched.
    """
    if yaml is None:
        return base_path  # no PyYAML available; fall back to the fixed config as-is
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["track_buffer"] = max(1, round(TARGET_OCCLUSION_SECONDS * fps))
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


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


def resolve_person_class(model) -> int:
    """Find the player/person class index by name rather than assuming
    COCO's index 0. A basketball-specific player dataset typically names
    its class "Player" (and may have a separate "Ref" class, which this
    deliberately does NOT match, so referees stay excluded). An explicit
    COURTIQ_PERSON_CLASS env var always wins if set.
    """
    override = os.environ.get("COURTIQ_PERSON_CLASS")
    if override is not None:
        return int(override)
    names = getattr(model, "names", None) or {}
    for idx, name in names.items():
        if str(name).strip().lower() in PERSON_CLASS_NAMES:
            return int(idx)
    return PERSON_CLASS_FALLBACK


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
    # Always load a separate ball_model instance, even if the paths happen
    # to match -- aliasing ball_model to the SAME object as player_model
    # let player_model.track(persist=True)'s internal tracker/predictor
    # state leak into the "separate" ball detection call on every frame,
    # which (measured on real footage) silently degraded ball detection
    # far below what the same weights produce when loaded independently.
    ball_model = _load_model(ball_model_path)
    ball_class = resolve_ball_class(ball_model)
    person_class = resolve_person_class(player_model)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total = min(frame_count, max_frames) if max_frames else frame_count
    tracker_config = scaled_tracker_config(TRACKER_CONFIG, fps)
    print(f"[courtiq_core] Video fps={fps:.1f} -> tracker occlusion buffer scaled to {TARGET_OCCLUSION_SECONDS}s of real time.")

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
        # possession logic already tolerates gaps via TARGET_POSSESSION_HOLD_SECONDS
        # -- is strided, to still cut overall inference cost.
        ok, frame = capture.read()
        if not ok:
            break
        t = frame_idx / fps

        # Player detection + tracking (ByteTrack keeps IDs stable across frames).
        track_result = player_model.track(
            frame, classes=[person_class], conf=PLAYER_DETECT_CONF, tracker=tracker_config,
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
            ball_result = ball_model(frame, classes=[ball_class], conf=BALL_DETECT_CONF, verbose=False)[0]
            ball_x_ft = ball_y_ft = None
            detected = False
            box_found = ball_result.boxes is not None and len(ball_result.boxes) > 0
            if box_found:
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

    # Drop tracks that never persisted long enough to plausibly be a real
    # player -- a real player is on court continuously across a possession,
    # not detected for a handful of frames. Short-lived tracks are almost
    # always detector noise (a misdetection, motion blur, momentary false
    # positive) or a fragment left over from an ID switch. Filtering these
    # out before team assignment also keeps that noise from polluting the
    # jersey-color KMeans clustering.
    survivors = {
        track_id
        for track_id in track_first_seen
        if (track_last_seen[track_id] - track_first_seen[track_id]) / fps >= MIN_TRACK_DURATION_SECONDS
    }
    track_colors = {tid: colors for tid, colors in track_colors.items() if tid in survivors}
    player_samples = [s for s in player_samples if s.track_id in survivors]
    track_first_seen = {tid: v for tid, v in track_first_seen.items() if tid in survivors}
    track_last_seen = {tid: v for tid, v in track_last_seen.items() if tid in survivors}

    team_by_track = assign_teams(track_colors)
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

    # frames_since_seen counts ball SAMPLES, not raw video frames -- ball
    # detection only runs every FRAME_STRIDE'th frame (see run_pipeline()),
    # so the real-time gap this hold represents is
    # hold_samples * FRAME_STRIDE / fps. Solve for hold_samples so the
    # actual tolerance is TARGET_POSSESSION_HOLD_SECONDS regardless of the
    # video's fps or FRAME_STRIDE.
    hold_samples = max(1, round(TARGET_POSSESSION_HOLD_SECONDS * fps / FRAME_STRIDE)) if fps else 6
    # A different player being nearest for a single sample is not enough to
    # switch possession -- in a crowded contested-ball moment (a rebound, a
    # screen, a double-team), the nearest player can flip between two or
    # three people from sample to sample purely from position jitter, with
    # the ball never actually changing hands. Require a candidate to be
    # nearest for several CONSECUTIVE samples before actually switching
    # current_holder -- real hysteresis, not just the "hold through a gap"
    # behavior above. Measured on real footage: without this, single-sample
    # flips fragmented one real possession into dozens of spurious ones (91
    # "possessions" in a 21-second clip).
    switch_confirm_samples = max(1, round(TARGET_POSSESSION_SWITCH_CONFIRM_SECONDS * fps / FRAME_STRIDE)) if fps else 2

    current_holder = None
    frames_since_seen = 0
    candidate_holder = None
    candidate_streak = 0
    holder_by_frame = {}

    for ball in ball_samples:
        candidates = players_by_frame.get(ball.frame, [])
        if ball.detected and candidates:
            nearest, dist = _nearest_player(ball.x_ft, ball.y_ft, candidates)
            if nearest is not None and dist <= MAX_POSSESSION_DIST_FT:
                frames_since_seen = 0
                if nearest.track_id == current_holder:
                    candidate_holder, candidate_streak = None, 0
                elif nearest.track_id == candidate_holder:
                    candidate_streak += 1
                else:
                    candidate_holder, candidate_streak = nearest.track_id, 1
                if candidate_holder is not None and candidate_streak >= switch_confirm_samples:
                    current_holder = candidate_holder
                    candidate_holder, candidate_streak = None, 0
            else:
                frames_since_seen += 1
        else:
            frames_since_seen += 1
        if frames_since_seen > hold_samples:
            current_holder = None
            candidate_holder, candidate_streak = None, 0
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
    parser.add_argument(
        "--camera", default=None,
        help="Name for this camera setup (e.g. 'home_gym_east_angle'). Enables "
             "calibrate_camera()'s automatic-first, manual-fallback-once strategy: tries "
             "automatic floor detection, and only if that fails, prompts for a one-time "
             "manual click -- then caches the result under this name in calibrations/ so "
             "every future video with the same --camera name reuses it with zero clicking. "
             "Takes priority over --interactive/automatic when set.",
    )
    parser.add_argument(
        "--force-manual", action="store_true",
        help="Used with --camera: skip automatic detection entirely and go straight to the "
             "manual click, even if detect_court_quad() would technically return a quad. Use "
             "this to (re)establish a correct calibration when automatic detection found A "
             "quad but the WRONG one (e.g. it found only part of the floor) -- "
             "detect_court_quad() only checks that a plausible-shaped/sized region was found, "
             "not that it's actually the full correct boundary, so it can be technically "
             "non-empty and still wrong.",
    )
    parser.add_argument(
        "--keypoint-model", default=None,
        help="Path to a trained court-keypoint model (see keypoint_model_homography() "
             "and KEYPOINT_MODEL_COURT_POINTS_FT). Overrides --interactive/automatic "
             "detection when set. UNVALIDATED against real footage -- verify the "
             "detection rate/accuracy yourself before trusting it.",
    )
    parser.add_argument(
        "--keypoint-conf", type=float, default=KEYPOINT_MODEL_CONF_THRESHOLD,
        help=f"Confidence threshold for --keypoint-model (default {KEYPOINT_MODEL_CONF_THRESHOLD}). "
             "A freshly-trained model's confidence calibration can be very different from "
             "whatever default this was set from -- check debug_keypoints.py's raw output "
             "and lower this if real detections are being filtered out.",
    )
    parser.add_argument(
        "--hybrid-model", default=None,
        help="Path to a trained court-keypoint model, used in HYBRID mode: the keypoint "
             "model handles far/small landmarks (even at low confidence -- it's finding "
             "the right area, just not confidently), and detect_court_quad()'s classical-CV "
             "floor detection supplements the near/close corners. See hybrid_homography(). "
             "Takes priority over --keypoint-model/--interactive/automatic when set.",
    )
    parser.add_argument(
        "--best-frame-model", default=None,
        help="Path to a trained court-keypoint model. Same hybrid approach as --hybrid-model, "
             "but scans multiple frames across the video (see BEST_FRAME_CANDIDATE_COUNT) and "
             "picks whichever one actually fits best, instead of assuming frame 0 is good. "
             "Slower (runs detection on several frames up front) but doesn't depend on frame 0 "
             "happening to have clear sightlines to every landmark. Takes priority over "
             "--hybrid-model/--keypoint-model/--interactive/automatic when set.",
    )
    args = parser.parse_args()

    if args.best_frame_model:
        homography = best_hybrid_homography(args.video, args.best_frame_model)
    elif args.hybrid_model:
        homography = hybrid_homography(args.video, args.hybrid_model)
    elif args.keypoint_model:
        homography = keypoint_model_homography(args.video, args.keypoint_model, conf_threshold=args.keypoint_conf)
    elif args.camera:
        homography = calibrate_camera(args.video, camera_name=args.camera, force_manual=args.force_manual)
    elif args.interactive:
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
