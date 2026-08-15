#!/usr/bin/env python3
"""CourtIQ: a local, privacy-friendly basketball action review app.

Run with: python3 app.py
Then open: http://127.0.0.1:8787
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    import cv2
except ImportError:  # A clearer startup error than a trace during upload.
    cv2 = None

import courtiq_core

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
THUMBS = DATA / "thumbnails"
STATE_FILE = DATA / "state.json"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}

# Long-running tracking/detection runs in a background thread; the frontend
# polls /api/analyze/status instead of blocking the upload/analyze request.
JOB_LOCK = threading.Lock()
JOB = {"status": "idle", "progress": 0, "total": 0, "error": None, "video_path": None}

LOCK = threading.Lock()


def default_state():
    return {"video": None, "actions": [], "tracks_summary": None}


def load_state():
    if not STATE_FILE.exists():
        return default_state()
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return default_state()


def save_state(state):
    DATA.mkdir(exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(STATE_FILE)


STATE = load_state()


def clean_filename(value: str) -> str:
    name = Path(unquote(value)).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    return name or "game.mp4"


def seconds_label(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


DECISION_TITLES = {
    "open_teammate_ignored": "Open teammate ignored",
    "shot_selection": "Shot selection",
    "spacing": "Spacing",
    "low_confidence": "Ball tracking too unreliable to grade",
}


def default_homography(width: int, height: int):
    """Best-effort fallback mapping the visible frame to a full court.

    TODO: this assumes the uploaded clip is a straight-on full-court view,
    which is rarely true for real broadcast/sideline footage. courtiq_core's
    pick_corners_interactive() supports calibrating from any 4+ visible
    landmarks (see COURT_LANDMARKS) and is the correct path for real
    accuracy -- it just needs a click-based calibration step in the web UI,
    which has not been built yet. Until then, court-position and possession
    output from the web app should be treated as approximate.
    """
    image_points = [(0, height), (width, height), (0, 0), (width, 0)]
    landmark_names = ["baseline_left", "baseline_right", "halfcourt_left", "halfcourt_right"]
    return courtiq_core.compute_homography(image_points, landmark_names)


def decisions_to_actions(data, interval=5.0):
    """Turn Stage 2 decisions into the timeline-window shape the existing
    UI already renders (id/start/end/label/title/analysis/note/thumbnail).
    Every window's label/analysis is derived from real decisions in
    data.decisions -- nothing here is invented per-window.
    """
    duration = data.duration
    windows = []
    start = 0.0
    while start < duration:
        windows.append((start, min(duration, start + interval)))
        start += interval
    windows = windows[:240]

    actions = []
    for index, (win_start, win_end) in enumerate(windows, 1):
        in_window = [d for d in data.decisions if win_start <= d["t"] < win_end]
        if not in_window:
            actions.append({
                "id": index, "start": win_start, "end": win_end,
                "label": "uncertain", "title": "No graded decision in this window",
                "analysis": "No possession-ending or spacing event was flagged by the decision rules in this window.",
                "note": "", "thumbnail": None,
            })
            continue
        bad = [d for d in in_window if d["verdict"] == "bad"]
        chosen = bad[0] if bad else in_window[0]
        actions.append({
            "id": index, "start": win_start, "end": win_end,
            "label": chosen["verdict"],
            "title": DECISION_TITLES.get(chosen["type"], chosen["type"]),
            "analysis": chosen["reason"],
            "note": "", "thumbnail": None,
        })
    return actions


def run_analysis_job(path: Path):
    """Runs in a background thread. Never call this from an HTTP handler
    directly -- tracking + detection across a full video can take minutes,
    and the POST handler must return immediately.
    """
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError("This video could not be opened. Try an MP4 encoded with H.264.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        if not width or not height:
            raise RuntimeError("The uploaded video has no readable dimensions.")

        def progress_cb(done, total):
            with JOB_LOCK:
                JOB["progress"] = done
                JOB["total"] = total

        homography = default_homography(width, height)
        # Prefer a downloaded basketball-trained ball model (see
        # setup_ball_model.py) over the COCO fallback baked into
        # courtiq_core's default, if one has been set up.
        basketball_ball_model = ROOT / "models" / "basketball_ball.pt"
        ball_model_path = str(basketball_ball_model) if basketball_ball_model.exists() else courtiq_core.BALL_MODEL_PATH
        data = courtiq_core.analyze_video(str(path), homography, ball_model_path=ball_model_path, progress_cb=progress_cb)
        actions = decisions_to_actions(data)
        tracks_path = path.with_suffix(".tracks.json")
        tracks_path.write_text(json.dumps(courtiq_core.to_json_dict(data), indent=2))

        with LOCK:
            STATE["video"].update({"duration": data.duration, "width": width, "height": height})
            STATE["actions"] = actions
            STATE["tracks_summary"] = {
                "ball_detection_rate": data.ball_detection_rate,
                "track_count": len(data.tracks),
                "possession_count": len(data.possessions),
                "decision_count": len(data.decisions),
                "iq_scores": {str(k): v for k, v in data.iq_scores.items()},
            }
            save_state(STATE)
        with JOB_LOCK:
            JOB["status"] = "done"
    except Exception as exc:  # noqa: BLE001 - reported to the polling client, not swallowed
        print(f"[analysis job] failed: {exc}")
        with JOB_LOCK:
            JOB["status"] = "error"
            JOB["error"] = str(exc)


def start_analysis_job(path: Path):
    with JOB_LOCK:
        if JOB["status"] == "running":
            raise ValueError("An analysis is already running.")
        JOB.update({"status": "running", "progress": 0, "total": 0, "error": None, "video_path": str(path)})
    thread = threading.Thread(target=run_analysis_job, args=(path,), daemon=True)
    thread.start()


class CourtIQHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}")

    def send_json(self, payload, status=HTTPStatus.OK):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size))

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/state":
            with LOCK:
                self.send_json(STATE)
            return
        if route == "/api/report":
            with LOCK:
                actions = STATE.get("actions", [])
                good = sum(a["label"] == "good" for a in actions)
                bad = sum(a["label"] == "bad" for a in actions)
                reviewed = good + bad
                score = round(100 * good / reviewed) if reviewed else None
                self.send_json({
                    "reviewed": reviewed, "good": good, "bad": bad, "score": score,
                    "actions": actions, "tracks_summary": STATE.get("tracks_summary"),
                })
            return
        if route == "/api/analyze/status":
            with JOB_LOCK:
                self.send_json(dict(JOB))
            return
        if route == "/" or route == "/index.html":
            self.path = "/index.html"
            return super().do_GET()
        if route.startswith("/data/"):
            self.path = route
            return super().do_GET()
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            if route == "/api/upload":
                self.upload_video()
            elif route == "/api/action":
                self.update_action()
            elif route == "/api/analyze":
                self.reanalyze()
            elif route == "/api/reset":
                self.reset()
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except (ValueError, RuntimeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            self.send_json({"error": "Something went wrong while processing the video."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def upload_video(self):
        size = int(self.headers.get("Content-Length", "0"))
        filename = clean_filename(self.headers.get("X-File-Name", "game.mp4"))
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Upload an MP4, MOV, M4V, AVI, or WebM video.")
        if not 0 < size <= MAX_UPLOAD_BYTES:
            raise ValueError("The video must be between 1 byte and 2 GB.")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        stored = f"{uuid.uuid4().hex}{suffix}"
        destination = UPLOADS / stored
        # ``rfile`` is a persistent HTTP connection. copyfileobj() reads until
        # EOF, which never arrives here while the browser is waiting for our
        # response. Read exactly Content-Length bytes instead.
        remaining = size
        with destination.open("wb") as output:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("The upload ended before the full video arrived.")
                output.write(chunk)
                remaining -= len(chunk)
        with LOCK:
            STATE["video"] = {"name": filename, "url": f"/data/uploads/{stored}", "duration": 0, "width": 0, "height": 0}
            STATE["actions"] = []
            STATE["tracks_summary"] = None
            save_state(STATE)
        # Tracking + detection across a full video can take minutes; never
        # run it synchronously inside this POST handler. The frontend polls
        # /api/analyze/status for progress.
        start_analysis_job(destination)
        self.send_json(STATE, HTTPStatus.CREATED)

    def update_action(self):
        data = self.read_json()
        action_id = int(data["id"])
        label = data.get("label", "unreviewed")
        if label not in {"unreviewed", "good", "bad", "uncertain"}:
            raise ValueError("Invalid action label.")
        with LOCK:
            action = next((item for item in STATE["actions"] if item["id"] == action_id), None)
            if action is None:
                raise ValueError("Action not found.")
            action["label"] = label
            action["note"] = str(data.get("note", ""))[:500]
            save_state(STATE)
        self.send_json(action)

    def reanalyze(self):
        """Kick off a fresh background analysis of the already-uploaded video."""
        with LOCK:
            video = STATE.get("video")
        if not video:
            raise ValueError("Upload a video before running AI analysis.")
        stored_name = Path(urlparse(video["url"]).path).name
        path = UPLOADS / stored_name
        if not path.exists():
            raise ValueError("The uploaded video is no longer available. Upload it again.")
        start_analysis_job(path)
        self.send_json(STATE)

    def reset(self):
        with LOCK:
            STATE.clear()
            STATE.update(default_state())
            save_state(STATE)
        self.send_json(STATE)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", 8787), CourtIQHandler)
    print("CourtIQ is running at http://127.0.0.1:8787")
    print("Press Ctrl+C to stop it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCourtIQ stopped.")
    finally:
        server.server_close()
