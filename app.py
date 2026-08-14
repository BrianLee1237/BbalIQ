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

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
THUMBS = DATA / "thumbnails"
STATE_FILE = DATA / "state.json"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}

LOCK = threading.Lock()
MODEL = None


def default_state():
    return {"video": None, "actions": []}


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


def detector():
    """Load the on-device detector only when the first video is uploaded."""
    global MODEL
    if YOLO is None:
        raise RuntimeError("AI analysis needs ultralytics. Install it with: pip install ultralytics")
    if MODEL is None:
        weights = ROOT / "yolo11n.pt"
        if not weights.exists():
            raise RuntimeError("The CourtIQ AI model is missing. Download yolo11n.pt into this folder, then restart.")
        MODEL = YOLO(str(weights))
    return MODEL


def analyze_frame(frame):
    """Return simple court signals from an on-device YOLO frame inference."""
    result = detector()(frame, classes=[0, 32], conf=0.18, verbose=False)[0]
    people, balls = [], []
    for box in result.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        (people if cls == 0 else balls).append(center)
    return people, balls


def ai_verdict(people, balls, frame_width, frame_height):
    """Generate an explainable play summary from visible spacing and the ball."""
    if len(people) < 2:
        return "Play not fully visible", "uncertain", "AI could not see enough of the court to grade this action."
    diagonal = max(1, (frame_width ** 2 + frame_height ** 2) ** 0.5)
    distances = [
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / diagonal
        for i, a in enumerate(people) for b in people[i + 1:]
    ]
    spacing = sum(distances) / len(distances) if distances else 0
    # COCO's sports-ball detector is imperfect on broadcast video, so we only
    # name a pass when the ball itself is visible rather than inventing one.
    if balls and spacing >= 0.17:
        return "Kickout pass for an open three", "good", "Positive: the ball and players show usable spacing for a perimeter kickout."
    if balls and spacing < 0.11:
        return "Attack into a crowded lane", "bad", "Negative: visible players are tightly clustered, reducing clean passing options."
    if balls:
        return "Half-court scoring decision", "good" if spacing >= 0.13 else "bad", ("Positive: spacing creates a workable scoring option." if spacing >= 0.13 else "Negative: limited spacing makes this a contested decision.")
    return "Off-ball spacing action", "good" if spacing >= 0.15 else "bad", ("Positive: players are spread, keeping driving and passing lanes available." if spacing >= 0.15 else "Negative: players are crowded, shrinking the available lanes.")


def build_actions(path: Path):
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("This video could not be opened. Try an MP4 encoded with H.264.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if duration <= 0:
        capture.release()
        raise RuntimeError("The uploaded video has no readable duration.")

    # Five-second windows are deliberately neutral: the app evaluates actions,
    # never teams or named players.
    interval = 5
    starts = list(range(0, max(1, int(duration)), interval))
    starts = starts[:240]  # prevents an accidentally huge game file from freezing the browser
    token = uuid.uuid4().hex
    actions = []
    THUMBS.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(starts, 1):
        middle = min(start + interval / 2, max(0, duration - 0.1))
        capture.set(cv2.CAP_PROP_POS_MSEC, middle * 1000)
        ok, frame = capture.read()
        thumbnail = None
        title, label, analysis = "Play not fully visible", "uncertain", "AI could not grade this action."
        if ok:
            people, balls = analyze_frame(frame)
            title, label, analysis = ai_verdict(people, balls, width, height)
            thumb_name = f"{token}-{index}.jpg"
            thumb_path = THUMBS / thumb_name
            scale = min(1.0, 480 / max(1, frame.shape[1]))
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            if cv2.imwrite(str(thumb_path), frame):
                thumbnail = f"/data/thumbnails/{thumb_name}"
        actions.append({
            "id": index,
            "start": start,
            "end": min(duration, start + interval),
            "label": label,
            "title": title,
            "analysis": analysis,
            "note": "",
            "thumbnail": thumbnail,
        })
    capture.release()
    return {"duration": duration, "width": width, "height": height, "actions": actions}


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
                self.send_json({"reviewed": reviewed, "good": good, "bad": bad, "score": score, "actions": actions})
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
        metadata = build_actions(destination)
        with LOCK:
            STATE["video"] = {"name": filename, "url": f"/data/uploads/{stored}", **{k: metadata[k] for k in ("duration", "width", "height")}}
            STATE["actions"] = metadata["actions"]
            save_state(STATE)
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
        """Apply the current AI to an already-uploaded video."""
        with LOCK:
            video = STATE.get("video")
        if not video:
            raise ValueError("Upload a video before running AI analysis.")
        stored_name = Path(urlparse(video["url"]).path).name
        path = UPLOADS / stored_name
        if not path.exists():
            raise ValueError("The uploaded video is no longer available. Upload it again.")
        metadata = build_actions(path)
        with LOCK:
            STATE["video"].update({k: metadata[k] for k in ("duration", "width", "height")})
            STATE["actions"] = metadata["actions"]
            save_state(STATE)
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
