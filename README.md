# CourtIQ action review

Run locally:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 app.py
```

Open <http://127.0.0.1:8787>, upload a game video, and label each five-second action as **Good action** or **Bad action**. The app stores video uploads and labels in `data/`, which is created automatically and intentionally ignored by version control.

Requirements: Python 3.12 and OpenCV (`pip install opencv-python`).

For AI analysis, install the dependencies and download the local YOLO model once:

```bash
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

Uploaded game video and generated analysis stay local in `data/` and are never included in the repository.
