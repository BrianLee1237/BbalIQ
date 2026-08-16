#!/usr/bin/env python3
"""Train a basketball-specific player detector from the Roboflow dataset
at universe.roboflow.com/workspace-5ujvu/basketball-players-fy4c2-vfsuv.

Honest expectation set before running this: our existing player detection
already uses YOLO's generic COCO "person" class, which is normally quite
accurate at finding people. The track fragmentation/ID-switch problems
this project fought earlier were mostly tracker-matching-parameter issues
(ByteTrack/BoT-SORT), not the person detector missing players. This
custom model is more likely to help with *appearance* (e.g. distinguishing
players from referees/bench occupants by basketball-specific training
data) than with the tracking issues already fixed. Worth trying, but
don't expect it alone to fix track counts.

Deliberately scaled down to something realistic without a dedicated
NVIDIA GPU: yolov8n.pt (nano detection model) instead of a larger variant,
and a modest epoch count. Increase MODEL/EPOCHS below if accuracy isn't
good enough and you're willing to spend more training time.

Usage:
    export ROBOFLOW_API_KEY="your key here"   # never hardcode it in this file
    python3 train_player_detector.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = "workspace-5ujvu"
PROJECT = "basketball-players-fy4c2-vfsuv"
VERSION = 1

MODEL = "yolov8n.pt"  # nano detection model
EPOCHS = 100
IMGSZ = 640
BATCH = 8

ROOT = Path(__file__).resolve().parent
OUT_MODEL = ROOT / "models" / "basketball_players.pt"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Set ROBOFLOW_API_KEY as an environment variable first:")
        print('  export ROBOFLOW_API_KEY="your key here"')
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("This needs the 'roboflow' package. Install it with: pip install roboflow")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("This needs 'ultralytics'. Install it with: pip install ultralytics")
        sys.exit(1)

    print(f"Downloading the player dataset from Roboflow ({WORKSPACE}/{PROJECT} v{VERSION})...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        print(f"Expected data.yaml at {data_yaml} but it's not there -- dataset download may have failed.")
        sys.exit(1)
    print(f"Dataset downloaded to {dataset.location}")

    print(f"Training {MODEL} for {EPOCHS} epochs (this will take a while -- keep the laptop plugged in)...")
    model = YOLO(MODEL)
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        task="detect",
        plots=True,
    )

    # Ultralytics saves the best checkpoint under runs/detect/train*/weights/best.pt
    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"Training finished but couldn't find {best_pt} -- check {run_dir} manually.")
        sys.exit(1)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.write_bytes(best_pt.read_bytes())
    print(f"Done. Copied trained weights to {OUT_MODEL}")
    print("The Roboflow API key is no longer needed from this point forward.")
    print()
    print("IMPORTANT: check the class names before wiring this into courtiq_core.py --")
    print("this dataset's classes may not be 'person' at index 0 like COCO's.")
    print(f"  python3 -c \"from ultralytics import YOLO; m = YOLO('{OUT_MODEL}'); print(m.names)\"")
    print("Then set COURTIQ_PLAYER_MODEL to this file's path and re-check PERSON_CLASS")
    print("in courtiq_core.py matches whatever class index that prints for the player class.")


if __name__ == "__main__":
    main()
