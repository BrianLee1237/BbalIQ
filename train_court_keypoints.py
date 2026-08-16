#!/usr/bin/env python3
"""Train a court-keypoint model from the Roboflow dataset referenced by
github.com/HanaFEKI/AI_BasketBall_Analysis_v1 (workspace fyp-3bwmg,
project reloc2-den7l), since that repo's own trained weights are not
publicly accessible (confirmed dead Google Drive link, unanswered GitHub
issue asking the same question since September).

Deliberately scaled down from that repo's own training config
(yolov8x-pose.pt for 500 epochs) to something realistic on a MacBook Air
with no dedicated NVIDIA GPU -- yolov8n-pose.pt (nano, far fewer
parameters) for far fewer epochs. This trades some accuracy for actually
finishing in a reasonable time on Apple Silicon (PyTorch's MPS backend,
not CUDA). Expect this to still take a while -- keep the laptop plugged
in. If results aren't accurate enough, EPOCHS/MODEL below are the first
things to increase, at the cost of training time.

Usage:
    export ROBOFLOW_API_KEY="your key here"   # never hardcode it in this file
    python3 train_court_keypoints.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL = "yolov8n-pose.pt"  # nano -- far smaller/faster than the repo's yolov8x-pose.pt
EPOCHS = 100                # scaled down from the repo's 500
IMGSZ = 640
BATCH = 8                   # smaller batch than their 16, more forgiving on limited memory

ROOT = Path(__file__).resolve().parent
OUT_MODEL = ROOT / "models" / "court_keypoints.pt"


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

    print("Downloading the court-keypoint dataset from Roboflow (fyp-3bwmg/reloc2-den7l v1)...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("fyp-3bwmg").project("reloc2-den7l")
    dataset = project.version(1).download("yolov8")
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
        task="pose",
        plots=True,
    )

    # Ultralytics saves the best checkpoint under runs/pose/train*/weights/best.pt
    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"Training finished but couldn't find {best_pt} -- check {run_dir} manually.")
        sys.exit(1)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.write_bytes(best_pt.read_bytes())
    print(f"Done. Copied trained weights to {OUT_MODEL}")
    print("The Roboflow API key is no longer needed from this point forward --")
    print(f"courtiq_core.py loads {OUT_MODEL} as a plain local file.")
    print()
    print("IMPORTANT: this model has NOT been validated against your real footage yet.")
    print("Run this next and check the results actually look right:")
    print(f"  python3 -c \"from ultralytics import YOLO; m = YOLO('{OUT_MODEL}'); print(m.task, m.names)\"")
    print(f"  python3 courtiq_core.py <your_video> --keypoint-model {OUT_MODEL}")


if __name__ == "__main__":
    main()
