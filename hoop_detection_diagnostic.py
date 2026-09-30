"""Dumps every hoop detection per sampled frame -- position and confidence --
instead of only the median that detect_hoop() returns.

detect_hoop() takes a median across frames, which is fine for scatter around
one true position but wrong if detections form two clusters (the real rim and
a false positive in the crowd): a median then lands on whichever cluster is
bigger, or in the empty space between them. This shows the actual
distribution so the combining rule can be based on it.

Usage: python3 hoop_detection_diagnostic.py <video> [conf]
Output: hoop_detections.png (every detection drawn on one frame)
"""
import sys
from collections import defaultdict

import cv2
import numpy as np

import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
conf_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

model = c._load_model(c.BALL_MODEL_PATH)
hoop_class = c.resolve_hoop_class(model)
print(f"model classes: {model.names}")
print(f"hoop class resolved to: {hoop_class}")
if hoop_class is None:
    raise SystemExit("No hoop class in this model.")

capture = cv2.VideoCapture(video_path)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
step = max(1, frame_count // 30)
print(f"\n{frame_count} frames, sampling every {step}\n")

first_frame = None
all_detections = []
for frame_idx in range(0, frame_count, step):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    if not ok:
        break
    if first_frame is None:
        first_frame = frame.copy()
    result = model(frame, classes=[hoop_class], conf=conf_threshold, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        print(f"  frame {frame_idx:5d}: no detection")
        continue
    entries = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        conf = float(box.conf[0])
        entries.append((cx, cy, conf))
        all_detections.append((cx, cy, conf))
    entries.sort(key=lambda e: -e[2])
    summary = ", ".join(f"({e[0]:.0f},{e[1]:.0f}) conf={e[2]:.2f}" for e in entries[:4])
    print(f"  frame {frame_idx:5d}: {len(entries)} detection(s): {summary}")
capture.release()

if not all_detections:
    raise SystemExit("\nNo hoop detections at all.")

arr = np.array(all_detections)
print(f"\ntotal detections: {len(all_detections)}")
print(f"median position:  ({np.median(arr[:,0]):.0f}, {np.median(arr[:,1]):.0f})  <-- what detect_hoop() returns")
best = max(all_detections, key=lambda d: d[2])
print(f"highest conf:     ({best[0]:.0f}, {best[1]:.0f}) conf={best[2]:.2f}")

# Cluster loosely by position to expose a split between the real rim and a
# false positive somewhere else in the frame.
clusters = defaultdict(list)
for cx, cy, conf in all_detections:
    clusters[(round(cx / 150), round(cy / 150))].append((cx, cy, conf))
print(f"\n{len(clusters)} position cluster(s):")
for members in sorted(clusters.values(), key=lambda m: -len(m)):
    marr = np.array(members)
    print(f"  ({marr[:,0].mean():6.0f},{marr[:,1].mean():6.0f})  n={len(members):3d}  "
          f"mean conf={marr[:,2].mean():.2f}  max conf={marr[:,2].max():.2f}")

for cx, cy, conf in all_detections:
    color = (0, 255, 0) if conf >= 0.5 else (0, 255, 255) if conf >= 0.25 else (0, 0, 255)
    cv2.circle(first_frame, (int(cx), int(cy)), 8, color, 2)
median_px = (int(np.median(arr[:, 0])), int(np.median(arr[:, 1])))
cv2.drawMarker(first_frame, median_px, (255, 0, 255), cv2.MARKER_CROSS, 40, 4)
cv2.putText(first_frame, "MEDIAN (what detect_hoop returns)", (median_px[0] + 25, median_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
cv2.imwrite("hoop_detections.png", first_frame)
print("\nWrote hoop_detections.png (green>=0.5, yellow>=0.25, red below; magenta cross = median)")
