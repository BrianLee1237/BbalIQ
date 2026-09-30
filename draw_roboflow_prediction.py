"""Draws a Roboflow keypoint-detection API prediction (paste the JSON dict
it returns) onto the source image, colored by confidence, so results can
be checked visually instead of trusting confidence numbers alone.

Usage: paste the prediction dict into predictions.py as PREDICTION = {...}
       python3 draw_roboflow_prediction.py <image> predictions.py
"""
import sys
import ast
import cv2

image_path = sys.argv[1]
pred_file = sys.argv[2]

with open(pred_file) as f:
    text = f.read()
data = ast.literal_eval(text[text.index("{"):])

frame = cv2.imread(image_path)
pred = data["predictions"][0]

x1 = int(pred["x"] - pred["width"] / 2)
y1 = int(pred["y"] - pred["height"] / 2)
x2 = int(pred["x"] + pred["width"] / 2)
y2 = int(pred["y"] + pred["height"] / 2)
cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
cv2.putText(frame, f"court conf={pred['confidence']:.2f}", (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

for kp in pred["keypoints"]:
    x, y, c = int(kp["x"]), int(kp["y"]), kp["confidence"]
    if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
        continue
    color = (0, 255, 0) if c >= 0.5 else (0, 255, 255) if c >= 0.15 else (0, 0, 255)
    cv2.circle(frame, (x, y), 7, color, -1)
    cv2.putText(frame, f"{kp['class']}:{c:.2f}", (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imwrite("roboflow_prediction.png", frame)
print("Wrote roboflow_prediction.png")
