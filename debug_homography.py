import sys
import cv2
import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

H = c.keypoint_model_homography(video_path, "models/court_keypoints.pt", conf_threshold=conf)
print("Homography matrix:")
print(H)

capture = cv2.VideoCapture(video_path)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture.release()
print(f"\nFrame size: {width}x{height}")

# Project the four frame corners and a center point through the homography
# to sanity-check the output range. A good homography should project most
# of the visible frame into roughly [-10, 60] x [-10, 100] ft (court is
# 50x94ft with a 4ft on-court margin) -- wildly larger/negative/NaN values
# mean the homography itself is bad, not the ball/player detectors.
test_points = {
    "top-left": (0, 0), "top-right": (width, 0),
    "bottom-left": (0, height), "bottom-right": (width, height),
    "center": (width / 2, height / 2),
}
print("\nProjected court coordinates (feet):")
for name, pt in test_points.items():
    x_ft, y_ft = c.project_point(H, pt)
    print(f"  {name:>12}: ({x_ft:.1f}, {y_ft:.1f})  on_court={c.is_on_court(x_ft, y_ft)}")
