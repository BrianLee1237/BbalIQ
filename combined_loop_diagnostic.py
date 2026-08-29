"""Reproduces run_pipeline()'s exact per-frame pattern (player_model.track()
every frame, then ball_model() every FRAME_STRIDE'th frame) with nothing
else, to isolate whether the two interleaved model calls are what's
tanking ball detection rate versus running the ball model alone.

Usage: python3 combined_loop_diagnostic.py <video> [num_frames]
"""
import sys
import cv2
import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1239

player_model = c._load_model("models/basketball_players.pt")
person_class = c.resolve_person_class(player_model)
ball_model = c._load_model("models/basketball_ball.pt")
ball_class = c.resolve_ball_class(ball_model)
tracker_config = c.scaled_tracker_config(c.TRACKER_CONFIG, 58.26)

capture = cv2.VideoCapture(video_path)
frame_idx = 0
box_found_count = 0
strided_count = 0

while frame_idx < num_frames:
    ok, frame = capture.read()
    if not ok:
        break

    # Same order as run_pipeline: player tracking every frame first.
    player_model.track(frame, classes=[person_class], conf=0.4, tracker=tracker_config,
                        persist=True, verbose=False)

    if frame_idx % 3 == 0:
        strided_count += 1
        ball_result = ball_model(frame, classes=[ball_class], conf=0.25, verbose=False)[0]
        found = ball_result.boxes is not None and len(ball_result.boxes) > 0
        if found:
            box_found_count += 1
        if frame_idx % 60 == 0:
            print(f"DEBUG frame={frame_idx} box_found={found}")

    frame_idx += 1

capture.release()
print(f"Processed {frame_idx} frames, {strided_count} strided ball-detection attempts.")
print(f"Box found: {box_found_count}/{strided_count} ({100.0*box_found_count/strided_count:.1f}%)")
