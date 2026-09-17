import type {
  BallDetection,
  TrackerEngineState,
} from "../../types/tracking";
import type { VisionTrackState } from "./ballTracker";

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function shiftDetection(
  detection: BallDetection | null,
  displacementX: number,
  displacementY: number,
): BallDetection | null {
  if (!detection) return null;
  return {
    ...detection,
    x: clamp(detection.x + displacementX),
    y: clamp(detection.y + displacementY),
  };
}

export function alignVisionTrackToRimShift(
  track: VisionTrackState,
  displacementX: number,
  displacementY: number,
): VisionTrackState {
  if (displacementX === 0 && displacementY === 0) return track;
  return {
    ...track,
    previous: shiftDetection(track.previous, displacementX, displacementY),
    current: shiftDetection(track.current, displacementX, displacementY),
    ...(track.observedHistory ? {observedHistory:track.observedHistory.map(sample =>
      shiftDetection(sample, displacementX, displacementY) ?? sample)} : {}),
  };
}

export function alignTrackerEngineToRimShift(
  state: TrackerEngineState,
  displacementX: number,
  displacementY: number,
): TrackerEngineState {
  if (displacementX === 0 && displacementY === 0) return state;
  return {
    ...state,
    previous: shiftDetection(state.previous, displacementX, displacementY),
    trajectory: state.trajectory.map((point) =>
      shiftDetection(point, displacementX, displacementY) ?? point
    ),
    apexY: state.trajectory.length > 0 ? clamp(state.apexY + displacementY) : state.apexY,
    entryX: state.enteredRim ? clamp(state.entryX + displacementX) : state.entryX,
    rimProximityX: state.rimProximityFrames > 0
      ? clamp(state.rimProximityX + displacementX)
      : state.rimProximityX,
    crossedRimX: state.crossedRimAt > 0
      ? clamp(state.crossedRimX + displacementX)
      : state.crossedRimX,
  };
}
