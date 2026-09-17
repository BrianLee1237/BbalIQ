import type { RimCalibration } from "../../types/tracking";
import type { VisionTrackState } from "./ballTracker";
import type { RimDetectionRegion } from "./rimDetectionRegion";
import { predictBallFlight } from "../tracking/flightPrediction";

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

/**
 * Builds a detector crop around the Kalman-like predicted ball position. The
 * crop is intentionally larger in the direction of travel so a fast sports
 * object stays visible despite detector latency or a short occlusion.
 */
export function computeTrackDetectionRegion(
  track: VisionTrackState,
  rim: RimCalibration,
  frameWidth: number,
  frameHeight: number,
  atMs: number,
): RimDetectionRegion | null {
  const current = track.current;
  if (!current) return null;
  const elapsedMs = atMs - current.at;
  if (elapsedMs < 0 || elapsedMs > 600) return null;

  const safeWidth = Math.max(1, Math.round(frameWidth));
  const safeHeight = Math.max(1, Math.round(frameHeight));
  const secondsAhead = elapsedMs / 1_000;
  const velocityX = track.velocityX ?? 0;
  const velocityY = track.velocityY ?? 0;
  const predictedX = clamp(current.x + velocityX * secondsAhead, 0, 1) * safeWidth;
  const predictedY = clamp(current.y + velocityY * secondsAhead, 0, 1) * safeHeight;
  const rimPixels = Math.max(6, rim.width * safeWidth);
  const speedPixels = Math.hypot(velocityX * safeWidth, velocityY * safeHeight);
  const halfWidth = clamp(
    Math.max(safeWidth * 0.08, rimPixels * 1.7, current.width * safeWidth * 3.2) +
      Math.abs(velocityX) * safeWidth * 0.05,
    48,
    safeWidth * 0.23,
  );
  const halfHeight = clamp(
    Math.max(safeHeight * 0.11, rimPixels * 2, current.height * safeHeight * 3.4) +
      Math.min(safeHeight * 0.08, speedPixels * 0.035),
    48,
    safeHeight * 0.28,
  );
  const flight = predictBallFlight(track.observedHistory ?? [], atMs);
  // Preserve the entire existing linear search area. Only extend it to cover
  // a well-supported curved path near an apex; never crop existing evidence out.
  const left = Math.floor(clamp(Math.min(predictedX - halfWidth,
    flight ? (flight.x-flight.toleranceX)*safeWidth : predictedX-halfWidth), 0, safeWidth-1));
  const top = Math.floor(clamp(Math.min(predictedY - halfHeight,
    flight ? (flight.y-flight.toleranceY)*safeHeight : predictedY-halfHeight), 0, safeHeight-1));
  const right = Math.ceil(clamp(Math.max(predictedX + halfWidth,
    flight ? (flight.x+flight.toleranceX)*safeWidth : predictedX+halfWidth), left+1, safeWidth));
  const bottom = Math.ceil(clamp(Math.max(predictedY + halfHeight,
    flight ? (flight.y+flight.toleranceY)*safeHeight : predictedY+halfHeight), top+1, safeHeight));
  return {
    left,
    top,
    width: Math.max(1, right - left),
    height: Math.max(1, bottom - top),
  };
}

export function regionsSubstantiallyOverlap(
  left: RimDetectionRegion,
  right: RimDetectionRegion,
): boolean {
  const intersectionWidth = Math.max(
    0,
    Math.min(left.left + left.width, right.left + right.width) -
      Math.max(left.left, right.left),
  );
  const intersectionHeight = Math.max(
    0,
    Math.min(left.top + left.height, right.top + right.height) -
      Math.max(left.top, right.top),
  );
  const intersection = intersectionWidth * intersectionHeight;
  const smallerArea = Math.min(left.width * left.height, right.width * right.height);
  return intersection / Math.max(1, smallerArea) >= 0.72;
}
