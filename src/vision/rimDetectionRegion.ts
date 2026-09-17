import type { RimCalibration } from "../../types/tracking";

export interface RimDetectionRegion {
  left: number;
  top: number;
  width: number;
  height: number;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

/**
 * Builds the focused detector crop used when whole-frame inference cannot see
 * both the calibrated hoop and a basketball. The crop is deliberately wider
 * than the rim so it retains the descending arc, adjacent airballs, and the
 * first part of the below-net exit while magnifying small screen-recorded
 * footage for the 640px detector input.
 */
export function computeRimDetectionRegion(
  rim: RimCalibration,
  frameWidth: number,
  frameHeight: number,
): RimDetectionRegion {
  const safeWidth = Math.max(1, Math.round(frameWidth));
  const safeHeight = Math.max(1, Math.round(frameHeight));
  const centerX = clamp(rim.x + rim.width / 2, 0, 1) * safeWidth;
  const centerY = clamp(rim.y + rim.height / 2, 0, 1) * safeHeight;
  const rimWidthPixels = Math.max(1, rim.width * safeWidth);
  // A coarse automatic box can include the backboard support. Cap its impact
  // so the first focused pass still meaningfully magnifies the actual rim.
  const halfWidth = Math.max(
    safeWidth * 0.13,
    Math.min(rimWidthPixels * 4, safeWidth * 0.2),
  );
  const halfHeight = Math.max(
    safeHeight * 0.2,
    Math.min(rimWidthPixels * 4, safeHeight * 0.25),
  );
  const left = Math.floor(clamp(centerX - halfWidth, 0, safeWidth - 1));
  const top = Math.floor(clamp(centerY - halfHeight, 0, safeHeight - 1));
  const right = Math.ceil(clamp(centerX + halfWidth, left + 1, safeWidth));
  const bottom = Math.ceil(clamp(centerY + halfHeight, top + 1, safeHeight));
  return {
    left,
    top,
    width: Math.max(1, right - left),
    height: Math.max(1, bottom - top),
  };
}
