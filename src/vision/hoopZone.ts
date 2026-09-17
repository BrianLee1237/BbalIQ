import type { BallDetection, RimCalibration } from "../../types/tracking";

/**
 * Keeps analysis centered on the only region that can produce a score. This
 * prevents people, court paint and a previously-made ball from holding the
 * single-ball tracker while the next shot approaches the rim.
 */
export function selectHoopZoneCandidates(
  candidates: BallDetection[],
  rim: RimCalibration,
  frameWidth: number,
  frameHeight: number,
): BallDetection[] {
  const rimWidthPixels = Math.max(4, rim.width * frameWidth);
  const rimCenterX = (rim.x + rim.width / 2) * frameWidth;
  const rimPlaneY = (rim.y + rim.height * 0.48) * frameHeight;
  return candidates.filter((candidate) => {
    const offsetX = candidate.x * frameWidth - rimCenterX;
    const offsetY = candidate.y * frameHeight - rimPlaneY;
    return (
      Math.abs(offsetX) <= rimWidthPixels * 2.8 &&
      offsetY >= -rimWidthPixels * 5.8 &&
      offsetY <= rimWidthPixels * 3.4
    );
  });
}
