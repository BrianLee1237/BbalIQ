import type { BallDetection, RimCalibration } from "../../types/tracking";
import { matchesFlightPrediction, predictBallFlight } from "../tracking/flightPrediction";

export interface VisionTrackState {
  previous: BallDetection | null;
  current: BallDetection | null;
  velocityX?: number;
  velocityY?: number;
  confirmedFrames?: number;
  missingFrames?: number;
  /** Actual selected detections only. Never append a predicted/interpolated point. */
  observedHistory?: BallDetection[];
}

export interface VisionSelection {
  detection: BallDetection | null;
  state: VisionTrackState;
}

export function createVisionTrackState(): VisionTrackState {
  "worklet";
  return {
    previous: null,
    current: null,
    velocityX: 0,
    velocityY: 0,
    confirmedFrames: 0,
    missingFrames: 0,
  };
}

// Keep worklet dependencies declared before the first worklet that captures
// them. The web worklet transform can lower later function declarations to a
// lexical binding, which otherwise causes a module-load temporal dead zone.
function clamp(value: number, minimum: number, maximum: number): number {
  "worklet";
  return Math.max(minimum, Math.min(maximum, value));
}

/**
 * Accepts pixel/OpenCV recovery only when a learned track already exists.
 * This is the observation-centric recovery path: it can bridge a missed YOLO
 * frame, but global motion or orange clothing can never initiate a shot.
 */
export function selectTrackGuidedRecoveryCandidates(
  candidates: BallDetection[],
  track: VisionTrackState,
  rim: RimCalibration,
  at: number,
): BallDetection[] {
  "worklet";
  const current = track.current;
  if (!current || (track.confirmedFrames ?? 0) < 2) return [];
  const elapsedMs = at - current.at;
  if (elapsedMs < 0 || elapsedMs > 500) return [];

  const secondsAhead = elapsedMs / 1_000;
  const velocityX = track.velocityX ?? 0;
  const velocityY = track.velocityY ?? 0;
  const predictedX = current.x + velocityX * secondsAhead;
  const predictedY = current.y + velocityY * secondsAhead;
  const speed = Math.hypot(velocityX, velocityY);
  const flight = predictBallFlight(track.observedHistory ?? [], at);
  const allowedDistance = clamp(
    Math.max(current.width, current.height) * 3.2 + speed * secondsAhead * 0.55,
    Math.max(0.045, rim.width * 0.46),
    0.24,
  );

  return candidates
    .filter((candidate) => {
      const distance = Math.hypot(candidate.x - predictedX, candidate.y - predictedY);
      const curvedRecovery = flight !== null && matchesFlightPrediction(candidate, flight) &&
        !(current.trackId !== undefined && candidate.trackId !== undefined && current.trackId !== candidate.trackId) &&
        (candidate.appearanceConfidence ?? 0) >= 0.35 &&
        (candidate.motionConfidence ?? 0) >= 0.7;
      if (distance > allowedDistance && !curvedRecovery) return false;
      const sizeRatio = Math.max(candidate.width, candidate.height) /
        Math.max(0.001, Math.max(current.width, current.height));
      if (sizeRatio < 0.42 || sizeRatio > 2.35) return false;
      if ((candidate.motionConfidence ?? 0) < 0.56) return false;
      if (
        (candidate.appearanceConfidence ?? 0) < 0.12 &&
        candidate.confidence < 0.72
      ) {
        return false;
      }

      const observedX = candidate.x - current.x;
      const observedY = candidate.y - current.y;
      const observedLength = Math.hypot(observedX, observedY);
      if (speed > 0.08 && observedLength > 0.008) {
        const directionCosine = (observedX * velocityX + observedY * velocityY) /
          Math.max(0.0001, observedLength * speed);
        const rimCenterX = rim.x + rim.width / 2;
        const rimPlaneY = rim.y + rim.height * 0.48;
        const inRimContactZone =
          Math.abs(candidate.x - rimCenterX) <= rim.width * 1.45 &&
          Math.abs(candidate.y - rimPlaneY) <= rim.width * 1.35;
        // A genuine rim bounce reverses direction abruptly. Permit that
        // reversal only inside the small rim-contact zone; away from the rim,
        // the same motion remains a distractor or camera-motion rejection.
        if (directionCosine < -0.18 && !inRimContactZone && !curvedRecovery) return false;
      }
      return true;
    })
    .sort((left, right) => {
      const leftDistance = Math.hypot(left.x - predictedX, left.y - predictedY);
      const rightDistance = Math.hypot(right.x - predictedX, right.y - predictedY);
      return leftDistance - rightDistance || right.confidence - left.confidence;
    })
    .slice(0, 2);
}

/**
 * Selects one candidate using predicted motion, size consistency and the rim
 * calibration. A low-quality discontinuous candidate is rejected instead of
 * being forced into the shot state machine.
 */
export function selectTrackedBall(
  candidates: BallDetection[],
  track: VisionTrackState,
  rim: RimCalibration,
  at: number,
): VisionSelection {
  "worklet";
  const current = track.current;
  const previous = track.previous;
  const elapsedSinceCurrent = current ? at - current.at : Number.POSITIVE_INFINITY;
  const hasFreshTrack = current !== null && elapsedSinceCurrent >= 0 && elapsedSinceCurrent <= 600;

  let predictedX = current?.x ?? 0;
  let predictedY = current?.y ?? 0;
  let velocityX = track.velocityX ?? 0;
  let velocityY = track.velocityY ?? 0;
  if (hasFreshTrack && current) {
    if (
      Math.abs(velocityX) < 0.0001 &&
      Math.abs(velocityY) < 0.0001 &&
      previous &&
      current.at > previous.at
    ) {
      const seconds = (current.at - previous.at) / 1_000;
      velocityX = (current.x - previous.x) / seconds;
      velocityY = (current.y - previous.y) / seconds;
    }
    const secondsAhead = elapsedSinceCurrent / 1_000;
    predictedX = current.x + velocityX * secondsAhead;
    predictedY = current.y + velocityY * secondsAhead;
  }

  let selected: BallDetection | null = null;
  let selectedScore = -1;

  for (const candidate of candidates) {
    let score = candidate.confidence;
    if (hasFreshTrack && current) {
      const distance = Math.hypot(candidate.x - predictedX, candidate.y - predictedY);
      const sizeDelta = Math.abs(candidate.width - current.width) +
        Math.abs(candidate.height - current.height);
      const speed = Math.hypot(velocityX, velocityY);
      const allowedJump = clamp(0.12 + speed * (elapsedSinceCurrent / 1_000) * 0.72, 0.16, 0.43);
      const continuity = 1 - clamp(distance / allowedJump, 0, 1);
      const sizeConsistency = 1 - clamp(sizeDelta / 0.18, 0, 1);
      const observedX = candidate.x - current.x;
      const observedY = candidate.y - current.y;
      const observedLength = Math.hypot(observedX, observedY);
      const expectedLength = Math.hypot(velocityX, velocityY);
      const directionConsistency = observedLength > 0.002 && expectedLength > 0.02
        ? clamp(
            (observedX * velocityX + observedY * velocityY) /
              (observedLength * expectedLength) * 0.5 + 0.5,
            0,
            1,
          )
        : 0.7;

      if (distance > allowedJump) {
        const prolongedGap = (track.missingFrames ?? 0) >= 2 || elapsedSinceCurrent >= 350;
        const plausibleReacquisition = distance <= allowedJump * 2.1;
        if (
          candidate.confidence < 0.93 ||
          !prolongedGap ||
          !plausibleReacquisition
        ) {
          continue;
        }
      }
      score =
        candidate.confidence * 0.34 +
        continuity * 0.32 +
        sizeConsistency * 0.12 +
        directionConsistency * 0.1 +
        (candidate.motionConfidence ?? 0.45) * 0.12;
    } else {
      const rimCenterX = rim.x + rim.width / 2;
      const rimPlaneY = rim.y + rim.height * 0.48;
      const distanceInRimWidths = Math.hypot(
        (candidate.x - rimCenterX) / Math.max(0.001, rim.width),
        (candidate.y - rimPlaneY) / Math.max(0.001, rim.width),
      );
      if (distanceInRimWidths > 9.5) continue;
      const motionConfidence = candidate.motionConfidence ?? 0;
      const highAppearanceNearRim =
        (candidate.appearanceConfidence ?? candidate.confidence) >= 0.82 &&
        distanceInRimWidths <= 2.4;
      if (motionConfidence < 0.36 && !highAppearanceNearRim) continue;
      const expectedWidth = rim.width * 0.5;
      const sizeScore = 1 - clamp(
        Math.abs(candidate.width - expectedWidth) / Math.max(0.001, expectedWidth * 1.45),
        0,
        1,
      );
      const proximity = 1 - clamp(distanceInRimWidths / 9.5, 0, 1);
      score =
        candidate.confidence * 0.5 +
        motionConfidence * 0.25 +
        sizeScore * 0.15 +
        proximity * 0.1;
    }

    const centeredOnStaticRim =
      candidate.x > rim.x &&
      candidate.x < rim.x + rim.width &&
      candidate.y > rim.y &&
      candidate.y < rim.y + rim.height &&
      candidate.width > rim.width * 0.46 &&
      (candidate.motionConfidence ?? 0) < 0.45;
    if (centeredOnStaticRim) score -= hasFreshTrack ? 0.3 : 0.48;

    if (score > selectedScore) {
      selectedScore = score;
      selected = candidate;
    }
  }

  if (!selected || selectedScore < (hasFreshTrack ? 0.52 : 0.44)) {
    // Additive fallback only: preserve every successful existing selection.
    // A predicted arc can nominate a real detection, never become one itself.
    const flight = hasFreshTrack ? predictBallFlight(track.observedHistory ?? [], at) : null;
    if (flight && current) {
      const recovery = candidates.filter(candidate =>
        !(current.trackId !== undefined && candidate.trackId !== undefined && current.trackId !== candidate.trackId) &&
        candidate.confidence >= 0.65 && (candidate.motionConfidence ?? 0) >= 0.7 &&
        (candidate.appearanceConfidence ?? 0) >= 0.35 &&
        candidate.width / current.width >= 0.5 && candidate.width / current.width <= 2 &&
        candidate.height / current.height >= 0.5 && candidate.height / current.height <= 2 &&
        matchesFlightPrediction(candidate, flight));
      // Multiple candidates in the corridor are ambiguous; don't choose by color.
      if (recovery.length === 1) selected = recovery[0]!;
      else selected = null;
    } else selected = null;
  }
  if (!selected) {
    const keepTrack = current !== null && elapsedSinceCurrent <= 600;
    return {
      detection: null,
      state: keepTrack
        ? { ...track, missingFrames: (track.missingFrames ?? 0) + 1 }
        : createVisionTrackState(),
    };
  }

  const seconds = current && selected.at > current.at
    ? (selected.at - current.at) / 1_000
    : 0;
  const measuredVelocityX = current && seconds > 0 ? (selected.x - current.x) / seconds : 0;
  const measuredVelocityY = current && seconds > 0 ? (selected.y - current.y) / seconds : 0;
  const blend = current ? 0.42 : 1;
  const nextVelocityX = clamp(velocityX * (1 - blend) + measuredVelocityX * blend, -4, 4);
  const nextVelocityY = clamp(velocityY * (1 - blend) + measuredVelocityY * blend, -4, 4);

  return {
    detection: selected,
    state: {
      previous: current,
      current: selected,
      velocityX: nextVelocityX,
      velocityY: nextVelocityY,
      confirmedFrames: (track.confirmedFrames ?? 0) + 1,
      missingFrames: 0,
      observedHistory: [
        ...(hasFreshTrack && elapsedSinceCurrent <= 160 &&
            !(current?.trackId !== undefined && selected.trackId !== undefined && current.trackId !== selected.trackId)
          ? track.observedHistory ?? (current ? [current] : []) : []),
        selected,
      ].filter(sample => selected.at - sample.at <= 450).slice(-10),
    },
  };
}
