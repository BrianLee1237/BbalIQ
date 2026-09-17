import type {
  BallDetection,
  RimCalibration,
  ShotKind,
  TrackerEngineState,
  TrackerStep,
} from "../../types/tracking";

export const SHOT_COOLDOWN_MS = 1_800;
export const LOST_BALL_MISS_MS = 900;
export const MAX_SHOT_FLIGHT_MS = 3_400;
export const MIN_AUTOMATIC_DECISION_CONFIDENCE = 0.86;
const MAX_TRAJECTORY_POINTS = 30;
const RIM_PROXIMITY_WINDOW_MS = 520;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

export function createTrackerEngineState(): TrackerEngineState {
  return {
    armed: false,
    armedAt: 0,
    enteredRim: false,
    entryAt: 0,
    entryX: 0,
    entryConfidence: 0,
    rimProximityAt: 0,
    rimProximityX: 0,
    rimProximityConfidence: 0,
    rimProximityFrames: 0,
    crossedRimAt: 0,
    crossedRimX: 0,
    crossedRimConfidence: 0,
    lastDetectedAt: 0,
    lastShotAt: -SHOT_COOLDOWN_MS,
    previous: null,
    trajectory: [],
    ascendingFrames: 0,
    descendingFrames: 0,
    apexY: 1,
  };
}

function clearShotState(state: TrackerEngineState): TrackerEngineState {
  return {
    ...state,
    armed: false,
    armedAt: 0,
    enteredRim: false,
    entryAt: 0,
    entryX: 0,
    entryConfidence: 0,
    rimProximityAt: 0,
    rimProximityX: 0,
    rimProximityConfidence: 0,
    rimProximityFrames: 0,
    crossedRimAt: 0,
    crossedRimX: 0,
    crossedRimConfidence: 0,
    lastDetectedAt: 0,
    previous: null,
    trajectory: [],
    ascendingFrames: 0,
    descendingFrames: 0,
    apexY: 1,
  };
}

function finishShot(
  state: TrackerEngineState,
  shot: ShotKind,
  now: number,
  confidence: number,
  reason: TrackerStep["reason"],
): TrackerStep {
  if (now - state.lastShotAt < SHOT_COOLDOWN_MS) {
    return {
      state: clearShotState(state),
      shot: null,
      confidence: 0,
      reason: "cooldown",
    };
  }

  return {
    state: {
      ...clearShotState(state),
      lastShotAt: now,
    },
    shot,
    confidence: clamp(confidence, 0, 1),
    reason,
  };
}

function interpolateCrossingX(
  previous: BallDetection,
  detection: BallDetection,
  planeY: number,
): number {
  const ratio = clamp(
    (planeY - previous.y) / Math.max(0.0001, detection.y - previous.y),
    0,
    1,
  );
  return previous.x + (detection.x - previous.x) * ratio;
}

function fittedCrossingX(
  trajectory: BallDetection[],
  planeY: number,
  rimHeight: number,
): number | null {
  const local = trajectory
    .filter((point) => Math.abs(point.y - planeY) <= Math.max(0.025, rimHeight * 5.5))
    .slice(-8);
  if (local.length < 3) return null;
  const meanY = local.reduce((sum, point) => sum + point.y, 0) / local.length;
  const meanX = local.reduce((sum, point) => sum + point.x, 0) / local.length;
  let covariance = 0;
  let variance = 0;
  for (const point of local) {
    covariance += (point.y - meanY) * (point.x - meanX);
    variance += (point.y - meanY) ** 2;
  }
  if (variance < 0.000005) return null;
  const slope = covariance / variance;
  return meanX + slope * (planeY - meanY);
}

export function stepTracker(
  current: TrackerEngineState,
  detection: BallDetection | null,
  rim: RimCalibration,
  now: number,
): TrackerStep {
  if (!detection) {
    if (
      current.armed &&
      current.lastDetectedAt > 0 &&
      now - current.lastDetectedAt > LOST_BALL_MISS_MS
    ) {
      const visualConfidence = current.previous?.confidence ?? 0.5;
      if (current.enteredRim) {
        return finishShot(
          current,
          "make",
          now,
          Math.min(0.82, 0.66 + current.entryConfidence * 0.13),
          "rim-entry-lost",
        );
      }
      return finishShot(
        current,
        "miss",
        now,
        0.54 + visualConfidence * 0.18,
        "lost",
      );
    }
    return { state: current, shot: null, confidence: 0, reason: "none" };
  }

  const previous = current.previous;
  const trajectory = [...current.trajectory, detection].slice(-MAX_TRAJECTORY_POINTS);
  let next: TrackerEngineState = {
    ...current,
    lastDetectedAt: now,
    previous: detection,
    trajectory,
    apexY: Math.min(current.apexY, detection.y),
  };

  if (!previous || detection.at <= previous.at) {
    return { state: next, shot: null, confidence: 0, reason: "none" };
  }

  const deltaY = detection.y - previous.y;
  const movingUp = deltaY < -0.0025;
  const movingDown = deltaY > 0.0025;
  next = {
    ...next,
    ascendingFrames: movingUp
      ? Math.min(10, current.ascendingFrames + 1)
      : current.armed
        ? current.ascendingFrames
        : Math.max(0, current.ascendingFrames - 1),
    descendingFrames: movingDown
      ? Math.min(10, current.descendingFrames + 1)
      : movingUp
        ? 0
        : current.descendingFrames,
  };

  const rimLeft = rim.x;
  const rimRight = rim.x + rim.width;
  const rimPlaneY = rim.y + rim.height * 0.48;
  const averageBallRadiusX = (previous.width + detection.width) * 0.25;
  const averageBallRadiusY = (previous.height + detection.height) * 0.25;
  const centerInset = clamp(
    Math.max(rim.width * 0.08, averageBallRadiusX * 0.38),
    rim.width * 0.08,
    rim.width * 0.25,
  );
  const safeLeft = rimLeft + centerInset;
  const safeRight = rimRight - centerInset;
  const approachLeft = rimLeft - rim.width * 4.8;
  const approachRight = rimRight + rim.width * 4.8;
  const localLeft = rimLeft - rim.width * 1.55;
  const localRight = rimRight + rim.width * 1.55;
  const aboveEntry = previous.y < rimPlaneY - Math.max(rim.height * 0.12, averageBallRadiusY * 0.12);
  const inBroadApproach = detection.x > approachLeft && detection.x < approachRight;
  const inLocalApproach = detection.x > localLeft && detection.x < localRight;
  const recentMotionEvidence = trajectory
    .slice(-3)
    .reduce((sum, point) => sum + (point.motionConfidence ?? 0), 0) /
    Math.max(1, Math.min(3, trajectory.length));

  const risingShot =
    inBroadApproach &&
    // Arm only once the rising ball reaches the hoop approach. A dribble or
    // low rebound below the basket must not start a second shot trajectory.
    detection.y < rimPlaneY + Math.min(0.3, Math.max(0.24, rim.width * 4.5)) &&
    movingUp &&
    next.ascendingFrames >= 2;
  const localDescendingShot =
    inLocalApproach &&
    aboveEntry &&
    movingDown &&
    next.descendingFrames >= 2 &&
    trajectory.length >= 3 &&
    recentMotionEvidence >= 0.5;

  if (!next.armed && (risingShot || localDescendingShot)) {
    next = {
      ...next,
      armed: true,
      armedAt: now,
      apexY: Math.min(previous.y, detection.y),
    };
  }

  if (!next.armed) {
    return { state: next, shot: null, confidence: 0, reason: "none" };
  }

  const crossedEntryPlane =
    movingDown &&
    previous.y < rimPlaneY &&
    detection.y >= rimPlaneY;

  // A detector can miss the thin rim-plane sample or put the ball center a few
  // pixels outside the calibrated opening. Preserve a short, descending
  // near-rim observation so that a later, centered exit below the net can
  // still verify the make. This is deliberately narrower than the general
  // approach lane so adjacent airballs never receive this evidence.
  const proximityTolerance = Math.max(rim.width * 0.14, averageBallRadiusX * 0.55);
  const proximityLeft = rimLeft - proximityTolerance;
  const proximityRight = rimRight + proximityTolerance;
  const proximityBandTop = rimPlaneY - Math.max(rim.height * 0.95, averageBallRadiusY * 0.8);
  const proximityBandBottom = rimPlaneY + Math.max(rim.height * 0.95, averageBallRadiusY * 0.8);
  const inRimProximityBand =
    movingDown &&
    detection.x > proximityLeft &&
    detection.x < proximityRight &&
    detection.y > proximityBandTop &&
    detection.y < proximityBandBottom;

  if (!next.enteredRim && inRimProximityBand) {
    const centerDistance = Math.abs(detection.x - (rimLeft + rimRight) / 2) /
      Math.max(0.001, rim.width / 2 + proximityTolerance);
    const centeredConfidence = 1 - clamp(centerDistance, 0, 1);
    const visualConfidence = Math.min(previous.confidence, detection.confidence);
    const proximityConfidence = clamp(
      0.79 + visualConfidence * 0.11 + centeredConfidence * 0.08,
      0,
      0.96,
    );
    next = {
      ...next,
      rimProximityAt: now,
      rimProximityX: detection.x,
      rimProximityConfidence: Math.max(next.rimProximityConfidence, proximityConfidence),
      rimProximityFrames: Math.min(4, next.rimProximityFrames + 1),
    };
  }

  if (!next.enteredRim && crossedEntryPlane) {
    const pairCrossingX = interpolateCrossingX(previous, detection, rimPlaneY);
    const fittedX = fittedCrossingX(trajectory, rimPlaneY, rim.height);
    const crossingX = fittedX !== null && Math.abs(fittedX - pairCrossingX) <= rim.width * 0.75
      ? pairCrossingX * 0.58 + fittedX * 0.42
      : pairCrossingX;
    const throughOpening = crossingX > safeLeft && crossingX < safeRight;
    const nearestBoundary = Math.min(
      Math.abs(crossingX - safeLeft),
      Math.abs(crossingX - safeRight),
    );
    const ambiguousBoundary = nearestBoundary < rim.width * 0.045;
    const visualConfidence = Math.min(previous.confidence, detection.confidence);
    const trajectorySpan = Math.max(0, rimPlaneY - next.apexY);
    const spanConfidence = clamp(trajectorySpan / Math.max(0.001, rim.height * 1.3), 0, 1);
    const entryConfidence = clamp(
      0.78 + visualConfidence * 0.14 + spanConfidence * 0.08,
      0,
      ambiguousBoundary ? 0.82 : 0.98,
    );

    next = {
      ...next,
      crossedRimAt: now,
      crossedRimX: crossingX,
      crossedRimConfidence: entryConfidence,
    };

    if (!throughOpening) {
      next = {
        ...next,
        rimProximityAt: now,
        rimProximityX: crossingX,
        rimProximityConfidence: Math.max(
          next.rimProximityConfidence,
          Math.min(0.92, entryConfidence),
        ),
        rimProximityFrames: Math.max(1, next.rimProximityFrames),
      };
    }

    if (throughOpening) {
      next = {
        ...next,
        enteredRim: true,
        entryAt: now,
        entryX: crossingX,
        entryConfidence,
      };
    }
  }

  if (next.enteredRim) {
    const exitPlaneY = rim.y + rim.height + Math.max(rim.height * 0.5, averageBallRadiusY * 0.5);
    const crossedExitPlane =
      movingDown &&
      previous.y < exitPlaneY &&
      detection.y >= exitPlaneY;
    const exitX = crossedExitPlane
      ? interpolateCrossingX(previous, detection, exitPlaneY)
      : detection.x;
    const insideNetCorridor =
      exitX > rimLeft - rim.width * 0.16 &&
      exitX < rimRight + rim.width * 0.16;

    if (crossedExitPlane && insideNetCorridor) {
      const visualConfidence = Math.min(previous.confidence, detection.confidence);
      const centerDistance = Math.abs(exitX - (rimLeft + rimRight) / 2) / Math.max(0.001, rim.width / 2);
      const centeredConfidence = 1 - clamp(centerDistance, 0, 1);
      const confidence =
        next.entryConfidence * 0.7 + visualConfidence * 0.2 + centeredConfidence * 0.1;
      return finishShot(next, "make", now, confidence, "rim-entry-exit");
    }

    const escapedNet =
      now - next.entryAt > 120 &&
      (detection.x < rimLeft - rim.width * 0.72 || detection.x > rimRight + rim.width * 0.72);
    if (escapedNet) {
      return finishShot(
        next,
        "miss",
        now,
        Math.min(0.84, 0.7 + detection.confidence * 0.12),
        "airball",
      );
    }
  }

  if (
    !next.enteredRim &&
    next.rimProximityFrames > 0 &&
    now - next.rimProximityAt <= RIM_PROXIMITY_WINDOW_MS
  ) {
    const exitPlaneY = rim.y + rim.height + Math.max(rim.height * 0.5, averageBallRadiusY * 0.5);
    const crossedExitPlane =
      movingDown &&
      previous.y < exitPlaneY &&
      detection.y >= exitPlaneY;
    const exitX = crossedExitPlane
      ? interpolateCrossingX(previous, detection, exitPlaneY)
      : detection.x;
    const insideTightNetCorridor =
      exitX > rimLeft + rim.width * 0.04 &&
      exitX < rimRight - rim.width * 0.04;

    if (crossedExitPlane && insideTightNetCorridor) {
      const visualConfidence = Math.min(previous.confidence, detection.confidence);
      const centerDistance = Math.abs(exitX - (rimLeft + rimRight) / 2) /
        Math.max(0.001, rim.width / 2);
      const centeredConfidence = 1 - clamp(centerDistance, 0, 1);
      const continuityBonus = Math.min(0.04, next.rimProximityFrames * 0.025);
      const confidence =
        next.rimProximityConfidence * 0.7 +
        visualConfidence * 0.2 +
        centeredConfidence * 0.1 +
        continuityBonus;
      return finishShot(next, "make", now, confidence, "rim-proximity-exit");
    }
  }

  if (
    !next.enteredRim &&
    next.crossedRimAt > 0 &&
    now - next.crossedRimAt <= RIM_PROXIMITY_WINDOW_MS + 180
  ) {
    const exitPlaneY = rim.y + rim.height + Math.max(rim.height * 0.5, averageBallRadiusY * 0.5);
    const crossedExitPlane =
      movingDown &&
      previous.y < exitPlaneY &&
      detection.y >= exitPlaneY;
    if (crossedExitPlane) {
      const exitX = interpolateCrossingX(previous, detection, exitPlaneY);
      const centeredExit = exitX > rimLeft + rim.width * 0.02 &&
        exitX < rimRight - rim.width * 0.02;
      if (centeredExit) {
        const visualConfidence = Math.min(previous.confidence, detection.confidence);
        const confidence =
          Math.max(next.crossedRimConfidence, next.rimProximityConfidence) * 0.68 +
          visualConfidence * 0.2 +
          0.12;
        return finishShot(next, "make", now, confidence, "rim-proximity-exit");
      }

      const outsideDistance = next.crossedRimX < rimLeft
        ? rimLeft - next.crossedRimX
        : next.crossedRimX > rimRight
          ? next.crossedRimX - rimRight
          : Math.min(Math.abs(exitX - rimLeft), Math.abs(exitX - rimRight));
      const geometryConfidence = clamp(outsideDistance / Math.max(0.001, rim.width * 0.55), 0, 1);
      const visualConfidence = Math.min(previous.confidence, detection.confidence);
      return finishShot(
        next,
        "miss",
        now,
        0.82 + visualConfidence * 0.08 + geometryConfidence * 0.08,
        "airball",
      );
    }
  }

  const belowRim = detection.y >
    rim.y + rim.height + Math.max(rim.height * 0.5, averageBallRadiusY * 0.5);
  const outsideOpening = detection.x <= safeLeft || detection.x >= safeRight;
  const leftLocalLane = detection.x < localLeft || detection.x > localRight;
  const timedOut = now - next.armedAt > MAX_SHOT_FLIGHT_MS;
  const hasClearDescent = movingDown && next.descendingFrames >= 1;

  if (
    !next.enteredRim &&
    hasClearDescent &&
    now - next.armedAt > 260 &&
    ((belowRim && outsideOpening) || (belowRim && leftLocalLane))
  ) {
    const visualConfidence = Math.min(previous.confidence, detection.confidence);
    return finishShot(
      next,
      "miss",
      now,
      0.84 + visualConfidence * 0.13,
      "airball",
    );
  }

  if (timedOut) {
    return finishShot(
      next,
      next.enteredRim ? "make" : "miss",
      now,
      next.enteredRim ? Math.min(0.82, next.entryConfidence) : 0.68,
      next.enteredRim ? "rim-entry-lost" : "timeout",
    );
  }

  return { state: next, shot: null, confidence: 0, reason: "none" };
}
