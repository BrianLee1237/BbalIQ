import type {
  BallDetection,
  RimCalibration,
  VideoShotDecision,
} from "../../types/tracking";
import { MIN_AUTOMATIC_DECISION_CONFIDENCE } from "./engine";

export interface BasketSemanticTrackerState {
  previous: BallDetection | null;
  approachFrames: number;
  lastApproachAt: number;
  occupancyFrames: number;
  firstOccupancyAt: number;
  lastOccupancyAt: number;
  occupancyConfidence: number;
  lastShotAt: number;
}

export interface BasketSemanticTrackerStep {
  state: BasketSemanticTrackerState;
  decision: VideoShotDecision | null;
}

const APPROACH_WINDOW_MS = 1_050;
const OCCUPANCY_EXIT_WINDOW_MS = 620;
// Attempt state is cleared immediately after a verified basket, so this timer
// only protects against fragmented duplicate exits. Keep it short enough for
// rapid-fire drills with independently observed approaches.
const SEMANTIC_COOLDOWN_MS = 600;
const SPARSE_CROSSING_WINDOW_MS = 320;

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function clearAttempt(
  state: BasketSemanticTrackerState,
  previous: BallDetection | null,
): BasketSemanticTrackerState {
  return {
    ...state,
    previous,
    approachFrames: 0,
    lastApproachAt: 0,
    occupancyFrames: 0,
    firstOccupancyAt: 0,
    lastOccupancyAt: 0,
    occupancyConfidence: 0,
  };
}

export function createBasketSemanticTrackerState(): BasketSemanticTrackerState {
  return {
    previous: null,
    approachFrames: 0,
    lastApproachAt: 0,
    occupancyFrames: 0,
    firstOccupancyAt: 0,
    lastOccupancyAt: 0,
    occupancyConfidence: 0,
    lastShotAt: Number.NEGATIVE_INFINITY,
  };
}

/**
 * SwishAI-inspired semantic basket evidence without depending on its missing
 * five-class weights. A tracked basketball occupying the locked rim opening
 * becomes an explicit "ball in basket" signal, but it can only score after an
 * above-rim approach and a centered, descending exit below the net.
 */
export function stepBasketSemanticTracker(
  current: BasketSemanticTrackerState,
  ball: BallDetection | null,
  rim: RimCalibration,
  atMs: number,
): BasketSemanticTrackerStep {
  let state = { ...current };
  if (!ball) {
    if (
      state.lastOccupancyAt > 0 &&
      atMs - state.lastOccupancyAt > OCCUPANCY_EXIT_WINDOW_MS
    ) {
      state = clearAttempt(state, null);
    }
    return { state, decision: null };
  }

  const previous = state.previous;
  const rimLeft = rim.x;
  const rimRight = rim.x + rim.width;
  const rimCenterX = rim.x + rim.width / 2;
  const rimPlaneY = rim.y + rim.height * 0.48;
  const exitPlaneY = rim.y + rim.height + Math.max(rim.height * 0.45, ball.height * 0.42);
  const movingDown = previous !== null && ball.at > previous.at && ball.y - previous.y > 0.0015;
  const visualConfidence = ball.appearanceConfidence ?? ball.confidence;
  const localHorizontal =
    ball.x > rimLeft - rim.width * 1.35 &&
    ball.x < rimRight + rim.width * 1.35;
  const aboveRim =
    ball.y < rimPlaneY - Math.max(rim.height * 0.28, ball.height * 0.12);

  if (
    localHorizontal &&
    aboveRim &&
    visualConfidence >= 0.28
  ) {
    state.approachFrames = Math.min(8, state.approachFrames + 1);
    state.lastApproachAt = atMs;
  }

  if (
    state.lastApproachAt > 0 &&
    atMs - state.lastApproachAt > APPROACH_WINDOW_MS &&
    state.occupancyFrames === 0
  ) {
    state = clearAttempt(state, ball);
  }

  const ballLeft = ball.x - ball.width / 2;
  const ballRight = ball.x + ball.width / 2;
  const horizontalOverlap = Math.max(
    0,
    Math.min(ballRight, rimRight) - Math.max(ballLeft, rimLeft),
  );
  const overlapRatio = horizontalOverlap / Math.max(0.001, ball.width);
  const occupancyTop = rimPlaneY - Math.max(rim.height * 0.7, ball.height * 0.58);
  const occupancyBottom = rim.y + rim.height + Math.max(rim.height * 0.35, ball.height * 0.28);
  const recentApproach =
    state.approachFrames >= 2 &&
    atMs - state.lastApproachAt <= APPROACH_WINDOW_MS;
  const occupiesBasket =
    recentApproach &&
    movingDown &&
    visualConfidence >= 0.22 &&
    overlapRatio >= 0.42 &&
    ball.x > rimLeft - rim.width * 0.08 &&
    ball.x < rimRight + rim.width * 0.08 &&
    ball.y >= occupancyTop &&
    ball.y <= occupancyBottom;

  if (occupiesBasket) {
    const centered = 1 - clamp(
      Math.abs(ball.x - rimCenterX) / Math.max(0.001, rim.width / 2),
    );
    const occupancyConfidence = clamp(
      0.72 + visualConfidence * 0.16 + overlapRatio * 0.06 + centered * 0.06,
    );
    state.occupancyFrames = Math.min(6, state.occupancyFrames + 1);
    state.firstOccupancyAt ||= atMs;
    state.lastOccupancyAt = atMs;
    state.occupancyConfidence = Math.max(
      state.occupancyConfidence,
      occupancyConfidence,
    );
  }

  const occupancyConfirmed =
    state.occupancyFrames >= 2 ||
    (state.occupancyFrames >= 1 && state.occupancyConfidence >= 0.92);
  const centeredExit =
    ball.x > rimLeft + rim.width * 0.01 &&
    ball.x < rimRight - rim.width * 0.01;
  const exitsBelowNet =
    occupancyConfirmed &&
    movingDown &&
    centeredExit &&
    ball.y >= exitPlaneY &&
    atMs - state.lastOccupancyAt <= OCCUPANCY_EXIT_WINDOW_MS;

  // A detector can briefly lose the ball while it is hidden by the rim or net.
  // Recover only a short, downward, high-confidence segment that starts above
  // the rim and finishes below the net, then interpolate its rim-plane crossing.
  // This preserves the reference trackers' line-crossing behavior without
  // treating a nearby rebound or an adjacent airball as a basket.
  const sparseGapMs = previous ? atMs - previous.at : Number.POSITIVE_INFINITY;
  const sparseVerticalTravel = previous ? ball.y - previous.y : 0;
  const sparseCrossingX = previous && sparseVerticalTravel > 0.0001
    ? previous.x +
      ((rimPlaneY - previous.y) / sparseVerticalTravel) * (ball.x - previous.x)
    : Number.NaN;
  const previousVisualConfidence = previous
    ? previous.appearanceConfidence ?? previous.confidence
    : 0;
  const sparseCenteredCrossing =
    Number.isFinite(sparseCrossingX) &&
    sparseCrossingX > rimLeft + rim.width * 0.01 &&
    sparseCrossingX < rimRight - rim.width * 0.01;
  const sparseOcclusionCrossing =
    state.approachFrames >= 2 &&
    previous !== null &&
    movingDown &&
    sparseGapMs >= 55 &&
    sparseGapMs <= SPARSE_CROSSING_WINDOW_MS &&
    previous.y < rimPlaneY &&
    ball.y >= exitPlaneY &&
    previousVisualConfidence >= 0.62 &&
    previous.confidence >= 0.72 &&
    ball.confidence >= 0.55 &&
    sparseCenteredCrossing;

  if (
    (exitsBelowNet || sparseOcclusionCrossing) &&
    atMs - state.lastShotAt >= SEMANTIC_COOLDOWN_MS
  ) {
    const centered = 1 - clamp(
      Math.abs(
        (sparseOcclusionCrossing ? sparseCrossingX : ball.x) - rimCenterX,
      ) / Math.max(0.001, rim.width / 2),
    );
    const continuity = clamp(state.occupancyFrames / 3);
    const sparseContinuity = clamp(
      1 - Math.max(0, sparseGapMs - 55) / (SPARSE_CROSSING_WINDOW_MS - 55),
    );
    const confidence = sparseOcclusionCrossing
      ? clamp(
        0.78 +
        centered * 0.06 +
        Math.min(previous.confidence, ball.confidence) * 0.06 +
        previousVisualConfidence * 0.04 +
        visualConfidence * 0.02 +
        sparseContinuity * 0.04,
      )
      : clamp(
        state.occupancyConfidence * 0.72 +
        visualConfidence * 0.12 +
        centered * 0.1 +
        continuity * 0.06,
      );
    const decision: VideoShotDecision = {
      id: `${atMs}-semantic-basket`,
      atSeconds: atMs / 1_000,
      suggestedKind: "make",
      finalKind: confidence >= MIN_AUTOMATIC_DECISION_CONFIDENCE ? "make" : null,
      confidence,
      reason: "semantic-basket",
    };
    state = {
      ...clearAttempt(state, ball),
      lastShotAt: atMs,
    };
    return { state, decision };
  }

  state.previous = ball;
  return { state, decision: null };
}
