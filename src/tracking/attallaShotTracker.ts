import type { ShotKind } from "../../types/tracking";
import { predictBallFlight } from "./flightPrediction";

/**
 * Detector observation consumed by the MIT-licensed Attalla shot pipeline.
 * Coordinates are pixels in one consistently sized decoded frame.
 */
export interface AttallaObjectDetection {
  kind: "ball" | "hoop";
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  /** ID supplied by the upstream multi-object tracker, when confirmed. */
  sourceTrackId?: number;
}

interface TimedObjectDetection extends AttallaObjectDetection {
  atMs: number;
}

interface TrackedHoop extends TimedObjectDetection {
  id: number;
}

interface TrackedBall {
  id: number;
  sourceTrackId?: number;
  detections: TimedObjectDetection[];
}

interface BallHoopPair {
  ballId: number;
  hoopId: number;
  pendingMissAt?: number;
  pendingMissCrossingX?: number;
  pendingMissConfidence?: number;
}

export interface AttallaShotTrackerState {
  hoops: TrackedHoop[];
  balls: TrackedBall[];
  armedPairs: BallHoopPair[];
  nextHoopId: number;
  nextBallId: number;
  lastShotAt: number;
  lastFrameAt?: number;
  lastShotSourceTrackId?: number;
}

export interface AttallaShotDecision {
  kind: ShotKind;
  atMs: number;
  confidence: number;
  crossingX: number;
  hoopId: number;
  ballId: number;
}

export interface AttallaShotTrackerStep {
  state: AttallaShotTrackerState;
  decisions: AttallaShotDecision[];
}

// The source drops a track after 20 input frames. The public reference clip is
// 28.89 FPS, so a timestamp window is the frame-rate-safe equivalent.
const TRACK_STALE_MS = 1_200;
// The upstream implementation retained 30 observations while evaluating
// roughly every second frame of a 29 FPS video (about two seconds of flight).
// Dense 30 FPS upload analysis needs a timestamp window instead of a hard
// 30-sample cap or the above-rim portion of a long arc is discarded before the
// below-net observation arrives.
const BALL_HISTORY_MS = 2_400;
const MAX_BALL_HISTORY_SAMPLES = 90;
// Track completion, not a long global timer, is the primary duplicate guard.
// Six hundred milliseconds still rejects detector fragmentation while allowing
// rapid passing drills to register the next independently tracked ball.
const SHOT_COOLDOWN_MS = 600;
const BALL_CONFIDENCE = 0.4;
// The reference OpenCV counter intentionally lowers its detector threshold in
// the rim zone, where net/rim occlusion makes the basketball response weaker.
const NEAR_HOOP_BALL_CONFIDENCE = 0.15;
const HOOP_CONFIDENCE = 0.3;
const FAST_ARM_BALL_CONFIDENCE = 0.75;
const FAST_ARM_MAX_SPAN_MS = 140;
const CLIP_START_ARM_WINDOW_MS = 700;
const CLIP_START_ARM_MAX_SPAN_MS = 260;
const CLIP_START_ARM_CONFIDENCE = 0.38;
// A near-edge descent can be the first contact of a rattling make. Hold only
// this narrow rim-contact band; clear adjacent airballs still score at once.
const RIM_BOUNCE_MISS_OFFSET = 1.35;
const RIM_BOUNCE_SETTLE_MS = 900;

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function distance(left: TimedObjectDetection, right: TimedObjectDetection): number {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

function diagonal(detection: TimedObjectDetection): number {
  return Math.hypot(detection.width, detection.height);
}

function hasPair(pairs: BallHoopPair[], ballId: number): boolean {
  return pairs.some((pair) => pair.ballId === ballId);
}

function cloneState(state: AttallaShotTrackerState): AttallaShotTrackerState {
  return {
    ...state,
    hoops: state.hoops.map((hoop) => ({ ...hoop })),
    balls: state.balls.map((ball) => ({
      ...ball,
      detections: ball.detections.map((detection) => ({ ...detection })),
    })),
    armedPairs: state.armedPairs.map((pair) => ({ ...pair })),
  };
}

function isInsideHoopArea(
  ball: TimedObjectDetection,
  hoops: TrackedHoop[],
): boolean {
  return hoops.some((hoop) => {
    // A calibrated rim is much thinner than a detector box around the complete
    // hoop assembly. Base the approach corridor on rim width as well as height
    // so the tracker can arm before the ball reaches the scoring plane.
    const approachHeight = Math.max(hoop.height * 3, hoop.width * 2.2);
    return ball.x > hoop.x - hoop.width * 2 &&
      ball.x < hoop.x + hoop.width * 2 &&
      ball.y > hoop.y + hoop.height / 2 - approachHeight &&
      ball.y < hoop.y + hoop.height / 2;
  });
}

function cleanTracks(state: AttallaShotTrackerState, atMs: number): void {
  state.balls = state.balls
    .filter((ball) => {
      const latest = ball.detections.at(-1);
      return latest !== undefined && atMs - latest.atMs <= TRACK_STALE_MS;
    })
    .map((ball) => ({
      ...ball,
      detections: ball.detections
        .filter((detection) => atMs - detection.atMs <= BALL_HISTORY_MS)
        .slice(-MAX_BALL_HISTORY_SAMPLES),
    }));
  state.hoops = state.hoops.filter((hoop) => atMs - hoop.atMs <= TRACK_STALE_MS);
  const ballIds = new Set(state.balls.map((ball) => ball.id));
  const hoopIds = new Set(state.hoops.map((hoop) => hoop.id));
  state.armedPairs = state.armedPairs.filter((pair) =>
    ballIds.has(pair.ballId) && hoopIds.has(pair.hoopId)
  );
}

function addHoop(state: AttallaShotTrackerState, detection: TimedObjectDetection): number | null {
  if (detection.confidence < HOOP_CONFIDENCE) return null;
  for (let index = 0; index < state.hoops.length; index += 1) {
    const existing = state.hoops[index];
    if (!existing) continue;
    if (distance(existing, detection) < diagonal(existing)) {
      state.hoops[index] = { ...detection, id: existing.id };
      return existing.id;
    }
  }
  const id = state.nextHoopId;
  state.nextHoopId += 1;
  state.hoops.push({ ...detection, id });
  return id;
}

function predictedDistance(
  ball: TrackedBall,
  detection: TimedObjectDetection,
): number {
  const latest = ball.detections.at(-1);
  if (!latest) return Number.POSITIVE_INFINITY;
  const previous = ball.detections.at(-2);
  if (!previous || latest.atMs <= previous.atMs) return distance(latest, detection);
  const historyMs = latest.atMs - previous.atMs;
  const predictionMs = Math.min(240, Math.max(0, detection.atMs - latest.atMs));
  const predictedX = latest.x + (latest.x - previous.x) * (predictionMs / historyMs);
  const predictedY = latest.y + (latest.y - previous.y) * (predictionMs / historyMs);
  return Math.hypot(predictedX - detection.x, predictedY - detection.y);
}

function detectionIoU(
  left: TimedObjectDetection,
  right: TimedObjectDetection,
): number {
  const leftEdge = Math.max(left.x - left.width / 2, right.x - right.width / 2);
  const rightEdge = Math.min(left.x + left.width / 2, right.x + right.width / 2);
  const topEdge = Math.max(left.y - left.height / 2, right.y - right.height / 2);
  const bottomEdge = Math.min(left.y + left.height / 2, right.y + right.height / 2);
  const intersection = Math.max(0, rightEdge - leftEdge) * Math.max(0, bottomEdge - topEdge);
  const union = left.width * left.height + right.width * right.height - intersection;
  return intersection / Math.max(1, union);
}

/**
 * Associates an entire frame at once. The old sequential loop could append
 * several simultaneous detections to one ball track, which is catastrophic in
 * ball-return drills where ten stationary basketballs sit under the rim.
 */
function addBalls(
  state: AttallaShotTrackerState,
  detections: TimedObjectDetection[],
): void {
  const accepted: TimedObjectDetection[] = [];
  for (const detection of [...detections].sort(
    (left, right) => right.confidence - left.confidence,
  )) {
    const nearHoop = isInsideHoopArea(detection, state.hoops);
    // A weak below-net observation may complete an established approach, but
    // must never acquire a new ball from rack/clothing/background responses.
    const recoveringExit = state.armedPairs.some((pair) => {
      const ball = state.balls.find((item) => item.id === pair.ballId);
      const hoop = state.hoops.find((item) => item.id === pair.hoopId);
      const latest = ball?.detections.at(-1);
      return ball && hoop && latest &&
        detection.atMs - latest.atMs <= 500 &&
        Math.abs(detection.x - hoop.x) <= hoop.width * 1.5 &&
        detection.y > hoop.y && detection.y < hoop.y + hoop.width * 2 &&
        predictedDistance(ball, detection) <= diagonal(latest) * 4;
    });
    const confident = detection.confidence >= BALL_CONFIDENCE ||
      ((nearHoop || recoveringExit) && detection.confidence >= NEAR_HOOP_BALL_CONFIDENCE);
    if (
      confident &&
      // Detectors often emit a tight ball plus a larger ball/rim composite at
      // the same center. Suppress that nested duplicate, while adjacent rack
      // balls remain separate because their boxes do not overlap.
      !accepted.some((existing) => detectionIoU(existing, detection) >= 0.2)
    ) {
      accepted.push(detection);
    }
  }
  const candidates: {
    ball: TrackedBall;
    detectionIndex: number;
    score: number;
    recovery: boolean;
  }[] = [];

  // Preserve detector identity before geometric association. This prevents a
  // fast shot from being swapped with a stationary rack ball when their boxes
  // pass close together under the hoop.
  const usedBallIds = new Set<number>();
  const usedDetectionIndexes = new Set<number>();
  for (let detectionIndex = 0; detectionIndex < accepted.length; detectionIndex += 1) {
    const detection = accepted[detectionIndex];
    if (detection?.sourceTrackId === undefined) continue;
    const exact = state.balls.find(
      (ball) => ball.sourceTrackId === detection.sourceTrackId,
    );
    const latest = exact?.detections.at(-1);
    if (!exact || !latest || usedBallIds.has(exact.id)) continue;
    // IDs are association hints, not proof: a recycled or swapped ID must not
    // join an above-rim ball to an unrelated observation across the court.
    if (predictedDistance(exact, detection) > diagonal(latest) * 8) {
      exact.sourceTrackId = undefined;
      continue;
    }
    exact.detections.push(detection);
    exact.detections = exact.detections
      .filter((sample) => detection.atMs - sample.atMs <= BALL_HISTORY_MS)
      .slice(-MAX_BALL_HISTORY_SAMPLES);
    usedBallIds.add(exact.id);
    usedDetectionIndexes.add(detectionIndex);
  }

  for (const ball of state.balls) {
    if (usedBallIds.has(ball.id)) continue;
    const latest = ball.detections.at(-1);
    if (!latest) continue;
    const flight = predictBallFlight(ball.detections.map(sample => ({
      x:sample.x,y:sample.y,width:sample.width,height:sample.height,
      at:sample.atMs,confidence:sample.confidence,trackId:sample.sourceTrackId,
    })), detections[0]?.atMs ?? latest.atMs);
    for (let detectionIndex = 0; detectionIndex < accepted.length; detectionIndex += 1) {
      if (usedDetectionIndexes.has(detectionIndex)) continue;
      const detection = accepted[detectionIndex];
      if (!detection) continue;
      const motionDistance = predictedDistance(ball, detection);
      const multiplier = hasPair(state.armedPairs, ball.id) ? 4 : 2;
      const recovery = motionDistance >= diagonal(latest) * multiplier;
      if (recovery && ball.sourceTrackId !== undefined && detection.sourceTrackId !== undefined &&
          ball.sourceTrackId !== detection.sourceTrackId) continue;
      const curvedDistance = flight ? Math.hypot(
        (detection.x-flight.x)/flight.toleranceX,
        (detection.y-flight.y)/flight.toleranceY) : Number.POSITIVE_INFINITY;
      if (recovery && (curvedDistance > 1 || detection.confidence < 0.4)) continue;
      const sizePenalty = Math.abs(Math.log(
        Math.max(1, diagonal(detection)) / Math.max(1, diagonal(latest)),
      ));
      candidates.push({
        ball,
        detectionIndex,
        recovery,
        // A stale rebound/rack track may be closer than the currently falling
        // ball. Prefer recent continuity; otherwise a crop-only observation
        // left under the net can steal the next shot's exit a second later.
        score: (recovery ? curvedDistance : motionDistance / Math.max(1, diagonal(latest))) + sizePenalty * 0.28 +
          Math.min(3, Math.max(0, detection.atMs - latest.atMs - 140) / 240),
      });
    }
  }

  candidates.sort((left, right) => Number(left.recovery) - Number(right.recovery) || left.score - right.score);
  for (const candidate of candidates) {
    if (
      usedBallIds.has(candidate.ball.id) ||
      usedDetectionIndexes.has(candidate.detectionIndex)
    ) {
      continue;
    }
    if (candidate.recovery) {
      // Curved recovery has less direct evidence than the legacy association.
      // Require a mutually unique match, not whichever near-identical ball
      // happened to sort first in a crowded return-rack scene.
      const available = candidates.filter(item => item.recovery &&
        !usedBallIds.has(item.ball.id) && !usedDetectionIndexes.has(item.detectionIndex));
      if (available.filter(item => item.ball.id === candidate.ball.id).length !== 1 ||
          available.filter(item => item.detectionIndex === candidate.detectionIndex).length !== 1) continue;
    }
    const detection = accepted[candidate.detectionIndex];
    if (!detection) continue;
    candidate.ball.detections.push(detection);
    candidate.ball.sourceTrackId = detection.sourceTrackId ?? candidate.ball.sourceTrackId;
    candidate.ball.detections = candidate.ball.detections
      .filter((sample) => detection.atMs - sample.atMs <= BALL_HISTORY_MS)
      .slice(-MAX_BALL_HISTORY_SAMPLES);
    usedBallIds.add(candidate.ball.id);
    usedDetectionIndexes.add(candidate.detectionIndex);
  }

  for (let detectionIndex = 0; detectionIndex < accepted.length; detectionIndex += 1) {
    if (usedDetectionIndexes.has(detectionIndex)) continue;
    const detection = accepted[detectionIndex];
    if (!detection) continue;
    const id = state.nextBallId;
    state.nextBallId += 1;
    state.balls.push({ id, sourceTrackId: detection.sourceTrackId, detections: [detection] });
  }
}

function armApproachingBalls(state: AttallaShotTrackerState, atMs: number): void {
  for (const ball of state.balls) {
    const recent = ball.detections.slice(-2);
    const reliableShortApproach = recent.length === 2 &&
      recent.every((detection) => detection.confidence >= FAST_ARM_BALL_CONFIDENCE) &&
      (recent[1]?.atMs ?? 0) - (recent[0]?.atMs ?? 0) <= FAST_ARM_MAX_SPAN_MS;
    if (hasPair(state.armedPairs, ball.id)) {
      continue;
    }
    const latest = ball.detections.at(-1);
    if (!latest) continue;
    for (const hoop of state.hoops) {
      // A calibrated scoring plane is intentionally thin, so comparing box
      // areas would reject a real basketball whose square detector box is
      // taller than the rim line. Physical plausibility is governed by ball
      // diameter versus rim opening width; oversized people/backboard
      // detections are still excluded.
      const ballDiameter = Math.sqrt(latest.width * latest.height);
      if (ballDiameter > hoop.width * 1.18) continue;
      const firstRecent = recent[0];
      const lastRecent = recent[1];
      const clipStartApproach =
        atMs <= CLIP_START_ARM_WINDOW_MS &&
        ball.detections.length === 2 &&
        firstRecent !== undefined &&
        lastRecent !== undefined &&
        lastRecent.atMs - firstRecent.atMs <= CLIP_START_ARM_MAX_SPAN_MS &&
        firstRecent.confidence >= CLIP_START_ARM_CONFIDENCE &&
        lastRecent.confidence >= CLIP_START_ARM_CONFIDENCE &&
        firstRecent.y < hoop.y &&
        lastRecent.y < hoop.y &&
        lastRecent.y - firstRecent.y >= Math.max(1, hoop.height * 0.08) &&
        lastRecent.x > hoop.x - hoop.width * 1.35 &&
        lastRecent.x < hoop.x + hoop.width * 1.35;
      if (ball.detections.length < 3 && !reliableShortApproach && !clipStartApproach) {
        continue;
      }
      const approachHeight = Math.max(hoop.height * 3, hoop.width * 2.2);
      const insideBackboardArea =
        latest.x > hoop.x - hoop.width * 2 &&
        latest.x < hoop.x + hoop.width * 2 &&
        latest.y > hoop.y - approachHeight &&
        latest.y < hoop.y;
      if (insideBackboardArea) {
        state.armedPairs.push({ ballId: ball.id, hoopId: hoop.id });
        break;
      }
    }
  }
}

function scoreCompletedPairs(
  state: AttallaShotTrackerState,
  atMs: number,
  forcePendingMisses = false,
): AttallaShotDecision[] {
  const decisions: AttallaShotDecision[] = [];
  const remainingPairs: BallHoopPair[] = [];
  const completedBallIds = new Set<number>();

  for (const pair of state.armedPairs) {
    const ball = state.balls.find((candidate) => candidate.id === pair.ballId);
    const hoop = state.hoops.find((candidate) => candidate.id === pair.hoopId);
    const below = ball?.detections.at(-1);
    if (!ball || !hoop || !below) continue;
    if (below.y <= hoop.y + hoop.height / 2) {
      remainingPairs.push(pair);
      continue;
    }

    const hoopTop = hoop.y - hoop.height / 2;
    const above = [...ball.detections].reverse().find((detection) => detection.y < hoopTop);
    if (!above) continue;
    const verticalTravel = below.y - above.y;
    if (verticalTravel <= 0.0001) continue;

    const crossingX = above.x +
      ((hoop.y - above.y) / verticalTravel) * (below.x - above.x);
    const halfWidth = Math.max(0.0001, hoop.width / 2);
    const normalizedOffset = Math.abs(crossingX - hoop.x) / halfWidth;
    const kind: ShotKind = normalizedOffset < 1 ? "make" : "miss";
    const geometryConfidence = kind === "make"
      ? clamp(1 - normalizedOffset)
      : clamp(normalizedOffset - 1);
    const detectionConfidence = clamp(
      (above.confidence + below.confidence + hoop.confidence) / 3,
    );
    const spanConfidence = clamp(verticalTravel / Math.max(1, hoop.height * 1.5));
    let confidence = clamp(
      0.72 + detectionConfidence * 0.18 + geometryConfidence * 0.07 + spanConfidence * 0.03,
    );
    // A long unobserved interval cannot prove a passage through the basket.
    // Keep the event for review instead of interpolating a certain result.
    const crossingSamples = ball.detections.filter(sample => sample.atMs >= above.atMs);
    const largestUnseenGap = crossingSamples.reduce((gap, sample, index) =>
      Math.max(gap, index > 0 ? sample.atMs - crossingSamples[index - 1]!.atMs : 0), 0);
    if (largestUnseenGap > 500) confidence = Math.min(confidence, 0.8);

    if (kind === "miss" && normalizedOffset < RIM_BOUNCE_MISS_OFFSET) {
      const pendingMissAt = pair.pendingMissAt ?? atMs;
      const pendingConfidence = Math.max(pair.pendingMissConfidence ?? 0, confidence);
      if (!forcePendingMisses && atMs - pendingMissAt < RIM_BOUNCE_SETTLE_MS) {
        remainingPairs.push({
          ...pair,
          pendingMissAt,
          pendingMissCrossingX: crossingX,
          pendingMissConfidence: pendingConfidence,
        });
        continue;
      }
    }
    completedBallIds.add(ball.id);

    // Distinct balls already observed concurrently may finish inside the
    // duplicate cooldown. Do not suppress that independently established shot.
    const concurrentAttempt = ball.sourceTrackId !== undefined &&
      state.lastShotSourceTrackId !== undefined &&
      ball.sourceTrackId !== state.lastShotSourceTrackId &&
      atMs - state.lastShotAt >= 200 &&
      ball.detections.filter(sample => sample.atMs <= state.lastShotAt).length >= 2;
    if (atMs - state.lastShotAt >= SHOT_COOLDOWN_MS || concurrentAttempt) {
      decisions.push({
        kind,
        atMs,
        confidence: kind === "miss"
          ? Math.max(confidence, pair.pendingMissConfidence ?? 0)
          : confidence,
        crossingX: kind === "miss"
          ? pair.pendingMissCrossingX ?? crossingX
          : crossingX,
        hoopId: hoop.id,
        ballId: ball.id,
      });
      state.lastShotAt = atMs;
      state.lastShotSourceTrackId = ball.sourceTrackId;
    }
  }

  // A completed ball must not be followed into its rebound or mistaken for the
  // next pass. A new physical ball now receives a fresh attempt track.
  state.balls = state.balls.filter((ball) => !completedBallIds.has(ball.id));
  state.armedPairs = remainingPairs.filter((pair) => !completedBallIds.has(pair.ballId));
  return decisions;
}

export function createAttallaShotTrackerState(): AttallaShotTrackerState {
  return {
    hoops: [],
    balls: [],
    armedPairs: [],
    nextHoopId: 0,
    nextBallId: 0,
    lastShotAt: Number.NEGATIVE_INFINITY,
  };
}

/**
 * TypeScript port of josephattalla/Basketball-Shot-Detection's detector-driven
 * ball/hoop association and above-rim -> below-net crossing classifier.
 * Source defects are fixed with timestamp-based expiry, a vertical-safe line
 * interpolation, deterministic nearest-track matching, and duplicate cooldown.
 */
export function stepAttallaShotTracker(
  current: AttallaShotTrackerState,
  detections: AttallaObjectDetection[],
  atMs: number,
): AttallaShotTrackerStep {
  if (!Number.isFinite(atMs) ||
      (current.lastFrameAt !== undefined && atMs <= current.lastFrameAt)) {
    return { state: current, decisions: [] };
  }
  const state = cloneState(current);
  state.lastFrameAt = atMs;
  cleanTracks(state, atMs);

  const valid: TimedObjectDetection[] = [];
  for (const detection of detections) {
    if (
      !Number.isFinite(detection.x) ||
      !Number.isFinite(detection.y) ||
      !Number.isFinite(detection.width) ||
      !Number.isFinite(detection.height) ||
      !Number.isFinite(detection.confidence) ||
      detection.confidence < 0 || detection.confidence > 1 ||
      detection.width <= 0 ||
      detection.height <= 0
    ) {
      continue;
    }
    valid.push({ ...detection, atMs });
  }

  // Update hoop context first, then perform a one-to-one ball assignment for
  // the whole frame. Detection array order must not alter the shot result.
  for (const detection of valid) {
    if (detection.kind === "hoop") addHoop(state, detection);
  }
  addBalls(
    state,
    valid.filter((detection) => detection.kind === "ball"),
  );
  armApproachingBalls(state, atMs);
  const decisions = scoreCompletedPairs(state, atMs);

  return { state, decisions };
}

/** Flushes only a deferred near-rim miss when an imported clip ends. */
export function finalizeAttallaShotTracker(
  current: AttallaShotTrackerState,
  atMs: number,
): AttallaShotTrackerStep {
  const state = cloneState(current);
  const decisions = scoreCompletedPairs(state, atMs, true);
  cleanTracks(state, atMs);
  return { state, decisions };
}
