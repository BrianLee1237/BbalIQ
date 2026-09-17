import assert from "node:assert/strict";
import test from "node:test";

import {
  createVisionTrackState,
  selectTrackGuidedRecoveryCandidates,
  selectTrackedBall,
  type VisionTrackState,
} from "../src/vision/ballTracker";
import type { BallDetection, RimCalibration } from "../types/tracking";

const rim: RimCalibration = { x: 0.65, y: 0.2, width: 0.2, height: 0.06 };

function candidate(
  x: number,
  y: number,
  at: number,
  confidence = 0.85,
  width = 0.05,
): BallDetection {
  return { x, y, at, width, height: width, confidence };
}

test("keeps the moving basketball instead of jumping to a stronger orange distractor", () => {
  const track: VisionTrackState = {
    previous: candidate(0.2, 0.7, 0, 0.82),
    current: candidate(0.3, 0.55, 100, 0.84),
  };
  const movingBall = candidate(0.4, 0.4, 200, 0.78);
  const staticDistractor = candidate(0.8, 0.75, 200, 0.96, 0.09);
  const result = selectTrackedBall([staticDistractor, movingBall], track, rim, 200);
  assert.equal(result.detection?.x, movingBall.x);
  assert.equal(result.detection?.y, movingBall.y);
});

test("rejects a discontinuous low-confidence candidate", () => {
  const track: VisionTrackState = {
    previous: candidate(0.2, 0.7, 0),
    current: candidate(0.28, 0.58, 100),
  };
  const result = selectTrackedBall(
    [candidate(0.92, 0.08, 200, 0.7)],
    track,
    rim,
    200,
  );
  assert.equal(result.detection, null);
  assert.equal(result.state.current?.x, track.current?.x);
});

test("clears a stale track after the ball has been missing", () => {
  const track: VisionTrackState = {
    previous: candidate(0.2, 0.7, 0),
    current: candidate(0.3, 0.55, 100),
  };
  const result = selectTrackedBall([], track, rim, 701);
  assert.deepEqual(result.state, createVisionTrackState());
});

test("penalizes a wide static candidate centered on the marked rim", () => {
  const rimArtifact = candidate(0.74, 0.23, 0, 0.94, 0.13);
  const ball = {
    ...candidate(0.48, 0.48, 0, 0.78, 0.05),
    motionConfidence: 0.9,
    appearanceConfidence: 0.78,
  };
  const result = selectTrackedBall([rimArtifact, ball], createVisionTrackState(), rim, 0);
  assert.equal(result.detection?.x, ball.x);
});

test("predicts through a short gap and reacquires the continuing ball", () => {
  const track: VisionTrackState = {
    previous: candidate(0.2, 0.7, 0),
    current: candidate(0.3, 0.55, 100),
    velocityX: 1,
    velocityY: -1.5,
    confirmedFrames: 2,
    missingFrames: 1,
  };
  const continuingBall = candidate(0.5, 0.25, 300, 0.8);
  const distractor = candidate(0.33, 0.53, 300, 0.94);
  const result = selectTrackedBall([distractor, continuingBall], track, rim, 300);
  assert.equal(result.detection?.x, continuingBall.x);
  assert.equal(result.state.missingFrames, 0);
});

test("allows a high-confidence reacquisition after a large but plausible gap", () => {
  const track: VisionTrackState = {
    previous: candidate(0.2, 0.7, 0),
    current: candidate(0.3, 0.55, 100),
    velocityX: 0.2,
    velocityY: -0.2,
  };
  const reacquired = candidate(0.66, 0.2, 500, 0.96);
  const result = selectTrackedBall([reacquired], track, rim, 500);
  assert.equal(result.detection?.x, reacquired.x);
});

test("rejects even a strong teleport candidate during a continuous track", () => {
  const track: VisionTrackState = {
    previous: candidate(0.2, 0.7, 0),
    current: candidate(0.28, 0.58, 100),
    velocityX: 0.8,
    velocityY: -1.2,
    confirmedFrames: 2,
    missingFrames: 0,
  };
  const teleport = candidate(0.92, 0.08, 200, 0.98);
  const result = selectTrackedBall([teleport], track, rim, 200);
  assert.equal(result.detection, null);
});

test("pixel recovery bridges only an established learned track", () => {
  const recovery = {
    ...candidate(0.5, 0.25, 300, 0.76),
    motionConfidence: 0.84,
    appearanceConfidence: 0.36,
  };
  assert.deepEqual(
    selectTrackGuidedRecoveryCandidates(
      [recovery],
      createVisionTrackState(),
      rim,
      300,
    ),
    [],
  );
  const track: VisionTrackState = {
    previous: candidate(0.3, 0.55, 100),
    current: candidate(0.4, 0.4, 200),
    velocityX: 1,
    velocityY: -1.5,
    confirmedFrames: 3,
  };
  assert.deepEqual(
    selectTrackGuidedRecoveryCandidates([recovery], track, rim, 300),
    [recovery],
  );
});

test("pixel recovery rejects strong camera-wide motion away from prediction", () => {
  const track: VisionTrackState = {
    previous: candidate(0.3, 0.55, 100),
    current: candidate(0.4, 0.4, 200),
    velocityX: 1,
    velocityY: -1.5,
    confirmedFrames: 4,
  };
  const distractor = {
    ...candidate(0.92, 0.82, 300, 0.96),
    motionConfidence: 0.98,
    appearanceConfidence: 0.5,
  };
  assert.deepEqual(
    selectTrackGuidedRecoveryCandidates([distractor], track, rim, 300),
    [],
  );
});

test("pixel recovery preserves a direction-reversing ball at rim contact", () => {
  const track: VisionTrackState = {
    previous: candidate(0.7, 0.1, 100),
    current: candidate(0.73, 0.19, 200),
    velocityX: 0.3,
    velocityY: 0.9,
    confirmedFrames: 4,
  };
  const rimBounce = {
    ...candidate(0.74, 0.17, 260, 0.78),
    motionConfidence: 0.88,
    appearanceConfidence: 0.42,
  };
  assert.deepEqual(
    selectTrackGuidedRecoveryCandidates([rimBounce], track, rim, 260),
    [rimBounce],
  );
});
