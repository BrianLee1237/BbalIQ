import assert from "node:assert/strict";
import test from "node:test";

import { MIN_AUTOMATIC_DECISION_CONFIDENCE } from "../src/tracking/engine";
import {
  classifyVideoTrajectories,
  type VideoTrajectorySample,
} from "../src/tracking/videoTrajectoryClassifier";
import type { RimCalibration } from "../types/tracking";

const rim: RimCalibration = { x: 0.4, y: 0.2, width: 0.2, height: 0.05 };

function sample(atSeconds: number, x: number, y: number): VideoTrajectorySample {
  return {
    atSeconds,
    rim,
    ball: {
      x,
      y,
      width: 0.045,
      height: 0.045,
      at: Math.round(atSeconds * 1_000),
      confidence: 0.92,
      appearanceConfidence: 0.88,
      motionConfidence: 0.9,
    },
  };
}

function madeShot(start: number, crossingX = 0.5): VideoTrajectorySample[] {
  return [
    sample(start, 0.32, 0.43),
    sample(start + 0.2, 0.38, 0.31),
    sample(start + 0.4, 0.44, 0.16),
    sample(start + 0.6, 0.48, 0.11),
    sample(start + 0.8, 0.5, 0.15),
    sample(start + 0.98, crossingX, 0.21),
    sample(start + 1.1, crossingX, 0.25),
    sample(start + 1.22, 0.51, 0.29),
    sample(start + 1.34, 0.52, 0.34),
  ];
}

function missedShot(start: number, x = 0.65): VideoTrajectorySample[] {
  return [
    sample(start, 0.33, 0.42),
    sample(start + 0.2, 0.4, 0.29),
    sample(start + 0.4, 0.55, 0.15),
    sample(start + 0.6, 0.62, 0.11),
    sample(start + 0.8, x, 0.16),
    sample(start + 0.98, x, 0.21),
    sample(start + 1.1, x, 0.25),
    sample(start + 1.22, x, 0.3),
    sample(start + 1.34, x, 0.35),
  ];
}

test("classifies every shot in a repeated all-makes upload", () => {
  const decisions = classifyVideoTrajectories([
    ...madeShot(0),
    ...madeShot(2.8),
    ...madeShot(5.6),
    ...madeShot(8.4),
  ]);
  assert.equal(decisions.length, 4);
  assert.deepEqual(decisions.map((decision) => decision.finalKind), [
    "make",
    "make",
    "make",
    "make",
  ]);
  assert.ok(decisions.every((decision) => decision.confidence >= MIN_AUTOMATIC_DECISION_CONFIDENCE));
});

test("classifies a right-adjacent descending trajectory as a miss", () => {
  const decisions = classifyVideoTrajectories(missedShot(0));
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "miss");
  assert.equal(decisions[0]?.reason, "airball");
});

test("uses the complete below-net path to recover a jittery rim-plane sample", () => {
  const samples = madeShot(0);
  const jittered = samples.map((value, index) =>
    index === 5 || index === 6
      ? sample(value.atSeconds, 0.63, value.ball.y)
      : value,
  );
  const decisions = classifyVideoTrajectories(jittered);
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "make");
});

test("interpolates a make across a short detector occlusion at the rim", () => {
  const samples = madeShot(0).filter((_, index) => index !== 5 && index !== 6);
  const decisions = classifyVideoTrajectories(samples);
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "make");
});

test("keeps the shot rim-relative through small hoop-lock corrections", () => {
  const shifted = madeShot(0).map((value, index) => {
    const shiftX = index * 0.003;
    const shiftY = index * 0.0015;
    return {
      ...value,
      rim: { ...value.rim, x: value.rim.x + shiftX, y: value.rim.y + shiftY },
      ball: { ...value.ball, x: value.ball.x + shiftX, y: value.ball.y + shiftY },
    };
  });
  const decisions = classifyVideoTrajectories(shifted);
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "make");
});

test("does not invent a miss from an incomplete or static track", () => {
  const incomplete = madeShot(0).slice(0, 6);
  const staticFalsePositive = Array.from({ length: 20 }, (_, index) =>
    sample(3 + index / 30, 0.51, 0.22),
  );
  assert.deepEqual(classifyVideoTrajectories([...incomplete, ...staticFalsePositive]), []);
});

test("keeps consecutive make and miss attempts separate", () => {
  const decisions = classifyVideoTrajectories([
    ...madeShot(0),
    ...missedShot(3),
    ...madeShot(6),
  ]);
  assert.deepEqual(decisions.map((decision) => decision.finalKind), ["make", "miss", "make"]);
});
