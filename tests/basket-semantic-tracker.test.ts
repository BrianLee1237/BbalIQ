import assert from "node:assert/strict";
import test from "node:test";

import {
  createBasketSemanticTrackerState,
  stepBasketSemanticTracker,
} from "../src/tracking/basketSemanticTracker";
import type { BallDetection, RimCalibration } from "../types/tracking";

const rim: RimCalibration = { x: 0.4, y: 0.2, width: 0.2, height: 0.05 };

function ball(at: number, x: number, y: number, confidence = 0.8): BallDetection {
  return {
    x,
    y,
    width: 0.04,
    height: 0.04,
    confidence: 0.55 + confidence * 0.45,
    appearanceConfidence: confidence,
    motionConfidence: 0.86,
    at,
  };
}

test("recovers a make from repeated basket occupancy and a centered net exit", () => {
  let state = createBasketSemanticTrackerState();
  const decisions = [];
  for (const detection of [
    ball(0, 0.48, 0.14),
    ball(50, 0.49, 0.17),
    ball(100, 0.5, 0.205),
    ball(133, 0.5, 0.225),
    ball(166, 0.5, 0.29),
  ]) {
    const step = stepBasketSemanticTracker(state, detection, rim, detection.at);
    state = step.state;
    if (step.decision) decisions.push(step.decision);
  }
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "make");
  assert.equal(decisions[0]?.reason, "semantic-basket");
});

test("does not score a ball that overlaps the rim without an approach", () => {
  let state = createBasketSemanticTrackerState();
  const decisions = [];
  for (const detection of [
    ball(0, 0.5, 0.205),
    ball(40, 0.5, 0.225),
    ball(80, 0.5, 0.29),
  ]) {
    const step = stepBasketSemanticTracker(state, detection, rim, detection.at);
    state = step.state;
    if (step.decision) decisions.push(step.decision);
  }
  assert.deepEqual(decisions, []);
});

test("does not turn a right-adjacent airball into a semantic make", () => {
  let state = createBasketSemanticTrackerState();
  const decisions = [];
  for (const detection of [
    ball(0, 0.62, 0.14),
    ball(50, 0.63, 0.17),
    ball(100, 0.64, 0.205),
    ball(133, 0.65, 0.225),
    ball(166, 0.66, 0.29),
  ]) {
    const step = stepBasketSemanticTracker(state, detection, rim, detection.at);
    state = step.state;
    if (step.decision) decisions.push(step.decision);
  }
  assert.deepEqual(decisions, []);
});

test("recovers a centered make when the detector skips the rim-occupancy frames", () => {
  let state = createBasketSemanticTrackerState();
  const decisions = [];
  for (const detection of [
    ball(0, 0.54, 0.15, 0.88),
    ball(50, 0.535, 0.19, 0.91),
    ball(283, 0.525, 0.29, 0.24),
  ]) {
    const step = stepBasketSemanticTracker(state, detection, rim, detection.at);
    state = step.state;
    if (step.decision) decisions.push(step.decision);
  }
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "make");
  assert.equal(decisions[0]?.reason, "semantic-basket");
});

test("does not recover a sparse crossing outside the rim opening", () => {
  let state = createBasketSemanticTrackerState();
  const decisions = [];
  for (const detection of [
    ball(0, 0.61, 0.15, 0.88),
    ball(50, 0.615, 0.19, 0.91),
    ball(283, 0.63, 0.29, 0.24),
  ]) {
    const step = stepBasketSemanticTracker(state, detection, rim, detection.at);
    state = step.state;
    if (step.decision) decisions.push(step.decision);
  }
  assert.deepEqual(decisions, []);
});

test("counts two complete semantic basket entries in a rapid drill", () => {
  let state = createBasketSemanticTrackerState();
  const decisions = [];
  const attempt = (start: number) => [
    ball(start, 0.48, 0.14),
    ball(start + 50, 0.49, 0.17),
    ball(start + 100, 0.5, 0.205),
    ball(start + 133, 0.5, 0.225),
    ball(start + 166, 0.5, 0.29),
  ];
  for (const detection of [...attempt(0), ...attempt(820)]) {
    const step = stepBasketSemanticTracker(state, detection, rim, detection.at);
    state = step.state;
    if (step.decision) decisions.push(step.decision);
  }
  assert.deepEqual(decisions.map((decision) => decision.finalKind), ["make", "make"]);
});
