import assert from "node:assert/strict";
import test from "node:test";

import { createTrackerEngineState } from "../src/tracking/engine";
import { createVisionTrackState } from "../src/vision/ballTracker";
import {
  alignTrackerEngineToRimShift,
  alignVisionTrackToRimShift,
} from "../src/vision/trackingAlignment";
import type { BallDetection } from "../types/tracking";

function ball(x: number, y: number, at: number): BallDetection {
  return {
    x,
    y,
    at,
    width: 0.05,
    height: 0.05,
    confidence: 0.9,
    motionConfidence: 0.9,
  };
}

test("moves the live ball lock with a panning camera", () => {
  const vision = {
    ...createVisionTrackState(),
    previous: ball(0.4, 0.3, 0),
    current: ball(0.5, 0.2, 33),
    velocityX: 1,
    velocityY: -1,
  };
  const shifted = alignVisionTrackToRimShift(vision, 0.12, -0.07);
  assert.equal(shifted.current?.x, 0.62);
  assert.equal(shifted.current?.y, 0.13);
  assert.equal(shifted.velocityX, vision.velocityX);
  assert.equal(shifted.velocityY, vision.velocityY);
});

test("keeps the armed trajectory aligned to the relocated hoop", () => {
  const state = {
    ...createTrackerEngineState(),
    armed: true,
    enteredRim: true,
    entryX: 0.52,
    previous: ball(0.52, 0.22, 99),
    trajectory: [ball(0.42, 0.36, 33), ball(0.52, 0.22, 99)],
    apexY: 0.18,
  };
  const shifted = alignTrackerEngineToRimShift(state, -0.08, 0.05);
  assert.equal(shifted.previous?.x, 0.44);
  assert.equal(shifted.previous?.y, 0.27);
  assert.equal(shifted.entryX, 0.44);
  assert.ok(Math.abs(shifted.apexY - 0.23) < 0.000001);
  assert.equal(shifted.armed, true);
  assert.equal(shifted.enteredRim, true);
});
