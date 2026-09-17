import assert from "node:assert/strict";
import test from "node:test";

import { computeRimDetectionRegion } from "../src/vision/rimDetectionRegion";
import {
  computeTrackDetectionRegion,
  regionsSubstantiallyOverlap,
} from "../src/vision/trackDetectionRegion";

test("magnifies a small centered rim while retaining the local shot corridor", () => {
  const region = computeRimDetectionRegion(
    { x: 0.385, y: 0.255, width: 0.03, height: 0.015 },
    640,
    376,
  );

  assert.deepEqual(region, {
    left: 172,
    top: 21,
    width: 168,
    height: 155,
  });
  assert.ok(region.width / 640 < 0.27);
  assert.ok(region.height / 376 < 0.42);
});

test("clamps the focused region at frame edges without producing an empty crop", () => {
  const region = computeRimDetectionRegion(
    { x: 0, y: 0, width: 0.04, height: 0.02 },
    640,
    360,
  );

  assert.equal(region.left, 0);
  assert.equal(region.top, 0);
  assert.ok(region.width > 1);
  assert.ok(region.height > 1);
  assert.ok(region.left + region.width <= 640);
  assert.ok(region.top + region.height <= 360);
});

test("predicted detector crop follows a fast ball and remains in frame", () => {
  const region = computeTrackDetectionRegion(
    {
      previous: { x: 0.2, y: 0.7, width: 0.04, height: 0.04, confidence: 0.9, at: 0 },
      current: { x: 0.3, y: 0.55, width: 0.04, height: 0.04, confidence: 0.9, at: 100 },
      velocityX: 1,
      velocityY: -1.5,
      confirmedFrames: 2,
    },
    { x: 0.68, y: 0.16, width: 0.16, height: 0.035 },
    640,
    360,
    300,
  );
  assert.ok(region);
  const predictedX = 0.5 * 640;
  const predictedY = 0.25 * 360;
  assert.ok(predictedX >= region.left && predictedX <= region.left + region.width);
  assert.ok(predictedY >= region.top && predictedY <= region.top + region.height);
  assert.ok(region.left >= 0 && region.left + region.width <= 640);
  assert.ok(region.top >= 0 && region.top + region.height <= 360);
});

test("skips a redundant track crop when the rim crop already covers it", () => {
  assert.equal(
    regionsSubstantiallyOverlap(
      { left: 100, top: 50, width: 200, height: 180 },
      { left: 110, top: 60, width: 180, height: 160 },
    ),
    true,
  );
});
