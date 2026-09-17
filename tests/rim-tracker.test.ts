import assert from "node:assert/strict";
import test from "node:test";

import { createRimTrackState, stepRimTracker } from "../src/vision/rimTracker";
import type { RimCalibration } from "../types/tracking";

const width = 120;
const height = 90;
const rim: RimCalibration = { x: 0.54, y: 0.24, width: 0.2, height: 0.06 };

function hoopFrame(offsetX: number, offsetY: number): Uint8Array {
  const frame = new Uint8Array(width * height).fill(28);
  const left = Math.round(rim.x * width) + offsetX;
  const right = Math.round((rim.x + rim.width) * width) + offsetX;
  const plane = Math.round((rim.y + rim.height * 0.5) * height) + offsetY;
  for (let x = left; x <= right; x += 1) {
    frame[plane * width + x] = 238;
    if (plane + 1 < height) frame[(plane + 1) * width + x] = 188;
  }
  for (let y = plane - 10; y <= plane + 8; y += 1) {
    if (y < 0 || y >= height) continue;
    frame[y * width + right] = 220;
  }
  for (let step = 0; step < 8; step += 1) {
    const x = left + 4 + step;
    const y = plane + 2 + step;
    if (x < width && y < height) frame[y * width + x] = 150;
  }
  return frame;
}

test("follows the calibrated hoop when it shifts between frames", () => {
  const state = createRimTrackState(hoopFrame(0, 0), width, height, rim);
  const result = stepRimTracker(hoopFrame(5, 3), width, height, state);
  assert.equal(result.found, true);
  assert.ok(result.confidence >= 0.56);
  assert.ok(Math.abs(result.rim.x - (rim.x + 5 / width)) < 0.025);
  assert.ok(Math.abs(result.rim.y - (rim.y + 3 / height)) < 0.03);
});

test("keeps the last hoop position instead of jumping on a bad match", () => {
  const state = createRimTrackState(hoopFrame(0, 0), width, height, rim);
  const blank = new Uint8Array(width * height).fill(28);
  const result = stepRimTracker(blank, width, height, state);
  assert.equal(result.found, false);
  assert.deepEqual(result.rim, rim);
});

test("reacquires the highlighted hoop across the full frame after a camera pan", () => {
  const state = createRimTrackState(hoopFrame(0, 0), width, height, rim);
  const result = stepRimTracker(hoopFrame(-40, 8), width, height, state);
  assert.equal(result.found, true);
  assert.equal(result.reacquired, true);
  assert.ok(Math.abs(result.rim.x - (rim.x - 40 / width)) < 0.04);
  assert.ok(Math.abs(result.rim.y - (rim.y + 8 / height)) < 0.045);
});
