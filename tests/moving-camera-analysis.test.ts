import assert from "node:assert/strict";
import test from "node:test";

import { createTrackerEngineState, stepTracker } from "../src/tracking/engine";
import { createVisionTrackState, selectTrackedBall } from "../src/vision/ballTracker";
import { selectHoopZoneCandidates } from "../src/vision/hoopZone";
import { detectBasketballCandidates } from "../src/vision/pixelBallDetector";
import { createRimTrackState, stepRimTracker } from "../src/vision/rimTracker";
import {
  alignTrackerEngineToRimShift,
  alignVisionTrackToRimShift,
} from "../src/vision/trackingAlignment";
import type { RimCalibration } from "../types/tracking";

const width = 160;
const height = 240;
const initialRim: RimCalibration = { x: 0.65, y: 0.15, width: 0.16, height: 0.035 };

function frame(
  cameraX: number,
  cameraY: number,
  ballX: number | null,
  ballY: number | null,
): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    data[offset] = 58;
    data[offset + 1] = 64;
    data[offset + 2] = 62;
    data[offset + 3] = 255;
  }
  const left = Math.round(initialRim.x * width) + cameraX;
  const right = Math.round((initialRim.x + initialRim.width) * width) + cameraX;
  const plane = Math.round((initialRim.y + initialRim.height * 0.48) * height) + cameraY;
  const paint = (x: number, y: number, value: number) => {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const offset = (y * width + x) * 4;
    data[offset] = value;
    data[offset + 1] = value;
    data[offset + 2] = value;
  };
  for (let x = left; x <= right; x += 1) {
    paint(x, plane, 236);
    paint(x, plane + 1, 182);
  }
  for (let y = plane - 18; y <= plane + 14; y += 1) paint(right, y, 214);
  for (let step = 0; step < 12; step += 1) {
    paint(left + 5 + step, plane + 3 + step, 145);
  }

  if (ballX !== null && ballY !== null) {
    const centerX = ballX + cameraX;
    const centerY = ballY + cameraY;
    for (let y = centerY - 4; y <= centerY + 4; y += 1) {
      for (let x = centerX - 4; x <= centerX + 4; x += 1) {
        if ((x - centerX) ** 2 + (y - centerY) ** 2 > 15) continue;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        const offset = (y * width + x) * 4;
        data[offset] = 126;
        data[offset + 1] = 74;
        data[offset + 2] = 38;
      }
    }
  }
  return data;
}

function gray(data: Uint8ClampedArray): Uint8Array {
  const result = new Uint8Array(width * height);
  for (let index = 0; index < result.length; index += 1) {
    const offset = index * 4;
    result[index] = Math.round(
      (data[offset] ?? 0) * 0.299 +
      (data[offset + 1] ?? 0) * 0.587 +
      (data[offset + 2] ?? 0) * 0.114,
    );
  }
  return result;
}

test("keeps a made-shot trajectory aligned through a large camera pan", () => {
  const calibrationFrame = frame(0, 0, null, null);
  let rimTrack = createRimTrackState(gray(calibrationFrame), width, height, initialRim);
  let previousGray: Uint8Array | null = null;
  let vision = createVisionTrackState();
  let tracker = createTrackerEngineState();
  const cameraOffsets = [0, -4, -8, -12, -55, -59, -63, -67];
  const ballPath = [
    [75, 160],
    [85, 120],
    [95, 90],
    [105, 58],
    [112, 30],
    [114, 35],
    [116, 43],
    [118, 52],
  ] as const;
  let makeCount = 0;
  let missCount = 0;

  ballPath.forEach(([ballX, ballY], index) => {
    const cameraX = cameraOffsets[index] ?? 0;
    const cameraY = index * 2;
    const pixels = frame(cameraX, cameraY, ballX, ballY);
    const timestamp = index * 33;
    const detectionFrame = detectBasketballCandidates(
      { width, height, data: pixels },
      previousGray,
      timestamp,
      rimTrack.rim,
    );
    previousGray = detectionFrame.gray;
    const previousLostFrames = rimTrack.consecutiveLostFrames;
    const rimStep = stepRimTracker(detectionFrame.gray, width, height, rimTrack);
    rimTrack = rimStep.state;
    assert.equal(rimStep.found, true);
    assert.ok(previousLostFrames < 4);
    vision = alignVisionTrackToRimShift(
      vision,
      rimStep.displacementX,
      rimStep.displacementY,
    );
    tracker = alignTrackerEngineToRimShift(
      tracker,
      rimStep.displacementX,
      rimStep.displacementY,
    );
    const candidates = selectHoopZoneCandidates(
      detectionFrame.candidates,
      rimStep.rim,
      width,
      height,
    );
    const selection = selectTrackedBall(candidates, vision, rimStep.rim, timestamp);
    vision = selection.state;
    const result = stepTracker(tracker, selection.detection, rimStep.rim, timestamp);
    tracker = result.state;
    if (result.shot === "make") makeCount += 1;
    if (result.shot === "miss") missCount += 1;
  });

  assert.ok(Math.abs(rimTrack.rim.x - (initialRim.x - 67 / width)) < 0.04);
  assert.equal(makeCount, 1);
  assert.equal(missCount, 0);
});
