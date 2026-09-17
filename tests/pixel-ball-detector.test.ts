import assert from "node:assert/strict";
import test from "node:test";

import { detectBasketballCandidates } from "../src/vision/pixelBallDetector";
import { createVisionTrackState, selectTrackedBall } from "../src/vision/ballTracker";
import { createTrackerEngineState, stepTracker } from "../src/tracking/engine";
import { selectHoopZoneCandidates } from "../src/vision/hoopZone";
import type { RimCalibration } from "../types/tracking";

const width = 120;
const height = 180;
const rim: RimCalibration = { x: 0.68, y: 0.16, width: 0.16, height: 0.035 };

function frameWithBall(centerX: number | null, centerY: number | null): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    data[offset] = 72;
    data[offset + 1] = 76;
    data[offset + 2] = 70;
    data[offset + 3] = 255;
  }
  if (centerX === null || centerY === null) return data;
  for (let y = centerY - 3; y <= centerY + 3; y += 1) {
    for (let x = centerX - 3; x <= centerX + 3; x += 1) {
      if ((x - centerX) ** 2 + (y - centerY) ** 2 > 10) continue;
      const offset = (y * width + x) * 4;
      data[offset] = 112;
      data[offset + 1] = 70;
      data[offset + 2] = 42;
    }
  }
  return data;
}

test("detects a small dim basketball near the marked hoop", () => {
  const previous = detectBasketballCandidates(
    { width, height, data: frameWithBall(null, null) },
    null,
    0,
    rim,
  );
  const result = detectBasketballCandidates(
    { width, height, data: frameWithBall(82, 48) },
    previous.gray,
    67,
    rim,
  );
  const ball = result.candidates.find(
    (candidate) => Math.abs(candidate.x - 82 / width) < 0.04,
  );
  assert.ok(ball);
  assert.ok((ball.motionConfidence ?? 0) >= 0.5);
  assert.ok((ball.appearanceConfidence ?? 0) >= 0.45);
});

test("turns small-ball pixels into a verified hoop entry and net exit", () => {
  const path = [
    [60, 110],
    [68, 85],
    [76, 60],
    [84, 38],
    [89, 22],
    [90, 27],
    [91, 34],
    [92, 42],
  ] as const;
  let previousGray: Uint8Array | null = null;
  let vision = createVisionTrackState();
  let tracker = createTrackerEngineState();
  let shot: string | null = null;

  path.forEach(([x, y], index) => {
    const timestamp = index * 67;
    const detection = detectBasketballCandidates(
      { width, height, data: frameWithBall(x, y) },
      previousGray,
      timestamp,
      rim,
    );
    previousGray = detection.gray;
    const hoopCandidates = selectHoopZoneCandidates(detection.candidates, rim, width, height);
    const selection = selectTrackedBall(hoopCandidates, vision, rim, timestamp);
    vision = selection.state;
    const step = stepTracker(tracker, selection.detection, rim, timestamp);
    tracker = step.state;
    shot = step.shot ?? shot;
  });

  assert.equal(shot, "make");
});

test("resets after a make and counts the next made shot in the same video", () => {
  const shotPath = [
    [60, 110],
    [68, 85],
    [76, 60],
    [84, 38],
    [89, 22],
    [90, 27],
    [91, 34],
    [92, 42],
  ] as const;
  let previousGray: Uint8Array | null = null;
  let vision = createVisionTrackState();
  let tracker = createTrackerEngineState();
  let makes = 0;

  const processFrame = (x: number | null, y: number | null, timestamp: number) => {
    const detection = detectBasketballCandidates(
      { width, height, data: frameWithBall(x, y) },
      previousGray,
      timestamp,
      rim,
    );
    previousGray = detection.gray;
    const hoopCandidates = selectHoopZoneCandidates(detection.candidates, rim, width, height);
    const selection = selectTrackedBall(hoopCandidates, vision, rim, timestamp);
    vision = selection.state;
    const result = stepTracker(tracker, selection.detection, rim, timestamp);
    tracker = result.state;
    if (result.shot === "make") makes += 1;
    if (result.shot || result.reason === "cooldown") {
      vision = createVisionTrackState();
    } else if (!vision.current && !tracker.armed && tracker.previous) {
      const lastShotAt = tracker.lastShotAt;
      tracker = { ...createTrackerEngineState(), lastShotAt };
    }
  };

  shotPath.forEach(([x, y], index) => processFrame(x, y, index * 33));
  for (let index = 0; index < 25; index += 1) {
    processFrame(null, null, 264 + index * 33);
  }
  shotPath.forEach(([x, y], index) => processFrame(x, y, 2_500 + index * 33));

  assert.equal(makes, 2);
});
