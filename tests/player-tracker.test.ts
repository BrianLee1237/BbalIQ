import assert from "node:assert/strict";
import test from "node:test";

import {
  createPlayerMotionState,
  trackMovingPlayer,
} from "../src/vision/playerTracker";

const width = 120;
const height = 80;

interface TestRectangle {
  x: number;
  y: number;
  width: number;
  height: number;
}

function frame(...rectangles: TestRectangle[]): Uint8Array {
  const pixels = new Uint8Array(width * height * 3);
  for (const rect of rectangles) {
    for (let y = rect.y; y < rect.y + rect.height; y += 1) {
      for (let x = rect.x; x < rect.x + rect.width; x += 1) {
        const index = (y * width + x) * 3;
        pixels[index] = 210;
        pixels[index + 1] = 210;
        pixels[index + 2] = 210;
      }
    }
  }
  return pixels;
}

test("acquires the dominant moving player after the background frame", () => {
  let state = createPlayerMotionState();
  state = trackMovingPlayer(frame(), width, height, state, 0).state;
  const result = trackMovingPlayer(
    frame({ x: 25, y: 15, width: 20, height: 52 }),
    width,
    height,
    state,
    100,
  );
  assert.ok(result.detection);
  assert.ok((result.detection?.height ?? 0) >= 0.5);
  assert.ok((result.detection?.confidence ?? 0) >= 0.6);
});

test("ignores isolated basketball-sized motion as a player", () => {
  let state = createPlayerMotionState();
  state = trackMovingPlayer(frame(), width, height, state, 0).state;
  const result = trackMovingPlayer(
    frame({ x: 80, y: 20, width: 5, height: 5 }),
    width,
    height,
    state,
    100,
  );
  assert.equal(result.detection, null);
});

test("retains the player briefly through an occluded or motionless frame", () => {
  let state = createPlayerMotionState();
  state = trackMovingPlayer(frame(), width, height, state, 0).state;
  const acquired = trackMovingPlayer(
    frame({ x: 25, y: 15, width: 20, height: 52 }),
    width,
    height,
    state,
    100,
  );
  const retained = trackMovingPlayer(
    frame({ x: 25, y: 15, width: 20, height: 52 }),
    width,
    height,
    acquired.state,
    180,
  );
  assert.ok(retained.detection);
  assert.ok((retained.detection?.confidence ?? 1) < (acquired.detection?.confidence ?? 0));
});

test("rejects camera-wide movement instead of boxing the entire frame", () => {
  let state = createPlayerMotionState();
  state = trackMovingPlayer(frame(), width, height, state, 0).state;
  const changed = new Uint8Array(width * height * 3).fill(255);
  const result = trackMovingPlayer(changed, width, height, state, 100);
  assert.equal(result.detection, null);
});

test("keeps the same player when a larger distant person enters", () => {
  let state = createPlayerMotionState();
  state = trackMovingPlayer(frame(), width, height, state, 0).state;
  state = trackMovingPlayer(
    frame({ x: 10, y: 15, width: 20, height: 52 }),
    width,
    height,
    state,
    100,
  ).state;
  const result = trackMovingPlayer(
    frame(
      { x: 13, y: 15, width: 20, height: 52 },
      { x: 85, y: 7, width: 27, height: 65 },
    ),
    width,
    height,
    state,
    180,
  );
  assert.ok(result.detection);
  assert.ok((result.detection?.x ?? 1) < 0.5);
});
