import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";

import {
  createAttallaShotTrackerState,
  finalizeAttallaShotTracker,
  stepAttallaShotTracker,
  type AttallaObjectDetection,
  type AttallaShotDecision,
} from "../src/tracking/attallaShotTracker";
import {
  calibratedRimToScoringHoop,
  rimFromAutomaticHoop,
} from "../src/vision/learnedTracking";

interface DetectionFixture {
  sourceSha256: string;
  groundTruth: { makes: number; misses: number };
  frames: {
    atMs: number;
    detections: AttallaObjectDetection[];
  }[];
}

function runFrames(
  frames: DetectionFixture["frames"],
): AttallaShotDecision[] {
  let state = createAttallaShotTrackerState();
  const decisions: AttallaShotDecision[] = [];
  for (const frame of frames) {
    const result = stepAttallaShotTracker(state, frame.detections, frame.atMs);
    state = result.state;
    decisions.push(...result.decisions);
  }
  return decisions;
}

test("classifies a centered above-rim to below-net crossing as a make", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const frames = [
    { atMs: 0, detections: [hoop, { kind: "ball" as const, x: 92, y: 65, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 70, detections: [hoop, { kind: "ball" as const, x: 96, y: 72, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 140, detections: [hoop, { kind: "ball" as const, x: 99, y: 82, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 210, detections: [hoop, { kind: "ball" as const, x: 101, y: 112, width: 10, height: 10, confidence: 0.9 }] },
  ];
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make"]);
});

test("classifies a completed adjacent crossing as a miss", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const frames = [
    { atMs: 0, detections: [hoop, { kind: "ball" as const, x: 140, y: 65, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 70, detections: [hoop, { kind: "ball" as const, x: 142, y: 72, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 140, detections: [hoop, { kind: "ball" as const, x: 144, y: 82, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 210, detections: [hoop, { kind: "ball" as const, x: 148, y: 112, width: 10, height: 10, confidence: 0.9 }] },
  ];
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["miss"]);
});

test("preserves the above-rim history during dense 30 FPS analysis", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const frames = Array.from({ length: 61 }, (_, index) => {
    const progress = index / 60;
    const y = progress < 0.72
      ? 66 + progress * 20
      : 80 + ((progress - 0.72) / 0.28) * 36;
    return {
      atMs: Math.round(index * (1_900 / 60)),
      detections: [
        hoop,
        {
          kind: "ball" as const,
          x: 96 + progress * 5,
          y,
          width: 10,
          height: 10,
          confidence: 0.9,
        },
      ],
    };
  });
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make"]);
});

test("recovers a high-confidence make after a short rim occlusion", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 250, y: 105, width: 19, height: 21, confidence: 0.88,
  };
  const frames = [
    { atMs: 5_983, detections: [hoop, { kind: "ball" as const, x: 248.6, y: 75, width: 9.6, height: 11.5, confidence: 0.86 }] },
    { atMs: 6_017, detections: [hoop, { kind: "ball" as const, x: 248.6, y: 75, width: 9.6, height: 11.5, confidence: 0.86 }] },
    { atMs: 6_050, detections: [hoop] },
    { atMs: 6_083, detections: [] },
    { atMs: 6_117, detections: [] },
    { atMs: 6_150, detections: [] },
    { atMs: 6_183, detections: [] },
    { atMs: 6_217, detections: [] },
    { atMs: 6_250, detections: [hoop, { kind: "ball" as const, x: 248.8, y: 118.8, width: 7.6, height: 9.8, confidence: 0.71 }] },
  ];

  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make"]);
});

test("does not fast-arm two weak above-rim observations", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const frames = [
    { atMs: 0, detections: [hoop, { kind: "ball" as const, x: 98, y: 70, width: 10, height: 10, confidence: 0.62 }] },
    { atMs: 40, detections: [hoop, { kind: "ball" as const, x: 98, y: 70, width: 10, height: 10, confidence: 0.68 }] },
    { atMs: 250, detections: [hoop, { kind: "ball" as const, x: 99, y: 115, width: 10, height: 10, confidence: 0.9 }] },
  ];

  assert.deepEqual(runFrames(frames), []);
});

test("counts a shot already descending during the opening frames", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const frames = [
    { atMs: 0, detections: [hoop, { kind: "ball" as const, x: 96, y: 70, width: 10, height: 10, confidence: 0.56 }] },
    { atMs: 60, detections: [hoop, { kind: "ball" as const, x: 98, y: 82, width: 10, height: 10, confidence: 0.58 }] },
    { atMs: 120, detections: [hoop, { kind: "ball" as const, x: 101, y: 113, width: 10, height: 10, confidence: 0.72 }] },
  ];
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make"]);
});

test("waits through a near-rim bounce and records the centered drop as a make", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const frames = [
    { atMs: 0, detections: [hoop, { kind: "ball" as const, x: 118, y: 65, width: 10, height: 10, confidence: 0.9, sourceTrackId: 4 }] },
    { atMs: 70, detections: [hoop, { kind: "ball" as const, x: 120, y: 76, width: 10, height: 10, confidence: 0.9, sourceTrackId: 4 }] },
    { atMs: 140, detections: [hoop, { kind: "ball" as const, x: 122, y: 87, width: 10, height: 10, confidence: 0.88, sourceTrackId: 4 }] },
    { atMs: 210, detections: [hoop, { kind: "ball" as const, x: 126, y: 114, width: 10, height: 10, confidence: 0.84, sourceTrackId: 4 }] },
    { atMs: 330, detections: [hoop, { kind: "ball" as const, x: 119, y: 88, width: 10, height: 10, confidence: 0.8, sourceTrackId: 4 }] },
    { atMs: 470, detections: [hoop, { kind: "ball" as const, x: 104, y: 116, width: 10, height: 10, confidence: 0.82, sourceTrackId: 4 }] },
  ];
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make"]);
});

test("flushes a deferred near-rim miss when an imported clip ends", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  let state = createAttallaShotTrackerState();
  for (const frame of [
    { atMs: 0, detections: [hoop, { kind: "ball" as const, x: 116, y: 65, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 70, detections: [hoop, { kind: "ball" as const, x: 119, y: 76, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 140, detections: [hoop, { kind: "ball" as const, x: 121, y: 86, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 210, detections: [hoop, { kind: "ball" as const, x: 126, y: 114, width: 10, height: 10, confidence: 0.9 }] },
  ]) {
    state = stepAttallaShotTracker(state, frame.detections, frame.atMs).state;
  }
  const final = finalizeAttallaShotTracker(state, 300);
  assert.deepEqual(final.decisions.map((decision) => decision.kind), ["miss"]);
});

test("counts two independently tracked makes in a rapid passing drill", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const shot = (start: number) => [
    { atMs: start, detections: [hoop, { kind: "ball" as const, x: 94, y: 65, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: start + 60, detections: [hoop, { kind: "ball" as const, x: 97, y: 74, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: start + 120, detections: [hoop, { kind: "ball" as const, x: 99, y: 83, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: start + 180, detections: [hoop, { kind: "ball" as const, x: 101, y: 113, width: 10, height: 10, confidence: 0.9 }] },
  ];
  const frames = [...shot(0), ...shot(820)];
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make", "make"]);
});

test("preserves a second in-flight attempt when the first ball scores", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 8, confidence: 0.98,
  };
  const ball = (
    sourceTrackId: number,
    x: number,
    y: number,
  ): AttallaObjectDetection => ({
    kind: "ball", x, y, width: 10, height: 10, confidence: 0.9, sourceTrackId,
  });
  const frames = [
    { atMs: 0, detections: [hoop, ball(1, 94, 65)] },
    { atMs: 60, detections: [hoop, ball(1, 97, 75)] },
    { atMs: 120, detections: [hoop, ball(1, 99, 85), ball(2, 92, 58)] },
    { atMs: 180, detections: [hoop, ball(1, 101, 108), ball(2, 94, 65)] },
    { atMs: 240, detections: [hoop, ball(2, 96, 72)] },
    { atMs: 500, detections: [hoop, ball(2, 98, 80)] },
    { atMs: 700, detections: [hoop, ball(2, 99, 88)] },
    { atMs: 820, detections: [hoop, ball(2, 101, 108)] },
  ];
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make", "make"]);
});

test("does not count the completed ball again while it rebounds below the rim", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const frames = [
    { atMs: 0, detections: [hoop, { kind: "ball" as const, x: 94, y: 65, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 60, detections: [hoop, { kind: "ball" as const, x: 97, y: 74, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 120, detections: [hoop, { kind: "ball" as const, x: 99, y: 83, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 180, detections: [hoop, { kind: "ball" as const, x: 101, y: 113, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 240, detections: [hoop, { kind: "ball" as const, x: 112, y: 124, width: 10, height: 10, confidence: 0.9 }] },
    { atMs: 300, detections: [hoop, { kind: "ball" as const, x: 126, y: 119, width: 10, height: 10, confidence: 0.9 }] },
  ];
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make"]);
});

test("keeps simultaneous basketballs on separate tracks regardless of detection order", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  let state = createAttallaShotTrackerState();
  for (const frame of [
    { atMs: 0, balls: [{ x: 91, y: 128 }, { x: 108, y: 127 }] },
    { atMs: 50, balls: [{ x: 108, y: 127 }, { x: 91, y: 128 }] },
    { atMs: 100, balls: [{ x: 91, y: 128 }, { x: 108, y: 127 }] },
  ]) {
    const step = stepAttallaShotTracker(
      state,
      [
        ...frame.balls.map((position) => ({
          kind: "ball" as const,
          ...position,
          width: 10,
          height: 10,
          confidence: 0.9,
        })),
        hoop,
      ],
      frame.atMs,
    );
    state = step.state;
  }
  assert.equal(state.balls.length, 2);
  assert.ok(state.balls.every((tracked) => tracked.detections.length === 3));
});

test("tracks a made shot through a stationary ball-return rack", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const rack = [
    { x: 84, y: 127 },
    { x: 99, y: 128 },
    { x: 114, y: 129 },
  ];
  const shot = [
    { x: 94, y: 64 },
    { x: 96, y: 72 },
    { x: 98, y: 82 },
    { x: 101, y: 112 },
  ];
  const frames = shot.map((position, index) => ({
    atMs: index * 70,
    detections: [
      ...rack.map((rackBall) => ({
        kind: "ball" as const,
        ...rackBall,
        width: 10,
        height: 10,
        confidence: 0.9,
      })),
      {
        kind: "ball" as const,
        ...position,
        width: 10,
        height: 10,
        confidence: 0.9,
      },
      hoop,
    ],
  }));
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make"]);
});

test("uses upstream track identity through a fast pass beside rack balls", () => {
  const hoop: AttallaObjectDetection = {
    kind: "hoop", x: 100, y: 100, width: 40, height: 18, confidence: 0.9,
  };
  const rack = [84, 99, 114];
  const shot = [
    { x: 93, y: 62 },
    { x: 97, y: 76 },
    { x: 100, y: 89 },
    { x: 102, y: 113 },
  ];
  const frames = shot.map((position, index) => ({
    atMs: index * 55,
    detections: [
      ...rack.map((x, rackIndex) => ({
        kind: "ball" as const,
        x,
        y: 127 + rackIndex,
        width: 10,
        height: 10,
        confidence: 0.9,
        sourceTrackId: 20 + rackIndex,
      })),
      {
        kind: "ball" as const,
        ...position,
        width: 10,
        height: 10,
        confidence: 0.9,
        sourceTrackId: 7,
      },
      hoop,
    ].reverse(),
  }));
  assert.deepEqual(runFrames(frames).map((decision) => decision.kind), ["make"]);
});

test("matches the three-make ground truth on the exact failed upload", () => {
  const fixturePath = path.join(
    process.cwd(),
    "tests",
    "fixtures",
    "attalla-reference-detections.json",
  );
  const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as DetectionFixture;
  assert.equal(
    fixture.sourceSha256,
    "6EC656830BE58954CFB3CB75BAE671C8C1FC549240B0ACE7153E4AA852466358",
  );
  const frameWidth = 2_304;
  const frameHeight = 1_440;
  const firstHoop = fixture.frames
    .flatMap((frame) => frame.detections)
    .find((detection) => detection.kind === "hoop");
  assert.ok(firstHoop);
  const rim = rimFromAutomaticHoop(
    {
      left: (firstHoop.x - firstHoop.width / 2) * frameWidth,
      top: (firstHoop.y - firstHoop.height / 2) * frameHeight,
      right: (firstHoop.x + firstHoop.width / 2) * frameWidth,
      bottom: (firstHoop.y + firstHoop.height / 2) * frameHeight,
      confidence: firstHoop.confidence,
    },
    frameWidth,
    frameHeight,
  );
  const scoringBox = calibratedRimToScoringHoop(rim, frameWidth, frameHeight);
  const scoringHoop: AttallaObjectDetection = {
    kind: "hoop",
    x: (scoringBox.left + scoringBox.right) / 2,
    y: (scoringBox.top + scoringBox.bottom) / 2,
    width: scoringBox.right - scoringBox.left,
    height: scoringBox.bottom - scoringBox.top,
    confidence: scoringBox.confidence,
  };
  const frames = fixture.frames.map((frame) => ({
    atMs: frame.atMs,
    detections: [
      ...frame.detections
        .filter((detection) => detection.kind === "ball")
        .map((detection) => ({
          ...detection,
          x: detection.x * frameWidth,
          y: detection.y * frameHeight,
          width: detection.width * frameWidth,
          height: detection.height * frameHeight,
        })),
      scoringHoop,
    ],
  }));
  const decisions = runFrames(frames);
  assert.equal(decisions.filter((decision) => decision.kind === "make").length, fixture.groundTruth.makes);
  assert.equal(decisions.filter((decision) => decision.kind === "miss").length, fixture.groundTruth.misses);
  assert.ok(decisions.every((decision) => decision.confidence >= 0.86));
});
