import assert from "node:assert/strict";
import test from "node:test";
import { createAttallaShotTrackerState, stepAttallaShotTracker,
  type AttallaObjectDetection } from "../src/tracking/attallaShotTracker";
import { sourceCropGeometry } from "../src/vision/sourceCrop";
import { validateVideoRimCalibration } from "../src/vision/videoAnalysisPolicy";

const hoop: AttallaObjectDetection = {
  kind: "hoop", x: 100, y: 100, width: 40, height: 8, confidence: 0.95,
};
function ball(x: number, y: number, confidence = 0.9): AttallaObjectDetection {
  return { kind: "ball", x, y, width: 10, height: 10, confidence, sourceTrackId: 7 };
}
function approach() {
  let state = createAttallaShotTrackerState();
  for (const [at, y] of [[0, 65], [50, 75], [100, 85]]) {
    state = stepAttallaShotTracker(state, [hoop, ball(100, y!)], at!).state;
  }
  return state;
}

test("recovers a weak below-net detection only after a tracked approach", () => {
  const recovered = stepAttallaShotTracker(approach(), [hoop, ball(100, 116, 0.22)], 200);
  assert.deepEqual(recovered.decisions.map(d => d.kind), ["make"]);
  const untracked = stepAttallaShotTracker(createAttallaShotTrackerState(),
    [hoop, ball(100, 116, 0.22)], 200);
  assert.equal(untracked.state.balls.length, 0);
});

test("a recycled tracker ID cannot teleport a ball through the hoop", () => {
  const result = stepAttallaShotTracker(approach(), [hoop, ball(100, 900)], 133);
  assert.equal(result.decisions.length, 0);
});

test("a stale below-net track cannot steal a fresh shot's exit", () => {
  let state = createAttallaShotTrackerState();
  for (const [at, y] of [[500,65], [550,75], [600,85]]) {
    state = stepAttallaShotTracker(state, [hoop, {...ball(100,y!), sourceTrackId:undefined}], at!).state;
  }
  state.balls.push({ id: state.nextBallId++, detections: [0,50].map(atMs =>
    ({...ball(100,116), sourceTrackId:undefined, atMs})) });
  const result = stepAttallaShotTracker(state,
    [hoop, {...ball(100,116), sourceTrackId:undefined}], 650);
  assert.deepEqual(result.decisions.map(d => d.kind), ["make"]);
});

test("duplicate and backward frames cannot manufacture an approach", () => {
  const initial = stepAttallaShotTracker(createAttallaShotTrackerState(), [hoop, ball(100, 65)], 100).state;
  for (const at of [100, 99, NaN]) {
    const result = stepAttallaShotTracker(initial, [hoop, ball(100, 85)], at);
    assert.equal(result.state, initial);
    assert.equal(result.decisions.length, 0);
  }
});

test("rejects nonfinite detector confidence", () => {
  const result = stepAttallaShotTracker(createAttallaShotTrackerState(),
    [hoop, ball(100, 65, NaN), ball(90, 70, Infinity)], 0);
  assert.equal(result.state.balls.length, 0);
});

test("long unseen rim passages remain reviewable", () => {
  const result = stepAttallaShotTracker(approach(), [hoop, ball(100, 150)], 800);
  assert.equal(result.decisions.length, 1);
  assert.ok(result.decisions[0]!.confidence < 0.86);
});

test("a continuously observed rim rattle is not a long unseen interval", () => {
  let state = approach();
  for (const at of [200,300,400,500,600,700]) {
    state = stepAttallaShotTracker(state, [hoop, ball(100, 99)], at).state;
  }
  const result = stepAttallaShotTracker(state, [hoop, ball(100, 116)], 800);
  assert.equal(result.decisions.length, 1);
  assert.equal(result.decisions[0]!.kind, "make");
  assert.ok(result.decisions[0]!.confidence >= .86);
});

test("make and adjacent miss geometry survives scale, mirror and translation", () => {
  for (const scale of [0.15, 0.3, 1, 3, 8]) {
    for (const mirror of [-1, 1]) {
      for (const offset of [0, 40]) {
        let state = createAttallaShotTrackerState();
        const decisions = [];
        for (const [i, y] of [65, 75, 85, 116].entries()) {
          const transform = (d: AttallaObjectDetection) => ({ ...d,
            x: 500 + d.x * scale * mirror, y: 200 + d.y * scale,
            width: d.width * scale, height: d.height * scale });
          const result = stepAttallaShotTracker(state,
            [transform(hoop), transform(ball(100 + offset, y))], i * 50);
          state = result.state;
          decisions.push(...result.decisions);
        }
        assert.deepEqual(decisions.map(d => d.kind), [offset === 0 ? "make" : "miss"]);
      }
    }
  }
});

test("native-resolution crops preserve detector coordinates in landscape and portrait", () => {
  for (const [aw, ah, sw, sh] of [[640, 360, 3840, 2160], [360, 640, 1080, 1920], [640, 400, 2304, 1440]]) {
    const region = { left: aw! * 0.25, top: ah! * 0.2, width: aw! * 0.25, height: ah! * 0.4 };
    const crop = sourceCropGeometry(region, aw!, ah!, sw!, sh!);
    assert.ok(Math.max(crop.outputWidth, crop.outputHeight) <= 640);
    assert.ok(crop.outputWidth > region.width);
    assert.ok(Math.abs(crop.outputWidth * 0.5 * crop.toAnalysisX + region.left - aw! * 0.375) < 1e-8);
    assert.ok(Math.abs(crop.outputHeight * crop.toAnalysisY - region.height) < 1e-8);
    assert.ok(Math.abs(crop.width / sw! - region.width / aw!) < 1e-8);
  }
});

test("a small high-resolution rim stays valid without widening its calibration", () => {
  const rim = { x: 0.6, y: 0.2, width: 0.015, height: 0.006 };
  assert.equal(validateVideoRimCalibration(rim, 16 / 9, 3840), null);
  assert.match(validateVideoRimCalibration(rim, 16 / 9, 640) ?? "", /detail/);
});

test("concurrently tracked balls can score inside the fragmentation cooldown", () => {
  let state = createAttallaShotTrackerState();
  const decisions = [];
  for (const [at, firstY, secondY] of [
    [0,65,25], [50,75,35], [100,85,45], [150,116,55],
    [250,0,65], [350,0,85], [450,0,116],
  ]) {
    const detections = [hoop, { ...ball(100, secondY!), sourceTrackId: 9 }];
    if (firstY) detections.push(ball(100, firstY));
    const result = stepAttallaShotTracker(state, detections, at!);
    state = result.state;
    decisions.push(...result.decisions);
  }
  assert.deepEqual(decisions.map(d => d.kind), ["make", "make"]);
});
