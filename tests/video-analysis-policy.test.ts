import assert from "node:assert/strict";
import test from "node:test";

import {
  applyVideoQualityGate,
  buildVideoAnalysisDiagnostics,
  buildVideoFrameTimes,
  consolidateVideoShotDecisions,
  createVideoStabilityState,
  IMPORT_ANALYSIS_FPS,
  resolveVideoSampleTiming,
  stepVideoStability,
  validateVideoRimCalibration,
} from "../src/vision/videoAnalysisPolicy";
import type { VideoShotDecision } from "../types/tracking";

test("builds ordered offline samples at the accuracy sampling rate", () => {
  const times = buildVideoFrameTimes(1);
  assert.equal(IMPORT_ANALYSIS_FPS, 30);
  assert.equal(times.length, IMPORT_ANALYSIS_FPS + 2);
  assert.ok(times.every((time, index) => index === 0 || time > (times[index - 1] ?? -1)));
  assert.ok(times.every((time) => time >= 0 && time < 1));
  assert.equal(times[0], 0);
  assert.ok((times.at(-1) ?? 0) >= 0.995);
});

test("keeps the decisive third-shot window in the dense upload cadence", () => {
  const times = buildVideoFrameTimes(21.85);
  const decisiveBelowNetFrame = 19.348;
  assert.ok(
    times.some((time) => Math.abs(time - decisiveBelowNetFrame) <= 1 / IMPORT_ANALYSIS_FPS),
  );
});

test("uses actual decoded frame time and rejects duplicate frames", () => {
  const first = resolveVideoSampleTiming(0.1, 0.133, null);
  assert.equal(first.timestampMs, 133);
  assert.equal(first.duplicate, false);

  const duplicate = resolveVideoSampleTiming(0.167, 0.133, first.timestampMs);
  assert.equal(duplicate.duplicate, true);
});

test("accepts a tight wide rim box and rejects implausible calibration", () => {
  assert.equal(
    validateVideoRimCalibration(
      { x: 0.65, y: 0.2, width: 0.18, height: 0.055 },
      16 / 9,
    ),
    null,
  );
  assert.match(
    validateVideoRimCalibration(
      { x: 0.65, y: 0.2, width: 0.06, height: 0.1 },
      16 / 9,
    ) ?? "",
    /wider/,
  );
  assert.match(
    validateVideoRimCalibration({ x: 0.8, y: 0.2, width: 0.3, height: 0.05 }) ?? "",
    /inside/,
  );
});

test("forces manual review when decoded timing is unreliable", () => {
  const diagnostics = buildVideoAnalysisDiagnostics(60, 40, 100);
  assert.equal(diagnostics.requiresFullReview, true);
  const decision: VideoShotDecision = {
    id: "shot-1",
    atSeconds: 1.2,
    suggestedKind: "make",
    finalKind: "make",
    confidence: 0.95,
    reason: "rim-crossing",
  };
  assert.equal(applyVideoQualityGate([decision], diagnostics)[0]?.finalKind, null);
});

test("collapses a tentative event followed by a confident result for the same attempt", () => {
  const tentative: VideoShotDecision = {
    id: "tentative",
    atSeconds: 4.1,
    suggestedKind: "make",
    finalKind: null,
    confidence: 0.81,
    reason: "rim-entry-lost",
  };
  const confirmed: VideoShotDecision = {
    id: "confirmed",
    atSeconds: 6.2,
    suggestedKind: "miss",
    finalKind: "miss",
    confidence: 0.96,
    reason: "airball",
  };
  assert.deepEqual(consolidateVideoShotDecisions([tentative, confirmed]), [confirmed]);
});

test("keeps a tentative rapid attempt after an already confirmed make", () => {
  const confirmed: VideoShotDecision = {
    id: "confirmed-first",
    atSeconds: 4.1,
    suggestedKind: "make",
    finalKind: "make",
    confidence: 0.94,
    reason: "rim-crossing",
  };
  const nextAttempt: VideoShotDecision = {
    id: "next-review",
    atSeconds: 5.2,
    suggestedKind: "make",
    finalKind: null,
    confidence: 0.82,
    reason: "rim-entry-lost",
  };
  assert.deepEqual(
    consolidateVideoShotDecisions([confirmed, nextAttempt]),
    [confirmed, nextAttempt],
  );
});

test("keeps automatic decisions when unique frame timing is continuous", () => {
  const diagnostics = buildVideoAnalysisDiagnostics(150, 0, 67);
  assert.equal(diagnostics.requiresFullReview, false);
  assert.deepEqual(diagnostics.warnings, []);
});

test("requires repeated broad motion before declaring a camera move", () => {
  let state = createVideoStabilityState();
  state = stepVideoStability(state, 0.42);
  assert.equal(state.cameraMotionEvents, 0);
  state = stepVideoStability(state, 0.44);
  assert.equal(state.cameraMotionEvents, 1);
});

test("an abrupt camera cut immediately disables automatic scoring", () => {
  const moved = stepVideoStability(createVideoStabilityState(), 0.8);
  const diagnostics = buildVideoAnalysisDiagnostics(150, 0, 67, moved.cameraMotionEvents);
  assert.equal(diagnostics.requiresFullReview, true);
  assert.match(diagnostics.warnings.join(" "), /Camera movement/);
});
