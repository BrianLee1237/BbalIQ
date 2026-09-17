import fs from "node:fs";
import path from "node:path";
import { Tracker } from "byte-track-ts";
import { createAttallaShotTrackerState, stepAttallaShotTracker, finalizeAttallaShotTracker } from "../src/tracking/attallaShotTracker";
import { calibratedRimToScoringHoop, filterPlausibleBasketballBoxes, mergeLearnedFrames,
  mergeTrackedAndRawBoxes, toByteTrackDetections, trackRowToPixelBox, type PixelBox } from "../src/vision/learnedTracking";
import { BALL_TRACKER_SETTINGS } from "../src/vision/trackerSettings";
import type { LearnedBasketballFrame, LearnedObjectDetection } from "../src/vision/learnedBasketballDetector.web";
import type { RimCalibration } from "../types/tracking";

interface Trace {
  source: string; width: number; height: number; rim: RimCalibration;
  frames: { atMs: number; full: LearnedObjectDetection[]; focused: LearnedObjectDetection[] }[];
}
function learned(objects: LearnedObjectDetection[]): LearnedBasketballFrame {
  return { objects, basketballs: objects.filter(o => o.label === "ball"),
    hoops: objects.filter(o => o.label === "hoop"), players: [] };
}
const directory = process.argv[2] ?? "work/recording-audit";
const report = [];
for (const file of fs.readdirSync(directory).filter(name => /^clip-\d+\.json$/.test(name))) {
  const trace: Trace = JSON.parse(fs.readFileSync(path.join(directory, file), "utf8"));
  const modes = [];
  for (const useCrop of [false, true]) {
    const association = new Tracker(BALL_TRACKER_SETTINGS);
    let state = createAttallaShotTrackerState();
    const decisions = [];
    let detectedFrames = 0;
    const hoop = calibratedRimToScoringHoop(trace.rim, trace.width, trace.height);
    for (const frame of trace.frames) {
      const objects = mergeLearnedFrames([learned(frame.full), ...(useCrop ? [learned(frame.focused)] : [])]);
      const raw = filterPlausibleBasketballBoxes(objects.basketballs, trace.rim, trace.width, trace.height);
      if (raw.length) detectedFrames++;
      const tracked = association.update(toByteTrackDetections(raw))
        .map(trackRowToPixelBox).filter((b): b is PixelBox => b !== null);
      const balls = mergeTrackedAndRawBoxes(tracked, raw);
      const result = stepAttallaShotTracker(state, [
        ...balls.map(b => ({ kind: "ball" as const, x: (b.left+b.right)/2,
          y: (b.top+b.bottom)/2, width: b.right-b.left, height: b.bottom-b.top,
          confidence: b.confidence, sourceTrackId: b.trackId })),
        { kind: "hoop", x: (hoop.left+hoop.right)/2, y: (hoop.top+hoop.bottom)/2,
          width: hoop.right-hoop.left, height: hoop.bottom-hoop.top, confidence: hoop.confidence },
      ], frame.atMs);
      state = result.state;
      decisions.push(...result.decisions);
    }
    decisions.push(...finalizeAttallaShotTracker(state, trace.frames.at(-1)?.atMs ?? 0).decisions);
    modes.push({ mode: useCrop ? "full-frame-and-original-pixel-crop" : "full-frame-only",
      detectedFrames, makes: decisions.filter(d => d.kind === "make" && d.confidence >= .86).length,
      misses: decisions.filter(d => d.kind === "miss" && d.confidence >= .86).length,
      reviews: decisions.filter(d => d.confidence < .86).length,
      events: decisions.map(d => ({ atSeconds: d.atMs/1000, kind: d.kind, confidence: d.confidence })) });
  }
  report.push({ clip: file, source: path.basename(trace.source), frames: trace.frames.length, modes });
}
fs.writeFileSync(path.join(directory, "report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
