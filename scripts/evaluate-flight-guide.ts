/** Cached detector replay: compares V22 selection with the additive V23 guide.
 * Not fresh inference, browser decoding, or a labeled accuracy benchmark.
 */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import {Tracker} from "byte-track-ts";
import {createVisionTrackState,selectTrackedBall} from "../src/vision/ballTracker";
import {predictBallFlight} from "../src/tracking/flightPrediction";
import {classifyVideoTrajectories,type VideoTrajectorySample} from "../src/tracking/videoTrajectoryClassifier";
import {mergeLearnedFrames,filterPlausibleBasketballBoxes,mergeTrackedAndRawBoxes,toByteTrackDetections,
  trackRowToPixelBox,pixelBoxToBallDetection,type PixelBox} from "../src/vision/learnedTracking";
import {BALL_TRACKER_SETTINGS} from "../src/vision/trackerSettings";
import {selectHoopZoneCandidates} from "../src/vision/hoopZone";
import type {RimCalibration} from "../types/tracking";
import type {LearnedObjectDetection} from "../src/vision/learnedBasketballDetector.web";

interface Trace {width:number;height:number;rim:RimCalibration;
  frames:{atMs:number;full:LearnedObjectDetection[];focused:LearnedObjectDetection[]}[]}
function learned(objects:LearnedObjectDetection[]) {
  return {objects,basketballs:objects.filter(o=>o.label==="ball"),hoops:objects.filter(o=>o.label==="hoop"),players:[]};
}
const dir=process.argv[2]??"work/recording-audit";
const report=[];
for (const file of fs.readdirSync(dir).filter(f=>/^clip-\d+\.json$/.test(f))) {
  const trace:Trace=JSON.parse(fs.readFileSync(path.join(dir,file),"utf8"));
  const tracker=new Tracker(BALL_TRACKER_SETTINGS);
  let legacy=createVisionTrackState(), guided=createVisionTrackState();
  const oldSamples:VideoTrajectorySample[]=[], newSamples:VideoTrajectorySample[]=[];
  let supportedPredictions=0, additiveRecoveries=0, disagreements=0;
  for (const frame of trace.frames) {
    const combined=mergeLearnedFrames([learned(frame.full),learned(frame.focused)]);
    const raw=filterPlausibleBasketballBoxes(combined.basketballs,trace.rim,trace.width,trace.height);
    const tracks=tracker.update(toByteTrackDetections(raw)).map(trackRowToPixelBox).filter((b):b is PixelBox=>b!==null);
    const candidates=selectHoopZoneCandidates(mergeTrackedAndRawBoxes(tracks,raw).map(b=>
      pixelBoxToBallDetection(b,trace.width,trace.height,frame.atMs)),trace.rim,trace.width,trace.height);
    // Counterfactual with identical state: adding a guide must never replace
    // a detection accepted by the legacy selector in that same state.
    const counterfactual=selectTrackedBall(candidates,{...guided,observedHistory:undefined},trace.rim,frame.atMs);
    const next=selectTrackedBall(candidates,guided,trace.rim,frame.atMs);
    if (counterfactual.detection) assert.equal(next.detection,counterfactual.detection);
    if (predictBallFlight(guided.observedHistory??[],frame.atMs)) supportedPredictions++;
    if (!counterfactual.detection && next.detection) additiveRecoveries++;
    const previous=selectTrackedBall(candidates,{...legacy,observedHistory:undefined},trace.rim,frame.atMs);
    if (previous.detection!==next.detection) disagreements++;
    legacy=previous.state; guided=next.state;
    if (previous.detection) oldSamples.push({atSeconds:frame.atMs/1000,ball:previous.detection,rim:trace.rim});
    if (next.detection) newSamples.push({atSeconds:frame.atMs/1000,ball:next.detection,rim:trace.rim});
  }
  const events=(samples:VideoTrajectorySample[])=>classifyVideoTrajectories(samples)
    .map(d=>({time:d.atSeconds,kind:d.finalKind,review:d.finalKind===null}));
  report.push({clip:file,frames:trace.frames.length,legacyTracked:oldSamples.length,guidedTracked:newSamples.length,
    supportedPredictions,additiveRecoveries,disagreements,legacyEvents:events(oldSamples),guidedEvents:events(newSamples)});
}
fs.writeFileSync(path.join(dir,"flight-guide-report.json"),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
