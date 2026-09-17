import assert from "node:assert/strict";
import test from "node:test";
import { predictBallFlight } from "../src/tracking/flightPrediction";
import {createVisionTrackState,selectTrackedBall,type VisionTrackState} from "../src/vision/ballTracker";
import {computeTrackDetectionRegion} from "../src/vision/trackDetectionRegion";
import {alignVisionTrackToRimShift} from "../src/vision/trackingAlignment";
import type {BallDetection} from "../types/tracking";
import {createAttallaShotTrackerState,stepAttallaShotTracker} from "../src/tracking/attallaShotTracker";

const rim = {x:0.6,y:0.3,width:0.15,height:0.04};
function observation(at: number, direction = 1, perspective = 0): BallDetection {
  const t = at/1000;
  const depth = 1+perspective*t;
  return {x:(0.4+direction*0.2*t)/depth, y:(0.4-0.6*t+1.1*t*t)/depth,
    width:0.025/depth,height:0.04/depth, at,confidence:0.9,
    motionConfidence:0.9,appearanceConfidence:0.9,trackId:4};
}
const history = [0,40,80,120,160,200,240].map(t=>observation(t));

test("temporal fits support vertical, mirrored and locally perspective-distorted arcs", () => {
  for (const direction of [-1,0,1]) for (const perspective of [-0.25,0,0.25]) {
    const samples = [0,35,80,125,160,200,240].map(t=>observation(t,direction,perspective));
    const result = predictBallFlight(samples,300);
    const expected = observation(300,direction,perspective);
    assert.ok(result);
    assert.ok(Math.abs(result.x-expected.x)<0.002);
    assert.ok(Math.abs(result.y-expected.y)<0.003);
  }
});

test("fits are stable across independent axis scales and positions", () => {
  const expected = predictBallFlight(history,300)!;
  for (const scaleX of [0.2,2,640]) for (const scaleY of [0.1,3,360]) {
    const result = predictBallFlight(history.map(p=>({...p,x:20+p.x*scaleX,y:40+p.y*scaleY,
      width:p.width*scaleX,height:p.height*scaleY})),300);
    assert.ok(result);
    assert.ok(Math.abs(result.x-(20+expected.x*scaleX))<1e-6);
    assert.ok(Math.abs(result.y-(40+expected.y*scaleY))<1e-6);
  }
});

test("one interior localization outlier cannot bend the prediction", () => {
  const dirty = history.map((p,i)=>i===2?{...p,x:p.x+0.14,y:p.y+0.13}:p);
  const result = predictBallFlight(dirty,300);
  assert.ok(result);
  assert.ok(Math.abs(result.x-observation(300).x)<1e-6);
  assert.ok(Math.abs(result.y-observation(300).y)<1e-6);
});

test("rejects impact discontinuities, long gaps, repeated timestamps and switched IDs", () => {
  assert.equal(predictBallFlight(history,500),null);
  assert.equal(predictBallFlight(history.slice(0,3),100),null);
  assert.equal(predictBallFlight(history.map((p,i)=>i===3?{...p,at:80}:p),300),null);
  assert.equal(predictBallFlight(history.map((p,i)=>i===3?{...p,trackId:99}:p),300),null);
  assert.equal(predictBallFlight(history.map((p,i)=>i===6?{...p,y:p.y-0.15}:p),300),null);
  assert.equal(predictBallFlight(history.map((p,i)=>i===6?{...p,width:NaN}:p),300),null);
});

function tracked(): VisionTrackState {
  return {previous:history.at(-2)!,current:history.at(-1)!,observedHistory:history,
    velocityX:0.2,velocityY:4,confirmedFrames:7,missingFrames:1};
}

test("curved fallback recovers an actual apex observation without synthesizing detections", () => {
  const current = tracked();
  const target = observation(360);
  // Deliberately stale linear velocity sends its search downward, away from the apex.
  const legacy = selectTrackedBall([target],{...current,observedHistory:undefined},rim,360);
  assert.equal(legacy.detection,null);
  const guided = selectTrackedBall([target],current,rim,360);
  assert.equal(guided.detection,target);
  const empty = selectTrackedBall([],current,rim,360);
  assert.equal(empty.detection,null);
  assert.equal(empty.state.observedHistory?.length,history.length);
});

test("ambiguous fallback does not pick between two balls and preserves a valid legacy match", () => {
  const first = observation(360);
  const second = {...first,x:first.x+0.005,trackId:undefined};
  assert.equal(selectTrackedBall([first,second],tracked(),rim,360).detection,null);
  const legacyMatch = {...first,y:0.8};
  const result = selectTrackedBall([legacyMatch,first],tracked(),rim,360);
  assert.equal(result.detection,legacyMatch);
});

test("curved vision fallback rejects a different known ball identity", () => {
  const result=selectTrackedBall([{...observation(360),trackId:99}],tracked(),rim,360);
  assert.equal(result.detection,null);
});

test("history resets on reacquisition and moves with a corrected rim", () => {
  const track = tracked();
  const shifted = alignVisionTrackToRimShift(track,0.03,-0.02);
  assert.equal(shifted.observedHistory![0]!.x,history[0]!.x+0.03);
  const next = {...observation(300),trackId:19};
  const result = selectTrackedBall([next],{...track,velocityY:0},rim,300);
  assert.equal(result.state.observedHistory?.length,1);
  assert.equal(createVisionTrackState().observedHistory,undefined);
});

test("curved detector search retains the complete old linear crop", () => {
  const oldRegion = computeTrackDetectionRegion({...tracked(),observedHistory:undefined},rim,640,360,300)!;
  const region = computeTrackDetectionRegion(tracked(),rim,640,360,300)!;
  assert.ok(region.left<=oldRegion.left && region.top<=oldRegion.top);
  assert.ok(region.left+region.width>=oldRegion.left+oldRegion.width);
  assert.ok(region.top+region.height>=oldRegion.top+oldRegion.height);
  const predicted=observation(300);
  assert.ok(predicted.x*640>=region.left && predicted.x*640<=region.left+region.width);
  assert.ok(predicted.y*360>=region.top && predicted.y*360<=region.top+region.height);
});

test("the shared detector tracker can reacquire a curved path without inventing a shot", () => {
  const state=createAttallaShotTrackerState();
  const samples=[0,40,80,120,160,200].map(atMs=>({
    kind:"ball" as const,x:100,y:100+2000*(atMs/1000-0.2)**2,
    width:5,height:5,confidence:0.9,atMs,
  }));
  state.balls=[{id:0,detections:samples}];state.nextBallId=1;state.lastFrameAt=200;
  const result=stepAttallaShotTracker(state,[{kind:"ball",x:100,y:128.8,
    width:5,height:5,confidence:0.9}],320);
  assert.equal(result.state.balls.length,1);
  assert.equal(result.state.balls[0]!.detections.length,7);
  assert.equal(result.decisions.length,0);
});

test("curved detector recovery cannot score a different tracked ball", () => {
  const state=createAttallaShotTrackerState();
  state.balls=[{id:0,sourceTrackId:4,detections:[0,40,80,120,160,200].map(atMs=>({
    kind:"ball" as const,x:100,y:100+2000*(atMs/1000-0.2)**2,
    width:5,height:5,confidence:0.9,atMs,sourceTrackId:4,
  }))}];
  state.hoops=[{id:0,kind:"hoop",x:100,y:120,width:15,height:2,confidence:0.95,atMs:200}];
  state.armedPairs=[{ballId:0,hoopId:0}];
  state.nextBallId=1;state.nextHoopId=1;state.lastFrameAt=200;
  const result=stepAttallaShotTracker(state,[{kind:"ball",x:100,y:128.8,
    width:5,height:5,confidence:0.9,sourceTrackId:99}],320);
  assert.equal(result.decisions.length,0);
  assert.equal(result.state.balls.find(b=>b.id===0)?.sourceTrackId,4);
});

test("curved detector recovery requires a mutually unique ball match", () => {
  for (const twoTracks of [false,true]) {
    const state=createAttallaShotTrackerState();
    state.balls=[0,...(twoTracks?[1]:[])].map(id=>({id,detections:[0,40,80,120,160,200].map(atMs=>({
      kind:"ball" as const,x:100+id*2,y:100+2000*(atMs/1000-0.2)**2,
      width:5,height:5,confidence:0.9,atMs,
    }))}));
    state.hoops=[{id:0,kind:"hoop",x:100,y:120,width:15,height:2,confidence:0.95,atMs:200}];
    state.armedPairs=state.balls.map(b=>({ballId:b.id,hoopId:0}));
    state.nextBallId=state.balls.length;state.nextHoopId=1;state.lastFrameAt=200;
    const result=stepAttallaShotTracker(state,(twoTracks?[101]:[98,102]).map(x=>({
      kind:"ball" as const,x,y:128.8,width:5,height:5,confidence:0.9,
    })),320);
    assert.equal(result.decisions.length,0);
    assert.equal(result.state.balls[0]!.detections.length,6);
  }
});
