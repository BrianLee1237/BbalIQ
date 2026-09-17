import type {
  BallDetection,
  RimCalibration,
  VideoShotDecision,
} from "../../types/tracking";
import {
  createTrackerEngineState,
  MIN_AUTOMATIC_DECISION_CONFIDENCE,
  stepTracker,
} from "./engine";

/**
 * A decoded-video observation. The rim is stored with every ball sample so
 * the trajectory remains rim-relative if the hoop lock is corrected.
 */
export interface VideoTrajectorySample {
  atSeconds: number;
  ball: BallDetection;
  rim: RimCalibration;
}

interface RelativeSample extends VideoTrajectorySample {
  /** Ball center in rim widths: 0 is the left edge and 1 is the right edge. */
  rimX: number;
  /** Ball center measured vertically from the rim plane. */
  rimY: number;
}

const MAX_CONTIGUOUS_GAP_MS = 560;

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[middle] ?? 0;
  return ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}

function verticalScale(rim: RimCalibration): number {
  // A very thin manual box should not amplify one pixel of vertical jitter.
  return Math.max(rim.height, rim.width * 0.2, 0.006);
}

function toRelative(sample: VideoTrajectorySample): RelativeSample {
  const rimWidth = Math.max(0.001, sample.rim.width);
  const rimPlaneY = sample.rim.y + sample.rim.height * 0.48;
  return {
    ...sample,
    rimX: (sample.ball.x - sample.rim.x) / rimWidth,
    rimY: (sample.ball.y - rimPlaneY) / verticalScale(sample.rim),
  };
}

function splitContinuousRuns(samples: RelativeSample[]): RelativeSample[][] {
  const runs: RelativeSample[][] = [];
  let run: RelativeSample[] = [];
  for (const sample of samples) {
    const previous = run.at(-1);
    const gapMs = previous ? (sample.atSeconds - previous.atSeconds) * 1_000 : 0;
    const jump = previous
      ? Math.hypot(sample.rimX - previous.rimX, sample.rimY - previous.rimY)
      : 0;
    const discontinuous = Boolean(previous) && (
      gapMs <= 0 ||
      gapMs > MAX_CONTIGUOUS_GAP_MS ||
      (gapMs < 220 && jump > 4.4)
    );
    if (discontinuous) {
      if (run.length >= 2) runs.push(run);
      run = [];
    }
    run.push(sample);
  }
  if (run.length >= 2) runs.push(run);
  return runs;
}

function smoothRun(run: RelativeSample[]): RelativeSample[] {
  return run.map((sample, index) => {
    const local = run.filter((candidate, candidateIndex) =>
      Math.abs(candidateIndex - index) <= 1 &&
      Math.abs(candidate.atSeconds - sample.atSeconds) <= 0.12,
    );
    return {
      ...sample,
      rimX: median(local.map((candidate) => candidate.rimX)),
      rimY: median(local.map((candidate) => candidate.rimY)),
    };
  });
}

function smoothedDetection(sample: RelativeSample): BallDetection {
  const rimPlaneY = sample.rim.y + sample.rim.height * 0.48;
  return {
    ...sample.ball,
    x: sample.rim.x + sample.rimX * sample.rim.width,
    y: rimPlaneY + sample.rimY * verticalScale(sample.rim),
    at: Math.round(sample.atSeconds * 1_000),
  };
}

/**
 * Cleans imported-video tracks with rim-relative median smoothing and then
 * runs the exact same pure shot state machine used by live ActiveTrack. The
 * state machine fits/interpolates the descending rim crossing and delays a
 * miss until the below-net path proves the ball exited outside the opening.
 */
export function classifyVideoTrajectories(
  samples: VideoTrajectorySample[],
): VideoShotDecision[] {
  const ordered = [...samples]
    .filter((sample) =>
      Number.isFinite(sample.atSeconds) &&
      Number.isFinite(sample.ball.x) &&
      Number.isFinite(sample.ball.y),
    )
    .sort((left, right) => left.atSeconds - right.atSeconds)
    .map(toRelative);
  const runs = splitContinuousRuns(ordered).map(smoothRun);
  const decisions: VideoShotDecision[] = [];
  let lastShotAt = Number.NEGATIVE_INFINITY;

  for (const run of runs) {
    let state = { ...createTrackerEngineState(), lastShotAt };
    for (const sample of run) {
      const timestamp = Math.round(sample.atSeconds * 1_000);
      const result = stepTracker(state, smoothedDetection(sample), sample.rim, timestamp);
      state = result.state;
      if (!result.shot) continue;
      decisions.push({
        id: `${timestamp}-${decisions.length}`,
        atSeconds: sample.atSeconds,
        suggestedKind: result.shot,
        finalKind: result.confidence >= MIN_AUTOMATIC_DECISION_CONFIDENCE
          ? result.shot
          : null,
        confidence: result.confidence,
        reason: result.reason,
      });
      lastShotAt = state.lastShotAt;
    }
    lastShotAt = Math.max(lastShotAt, state.lastShotAt);
  }

  return decisions;
}
