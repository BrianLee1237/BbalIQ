import type {
  RimCalibration,
  VideoAnalysisDiagnostics,
  VideoShotDecision,
} from "../../types/tracking";

// Uploaded footage needs a dense pass around the rim. A 15 FPS cadence can
// skip the only decoded frame that proves the ball exited below the net, which
// caused a verified three-make clip to return only two makes. Thirty requested
// samples per second covers every frame in typical 30 FPS footage; duplicate
// decoded frames are still rejected below using their actual timestamps.
export const IMPORT_ANALYSIS_FPS = 30;
export const MAX_IMPORT_DURATION_SECONDS = 300;
const DUPLICATE_TIMESTAMP_TOLERANCE_MS = 2;
const MAX_RELIABLE_FRAME_GAP_MS = 350;
// Oversampling a legitimate 24/25 FPS source at the 30 FPS accuracy cadence
// naturally returns some duplicate decoded frames. Keep that case valid while
// still rejecting severely damaged or very-low-frame-rate footage.
const MAX_DUPLICATE_SAMPLE_RATIO = 0.28;
const FRAME_END_GUARD_SECONDS = 0.004;

export interface VideoSampleTiming {
  timestampMs: number;
  atSeconds: number;
  gapMs: number;
  duplicate: boolean;
}

export interface VideoStabilityState {
  highMotionStreak: number;
  cooldownFrames: number;
  cameraMotionEvents: number;
}

export interface VideoTrackingQuality {
  rimTrackedFrames?: number;
  rimTrackingLostFrames?: number;
  averageRimTrackingConfidence?: number;
  rimGlobalReacquisitions?: number;
  ballCandidateFrames?: number;
  ballTrackedFrames?: number;
  learnedBallDetectionFrames?: number;
  learnedHoopDetectionFrames?: number;
  learnedPlayerDetectionFrames?: number;
  playerTrackedFrames?: number;
  learnedDetectorBackend?: "webgpu" | "wasm" | "native";
}

export function createVideoStabilityState(): VideoStabilityState {
  return { highMotionStreak: 0, cooldownFrames: 0, cameraMotionEvents: 0 };
}

export function stepVideoStability(
  state: VideoStabilityState,
  changedPixelRatio: number,
): VideoStabilityState {
  const cooldownFrames = Math.max(0, state.cooldownFrames - 1);
  const highMotion = changedPixelRatio >= 0.38;
  const highMotionStreak = highMotion ? state.highMotionStreak + 1 : 0;
  const cameraMotion = cooldownFrames === 0 &&
    (changedPixelRatio >= 0.72 || highMotionStreak >= 2);
  return {
    highMotionStreak: cameraMotion ? 0 : highMotionStreak,
    cooldownFrames: cameraMotion ? IMPORT_ANALYSIS_FPS : cooldownFrames,
    cameraMotionEvents: state.cameraMotionEvents + (cameraMotion ? 1 : 0),
  };
}

export function buildVideoFrameTimes(durationSeconds: number): number[] {
  const safeDuration = Math.max(0.02, durationSeconds);
  const frameCount = Math.max(1, Math.floor(safeDuration * IMPORT_ANALYSIS_FPS));
  const centeredSamples = Array.from({ length: frameCount }, (_, index) =>
    Math.max(
      0,
      Math.min(
        safeDuration - FRAME_END_GUARD_SECONDS,
        (index + 0.5) / IMPORT_ANALYSIS_FPS,
      ),
    )
  );
  // Explicit boundary probes protect attempts already in flight at time zero
  // and a decisive below-net observation on the final decoded frame.
  return Array.from(new Set([
    0,
    ...centeredSamples,
    Math.max(0, safeDuration - FRAME_END_GUARD_SECONDS),
  ].map((time) => Math.round(time * 1_000_000) / 1_000_000))).sort(
    (left, right) => left - right,
  );
}

export function resolveVideoSampleTiming(
  requestedTimeSeconds: number,
  actualTimeSeconds: number | null | undefined,
  previousTimestampMs: number | null,
): VideoSampleTiming {
  const resolvedSeconds = Number.isFinite(actualTimeSeconds)
    ? Math.max(0, actualTimeSeconds ?? requestedTimeSeconds)
    : Math.max(0, requestedTimeSeconds);
  const timestampMs = Math.round(resolvedSeconds * 1_000);
  const gapMs = previousTimestampMs === null
    ? 0
    : Math.max(0, timestampMs - previousTimestampMs);
  return {
    timestampMs,
    atSeconds: resolvedSeconds,
    gapMs,
    duplicate: previousTimestampMs !== null &&
      timestampMs <= previousTimestampMs + DUPLICATE_TIMESTAMP_TOLERANCE_MS,
  };
}

export function validateVideoRimCalibration(
  rim: RimCalibration,
  frameAspectRatio = 1,
  sourceWidth = 640,
): string | null {
  const values = [rim.x, rim.y, rim.width, rim.height];
  if (values.some((value) => !Number.isFinite(value))) {
    return "Draw the rim box again.";
  }
  if (
    rim.x < 0 ||
    rim.y < 0 ||
    rim.width <= 0 ||
    rim.height <= 0 ||
    rim.x + rim.width > 1.001 ||
    rim.y + rim.height > 1.001
  ) {
    return "Keep the complete rim box inside the video frame.";
  }
  if (rim.width * sourceWidth < 12 ||
      rim.height * sourceWidth / Math.max(0.01, frameAspectRatio) < 2) {
    return "The rim has too little image detail. Use a closer or higher-resolution recording; keep the box tight around the opening.";
  }
  if (rim.width > 0.5 || rim.height > 0.28) {
    return "The rim box is too large. Include only the hoop opening.";
  }
  // Rim coordinates are normalized to the frame. Convert them back to pixel
  // proportions before judging their shape or a landscape video makes a
  // correctly drawn wide box look artificially narrow.
  const aspectRatio = (rim.width * Math.max(0.01, frameAspectRatio)) / rim.height;
  if (aspectRatio < 1.4) {
    return "The rim box should be wider than it is tall.";
  }
  if (aspectRatio > 14) {
    return "The rim box is too thin. Include the full height of the opening.";
  }
  return null;
}

export function consolidateVideoShotDecisions(
  decisions: VideoShotDecision[],
  maximumSameAttemptGapMs = 2_400,
): VideoShotDecision[] {
  const consolidated: VideoShotDecision[] = [];
  for (const decision of decisions) {
    const previous = consolidated.at(-1);
    if (
      previous &&
      (decision.atSeconds - previous.atSeconds) * 1_000 <= maximumSameAttemptGapMs &&
      previous.finalKind === null
    ) {
      // A tentative entry/lost-ball review followed by a confident exit or
      // miss can belong to one physical attempt. A tentative event *after* a
      // confirmed shot is never swallowed: in a rapid drill it may be the next
      // attempt, and hiding it made the displayed shot total too small.
      if (
        decision.finalKind !== null ||
        (previous.finalKind === null && decision.confidence > previous.confidence)
      ) {
        consolidated[consolidated.length - 1] = decision;
      }
      continue;
    }
    consolidated.push(decision);
  }
  return consolidated;
}

export function buildVideoAnalysisDiagnostics(
  framesAnalyzed: number,
  duplicateFramesSkipped: number,
  largestFrameGapMs: number,
  cameraMotionEvents = 0,
  tracking: VideoTrackingQuality = {},
): VideoAnalysisDiagnostics {
  const totalSamples = framesAnalyzed + duplicateFramesSkipped;
  const duplicateRatio = totalSamples > 0
    ? duplicateFramesSkipped / totalSamples
    : 0;
  const warnings: string[] = [];
  if (duplicateRatio > MAX_DUPLICATE_SAMPLE_RATIO) {
    warnings.push(
      "The video decoder returned too many repeated or unreadable frames, so every detected shot requires review.",
    );
  }
  if (largestFrameGapMs > MAX_RELIABLE_FRAME_GAP_MS) {
    warnings.push(
      "The video contains a long frame gap, so every detected shot requires review.",
    );
  }
  const rimTrackedFrames = tracking.rimTrackedFrames ?? 0;
  const rimTrackingLostFrames = tracking.rimTrackingLostFrames ?? 0;
  const rimTrackingSamples = rimTrackedFrames + rimTrackingLostFrames;
  const rimLostRatio = rimTrackingSamples > 0
    ? rimTrackingLostFrames / rimTrackingSamples
    : 1;
  const rimTrackingVerified =
    rimTrackingSamples > 0 &&
    rimLostRatio <= 0.12 &&
    (tracking.averageRimTrackingConfidence ?? 0) >= 0.56;
  if (cameraMotionEvents > 0 && !rimTrackingVerified) {
    warnings.push(
      "Camera movement or a cut exceeded the hoop track, so every detected shot requires review.",
    );
  }
  if (rimTrackingSamples > 0 && rimLostRatio > 0.12) {
    warnings.push(
      "The hoop could not be locked in enough frames, so every detected shot requires review.",
    );
  }
  return {
    analysisFps: IMPORT_ANALYSIS_FPS,
    duplicateFramesSkipped,
    largestFrameGapMs,
    cameraMotionEvents,
    rimTrackedFrames,
    rimTrackingLostFrames,
    averageRimTrackingConfidence: tracking.averageRimTrackingConfidence ?? 0,
    rimGlobalReacquisitions: tracking.rimGlobalReacquisitions ?? 0,
    ballCandidateFrames: tracking.ballCandidateFrames ?? 0,
    ballTrackedFrames: tracking.ballTrackedFrames ?? 0,
    learnedBallDetectionFrames: tracking.learnedBallDetectionFrames ?? 0,
    learnedHoopDetectionFrames: tracking.learnedHoopDetectionFrames ?? 0,
    learnedPlayerDetectionFrames: tracking.learnedPlayerDetectionFrames ?? 0,
    playerTrackedFrames: tracking.playerTrackedFrames ?? 0,
    learnedDetectorBackend: tracking.learnedDetectorBackend ?? "native",
    requiresFullReview: warnings.length > 0,
    warnings,
  };
}

export function applyVideoQualityGate(
  decisions: VideoShotDecision[],
  diagnostics: VideoAnalysisDiagnostics,
): VideoShotDecision[] {
  if (!diagnostics.requiresFullReview) return decisions;
  return decisions.map((decision) => ({ ...decision, finalKind: null }));
}
