export type ShotKind = "make" | "miss";
export type ShotMethod = "tracked" | "manual" | "demo";

export interface BallDetection {
  /** Persistent detector/tracker identity when one is available. */
  trackId?: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  motionConfidence?: number;
  appearanceConfidence?: number;
  at: number;
}

export interface PlayerDetection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  at: number;
}

export interface RimCalibration {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ShotEvent {
  id: string;
  kind: ShotKind;
  method: ShotMethod;
  elapsedSeconds: number;
  createdAt: string;
}

export interface SessionRecord {
  id: string;
  startedAt: string;
  endedAt: string;
  durationSeconds: number;
  makes: number;
  misses: number;
  accuracy: number;
  recordingUri: string | null;
}

export interface TrackerEngineState {
  armed: boolean;
  armedAt: number;
  enteredRim: boolean;
  entryAt: number;
  entryX: number;
  entryConfidence: number;
  rimProximityAt: number;
  rimProximityX: number;
  rimProximityConfidence: number;
  rimProximityFrames: number;
  crossedRimAt: number;
  crossedRimX: number;
  crossedRimConfidence: number;
  lastDetectedAt: number;
  lastShotAt: number;
  previous: BallDetection | null;
  trajectory: BallDetection[];
  ascendingFrames: number;
  descendingFrames: number;
  apexY: number;
}

export interface TrackerStep {
  state: TrackerEngineState;
  shot: ShotKind | null;
  confidence: number;
  reason:
    | "none"
    | "rim-crossing"
    | "rim-entry-exit"
    | "rim-proximity-exit"
    | "rim-entry-lost"
    | "semantic-basket"
    | "evidence-fusion"
    | "evidence-conflict"
    | "airball"
    | "lost"
    | "timeout"
    | "cooldown";
}

export interface VideoShotDecision {
  id: string;
  atSeconds: number;
  suggestedKind: ShotKind;
  finalKind: ShotKind | null;
  confidence: number;
  reason: TrackerStep["reason"];
}

export interface VideoAnalysisResult {
  durationSeconds: number;
  framesAnalyzed: number;
  decisions: VideoShotDecision[];
  diagnostics: VideoAnalysisDiagnostics;
}

export interface VideoAnalysisDiagnostics {
  analysisFps: number;
  duplicateFramesSkipped: number;
  largestFrameGapMs: number;
  cameraMotionEvents: number;
  rimTrackedFrames: number;
  rimTrackingLostFrames: number;
  averageRimTrackingConfidence: number;
  rimGlobalReacquisitions: number;
  ballCandidateFrames: number;
  ballTrackedFrames: number;
  learnedBallDetectionFrames: number;
  learnedHoopDetectionFrames: number;
  learnedPlayerDetectionFrames: number;
  playerTrackedFrames: number;
  learnedDetectorBackend: "webgpu" | "wasm" | "native";
  requiresFullReview: boolean;
  warnings: string[];
}
