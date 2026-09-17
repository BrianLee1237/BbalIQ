import { Tracker as ByteTracker } from "byte-track-ts";
import { sourceCropGeometry } from "./sourceCrop";
import {
  createAttallaShotTrackerState,
  finalizeAttallaShotTracker,
  stepAttallaShotTracker,
  type AttallaObjectDetection,
} from "../tracking/attallaShotTracker";
import {
  createBasketSemanticTrackerState,
  stepBasketSemanticTracker,
} from "../tracking/basketSemanticTracker";
import { MIN_AUTOMATIC_DECISION_CONFIDENCE } from "../tracking/engine";
import {
  fuseShotEvidence,
  type SourcedShotDecision,
} from "../tracking/shotEvidenceFusion";
import {
  classifyVideoTrajectories,
  type VideoTrajectorySample,
} from "../tracking/videoTrajectoryClassifier";
import type {
  RimCalibration,
  VideoAnalysisResult,
  VideoShotDecision,
} from "../../types/tracking";
import {
  loadLearnedBasketballDetector,
  type LearnedBasketballDetector,
  type LearnedBasketballFrame,
  type LearnedObjectDetection,
} from "./learnedBasketballDetector.web";
import {
  chooseBoxNearReference,
  chooseAutomaticHoop,
  chooseCalibrationHoop,
  calibratedRimToScoringHoop,
  createHoopRimAnchor,
  filterPlausibleBasketballBoxes,
  learnedDetectionToPixelBox,
  mergeLearnedAndMotionCandidates,
  mergeLearnedFrames,
  mergeTrackedAndRawBoxes,
  pixelBoxToBallDetection,
  rimFromAutomaticHoop,
  rimFromTrackedHoop,
  toByteTrackDetections,
  trackRowToPixelBox,
  type HoopRimAnchor,
  type PixelBox,
} from "./learnedTracking";
import {
  createVisionTrackState,
  selectTrackGuidedRecoveryCandidates,
  selectTrackedBall,
  type VisionTrackState,
} from "./ballTracker";
import {
  detectBasketballCandidates,
  type PixelBallDetectionResult,
} from "./pixelBallDetector";
import { selectHoopZoneCandidates } from "./hoopZone";
import {
  createRimTrackState,
  stepFixedRimTracker,
  stepRimTracker,
  stepRimTrackerFromDetection,
} from "./rimTracker";
import {
  applyVideoQualityGate,
  buildVideoAnalysisDiagnostics,
  buildVideoFrameTimes,
  consolidateVideoShotDecisions,
  createVideoStabilityState,
  IMPORT_ANALYSIS_FPS,
  MAX_IMPORT_DURATION_SECONDS,
  resolveVideoSampleTiming,
  stepVideoStability,
} from "./videoAnalysisPolicy";
import {
  BALL_TRACKER_SETTINGS,
  HOOP_TRACKER_SETTINGS,
  MIN_AUTOMATIC_HOOP_CONFIDENCE,
} from "./trackerSettings";
import {
  computeRimDetectionRegion,
  type RimDetectionRegion,
} from "./rimDetectionRegion";
import {
  computeTrackDetectionRegion,
  regionsSubstantiallyOverlap,
} from "./trackDetectionRegion";

export { IMPORT_ANALYSIS_FPS, MAX_IMPORT_DURATION_SECONDS };
const ANALYSIS_MAX_WIDTH = 640;
const ANALYSIS_MAX_HEIGHT = 640;

export interface VideoPreview {
  sourceWidth?: number;
  uri: string;
  width: number;
  height: number;
  durationSeconds: number;
  atSeconds: number;
  automaticRim: RimCalibration | null;
  automaticRimConfidence: number;
  automaticHoopCandidates: number;
  automaticHoopAmbiguous: boolean;
}

export interface VideoAnalysisOptions {
  durationSeconds?: number;
  rimCalibrationTimeSeconds?: number;
  onProgress?: (completedFrames: number, totalFrames: number) => void;
  isCancelled?: () => boolean;
}

export async function releaseVideoPreview(): Promise<void> {
  // Browser previews are data URLs and do not create a temporary file.
}

function createLoadedVideo(uri: string): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    let settled = false;
    const timeout = window.setTimeout(
      () => finish(new Error("The selected video took too long to load.")),
      15_000,
    );

    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", handleReady);
      video.removeEventListener("error", handleError);
    };
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (error) reject(error);
      else resolve(video);
    };
    const handleReady = () => finish();
    const handleError = () => finish(new Error("The selected video could not be opened in this browser."));

    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.addEventListener("loadeddata", handleReady);
    video.addEventListener("error", handleError);
    video.src = uri;
    video.load();
  });
}

function disposeVideo(video: HTMLVideoElement): void {
  video.pause();
  video.removeAttribute("src");
  video.load();
}

function seekVideo(video: HTMLVideoElement, requestedTime: number): Promise<void> {
  const target = Math.max(0, Math.min(requestedTime, Math.max(0, video.duration - 0.002)));
  if (Math.abs(video.currentTime - target) < 0.002 && video.readyState >= 2) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = window.setTimeout(
      () => finish(new Error("A frame could not be read from this video.")),
      10_000,
    );
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("seeked", handleSeeked);
      video.removeEventListener("error", handleError);
    };
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (error) reject(error);
      else resolve();
    };
    const handleSeeked = () => finish();
    const handleError = () => finish(new Error("A frame could not be read from this video."));

    video.addEventListener("seeked", handleSeeked);
    video.addEventListener("error", handleError);
    video.currentTime = target;
  });
}

function createFrameCanvas(
  video: HTMLVideoElement,
  maximumWidth: number,
  maximumHeight: number,
): HTMLCanvasElement {
  const sourceWidth = Math.max(1, video.videoWidth);
  const sourceHeight = Math.max(1, video.videoHeight);
  const scale = Math.min(maximumWidth / sourceWidth, maximumHeight / sourceHeight, 1);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(sourceWidth * scale));
  canvas.height = Math.max(1, Math.round(sourceHeight * scale));
  return canvas;
}

function mapDetectionFromRegion(
  detection: LearnedObjectDetection,
  region: RimDetectionRegion,
  scaleX = 1,
  scaleY = 1,
): LearnedObjectDetection {
  return {
    ...detection,
    left: detection.left * scaleX + region.left,
    top: detection.top * scaleY + region.top,
    right: detection.right * scaleX + region.left,
    bottom: detection.bottom * scaleY + region.top,
  };
}

function mapFrameFromRegion(
  frame: LearnedBasketballFrame,
  region: RimDetectionRegion,
  scaleX = 1,
  scaleY = 1,
): LearnedBasketballFrame {
  const objects = frame.objects.map((detection) => mapDetectionFromRegion(detection, region, scaleX, scaleY));
  return {
    objects,
    basketballs: objects.filter((detection) => detection.label === "ball"),
    hoops: objects.filter((detection) => detection.label === "hoop"),
    players: frame.players,
  };
}

async function detectRegion(
  detector: LearnedBasketballDetector,
  frameCanvas: HTMLCanvasElement,
  focusedCanvas: HTMLCanvasElement,
  focusedContext: CanvasRenderingContext2D,
  region: RimDetectionRegion,
  sourceVideo: HTMLVideoElement,
): Promise<LearnedBasketballFrame> {
  const crop = sourceCropGeometry(region, frameCanvas.width, frameCanvas.height,
    sourceVideo.videoWidth, sourceVideo.videoHeight);
  if (focusedCanvas.width !== crop.outputWidth) focusedCanvas.width = crop.outputWidth;
  if (focusedCanvas.height !== crop.outputHeight) focusedCanvas.height = crop.outputHeight;
  focusedContext.clearRect(0, 0, crop.outputWidth, crop.outputHeight);
  focusedContext.drawImage(
    sourceVideo,
    crop.left,
    crop.top,
    crop.width,
    crop.height,
    0,
    0,
    crop.outputWidth,
    crop.outputHeight,
  );
  return mapFrameFromRegion(
    await detector.detect(focusedCanvas),
    region,
    crop.toAnalysisX,
    crop.toAnalysisY,
  );
}

/**
 * Runs complementary detector scales instead of permanently switching to a
 * rim-only crop. Full scans acquire each new shot on every sample, the rim crop keeps
 * the entry/exit visible, and the predicted crop follows fast balls between
 * those two regions. Detections are merged before ByteTrack association.
 */
async function detectFrameMultiScale(
  detector: LearnedBasketballDetector,
  frameCanvas: HTMLCanvasElement,
  focusedCanvas: HTMLCanvasElement,
  focusedContext: CanvasRenderingContext2D,
  rim: RimCalibration,
  track: VisionTrackState,
  atMs: number,
  sourceVideo: HTMLVideoElement,
): Promise<LearnedBasketballFrame> {
  const rimRegion = computeRimDetectionRegion(
    rim,
    frameCanvas.width,
    frameCanvas.height,
  );
  const frames: LearnedBasketballFrame[] = [];
  // Uploaded videos are processed offline: keep full-frame evidence on every
  // sample so a crop cannot hide a second ball or lose the approach trajectory.
  frames.push(await detector.detect(frameCanvas));
  frames.push(await detectRegion(
    detector,
    frameCanvas,
    focusedCanvas,
    focusedContext,
    rimRegion,
    sourceVideo,
  ));

  const trackRegion = computeTrackDetectionRegion(
    track,
    rim,
    frameCanvas.width,
    frameCanvas.height,
    atMs,
  );
  if (trackRegion && !regionsSubstantiallyOverlap(trackRegion, rimRegion)) {
    frames.push(await detectRegion(
      detector,
      frameCanvas,
      focusedCanvas,
      focusedContext,
      trackRegion,
      sourceVideo,
    ));
  }
  return mergeLearnedFrames(frames);
}

function frameGray(pixels: ImageData): Uint8Array {
  const gray = new Uint8Array(pixels.width * pixels.height);
  for (let index = 0; index < gray.length; index += 1) {
    const offset = index * 4;
    gray[index] = Math.round(
      (pixels.data[offset] ?? 0) * 0.299 +
      (pixels.data[offset + 1] ?? 0) * 0.587 +
      (pixels.data[offset + 2] ?? 0) * 0.114,
    );
  }
  return gray;
}


function buildAutomaticHoopScanTimes(
  requestedTime: number,
  durationSeconds: number,
): number[] {
  const maximum = Math.max(0, durationSeconds - 0.05);
  const offsets = [0, 0.45, -0.45, 0.9, 1.6, 2.5];
  return Array.from(new Set(offsets.map((offset) =>
    Math.round(Math.max(0, Math.min(maximum, requestedTime + offset)) * 1_000) / 1_000
  )));
}

function isWarmRimPixel(red: number, green: number, blue: number): boolean {
  const maximum = Math.max(red, green, blue);
  const minimum = Math.min(red, green, blue);
  return red >= 85 && red - minimum >= 34 && red >= green * 1.08 && red >= blue * 1.22 && maximum - minimum >= 38;
}

/** Refines the broad learned hoop box to the painted horizontal rim line. */
function refineAutomaticRim(
  fallback: RimCalibration,
  hoop: PixelBox,
  pixels: ImageData,
): RimCalibration {
  const left = Math.max(0, Math.floor(hoop.left));
  const right = Math.min(pixels.width - 1, Math.ceil(hoop.right));
  const top = Math.max(0, Math.floor(hoop.top));
  const bottom = Math.min(
    pixels.height - 1,
    Math.ceil(hoop.top + (hoop.bottom - hoop.top) * 0.72),
  );
  let bestRow = -1;
  let bestLeft = 0;
  let bestRight = 0;
  let bestScore = 0;
  for (let y = top; y <= bottom; y += 1) {
    let count = 0;
    let rowLeft = right;
    let rowRight = left;
    for (let x = left; x <= right; x += 1) {
      const offset = (y * pixels.width + x) * 4;
      if (!isWarmRimPixel(
        pixels.data[offset] ?? 0,
        pixels.data[offset + 1] ?? 0,
        pixels.data[offset + 2] ?? 0,
      )) continue;
      count += 1;
      rowLeft = Math.min(rowLeft, x);
      rowRight = Math.max(rowRight, x);
    }
    const span = Math.max(0, rowRight - rowLeft);
    const score = count + span * 0.45;
    if (score > bestScore) {
      bestScore = score;
      bestRow = y;
      bestLeft = rowLeft;
      bestRight = rowRight;
    }
  }
  const hoopWidth = Math.max(1, hoop.right - hoop.left);
  if (bestRow < 0 || bestRight - bestLeft < hoopWidth * 0.28 || bestScore < 8) {
    return fallback;
  }
  const padding = Math.max(1, (bestRight - bestLeft) * 0.05);
  const widthPixels = Math.max(pixels.width * 0.035, bestRight - bestLeft + padding * 2);
  const heightPixels = Math.max(pixels.height * 0.012, widthPixels / 4.2);
  const width = Math.min(0.5, widthPixels / pixels.width);
  const height = Math.min(0.28, heightPixels / pixels.height);
  const centerX = (bestLeft + bestRight) / 2 / pixels.width;
  const centerY = bestRow / pixels.height;
  return {
    x: Math.max(0, Math.min(1 - width, centerX - width / 2)),
    y: Math.max(0, Math.min(1 - height, centerY - height / 2)),
    width,
    height,
  };
}

export async function createVideoPreview(
  uri: string,
  requestedTimeSeconds = 0.25,
): Promise<VideoPreview> {
  const video = await createLoadedVideo(uri);
  try {
    const durationSeconds = video.duration;
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      throw new Error("The selected video has no readable duration.");
    }
    const canvas = createFrameCanvas(video, 960, 960);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("This browser cannot read video frames.");
    const detector = await loadLearnedBasketballDetector();
    const focusedCanvas = document.createElement("canvas");
    const focusedContext = focusedCanvas.getContext("2d", { willReadFrequently: true });
    if (!focusedContext) throw new Error("This browser cannot refine the automatic hoop lock.");
    const scanTimes = buildAutomaticHoopScanTimes(requestedTimeSeconds, durationSeconds);
    let bestPreview: {
      uri: string;
      atSeconds: number;
      rim: RimCalibration;
      confidence: number;
      candidates: number;
      ambiguous: boolean;
      hoop: PixelBox;
    } | null = null;
    let fallbackUri = "";
    let fallbackTime = 0;

    for (const scanTime of scanTimes) {
      await seekVideo(video, scanTime);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameUri = canvas.toDataURL("image/jpeg", 0.92);
      if (!fallbackUri) {
        fallbackUri = frameUri;
        fallbackTime = video.currentTime;
      }
      const learned = await detector.detect(canvas);
      const choice = chooseAutomaticHoop(
        learned.hoops.map(learnedDetectionToPixelBox),
        canvas.width,
        canvas.height,
      );
      if (!choice || choice.confidence < MIN_AUTOMATIC_HOOP_CONFIDENCE) continue;
      const framePixels = context.getImageData(0, 0, canvas.width, canvas.height);
      const fallbackRim = rimFromAutomaticHoop(
        choice.hoop,
        canvas.width,
        canvas.height,
      );
      const rim = refineAutomaticRim(fallbackRim, choice.hoop, framePixels);
      const candidate = {
        uri: frameUri,
        atSeconds: video.currentTime,
        rim,
        confidence: choice.confidence,
        candidates: learned.hoops.length,
        ambiguous: choice.ambiguous,
        hoop: choice.hoop,
      };
      if (!bestPreview || candidate.confidence > bestPreview.confidence) {
        bestPreview = candidate;
      }
      if (choice.confidence >= 0.72 && !choice.ambiguous) break;
    }

    if (bestPreview) {
      await seekVideo(video, bestPreview.atSeconds);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const region = computeRimDetectionRegion(
        bestPreview.rim,
        canvas.width,
        canvas.height,
      );
      const focused = await detectRegion(
        detector, canvas, focusedCanvas, focusedContext, region, video,
      );
      const refinedChoice = chooseAutomaticHoop(
        focused.hoops.map(learnedDetectionToPixelBox),
        canvas.width,
        canvas.height,
      );
      if (refinedChoice && refinedChoice.confidence > bestPreview.confidence + 0.12) {
        const framePixels = context.getImageData(0, 0, canvas.width, canvas.height);
        const fallbackRim = rimFromAutomaticHoop(
          refinedChoice.hoop,
          canvas.width,
          canvas.height,
        );
        bestPreview = {
          ...bestPreview,
          uri: canvas.toDataURL("image/jpeg", 0.92),
          rim: refineAutomaticRim(fallbackRim, refinedChoice.hoop, framePixels),
          confidence: refinedChoice.confidence,
          candidates: focused.hoops.length,
          ambiguous: refinedChoice.ambiguous,
          hoop: refinedChoice.hoop,
        };
      }
    }

    return {
      uri: bestPreview?.uri ?? fallbackUri,
      sourceWidth: video.videoWidth,
      width: canvas.width,
      height: canvas.height,
      durationSeconds,
      atSeconds: bestPreview?.atSeconds ?? fallbackTime,
      automaticRim: bestPreview?.rim ?? null,
      automaticRimConfidence: bestPreview?.confidence ?? 0,
      automaticHoopCandidates: bestPreview?.candidates ?? 0,
      automaticHoopAmbiguous: bestPreview?.ambiguous ?? false,
    };
  } finally {
    disposeVideo(video);
  }
}

export async function analyzeBasketballVideo(
  uri: string,
  rim: RimCalibration,
  options: VideoAnalysisOptions = {},
): Promise<VideoAnalysisResult> {
  const video = await createLoadedVideo(uri);
  let shotTracker = createAttallaShotTrackerState();
  const shotDecisions: VideoShotDecision[] = [];
  let semanticTracker = createBasketSemanticTrackerState();
  const semanticDecisions: VideoShotDecision[] = [];
  let visionTrack = createVisionTrackState();
  const trajectorySamples: VideoTrajectorySample[] = [];
  let framesAnalyzed = 0;
  let samplesCompleted = 0;
  let duplicateFramesSkipped = 0;
  let largestFrameGapMs = 0;
  let previousTimestampMs: number | null = null;
  let previousGray: Uint8Array | null = null;
  let stability = createVideoStabilityState();
  let ballCandidateFrames = 0;
  let ballTrackedFrames = 0;
  let learnedBallDetectionFrames = 0;
  let learnedHoopDetectionFrames = 0;
  let learnedPlayerDetectionFrames = 0;
  let playerTrackedFrames = 0;
  const learnedDetector = await loadLearnedBasketballDetector();
  const hoopAssociation = new ByteTracker(HOOP_TRACKER_SETTINGS);
  const ballAssociation = new ByteTracker(BALL_TRACKER_SETTINGS);

  try {
    const durationSeconds = options.durationSeconds && options.durationSeconds > 0
      ? Math.min(options.durationSeconds, video.duration)
      : video.duration;
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      throw new Error("The selected video has no readable duration.");
    }
    if (durationSeconds > MAX_IMPORT_DURATION_SECONDS) {
      throw new Error(`Choose a video that is ${MAX_IMPORT_DURATION_SECONDS / 60} minutes or shorter.`);
    }

    const frameTimes = buildVideoFrameTimes(durationSeconds);
    const canvas = createFrameCanvas(video, ANALYSIS_MAX_WIDTH, ANALYSIS_MAX_HEIGHT);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("This browser cannot analyze video frames.");
    const focusedCanvas = document.createElement("canvas");
    const focusedContext = focusedCanvas.getContext("2d", { willReadFrequently: true });
    if (!focusedContext) throw new Error("This browser cannot focus detector frames around the rim.");

    const calibrationTime = Math.max(
      0,
      Math.min(
        options.rimCalibrationTimeSeconds ?? Math.min(1, durationSeconds * 0.08),
        Math.max(0, durationSeconds - 0.02),
      ),
    );
    await seekVideo(video, calibrationTime);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const calibrationPixels = context.getImageData(0, 0, canvas.width, canvas.height);
    const calibrationLearned = await learnedDetector.detect(canvas);
    const calibrationHoop = chooseCalibrationHoop(
      calibrationLearned.hoops.map(learnedDetectionToPixelBox),
      rim,
      canvas.width,
      canvas.height,
    );
    let hoopAnchor: HoopRimAnchor | null = null;
    let lastHoopBox: PixelBox | null = calibrationHoop;
    let preferredHoopTrackId: number | undefined;
    let consecutiveLearnedHoopFrames = calibrationHoop ? 1 : 0;
    let rimTrack = createRimTrackState(
      frameGray(calibrationPixels),
      canvas.width,
      canvas.height,
      rim,
    );
    options.onProgress?.(0, frameTimes.length);

    for (const requestedTime of frameTimes) {
      if (options.isCancelled?.()) throw new Error("Video analysis was cancelled.");
      await seekVideo(video, requestedTime);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      const timing = resolveVideoSampleTiming(
        requestedTime,
        video.currentTime,
        previousTimestampMs,
      );
      if (timing.duplicate) {
        duplicateFramesSkipped += 1;
        samplesCompleted += 1;
        options.onProgress?.(samplesCompleted, frameTimes.length);
        continue;
      }
      previousTimestampMs = timing.timestampMs;
      largestFrameGapMs = Math.max(largestFrameGapMs, timing.gapMs);
      const timestamp = timing.timestampMs;

      const learnedFrame = await detectFrameMultiScale(
        learnedDetector,
        canvas,
        focusedCanvas,
        focusedContext,
        rimTrack.rim,
        visionTrack,
        timestamp,
        video,
      );
      if (learnedFrame.basketballs.length > 0) learnedBallDetectionFrames += 1;
      if (learnedFrame.hoops.length > 0) learnedHoopDetectionFrames += 1;
      if (learnedFrame.players.length > 0) learnedPlayerDetectionFrames += 1;
      // The detector's hoop box is intentionally used for visual reacquisition
      // below, but the scoring plane must come from the approved rim opening.
      // Detector hoop boxes vary between the rim, net, and whole backboard and
      // can otherwise make an identical swish count in one frame and vanish in
      // the next.
      const scoringHoop = calibratedRimToScoringHoop(
        rimTrack.rim,
        canvas.width,
        canvas.height,
      );
      const rawBallBoxes = filterPlausibleBasketballBoxes(
        learnedFrame.basketballs.map(learnedDetectionToPixelBox),
        rimTrack.rim,
        canvas.width,
        canvas.height,
      );
      const trackedBallBoxes = ballAssociation
        .update(toByteTrackDetections(rawBallBoxes))
        .map(trackRowToPixelBox)
        .filter((box): box is PixelBox => box !== null);
      const associatedBallBoxes = mergeTrackedAndRawBoxes(
        trackedBallBoxes,
        rawBallBoxes,
      );
      const learnedBallCandidates = associatedBallBoxes.map((box) =>
        pixelBoxToBallDetection(box, canvas.width, canvas.height, timestamp)
      );
      const pixelDetection: PixelBallDetectionResult = visionTrack.current && previousGray
        ? detectBasketballCandidates(pixels, previousGray, timestamp, rimTrack.rim)
        : { candidates: [], gray: frameGray(pixels) };
      const recoveryCandidates = selectTrackGuidedRecoveryCandidates(
        pixelDetection.candidates,
        visionTrack,
        rimTrack.rim,
        timestamp,
      );
      const ballCandidates = selectHoopZoneCandidates(
        mergeLearnedAndMotionCandidates(learnedBallCandidates, recoveryCandidates),
        rimTrack.rim,
        canvas.width,
        canvas.height,
      );
      const ballSelection = selectTrackedBall(
        ballCandidates,
        visionTrack,
        rimTrack.rim,
        timestamp,
      );
      visionTrack = ballSelection.state;
      if (ballSelection.detection) {
        ballTrackedFrames += 1;
        trajectorySamples.push({
          atSeconds: timestamp / 1_000,
          ball: ballSelection.detection,
          rim: rimTrack.rim,
        });
      }
      const semanticStep = stepBasketSemanticTracker(
        semanticTracker,
        ballSelection.detection,
        rimTrack.rim,
        timestamp,
      );
      semanticTracker = semanticStep.state;
      if (semanticStep.decision) semanticDecisions.push(semanticStep.decision);
      const recoveredBall = ballSelection.detection &&
          recoveryCandidates.includes(ballSelection.detection)
        ? ballSelection.detection
        : null;
      // Feed the crossing tracker only geometrically plausible basketballs.
      // This lets the near-rim threshold recover weak occluded balls without
      // exposing the state machine to every low-score detector response.
      const detectorObjects: AttallaObjectDetection[] = [
        ...associatedBallBoxes.map((box) => ({
          kind: "ball" as const,
          x: (box.left + box.right) / 2,
          y: (box.top + box.bottom) / 2,
          width: box.right - box.left,
          height: box.bottom - box.top,
          confidence: box.confidence,
          sourceTrackId: box.trackId,
        })),
        ...(recoveredBall
          ? [{
            kind: "ball" as const,
            x: recoveredBall.x * canvas.width,
            y: recoveredBall.y * canvas.height,
            width: recoveredBall.width * canvas.width,
            height: recoveredBall.height * canvas.height,
            confidence: recoveredBall.confidence,
          }]
          : []),
        {
          kind: "hoop" as const,
          x: (scoringHoop.left + scoringHoop.right) / 2,
          y: (scoringHoop.top + scoringHoop.bottom) / 2,
          width: scoringHoop.right - scoringHoop.left,
          height: scoringHoop.bottom - scoringHoop.top,
          confidence: scoringHoop.confidence,
        },
      ];
      const shotStep = stepAttallaShotTracker(shotTracker, detectorObjects, timestamp);
      shotTracker = shotStep.state;
      if (learnedFrame.basketballs.length > 0) ballCandidateFrames += 1;
      for (const decision of shotStep.decisions) {
        shotDecisions.push({
          id: `${decision.atMs}-${shotDecisions.length}`,
          atSeconds: decision.atMs / 1_000,
          suggestedKind: decision.kind,
          finalKind: decision.confidence >= MIN_AUTOMATIC_DECISION_CONFIDENCE
            ? decision.kind
            : null,
          confidence: decision.confidence,
          reason: "rim-crossing",
        });
      }
      if (shotStep.decisions.length > 0) {
        // Start the single-ball semantic path cleanly instead of following the
        // scored ball's rebound. Do not restart ByteTrack here: another ball
        // may already be in flight, and resetting IDs at this exact moment can
        // merge or discard that independent attempt. The crossing tracker
        // removes only the completed physical ball from its own state.
        visionTrack = createVisionTrackState();
        semanticTracker = {
          ...createBasketSemanticTrackerState(),
          lastShotAt: semanticTracker.lastShotAt,
        };
      }
      const rawHoopBoxes = learnedFrame.hoops.map(learnedDetectionToPixelBox);
      const trackedHoopBoxes = hoopAssociation
        .update(toByteTrackDetections(rawHoopBoxes))
        .map(trackRowToPixelBox)
        .filter((box): box is PixelBox => box !== null);
      const observedHoop = chooseCalibrationHoop(
        rawHoopBoxes,
        rimTrack.rim,
        canvas.width,
        canvas.height,
      );
      consecutiveLearnedHoopFrames = observedHoop
        ? consecutiveLearnedHoopFrames + 1
        : 0;
      if (!hoopAnchor && observedHoop && consecutiveLearnedHoopFrames >= 3) {
        hoopAnchor = createHoopRimAnchor(
          rimTrack.rim,
          observedHoop,
          canvas.width,
          canvas.height,
        );
        lastHoopBox = observedHoop;
      }
      const hoopReference: PixelBox | null = lastHoopBox ?? observedHoop;
      const trackedHoop: PixelBox | null = hoopAnchor && hoopReference
        ? chooseBoxNearReference(trackedHoopBoxes, hoopReference, preferredHoopTrackId) ??
          chooseBoxNearReference(rawHoopBoxes, hoopReference)
        : null;
      if (trackedHoop) {
        lastHoopBox = trackedHoop;
        preferredHoopTrackId = trackedHoop.trackId ?? preferredHoopTrackId;
      }

      const gray = pixelDetection.gray;
      const learnedRim = trackedHoop && hoopAnchor
        ? rimFromTrackedHoop(trackedHoop, hoopAnchor, canvas.width, canvas.height)
        : null;
      const templateRimStep = learnedRim && trackedHoop
        ? null
        : stepRimTracker(
          gray,
          canvas.width,
          canvas.height,
          rimTrack,
        );
      const rimStep = learnedRim && trackedHoop
        ? stepRimTrackerFromDetection(rimTrack, learnedRim, trackedHoop.confidence)
        : templateRimStep?.found
          ? templateRimStep
          : stepFixedRimTracker(rimTrack);
      rimTrack = rimStep.state;

      let changedPixels = 0;
      if (previousGray && previousGray.length === gray.length) {
        for (let index = 0; index < gray.length; index += 1) {
          if (Math.abs((gray[index] ?? 0) - (previousGray[index] ?? 0)) >= 38) {
            changedPixels += 1;
          }
        }
      }
      const changedPixelRatio = previousGray
        ? changedPixels / Math.max(1, gray.length)
        : 0;
      stability = stepVideoStability(stability, changedPixelRatio);
      previousGray = gray;

      if (!rimStep.found) {
        framesAnalyzed += 1;
        samplesCompleted += 1;
        options.onProgress?.(samplesCompleted, frameTimes.length);
        if (framesAnalyzed % 3 === 0) {
          await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
        }
        continue;
      }

      framesAnalyzed += 1;
      samplesCompleted += 1;
      options.onProgress?.(samplesCompleted, frameTimes.length);
      if (framesAnalyzed % 3 === 0) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      }
    }

    const finalShotStep = finalizeAttallaShotTracker(
      shotTracker,
      Math.round(durationSeconds * 1_000),
    );
    shotTracker = finalShotStep.state;
    for (const decision of finalShotStep.decisions) {
      shotDecisions.push({
        id: `${decision.atMs}-${shotDecisions.length}-final`,
        atSeconds: decision.atMs / 1_000,
        suggestedKind: decision.kind,
        finalKind: decision.confidence >= MIN_AUTOMATIC_DECISION_CONFIDENCE
          ? decision.kind
          : null,
        confidence: decision.confidence,
        reason: "rim-crossing",
      });
    }

    const diagnostics = buildVideoAnalysisDiagnostics(
      framesAnalyzed,
      duplicateFramesSkipped,
      largestFrameGapMs,
      stability.cameraMotionEvents,
      {
        rimTrackedFrames: rimTrack.trackedFrames,
        rimTrackingLostFrames: rimTrack.lostFrames,
        averageRimTrackingConfidence:
          rimTrack.confidenceTotal / Math.max(1, rimTrack.framesProcessed),
        rimGlobalReacquisitions: rimTrack.globalReacquisitions,
        ballCandidateFrames,
        ballTrackedFrames,
        learnedBallDetectionFrames,
        learnedHoopDetectionFrames,
        learnedPlayerDetectionFrames,
        playerTrackedFrames,
        learnedDetectorBackend: learnedDetector.backend,
      },
    );
    const trajectoryDecisions = classifyVideoTrajectories(trajectorySamples);
    const evidence: SourcedShotDecision[] = [
      ...shotDecisions.map((decision) => ({ source: "detector" as const, decision })),
      ...trajectoryDecisions.map((decision) => ({ source: "trajectory" as const, decision })),
      ...semanticDecisions.map((decision) => ({ source: "semantic" as const, decision })),
    ];
    const fusedDecisions = consolidateVideoShotDecisions(
      fuseShotEvidence(evidence),
    );
    return {
      durationSeconds,
      framesAnalyzed,
      decisions: applyVideoQualityGate(
        fusedDecisions,
        diagnostics,
      ),
      diagnostics,
    };
  } finally {
    disposeVideo(video);
  }
}
