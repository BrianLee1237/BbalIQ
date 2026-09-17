import * as FileSystem from "expo-file-system/legacy";
import { ImageManipulator, SaveFormat } from "expo-image-manipulator";
import { createVideoPlayer, type VideoPlayer, type VideoThumbnail } from "expo-video";
import {
  ColorConversionCodes,
  DataTypes,
  Mat,
  OpenCV,
  Size,
  ThresholdTypes,
  type Mat as OpenCvMat,
} from "react-native-fast-opencv";

import {
  classifyVideoTrajectories,
  type VideoTrajectorySample,
} from "../tracking/videoTrajectoryClassifier";
import type {
  RimCalibration,
  VideoAnalysisResult,
} from "../../types/tracking";
import {
  detectOrangeBallCandidates,
} from "./ballDetector";
import { selectHoopZoneCandidates } from "./hoopZone";
import { createVisionTrackState, selectTrackedBall } from "./ballTracker";
import {
  applyVideoQualityGate,
  buildVideoAnalysisDiagnostics,
  buildVideoFrameTimes,
  createVideoStabilityState,
  IMPORT_ANALYSIS_FPS,
  MAX_IMPORT_DURATION_SECONDS,
  resolveVideoSampleTiming,
  stepVideoStability,
} from "./videoAnalysisPolicy";

export { IMPORT_ANALYSIS_FPS, MAX_IMPORT_DURATION_SECONDS };
const THUMBNAIL_WIDTH = 640;
const THUMBNAIL_HEIGHT = 640;
const THUMBNAIL_CHUNK_SIZE = 12;

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

export async function releaseVideoPreview(uri: string): Promise<void> {
  await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => undefined);
}

function waitForPlayerReady(player: VideoPlayer): Promise<void> {
  if (player.status === "readyToPlay") return Promise.resolve();
  if (player.status === "error") return Promise.reject(new Error("The selected video could not be opened."));

  return new Promise((resolve, reject) => {
    let settled = false;
    let subscription: { remove: () => void } | null = null;
    let timeout: ReturnType<typeof setTimeout> | null = null;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      if (timeout) clearTimeout(timeout);
      subscription?.remove();
      if (error) reject(error);
      else resolve();
    };
    subscription = player.addListener("statusChange", ({ status, error }) => {
      if (status === "readyToPlay") finish();
      if (status === "error") finish(new Error(error?.message ?? "The selected video could not be opened."));
    });
    if (settled) {
      subscription.remove();
      return;
    }
    timeout = setTimeout(
      () => finish(new Error("The selected video took too long to load.")),
      15_000,
    );
  });
}

async function createReadyPlayer(uri: string): Promise<VideoPlayer> {
  const player = createVideoPlayer(null);
  player.muted = true;
  try {
    await player.replaceAsync({ uri });
    await waitForPlayerReady(player);
    return player;
  } catch (error) {
    player.release();
    throw error;
  }
}

async function saveThumbnail(
  thumbnail: VideoThumbnail,
  includeBase64: boolean,
): Promise<{ uri: string; width: number; height: number; base64?: string }> {
  const context = ImageManipulator.manipulate(thumbnail);
  let image: Awaited<ReturnType<typeof context.renderAsync>> | null = null;
  try {
    image = await context.renderAsync();
    return await image.saveAsync({
      base64: includeBase64,
      compress: 0.9,
      format: SaveFormat.JPEG,
    });
  } finally {
    image?.release();
    context.release();
  }
}

export async function createVideoPreview(
  uri: string,
  requestedTimeSeconds = 0.25,
): Promise<VideoPreview> {
  const player = await createReadyPlayer(uri);
  try {
    const durationSeconds = player.duration;
    const safeTime = Math.max(0, Math.min(requestedTimeSeconds, Math.max(0, durationSeconds - 0.05)));
    const thumbnails = await player.generateThumbnailsAsync(safeTime, {
      maxWidth: 960,
      maxHeight: 540,
    });
    const thumbnail = thumbnails[0];
    if (!thumbnail) throw new Error("A preview frame could not be read from this video.");
    try {
      const saved = await saveThumbnail(thumbnail, false);
      return {
        uri: saved.uri,
        width: saved.width,
        height: saved.height,
        durationSeconds,
        atSeconds: Number.isFinite(thumbnail.actualTime)
          ? thumbnail.actualTime
          : safeTime,
        automaticRim: null,
        automaticRimConfidence: 0,
        automaticHoopCandidates: 0,
        automaticHoopAmbiguous: false,
      };
    } finally {
      thumbnail.release();
    }
  } finally {
    player.release();
  }
}

function chunks<T>(values: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let index = 0; index < values.length; index += size) {
    result.push(values.slice(index, index + size));
  }
  return result;
}

function measureGlobalFrameChange(
  source: OpenCvMat,
  previousGray: OpenCvMat | null,
): { gray: ReturnType<typeof Mat.create>; changedPixelRatio: number } {
  const gray = Mat.create(0, 0, DataTypes.CV_8U);
  const difference = Mat.create(0, 0, DataTypes.CV_8U);
  const changedMask = Mat.create(0, 0, DataTypes.CV_8U);
  const blurSize = Size.create(5, 5);
  try {
    OpenCV.cvtColor(source, gray, ColorConversionCodes.COLOR_BGR2GRAY);
    OpenCV.GaussianBlur(gray, gray, blurSize, 0);
    let changedPixelRatio = 0;
    if (
      previousGray &&
      previousGray.cols === gray.cols &&
      previousGray.rows === gray.rows
    ) {
      OpenCV.absdiff(previousGray, gray, difference);
      OpenCV.threshold(
        difference,
        changedMask,
        32,
        255,
        ThresholdTypes.THRESH_BINARY,
      );
      changedPixelRatio = OpenCV.countNonZero(changedMask).value /
        Math.max(1, gray.cols * gray.rows);
    }
    return { gray, changedPixelRatio };
  } catch (error) {
    gray.release();
    throw error;
  } finally {
    difference.release();
    changedMask.release();
    blurSize.release();
  }
}

export async function analyzeBasketballVideo(
  uri: string,
  rim: RimCalibration,
  options: VideoAnalysisOptions = {},
): Promise<VideoAnalysisResult> {
  const player = await createReadyPlayer(uri);
  let framesAnalyzed = 0;
  let visionTrack = createVisionTrackState();
  const trajectorySamples: VideoTrajectorySample[] = [];
  let samplesCompleted = 0;
  let duplicateFramesSkipped = 0;
  let largestFrameGapMs = 0;
  let previousTimestampMs: number | null = null;
  let previousGray: ReturnType<typeof Mat.create> | null = null;
  let stability = createVideoStabilityState();
  let ballCandidateFrames = 0;
  let ballTrackedFrames = 0;

  try {
    const durationSeconds = options.durationSeconds && options.durationSeconds > 0
      ? Math.min(options.durationSeconds, player.duration)
      : player.duration;
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      throw new Error("The selected video has no readable duration.");
    }
    if (durationSeconds > MAX_IMPORT_DURATION_SECONDS) {
      throw new Error(`Choose a video that is ${MAX_IMPORT_DURATION_SECONDS / 60} minutes or shorter.`);
    }

    const frameTimes = buildVideoFrameTimes(durationSeconds);
    options.onProgress?.(0, frameTimes.length);

    for (const timeChunk of chunks(frameTimes, THUMBNAIL_CHUNK_SIZE)) {
      if (options.isCancelled?.()) throw new Error("Video analysis was cancelled.");
      const thumbnails = await player.generateThumbnailsAsync(timeChunk, {
        maxWidth: THUMBNAIL_WIDTH,
        maxHeight: THUMBNAIL_HEIGHT,
      });
      let releasedThroughIndex = -1;

      try {
        for (let index = 0; index < thumbnails.length; index += 1) {
          if (options.isCancelled?.()) throw new Error("Video analysis was cancelled.");
          const thumbnail = thumbnails[index];
          if (!thumbnail) {
            releasedThroughIndex = index;
            samplesCompleted += 1;
            options.onProgress?.(samplesCompleted, frameTimes.length);
            continue;
          }
          const requestedTime = thumbnail.requestedTime ?? timeChunk[index] ?? 0;
          let savedUri = "";
          let source: ReturnType<typeof Mat.createFromBase64> | null = null;

          try {
            const saved = await saveThumbnail(thumbnail, true);
            savedUri = saved.uri;
            if (!saved.base64) throw new Error("A sampled video frame could not be decoded.");

            source = Mat.createFromBase64(saved.base64);
            if (source.cols <= 0 || source.rows <= 0) {
              throw new Error("A sampled video frame had invalid dimensions.");
            }
            const timing = resolveVideoSampleTiming(
              requestedTime,
              thumbnail.actualTime,
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
            const stabilityFrame = measureGlobalFrameChange(source, previousGray);
            previousGray?.release();
            previousGray = stabilityFrame.gray;
            stability = stepVideoStability(stability, stabilityFrame.changedPixelRatio);
            const candidates = detectOrangeBallCandidates(
              source,
              source.cols,
              source.rows,
              timestamp,
            );
            const hoopCandidates = selectHoopZoneCandidates(
              candidates,
              rim,
              source.cols,
              source.rows,
            );
            if (hoopCandidates.length > 0) ballCandidateFrames += 1;
            const selection = selectTrackedBall(hoopCandidates, visionTrack, rim, timestamp);
            visionTrack = selection.state;
            if (selection.detection) {
              ballTrackedFrames += 1;
              trajectorySamples.push({
                atSeconds: timing.atSeconds,
                ball: selection.detection,
                rim,
              });
            }
          } finally {
            source?.release();
            thumbnail.release();
            releasedThroughIndex = index;
            if (savedUri) {
              await FileSystem.deleteAsync(savedUri, { idempotent: true }).catch(() => undefined);
            }
          }

          framesAnalyzed += 1;
          samplesCompleted += 1;
          options.onProgress?.(samplesCompleted, frameTimes.length);
          if (framesAnalyzed % 6 === 0) {
            await new Promise<void>((resolve) => setTimeout(resolve, 0));
          }
        }
      } finally {
        // Release any native thumbnails left in this chunk if processing or
        // navigation interrupts the loop before every frame is consumed.
        for (let index = releasedThroughIndex + 1; index < thumbnails.length; index += 1) {
          thumbnails[index]?.release();
        }
      }
      const missingSamples = Math.max(0, timeChunk.length - thumbnails.length);
      if (missingSamples > 0) {
        duplicateFramesSkipped += missingSamples;
        samplesCompleted += missingSamples;
        options.onProgress?.(samplesCompleted, frameTimes.length);
      }
    }

    const decisions = classifyVideoTrajectories(trajectorySamples);
    const diagnostics = buildVideoAnalysisDiagnostics(
      framesAnalyzed,
      duplicateFramesSkipped,
      largestFrameGapMs,
      stability.cameraMotionEvents,
      {
        rimTrackedFrames: stability.cameraMotionEvents === 0 ? framesAnalyzed : 0,
        rimTrackingLostFrames: stability.cameraMotionEvents === 0 ? 0 : framesAnalyzed,
        averageRimTrackingConfidence: stability.cameraMotionEvents === 0 ? 1 : 0,
        rimGlobalReacquisitions: 0,
        ballCandidateFrames,
        ballTrackedFrames,
        learnedBallDetectionFrames: 0,
        learnedHoopDetectionFrames: 0,
        learnedPlayerDetectionFrames: 0,
        playerTrackedFrames: 0,
        learnedDetectorBackend: "native",
      },
    );
    return {
      durationSeconds,
      framesAnalyzed,
      decisions: applyVideoQualityGate(decisions, diagnostics),
      diagnostics,
    };
  } finally {
    previousGray?.release();
    player.release();
  }
}
