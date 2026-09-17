import type { RimCalibration } from "../../types/tracking";

const TEMPLATE_COLUMNS = 15;
const TEMPLATE_ROWS = 11;
const MIN_LOCAL_MATCH_CONFIDENCE = 0.63;
const MIN_GLOBAL_MATCH_CONFIDENCE = 0.62;

export interface RimAppearanceTemplate {
  columns: number;
  rows: number;
  patchWidth: number;
  patchHeight: number;
  normalizedSamples: Float32Array;
}

export interface RimTrackState {
  rim: RimCalibration;
  template: RimAppearanceTemplate;
  framesProcessed: number;
  trackedFrames: number;
  lostFrames: number;
  confidenceTotal: number;
  consecutiveLostFrames: number;
  velocityX: number;
  velocityY: number;
  globalReacquisitions: number;
}

export interface RimTrackStep {
  rim: RimCalibration;
  confidence: number;
  found: boolean;
  reacquired: boolean;
  displacementX: number;
  displacementY: number;
  state: RimTrackState;
}

interface Match {
  centerX: number;
  centerY: number;
  confidence: number;
  score: number;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function normalizeSamples(samples: Float32Array): Float32Array {
  let mean = 0;
  for (const sample of samples) mean += sample;
  mean /= Math.max(1, samples.length);

  let variance = 0;
  for (const sample of samples) variance += (sample - mean) ** 2;
  const deviation = Math.sqrt(variance / Math.max(1, samples.length));
  const scale = Math.max(7, deviation);
  const normalized = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    normalized[index] = ((samples[index] ?? mean) - mean) / scale;
  }
  return normalized;
}

function samplePatch(
  gray: Uint8Array,
  frameWidth: number,
  frameHeight: number,
  centerX: number,
  centerY: number,
  patchWidth: number,
  patchHeight: number,
  columns: number,
  rows: number,
): Float32Array {
  const samples = new Float32Array(columns * rows);
  const left = centerX - patchWidth / 2;
  const top = centerY - patchHeight / 2;
  let target = 0;

  for (let row = 0; row < rows; row += 1) {
    const sampleY = clamp(
      Math.round(top + ((row + 0.5) / rows) * patchHeight),
      1,
      Math.max(1, frameHeight - 2),
    );
    for (let column = 0; column < columns; column += 1) {
      const sampleX = clamp(
        Math.round(left + ((column + 0.5) / columns) * patchWidth),
        1,
        Math.max(1, frameWidth - 2),
      );
      const index = sampleY * frameWidth + sampleX;
      const horizontalEdge = Math.abs((gray[index + 1] ?? 0) - (gray[index - 1] ?? 0));
      const verticalEdge = Math.abs(
        (gray[index + frameWidth] ?? 0) - (gray[index - frameWidth] ?? 0),
      );
      const diagonalEdge = Math.abs(
        (gray[index + frameWidth + 1] ?? 0) - (gray[index - frameWidth - 1] ?? 0),
      );
      const intensity = gray[index] ?? 0;
      samples[target] = intensity * 0.28 +
        horizontalEdge * 0.28 +
        verticalEdge * 0.3 +
        diagonalEdge * 0.14;
      target += 1;
    }
  }
  return samples;
}

function correlation(left: Float32Array, right: Float32Array): number {
  let dot = 0;
  let leftEnergy = 0;
  let rightEnergy = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftEnergy += a * a;
    rightEnergy += b * b;
  }
  const denominator = Math.sqrt(leftEnergy * rightEnergy);
  return denominator > 0.0001 ? clamp(dot / denominator, -1, 1) : -1;
}

function evaluateMatch(
  gray: Uint8Array,
  frameWidth: number,
  frameHeight: number,
  template: RimAppearanceTemplate,
  centerX: number,
  centerY: number,
  referenceX: number,
  referenceY: number,
  radiusX: number,
  radiusY: number,
  distancePenalty: number,
): Match {
  const samples = normalizeSamples(
    samplePatch(
      gray,
      frameWidth,
      frameHeight,
      centerX,
      centerY,
      template.patchWidth,
      template.patchHeight,
      template.columns,
      template.rows,
    ),
  );
  const match = correlation(template.normalizedSamples, samples);
  const confidence = clamp((match + 1) / 2, 0, 1);
  const normalizedDistance = Math.hypot(
    (centerX - referenceX) / Math.max(1, radiusX),
    (centerY - referenceY) / Math.max(1, radiusY),
  );
  return {
    centerX,
    centerY,
    confidence,
    score: confidence - normalizedDistance * distancePenalty,
  };
}

function searchGrid(
  gray: Uint8Array,
  frameWidth: number,
  frameHeight: number,
  template: RimAppearanceTemplate,
  minimumX: number,
  maximumX: number,
  minimumY: number,
  maximumY: number,
  stride: number,
  referenceX: number,
  referenceY: number,
  radiusX: number,
  radiusY: number,
  distancePenalty: number,
): Match {
  let best: Match = {
    centerX: referenceX,
    centerY: referenceY,
    confidence: 0,
    score: -1,
  };
  const left = Math.max(1, minimumX);
  const right = Math.min(frameWidth - 2, maximumX);
  const top = Math.max(1, minimumY);
  const bottom = Math.min(frameHeight - 2, maximumY);

  for (let centerY = top; centerY <= bottom; centerY += stride) {
    for (let centerX = left; centerX <= right; centerX += stride) {
      const candidate = evaluateMatch(
        gray,
        frameWidth,
        frameHeight,
        template,
        centerX,
        centerY,
        referenceX,
        referenceY,
        radiusX,
        radiusY,
        distancePenalty,
      );
      if (candidate.score > best.score) best = candidate;
    }
  }
  return best;
}

export function createRimTrackState(
  gray: Uint8Array,
  frameWidth: number,
  frameHeight: number,
  rim: RimCalibration,
): RimTrackState {
  const rimWidthPixels = Math.max(3, rim.width * frameWidth);
  const rimHeightPixels = Math.max(2, rim.height * frameHeight);
  const patchWidth = Math.round(clamp(rimWidthPixels * 3.2, 36, frameWidth * 0.58));
  const patchHeight = Math.round(
    clamp(
      Math.max(rimHeightPixels * 6, rimWidthPixels * 2.45),
      30,
      frameHeight * 0.45,
    ),
  );
  const centerX = (rim.x + rim.width / 2) * frameWidth;
  const centerY = (rim.y + rim.height / 2) * frameHeight;
  const samples = samplePatch(
    gray,
    frameWidth,
    frameHeight,
    centerX,
    centerY,
    patchWidth,
    patchHeight,
    TEMPLATE_COLUMNS,
    TEMPLATE_ROWS,
  );

  return {
    rim: { ...rim },
    template: {
      columns: TEMPLATE_COLUMNS,
      rows: TEMPLATE_ROWS,
      patchWidth,
      patchHeight,
      normalizedSamples: normalizeSamples(samples),
    },
    framesProcessed: 0,
    trackedFrames: 0,
    lostFrames: 0,
    confidenceTotal: 0,
    consecutiveLostFrames: 0,
    velocityX: 0,
    velocityY: 0,
    globalReacquisitions: 0,
  };
}

export function stepRimTracker(
  gray: Uint8Array,
  frameWidth: number,
  frameHeight: number,
  current: RimTrackState,
): RimTrackStep {
  const currentCenterX = (current.rim.x + current.rim.width / 2) * frameWidth;
  const currentCenterY = (current.rim.y + current.rim.height / 2) * frameHeight;
  const predictedCenterX = currentCenterX + current.velocityX * frameWidth;
  const predictedCenterY = currentCenterY + current.velocityY * frameHeight;
  const rimWidthPixels = Math.max(3, current.rim.width * frameWidth);
  const radiusX = Math.round(clamp(rimWidthPixels * 1.35, 10, frameWidth * 0.16));
  const radiusY = Math.round(clamp(rimWidthPixels * 1.1, 9, frameHeight * 0.14));
  const localStride = Math.max(radiusX, radiusY) > 24 ? 2 : 1;
  let best = searchGrid(
    gray,
    frameWidth,
    frameHeight,
    current.template,
    predictedCenterX - radiusX,
    predictedCenterX + radiusX,
    predictedCenterY - radiusY,
    predictedCenterY + radiusY,
    localStride,
    predictedCenterX,
    predictedCenterY,
    radiusX,
    radiusY,
    0.025,
  );

  let reacquired = false;
  if (best.confidence < MIN_LOCAL_MATCH_CONFIDENCE) {
    const coarseStride = Math.round(clamp(rimWidthPixels * 0.3, 3, 10));
    const coarse = searchGrid(
      gray,
      frameWidth,
      frameHeight,
      current.template,
      0,
      frameWidth,
      0,
      frameHeight,
      coarseStride,
      currentCenterX,
      currentCenterY,
      frameWidth,
      frameHeight,
      0,
    );
    const refineRadius = coarseStride * 2.5;
    const refined = searchGrid(
      gray,
      frameWidth,
      frameHeight,
      current.template,
      coarse.centerX - refineRadius,
      coarse.centerX + refineRadius,
      coarse.centerY - refineRadius,
      coarse.centerY + refineRadius,
      1,
      coarse.centerX,
      coarse.centerY,
      refineRadius,
      refineRadius,
      0.008,
    );
    if (refined.confidence > best.confidence) best = refined;
    reacquired = best.confidence >= MIN_GLOBAL_MATCH_CONFIDENCE;
  }

  const found = reacquired || best.confidence >= MIN_LOCAL_MATCH_CONFIDENCE;
  const smoothing = reacquired ? 1 : best.confidence >= 0.78 ? 0.93 : 0.78;
  const resolvedCenterX = found
    ? currentCenterX + (best.centerX - currentCenterX) * smoothing
    : currentCenterX;
  const resolvedCenterY = found
    ? currentCenterY + (best.centerY - currentCenterY) * smoothing
    : currentCenterY;
  const displacementX = found ? (resolvedCenterX - currentCenterX) / frameWidth : 0;
  const displacementY = found ? (resolvedCenterY - currentCenterY) / frameHeight : 0;
  const rim = found
    ? {
        ...current.rim,
        x: clamp(resolvedCenterX / frameWidth - current.rim.width / 2, 0, 1 - current.rim.width),
        y: clamp(resolvedCenterY / frameHeight - current.rim.height / 2, 0, 1 - current.rim.height),
      }
    : current.rim;
  const velocityBlend = reacquired ? 0.25 : 0.62;
  const state: RimTrackState = {
    ...current,
    rim,
    framesProcessed: current.framesProcessed + 1,
    trackedFrames: current.trackedFrames + (found ? 1 : 0),
    lostFrames: current.lostFrames + (found ? 0 : 1),
    confidenceTotal: current.confidenceTotal + best.confidence,
    consecutiveLostFrames: found ? 0 : current.consecutiveLostFrames + 1,
    velocityX: found
      ? current.velocityX * (1 - velocityBlend) + displacementX * velocityBlend
      : current.velocityX * 0.45,
    velocityY: found
      ? current.velocityY * (1 - velocityBlend) + displacementY * velocityBlend
      : current.velocityY * 0.45,
    globalReacquisitions: current.globalReacquisitions + (reacquired ? 1 : 0),
  };

  return {
    rim,
    confidence: best.confidence,
    found,
    reacquired,
    displacementX,
    displacementY,
    state,
  };
}

/**
 * Lets a learned hoop detector provide the authoritative rim location while
 * preserving the same motion and diagnostics state used by template fallback.
 */
export function stepRimTrackerFromDetection(
  current: RimTrackState,
  detectedRim: RimCalibration,
  confidence: number,
): RimTrackStep {
  const currentCenterX = current.rim.x + current.rim.width / 2;
  const currentCenterY = current.rim.y + current.rim.height / 2;
  const detectedCenterX = detectedRim.x + detectedRim.width / 2;
  const detectedCenterY = detectedRim.y + detectedRim.height / 2;
  const displacementX = detectedCenterX - currentCenterX;
  const displacementY = detectedCenterY - currentCenterY;
  const resolvedConfidence = clamp(confidence, 0, 1);
  const state: RimTrackState = {
    ...current,
    rim: { ...detectedRim },
    framesProcessed: current.framesProcessed + 1,
    trackedFrames: current.trackedFrames + 1,
    confidenceTotal: current.confidenceTotal + resolvedConfidence,
    consecutiveLostFrames: 0,
    velocityX: current.velocityX * 0.3 + displacementX * 0.7,
    velocityY: current.velocityY * 0.3 + displacementY * 0.7,
    globalReacquisitions:
      current.globalReacquisitions + (current.consecutiveLostFrames > 0 ? 1 : 0),
  };
  return {
    rim: state.rim,
    confidence: resolvedConfidence,
    found: true,
    reacquired: current.consecutiveLostFrames > 0,
    displacementX,
    displacementY,
    state,
  };
}

/**
 * Imported-video analysis is calibrated for a fixed camera. When neither the
 * learned detector nor appearance template has a trustworthy match, retain
 * the user's exact rim instead of discarding otherwise valid ball detections.
 * Camera-motion diagnostics still force those videos to manual review.
 */
export function stepFixedRimTracker(current: RimTrackState): RimTrackStep {
  return stepRimTrackerFromDetection(current, current.rim, 0.84);
}
