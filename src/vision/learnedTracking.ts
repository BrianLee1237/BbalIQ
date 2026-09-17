import type { BallDetection, RimCalibration } from "../../types/tracking";
import type {
  LearnedBasketballFrame,
  LearnedObjectDetection,
} from "./learnedBasketballDetector.web";

export interface PixelBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
  confidence: number;
  trackId?: number;
}

/**
 * Converts the calibrated rim opening into the pixel-space hoop observation
 * consumed by the shot state machine. The learned detector may describe the
 * whole backboard/net assembly, which is useful for reacquisition but too
 * unstable to define the actual scoring plane. Imported-video scoring should
 * therefore remain anchored to the rim the user approved.
 */
export function calibratedRimToScoringHoop(
  rim: RimCalibration,
  frameWidth: number,
  frameHeight: number,
): PixelBox {
  const width = Math.max(4, rim.width * frameWidth);
  // Keep a non-zero rim plane even when a very thin box was drawn. A modest
  // fraction of the rim width matches the visible front/back rim thickness
  // without expanding the crossing zone into the backboard or net.
  const height = Math.max(4, rim.height * frameHeight, width * 0.16);
  const centerX = (rim.x + rim.width / 2) * frameWidth;
  const centerY = (rim.y + rim.height / 2) * frameHeight;
  return {
    left: centerX - width / 2,
    top: centerY - height / 2,
    right: centerX + width / 2,
    bottom: centerY + height / 2,
    confidence: 0.98,
  };
}

export interface HoopRimAnchor {
  centerOffsetX: number;
  centerOffsetY: number;
  widthScale: number;
  heightScale: number;
}

export interface AutomaticHoopChoice {
  hoop: PixelBox;
  confidence: number;
  ambiguous: boolean;
}

function intersectionOverUnion(left: PixelBox, right: PixelBox): number {
  const intersectionWidth = Math.max(
    0,
    Math.min(left.right, right.right) - Math.max(left.left, right.left),
  );
  const intersectionHeight = Math.max(
    0,
    Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top),
  );
  const intersection = intersectionWidth * intersectionHeight;
  const union = boxWidth(left) * boxHeight(left) +
    boxWidth(right) * boxHeight(right) - intersection;
  return intersection / Math.max(1, union);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function boxWidth(box: PixelBox): number {
  return Math.max(1, box.right - box.left);
}

function boxHeight(box: PixelBox): number {
  return Math.max(1, box.bottom - box.top);
}

function boxCenter(box: PixelBox): { x: number; y: number } {
  return {
    x: (box.left + box.right) / 2,
    y: (box.top + box.bottom) / 2,
  };
}

/**
 * Selects the most useful hoop without a user-provided spatial hint. Detector
 * confidence remains the primary signal; visible size is only a tie-breaker
 * so a tiny high-confidence target is not displaced by a large false box.
 */
export function chooseAutomaticHoop(
  hoops: PixelBox[],
  frameWidth: number,
  frameHeight: number,
): AutomaticHoopChoice | null {
  const frameArea = Math.max(1, frameWidth * frameHeight);
  const ranked = hoops
    .filter((hoop) => boxWidth(hoop) >= 8 && boxHeight(hoop) >= 5)
    .map((hoop) => {
      const visibleArea = boxWidth(hoop) * boxHeight(hoop) / frameArea;
      return {
        hoop,
        score: hoop.confidence * 0.88 + Math.min(0.12, Math.sqrt(visibleArea) * 0.42),
      };
    })
    .sort((left, right) => right.score - left.score);
  const first = ranked[0];
  if (!first) return null;
  const second = ranked[1];
  return {
    hoop: first.hoop,
    confidence: clamp(first.score, 0, 1),
    ambiguous: Boolean(second && first.score - second.score < 0.08),
  };
}

/**
 * Converts the learned detector's broader hoop object box into the narrow rim
 * opening used by the entry/exit state machine. This geometry is the fallback
 * for rims whose paint cannot be refined from frame pixels.
 */
export function rimFromAutomaticHoop(
  hoop: PixelBox,
  frameWidth: number,
  frameHeight: number,
): RimCalibration {
  const hoopWidth = boxWidth(hoop);
  const hoopHeight = boxHeight(hoop);
  const center = boxCenter(hoop);
  const rimWidthPixels = clamp(hoopWidth * 0.64, frameWidth * 0.035, frameWidth * 0.5);
  const rimHeightPixels = clamp(
    rimWidthPixels / 4.2,
    frameHeight * 0.012,
    Math.min(frameHeight * 0.28, hoopHeight * 0.3),
  );
  const rimCenterY = hoop.top + hoopHeight * 0.36;
  const width = rimWidthPixels / frameWidth;
  const height = rimHeightPixels / frameHeight;
  return {
    x: clamp(center.x / frameWidth - width / 2, 0, 1 - width),
    y: clamp(rimCenterY / frameHeight - height / 2, 0, 1 - height),
    width,
    height,
  };
}

export function learnedDetectionToPixelBox(detection: LearnedObjectDetection): PixelBox {
  return {
    left: detection.left,
    top: detection.top,
    right: detection.right,
    bottom: detection.bottom,
    confidence: detection.confidence,
  };
}

/**
 * Class-aware NMS across full-frame, rim, and predicted-track detector scales.
 * A focused crop commonly reports the same ball at higher confidence; keeping
 * both would split one physical shot into two tracker observations.
 */
export function mergeLearnedFrames(
  frames: LearnedBasketballFrame[],
): LearnedBasketballFrame {
  const selected: LearnedObjectDetection[] = [];
  const detections = frames
    .flatMap((frame) => frame.objects)
    .sort((left, right) => right.confidence - left.confidence);
  for (const detection of detections) {
    const box = learnedDetectionToPixelBox(detection);
    const duplicate = selected.some((existing) =>
      existing.classId === detection.classId &&
      intersectionOverUnion(learnedDetectionToPixelBox(existing), box) >= 0.28
    );
    if (!duplicate) selected.push(detection);
  }
  return {
    objects: selected,
    basketballs: selected.filter((detection) => detection.label === "ball"),
    hoops: selected.filter((detection) => detection.label === "hoop"),
    players: frames.flatMap((frame) => frame.players),
  };
}

export function trackRowToPixelBox(row: number[]): PixelBox | null {
  const [left, top, right, bottom, trackId, confidence] = row;
  if (
    !Number.isFinite(left) ||
    !Number.isFinite(top) ||
    !Number.isFinite(right) ||
    !Number.isFinite(bottom) ||
    !Number.isFinite(confidence) ||
    (right ?? 0) <= (left ?? 0) ||
    (bottom ?? 0) <= (top ?? 0)
  ) {
    return null;
  }
  return {
    left: left ?? 0,
    top: top ?? 0,
    right: right ?? 0,
    bottom: bottom ?? 0,
    confidence: confidence ?? 0,
    trackId: Number.isFinite(trackId) ? trackId : undefined,
  };
}

export function toByteTrackDetections(boxes: PixelBox[]): {
  xywh: number[][];
  conf: number[];
  cls: number[];
} {
  return {
    xywh: boxes.map((box) => [
      (box.left + box.right) / 2,
      (box.top + box.bottom) / 2,
      boxWidth(box),
      boxHeight(box),
    ]),
    conf: boxes.map((box) => box.confidence),
    cls: boxes.map(() => 0),
  };
}

export function chooseBoxNearReference(
  boxes: PixelBox[],
  reference: PixelBox,
  preferredTrackId?: number,
): PixelBox | null {
  if (boxes.length === 0) return null;
  const preferred = preferredTrackId === undefined
    ? null
    : boxes.find((box) => box.trackId === preferredTrackId) ?? null;
  if (preferred) return preferred;

  const referenceCenter = boxCenter(reference);
  const referenceDiagonal = Math.max(1, Math.hypot(boxWidth(reference), boxHeight(reference)));
  let best: PixelBox | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const box of boxes) {
    const center = boxCenter(box);
    const normalizedDistance = Math.hypot(
      center.x - referenceCenter.x,
      center.y - referenceCenter.y,
    ) / referenceDiagonal;
    const sizeChange = Math.abs(Math.log(boxWidth(box) / boxWidth(reference))) +
      Math.abs(Math.log(boxHeight(box) / boxHeight(reference)));
    const score = box.confidence * 0.48 - normalizedDistance * 0.38 - sizeChange * 0.14;
    if (score > bestScore) {
      best = box;
      bestScore = score;
    }
  }
  return best;
}

export function chooseCalibrationHoop(
  hoops: PixelBox[],
  rim: RimCalibration,
  frameWidth: number,
  frameHeight: number,
): PixelBox | null {
  if (hoops.length === 0) return null;
  const rimCenterX = (rim.x + rim.width / 2) * frameWidth;
  const rimCenterY = (rim.y + rim.height / 2) * frameHeight;
  let best: PixelBox | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const hoop of hoops) {
    const width = boxWidth(hoop);
    const height = boxHeight(hoop);
    const center = boxCenter(hoop);
    const containsRimCenter =
      rimCenterX >= hoop.left - width * 0.2 &&
      rimCenterX <= hoop.right + width * 0.2 &&
      rimCenterY >= hoop.top - height * 0.2 &&
      rimCenterY <= hoop.bottom + height * 0.2;
    const normalizedDistance = Math.hypot(
      center.x - rimCenterX,
      center.y - rimCenterY,
    ) / Math.max(1, Math.hypot(width, height));
    // A manually marked rim is a strong spatial prior. Do not let a weak
    // detector response elsewhere on the court steal the hoop lock.
    if (!containsRimCenter && normalizedDistance > 1.8) continue;
    const score = hoop.confidence + (containsRimCenter ? 1 : 0) - normalizedDistance * 0.45;
    if (score > bestScore) {
      best = hoop;
      bestScore = score;
    }
  }
  return best;
}

/**
 * Adapts avishah3's basketball data cleaning before association. Very thin
 * boxes, oversized rim/support artifacts, and implausibly tiny detections are
 * removed, while mild motion blur remains valid.
 */
export function filterPlausibleBasketballBoxes(
  boxes: PixelBox[],
  rim: RimCalibration,
  frameWidth: number,
  frameHeight: number,
): PixelBox[] {
  const rimWidthPixels = Math.max(4, rim.width * frameWidth);
  return boxes.filter((box) => {
    const width = boxWidth(box);
    const height = boxHeight(box);
    const aspect = Math.max(width / height, height / width);
    const diameter = Math.sqrt(width * height);
    const center = boxCenter(box);
    const rimCenterX = (rim.x + rim.width / 2) * frameWidth;
    const rimPlaneY = (rim.y + rim.height * 0.48) * frameHeight;
    const nearRim =
      Math.abs(center.x - rimCenterX) <= rimWidthPixels * 3.1 &&
      center.y >= rimPlaneY - rimWidthPixels * 6.2 &&
      center.y <= rimPlaneY + rimWidthPixels * 3.6;
    const minimumDiameter = nearRim ? rimWidthPixels * 0.11 : rimWidthPixels * 0.08;
    return (
      aspect <= 1.85 &&
      diameter >= Math.max(2, minimumDiameter) &&
      diameter <= rimWidthPixels * 1.18
    );
  });
}

export function createHoopRimAnchor(
  rim: RimCalibration,
  hoop: PixelBox,
  frameWidth: number,
  frameHeight: number,
): HoopRimAnchor {
  const hoopWidth = boxWidth(hoop);
  const hoopHeight = boxHeight(hoop);
  const center = boxCenter(hoop);
  const rimCenterX = (rim.x + rim.width / 2) * frameWidth;
  const rimCenterY = (rim.y + rim.height / 2) * frameHeight;
  return {
    centerOffsetX: (rimCenterX - center.x) / hoopWidth,
    centerOffsetY: (rimCenterY - center.y) / hoopHeight,
    widthScale: (rim.width * frameWidth) / hoopWidth,
    heightScale: (rim.height * frameHeight) / hoopHeight,
  };
}

export function rimFromTrackedHoop(
  hoop: PixelBox,
  anchor: HoopRimAnchor,
  frameWidth: number,
  frameHeight: number,
): RimCalibration {
  const hoopWidth = boxWidth(hoop);
  const hoopHeight = boxHeight(hoop);
  const center = boxCenter(hoop);
  const rimWidth = clamp(hoopWidth * anchor.widthScale / frameWidth, 0.02, 0.5);
  const rimHeight = clamp(hoopHeight * anchor.heightScale / frameHeight, 0.008, 0.28);
  const rimCenterX = (center.x + anchor.centerOffsetX * hoopWidth) / frameWidth;
  const rimCenterY = (center.y + anchor.centerOffsetY * hoopHeight) / frameHeight;
  return {
    x: clamp(rimCenterX - rimWidth / 2, 0, 1 - rimWidth),
    y: clamp(rimCenterY - rimHeight / 2, 0, 1 - rimHeight),
    width: rimWidth,
    height: rimHeight,
  };
}

export function pixelBoxToBallDetection(
  box: PixelBox,
  frameWidth: number,
  frameHeight: number,
  at: number,
): BallDetection {
  const width = boxWidth(box) / frameWidth;
  const height = boxHeight(box) / frameHeight;
  return {
    trackId: box.trackId,
    x: ((box.left + box.right) / 2) / frameWidth,
    y: ((box.top + box.bottom) / 2) / frameHeight,
    width,
    height,
    // Keep the detector score honest. ByteTrack continuity raises motion
    // confidence, but a weak raw box must never become a near-certain ball.
    confidence: clamp(0.55 + box.confidence * 0.45, 0, 1),
    motionConfidence: box.trackId === undefined ? 0.3 : 0.86,
    appearanceConfidence: clamp(box.confidence, 0, 1),
    at,
  };
}

/**
 * Keeps ByteTrack's persistent observations while retaining unmatched weak
 * detections for its downstream low-confidence/rim-zone recovery pass.
 * Duplicate raw boxes must not create a second logical basketball.
 */
export function mergeTrackedAndRawBoxes(
  tracked: PixelBox[],
  raw: PixelBox[],
): PixelBox[] {
  if (tracked.length === 0) return raw;
  const merged = [...tracked];
  for (const candidate of raw) {
    const duplicate = tracked.some((existing) => {
      const centerDistance = Math.hypot(
        boxCenter(existing).x - boxCenter(candidate).x,
        boxCenter(existing).y - boxCenter(candidate).y,
      );
      return intersectionOverUnion(existing, candidate) >= 0.18 ||
        centerDistance <= Math.max(
          2,
          Math.min(boxWidth(existing), boxHeight(existing)) * 0.42,
        );
    });
    if (!duplicate) merged.push(candidate);
  }
  return merged;
}

export function mergeLearnedAndMotionCandidates(
  learned: BallDetection[],
  motion: BallDetection[],
): BallDetection[] {
  if (learned.length === 0) return motion;
  const merged = [...learned];
  for (const candidate of motion) {
    const duplicate = learned.some((detection) => {
      const allowedDistance = Math.max(
        0.025,
        Math.max(detection.width, detection.height, candidate.width, candidate.height) * 1.4,
      );
      return Math.hypot(detection.x - candidate.x, detection.y - candidate.y) <= allowedDistance;
    });
    if (!duplicate) merged.push(candidate);
  }
  return merged;
}
