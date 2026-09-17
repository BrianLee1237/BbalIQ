import type { BallDetection, RimCalibration } from "../../types/tracking";

const MAX_CANDIDATES = 20;

export interface PixelFrame {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface PixelBallDetectionResult {
  candidates: BallDetection[];
  gray: Uint8Array;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function isBasketballColor(red: number, green: number, blue: number): boolean {
  const maximum = Math.max(red, green, blue);
  const minimum = Math.min(red, green, blue);
  if (maximum < 34 || maximum === minimum) return false;
  const saturation = (maximum - minimum) / maximum;
  const warmDominance = red - blue;
  return (
    saturation >= 0.16 &&
    warmDominance >= 7 &&
    red >= green * 0.94 &&
    green >= blue * 0.72
  );
}

function overlapRatio(left: BallDetection, right: BallDetection): number {
  const leftEdge = Math.max(left.x - left.width / 2, right.x - right.width / 2);
  const rightEdge = Math.min(left.x + left.width / 2, right.x + right.width / 2);
  const topEdge = Math.max(left.y - left.height / 2, right.y - right.height / 2);
  const bottomEdge = Math.min(left.y + left.height / 2, right.y + right.height / 2);
  const intersection = Math.max(0, rightEdge - leftEdge) * Math.max(0, bottomEdge - topEdge);
  const union = left.width * left.height + right.width * right.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function collectComponents(
  mask: Uint8Array,
  secondaryMask: Uint8Array,
  width: number,
  height: number,
  rim: RimCalibration,
  at: number,
  source: "appearance" | "motion",
): BallDetection[] {
  const visited = new Uint8Array(mask.length);
  const queue = new Int32Array(mask.length);
  const candidates: BallDetection[] = [];
  const rimWidthPixels = Math.max(4, rim.width * width);
  const expectedBallDiameter = Math.max(3, rimWidthPixels * 0.5);
  const maximumDimension = Math.max(12, rimWidthPixels * 1.45);
  const minimumArea = source === "appearance" ? 3 : 2;
  const rimCenterX = (rim.x + rim.width / 2) * width;
  const rimPlaneY = (rim.y + rim.height * 0.48) * height;

  for (let seed = 0; seed < mask.length; seed += 1) {
    if (mask[seed] !== 1 || visited[seed] === 1) continue;
    let head = 0;
    let tail = 1;
    queue[0] = seed;
    visited[seed] = 1;
    let area = 0;
    let secondaryPixels = 0;
    let minimumX = width;
    let maximumX = 0;
    let minimumY = height;
    let maximumY = 0;

    while (head < tail) {
      const index = queue[head] ?? 0;
      head += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      area += 1;
      secondaryPixels += secondaryMask[index] ?? 0;
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);

      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          if (offsetX === 0 && offsetY === 0) continue;
          const nextX = x + offsetX;
          const nextY = y + offsetY;
          if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) continue;
          const next = nextY * width + nextX;
          if (mask[next] !== 1 || visited[next] === 1) continue;
          visited[next] = 1;
          queue[tail] = next;
          tail += 1;
        }
      }
    }

    if (area < minimumArea) continue;
    const rectWidth = maximumX - minimumX + 1;
    const rectHeight = maximumY - minimumY + 1;
    if (
      rectWidth < 2 ||
      rectHeight < 2 ||
      rectWidth > maximumDimension ||
      rectHeight > maximumDimension * 1.35
    ) {
      continue;
    }
    const ratio = rectWidth / Math.max(1, rectHeight);
    if (ratio < 0.25 || ratio > 4) continue;
    const rectArea = Math.max(1, rectWidth * rectHeight);
    const fill = area / rectArea;
    if (fill < (source === "appearance" ? 0.18 : 0.08)) continue;

    const centerX = minimumX + rectWidth / 2;
    const centerY = minimumY + rectHeight / 2;
    const distanceFromRim = Math.hypot(
      (centerX - rimCenterX) / rimWidthPixels,
      (centerY - rimPlaneY) / rimWidthPixels,
    );
    if (distanceFromRim > 9.5) continue;

    const roundness = 1 - clamp(Math.abs(1 - ratio) / 1.6, 0, 1);
    const diameter = Math.sqrt(rectWidth * rectHeight);
    const sizeScore = 1 - clamp(
      Math.abs(diameter - expectedBallDiameter) / Math.max(expectedBallDiameter, 4),
      0,
      1,
    );
    const secondaryRatio = clamp(secondaryPixels / Math.max(1, area), 0, 1);
    const proximityScore = 1 - clamp(distanceFromRim / 9.5, 0, 1);
    const appearanceConfidence = source === "appearance"
      ? clamp(0.46 + fill * 0.2 + roundness * 0.2 + sizeScore * 0.14, 0, 0.99)
      : secondaryRatio;
    const motionConfidence = source === "motion"
      ? clamp(0.5 + fill * 0.16 + roundness * 0.16 + sizeScore * 0.1, 0, 0.99)
      : secondaryRatio;
    const confidence = source === "appearance"
      ? clamp(
          appearanceConfidence * 0.58 + motionConfidence * 0.28 + proximityScore * 0.14,
          0,
          0.99,
        )
      : clamp(
          motionConfidence * 0.5 + appearanceConfidence * 0.24 +
            sizeScore * 0.14 + proximityScore * 0.12,
          0,
          0.96,
        );
    if (confidence < (source === "appearance" ? 0.42 : 0.44)) continue;

    candidates.push({
      x: centerX / width,
      y: centerY / height,
      width: rectWidth / width,
      height: rectHeight / height,
      confidence,
      motionConfidence,
      appearanceConfidence,
      at,
    });
  }
  return candidates;
}

export function detectBasketballCandidates(
  pixels: PixelFrame,
  previousGray: Uint8Array | null,
  at: number,
  rim: RimCalibration,
): PixelBallDetectionResult {
  const { width, height, data } = pixels;
  const pixelCount = width * height;
  const gray = new Uint8Array(pixelCount);
  const appearanceMask = new Uint8Array(pixelCount);
  const motionMask = new Uint8Array(pixelCount);

  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    const red = data[offset] ?? 0;
    const green = data[offset + 1] ?? 0;
    const blue = data[offset + 2] ?? 0;
    const luminance = Math.round(red * 0.299 + green * 0.587 + blue * 0.114);
    gray[index] = luminance;
    const basketballColor = isBasketballColor(red, green, blue);
    if (basketballColor) appearanceMask[index] = 1;
    if (
      previousGray &&
      previousGray.length === pixelCount &&
      (
        Math.abs(luminance - (previousGray[index] ?? luminance)) >= 17 ||
        (basketballColor && Math.abs(luminance - (previousGray[index] ?? luminance)) >= 3)
      )
    ) {
      motionMask[index] = 1;
    }
  }

  const candidates = [
    ...collectComponents(
      appearanceMask,
      motionMask,
      width,
      height,
      rim,
      at,
      "appearance",
    ),
    ...(previousGray
      ? collectComponents(
          motionMask,
          appearanceMask,
          width,
          height,
          rim,
          at,
          "motion",
        )
      : []),
  ].sort((left, right) => right.confidence - left.confidence);

  const merged: BallDetection[] = [];
  for (const candidate of candidates) {
    const duplicate = merged.find(
      (existing) =>
        overlapRatio(existing, candidate) >= 0.28 ||
        Math.hypot(existing.x - candidate.x, existing.y - candidate.y) <
          Math.max(existing.width, candidate.width) * 0.45,
    );
    if (duplicate) {
      duplicate.confidence = Math.max(duplicate.confidence, candidate.confidence);
      duplicate.motionConfidence = Math.max(
        duplicate.motionConfidence ?? 0,
        candidate.motionConfidence ?? 0,
      );
      duplicate.appearanceConfidence = Math.max(
        duplicate.appearanceConfidence ?? 0,
        candidate.appearanceConfidence ?? 0,
      );
      continue;
    }
    merged.push(candidate);
    if (merged.length >= MAX_CANDIDATES) break;
  }

  return { candidates: merged, gray };
}
