import {
  ColorConversionCodes,
  ContourApproximationModes,
  DataTypes,
  Mat,
  MatVector,
  MorphShapes,
  MorphTypes,
  OpenCV,
  RetrievalModes,
  Scalar,
  Size,
  type Mat as OpenCvMat,
} from "react-native-fast-opencv";

import type { BallDetection } from "../../types/tracking";

const MAX_CANDIDATES = 8;

function clamp(value: number, minimum: number, maximum: number): number {
  "worklet";
  return Math.max(minimum, Math.min(maximum, value));
}

/**
 * Produces shape-scored basketball candidates from a BGR OpenCV Mat.
 * Color is only the first gate; fill, aspect ratio, circularity and size reject
 * most orange court markings, rims, clothing and compression artifacts.
 */
export function detectOrangeBallCandidates(
  source: OpenCvMat,
  width: number,
  height: number,
  at: number,
): BallDetection[] {
  "worklet";
  const hsv = Mat.create(0, 0, DataTypes.CV_8U);
  const mask = Mat.create(0, 0, DataTypes.CV_8U);
  const lowerOrange = Scalar.create(0, 42, 34);
  const upperOrange = Scalar.create(42, 255, 255);
  const kernelSize = Size.create(3, 3);
  const kernel = OpenCV.getStructuringElement(MorphShapes.MORPH_ELLIPSE, kernelSize);
  const contours = MatVector.create();

  try {
    OpenCV.cvtColor(source, hsv, ColorConversionCodes.COLOR_BGR2HSV);
    OpenCV.inRange(hsv, lowerOrange, upperOrange, mask);
    OpenCV.medianBlur(mask, mask, 3);
    OpenCV.morphologyEx(mask, mask, MorphTypes.MORPH_CLOSE, kernel);
    OpenCV.findContours(
      mask,
      contours,
      RetrievalModes.RETR_EXTERNAL,
      ContourApproximationModes.CHAIN_APPROX_SIMPLE,
    );

    const frameArea = Math.max(1, width * height);
    const minimumArea = Math.max(3, frameArea * 0.000035);
    const maximumArea = frameArea * 0.12;
    const candidates: BallDetection[] = [];

    for (let index = 0; index < contours.length; index += 1) {
      const contour = contours.get(index);
      const area = OpenCV.contourArea(contour, false).value;
      if (area < minimumArea || area > maximumArea) continue;

      const rect = OpenCV.boundingRect(contour);
      const rectArea = Math.max(1, rect.width * rect.height);
      const ratio = rect.width / Math.max(1, rect.height);
      const fill = clamp(area / rectArea, 0, 1);
      const perimeter = OpenCV.arcLength(contour, true).value;
      const circularity = perimeter > 0
        ? clamp((4 * Math.PI * area) / (perimeter * perimeter), 0, 1)
        : 0;
      const roundness = 1 - clamp(Math.abs(1 - ratio), 0, 1);

      const validShape =
        rect.width >= 2 &&
        rect.height >= 2 &&
        rect.width <= width * 0.26 &&
        rect.height <= height * 0.4 &&
        ratio >= 0.4 &&
        ratio <= 2.4 &&
        fill >= 0.27 &&
        circularity >= 0.24;

      if (!validShape) continue;

      const confidence = clamp(
        0.28 + fill * 0.22 + roundness * 0.2 + circularity * 0.34,
        0,
        0.99,
      );
      if (confidence < 0.52) continue;

      candidates.push({
        x: (rect.x + rect.width / 2) / width,
        y: (rect.y + rect.height / 2) / height,
        width: rect.width / width,
        height: rect.height / height,
        confidence,
        appearanceConfidence: confidence,
        motionConfidence: 0.5,
        at,
      });
    }

    candidates.sort((left, right) => right.confidence - left.confidence);
    return candidates.slice(0, MAX_CANDIDATES);
  } finally {
    hsv.release();
    mask.release();
    lowerOrange.release();
    upperOrange.release();
    kernelSize.release();
    kernel.release();
    contours.release();
  }
}
