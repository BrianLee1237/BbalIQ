import type { RimDetectionRegion } from "./rimDetectionRegion";

/** Crop original decoded pixels, then map detections back to analysis space. */
export function sourceCropGeometry(
  region: RimDetectionRegion,
  analysisWidth: number,
  analysisHeight: number,
  sourceWidth: number,
  sourceHeight: number,
) {
  const scaleX = sourceWidth / analysisWidth;
  const scaleY = sourceHeight / analysisHeight;
  const width = region.width * scaleX;
  const height = region.height * scaleY;
  const gain = Math.min(1, 640 / Math.max(width, height));
  const outputWidth = Math.max(1, Math.round(width * gain));
  const outputHeight = Math.max(1, Math.round(height * gain));
  return {
    left: region.left * scaleX,
    top: region.top * scaleY,
    width,
    height,
    outputWidth,
    outputHeight,
    toAnalysisX: region.width / outputWidth,
    toAnalysisY: region.height / outputHeight,
  };
}
