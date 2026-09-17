import type { BallDetection } from "../../types/tracking";

export interface FlightPrediction {
  x: number;
  y: number;
  /** Prediction-only search tolerances, in the input coordinate system. */
  toleranceX: number;
  toleranceY: number;
  residual: number;
  samples: number;
}

function polynomial(coefficients: number[], time: number): number {
  "worklet";
  return coefficients[0]! + coefficients[1]! * time + coefficients[2]! * time * time;
}

/** Small, time-normalized least-squares system; no image-space y=f(x) singularity. */
function fitAxis(times: number[], values: number[]): number[] | null {
  "worklet";
  const matrix = [[0,0,0,0], [0,0,0,0], [0,0,0,0]];
  for (let i = 0; i < times.length; i += 1) {
    const basis = [1, times[i]!, times[i]! * times[i]!];
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) matrix[row]![col]! += basis[row]! * basis[col]!;
      matrix[row]![3]! += basis[row]! * values[i]!;
    }
  }
  for (let col = 0; col < 3; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 3; row += 1) {
      if (Math.abs(matrix[row]![col]!) > Math.abs(matrix[pivot]![col]!)) pivot = row;
    }
    if (Math.abs(matrix[pivot]![col]!) < 1e-8) return null;
    [matrix[col], matrix[pivot]] = [matrix[pivot]!, matrix[col]!];
    const divisor = matrix[col]![col]!;
    for (let j = col; j < 4; j += 1) matrix[col]![j]! /= divisor;
    for (let row = 0; row < 3; row += 1) {
      if (row === col) continue;
      const factor = matrix[row]![col]!;
      for (let j = col; j < 4; j += 1) matrix[row]![j]! -= factor * matrix[col]![j]!;
    }
  }
  return matrix.map(row => row[3]!);
}

/**
 * Locally fits x(t), y(t) to OBSERVATIONS to guide reacquisition, never scoring.
 * This is a short-horizon image-space approximation, not a reconstructed 3D
 * trajectory. Unlike a floor homography, it does not pretend the airborne ball
 * lies on the court. Standard regression/consensus methods, independently
 * implemented after reviewing HanaFEKI and nvan21 (see research notes).
 */
export function predictBallFlight(history: BallDetection[], atMs: number): FlightPrediction | null {
  "worklet";
  const latest = history.at(-1);
  if (!latest || !Number.isFinite(atMs)) return null;
  const horizon = atMs - latest.at;
  if (horizon < 0 || horizon > 150) return null;
  const samples = history.filter(sample => latest.at - sample.at <= 450).slice(-10);
  if (samples.length < 5) return null;
  const span = latest.at - samples[0]!.at;
  if (span < 100 || horizon > span * 0.6) return null;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i]!;
    if (![sample.x,sample.y,sample.width,sample.height,sample.at,sample.confidence].every(Number.isFinite) ||
        sample.width <= 0 || sample.height <= 0 || sample.confidence < 0.35 || sample.confidence > 1 ||
        (i > 0 && (sample.at <= samples[i-1]!.at || sample.at - samples[i-1]!.at > 160)) ||
        (sample.trackId !== undefined && latest.trackId !== undefined && sample.trackId !== latest.trackId)) return null;
    const widthRatio = sample.width / latest.width;
    const heightRatio = sample.height / latest.height;
    if (widthRatio < 0.5 || widthRatio > 2 || heightRatio < 0.5 || heightRatio > 2) return null;
  }
  const times = samples.map(sample => (sample.at - latest.at) / span);
  // The latest observation must agree. Permit one older bad localization only;
  // do not fit through a recent impact or extrapolate across multiple outliers.
  let best: { x: number[]; y: number[]; error: number; retained: number } | null = null;
  for (let excluded = -1; excluded < samples.length - 1; excluded += 1) {
    if (excluded >= 0 && samples.length < 6) break;
    const indices = samples.map((_, i) => i).filter(i => i !== excluded);
    const x = fitAxis(indices.map(i => times[i]!), indices.map(i => samples[i]!.x));
    const y = fitAxis(indices.map(i => times[i]!), indices.map(i => samples[i]!.y));
    if (!x || !y) continue;
    const errors = samples.map((sample, i) => Math.hypot(
      (polynomial(x,times[i]!) - sample.x) / latest.width,
      (polynomial(y,times[i]!) - sample.y) / latest.height));
    const inliers = errors.filter(error => error <= 0.4).length;
    const error = Math.sqrt(indices.reduce((sum,i) => sum + errors[i]! ** 2,0) / indices.length);
    if (inliers < samples.length - (samples.length >= 6 ? 1 : 0) ||
        errors.at(-1)! > 0.25 || errors.at(-2)! > 0.4 || error > 0.22) continue;
    if (!best || error < best.error) best = {x,y,error,retained:indices.length};
  }
  if (!best) return null;
  const t = horizon / span;
  const x = polynomial(best.x,t);
  const y = polynomial(best.y,t);
  const travel = Math.hypot((x-latest.x)/latest.width,(y-latest.y)/latest.height);
  if (!Number.isFinite(travel) || travel > 6) return null;
  const tolerance = 0.6 + best.error * 2 + t * 0.6;
  return {x,y,toleranceX:latest.width*tolerance,toleranceY:latest.height*tolerance,
    residual:best.error,samples:best.retained};
}

export function matchesFlightPrediction(candidate: BallDetection, prediction: FlightPrediction): boolean {
  "worklet";
  return Math.hypot((candidate.x-prediction.x)/prediction.toleranceX,
    (candidate.y-prediction.y)/prediction.toleranceY) <= 1;
}
