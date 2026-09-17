import type { PlayerDetection } from "../../types/tracking";

export interface PlayerMotionState {
  previousGray: Uint8Array;
  gridWidth: number;
  gridHeight: number;
  current: PlayerDetection | null;
  missingFrames: number;
}

export interface PlayerMotionResult {
  detection: PlayerDetection | null;
  state: PlayerMotionState;
}

export function createPlayerMotionState(): PlayerMotionState {
  "worklet";
  return {
    previousGray: new Uint8Array(0),
    gridWidth: 0,
    gridHeight: 0,
    current: null,
    missingFrames: 0,
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  "worklet";
  return Math.max(minimum, Math.min(maximum, value));
}

function dilateMask(source: Uint8Array, width: number, height: number): Uint8Array {
  "worklet";
  const output = new Uint8Array(source.length);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      if (
        source[index] ||
        source[index - 1] ||
        source[index + 1] ||
        source[index - width] ||
        source[index + width]
      ) {
        output[index] = 1;
      }
    }
  }
  return output;
}

function smoothDetection(
  current: PlayerDetection | null,
  candidate: PlayerDetection,
): PlayerDetection {
  "worklet";
  if (!current || candidate.at - current.at > 900) return candidate;
  const alpha = 0.42;
  return {
    x: current.x * (1 - alpha) + candidate.x * alpha,
    y: current.y * (1 - alpha) + candidate.y * alpha,
    width: current.width * (1 - alpha) + candidate.width * alpha,
    height: current.height * (1 - alpha) + candidate.height * alpha,
    confidence: clamp(current.confidence * 0.35 + candidate.confidence * 0.65, 0, 0.99),
    at: candidate.at,
  };
}

/**
 * Tracks the dominant moving person for a fixed courtside camera. This is a
 * deliberately separate signal from basketball color detection, so clothing
 * and skin tones cannot be mistaken for the ball. Camera-wide motion is
 * rejected as a shake/reposition event.
 */
export function trackMovingPlayer(
  bgrPixels: Uint8Array,
  width: number,
  height: number,
  state: PlayerMotionState,
  at: number,
): PlayerMotionResult {
  "worklet";
  const samplingStep = width >= 200 ? 2 : 1;
  const gridWidth = Math.max(1, Math.floor(width / samplingStep));
  const gridHeight = Math.max(1, Math.floor(height / samplingStep));
  const gray = new Uint8Array(gridWidth * gridHeight);

  for (let y = 0; y < gridHeight; y += 1) {
    for (let x = 0; x < gridWidth; x += 1) {
      const sourceX = x * samplingStep;
      const sourceY = y * samplingStep;
      const sourceIndex = (sourceY * width + sourceX) * 3;
      const blue = bgrPixels[sourceIndex] ?? 0;
      const green = bgrPixels[sourceIndex + 1] ?? 0;
      const red = bgrPixels[sourceIndex + 2] ?? 0;
      gray[y * gridWidth + x] = Math.round(red * 0.299 + green * 0.587 + blue * 0.114);
    }
  }

  if (
    state.previousGray.length !== gray.length ||
    state.gridWidth !== gridWidth ||
    state.gridHeight !== gridHeight
  ) {
    return {
      detection: null,
      state: {
        previousGray: gray,
        gridWidth,
        gridHeight,
        current: null,
        missingFrames: 0,
      },
    };
  }

  const motion = new Uint8Array(gray.length);
  let changedPixels = 0;
  for (let index = 0; index < gray.length; index += 1) {
    if (Math.abs((gray[index] ?? 0) - (state.previousGray[index] ?? 0)) >= 19) {
      motion[index] = 1;
      changedPixels += 1;
    }
  }

  const cameraMoved = changedPixels / Math.max(1, gray.length) > 0.28;
  let expandedMotion = dilateMask(motion, gridWidth, gridHeight);
  expandedMotion = dilateMask(expandedMotion, gridWidth, gridHeight);
  const visited = new Uint8Array(expandedMotion.length);
  const queue = new Int32Array(expandedMotion.length);
  let best: PlayerDetection | null = null;
  let bestScore = -1;

  if (!cameraMoved) {
    for (let seed = 0; seed < expandedMotion.length; seed += 1) {
      if (!expandedMotion[seed] || visited[seed]) continue;
      let head = 0;
      let tail = 0;
      queue[tail] = seed;
      tail += 1;
      visited[seed] = 1;
      let area = 0;
      let minimumX = gridWidth;
      let maximumX = 0;
      let minimumY = gridHeight;
      let maximumY = 0;

      while (head < tail) {
        const index = queue[head] ?? 0;
        head += 1;
        const x = index % gridWidth;
        const y = Math.floor(index / gridWidth);
        area += 1;
        minimumX = Math.min(minimumX, x);
        maximumX = Math.max(maximumX, x);
        minimumY = Math.min(minimumY, y);
        maximumY = Math.max(maximumY, y);

        const left = index - 1;
        const right = index + 1;
        const above = index - gridWidth;
        const below = index + gridWidth;
        if (x > 0 && expandedMotion[left] && !visited[left]) {
          visited[left] = 1;
          queue[tail] = left;
          tail += 1;
        }
        if (x < gridWidth - 1 && expandedMotion[right] && !visited[right]) {
          visited[right] = 1;
          queue[tail] = right;
          tail += 1;
        }
        if (y > 0 && expandedMotion[above] && !visited[above]) {
          visited[above] = 1;
          queue[tail] = above;
          tail += 1;
        }
        if (y < gridHeight - 1 && expandedMotion[below] && !visited[below]) {
          visited[below] = 1;
          queue[tail] = below;
          tail += 1;
        }
      }

      const componentWidth = maximumX - minimumX + 1;
      const componentHeight = maximumY - minimumY + 1;
      const normalizedWidth = componentWidth / gridWidth;
      const normalizedHeight = componentHeight / gridHeight;
      const normalizedArea = area / Math.max(1, gridWidth * gridHeight);
      if (
        normalizedHeight < 0.19 ||
        normalizedWidth < 0.025 ||
        normalizedWidth > 0.66 ||
        normalizedArea < 0.004
      ) {
        continue;
      }

      const boxWidth = clamp(normalizedWidth * 2.05, 0.13, 0.58);
      const boxHeight = clamp(normalizedHeight * 1.55, 0.35, 0.96);
      const motionCenterX = (minimumX + componentWidth / 2) / gridWidth;
      const motionCenterY = (minimumY + componentHeight / 2) / gridHeight;
      const x = clamp(motionCenterX - boxWidth / 2, 0, 1 - boxWidth);
      const y = clamp(motionCenterY - boxHeight * 0.56, 0, 1 - boxHeight);
      const centerDistance = state.current
        ? Math.hypot(
            x + boxWidth / 2 - (state.current.x + state.current.width / 2),
            y + boxHeight / 2 - (state.current.y + state.current.height / 2),
          )
        : 0;
      if (
        state.current &&
        state.current.confidence > 0.35 &&
        state.missingFrames < 4 &&
        centerDistance > 0.32
      ) {
        continue;
      }
      const continuity = state.current
        ? 1 - clamp(centerDistance / 0.5, 0, 1)
        : 0.65;
      const heightScore = clamp((boxHeight - 0.3) / 0.55, 0, 1);
      const areaScore = clamp(normalizedArea / 0.08, 0, 1);
      const score = continuity * 0.58 + heightScore * 0.25 + areaScore * 0.17;

      if (score > bestScore) {
        bestScore = score;
        best = {
          x,
          y,
          width: boxWidth,
          height: boxHeight,
          confidence: clamp(0.55 + score * 0.4, 0, 0.96),
          at,
        };
      }
    }
  }

  if (best) {
    const detection = smoothDetection(state.current, best);
    return {
      detection,
      state: {
        previousGray: gray,
        gridWidth,
        gridHeight,
        current: detection,
        missingFrames: 0,
      },
    };
  }

  const missingFrames = state.missingFrames + 1;
  const retained = state.current && missingFrames <= 24
    ? {
        ...state.current,
        confidence: state.current.confidence * 0.94,
        at,
      }
    : null;
  return {
    detection: retained,
    state: {
      previousGray: gray,
      gridWidth,
      gridHeight,
      current: retained,
      missingFrames,
    },
  };
}
