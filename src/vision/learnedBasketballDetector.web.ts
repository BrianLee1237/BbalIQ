const MODEL_URL = "/models/attalla-basketball-yolov8n.onnx";
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
const RUNTIME_URL = `${ORT_CDN}ort.all.min.js`;
const WASM_PATH = ORT_CDN;
const LABELS = ["ball", "hoop"] as const;
const MODEL_SIZE = 640;
const CLASS_CONFIDENCE: Record<LearnedBasketballLabel, number> = {
  // Retain weak ball candidates for the rim-zone gate. Net/rim occlusion often
  // depresses confidence exactly where the crossing must be observed; strict
  // shape, scale, location, association, and shot-state checks run downstream.
  ball: 0.12,
  hoop: 0.2,
};
const NMS_IOU = 0.7;

export type LearnedBasketballLabel = (typeof LABELS)[number];

export interface LearnedObjectDetection {
  classId: number;
  label: LearnedBasketballLabel;
  confidence: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface LearnedBasketballFrame {
  objects: LearnedObjectDetection[];
  basketballs: LearnedObjectDetection[];
  hoops: LearnedObjectDetection[];
  players: LearnedObjectDetection[];
}

export interface LearnedBasketballDetector {
  backend: "webgpu" | "wasm";
  detect(source: CanvasImageSource): Promise<LearnedBasketballFrame>;
}

interface OrtTensorResult {
  data: Float32Array | Uint16Array;
  dims: readonly number[];
}

interface OrtSession {
  inputNames: readonly string[];
  outputNames: readonly string[];
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensorResult>>;
}

interface OrtTensor {
  data: Float32Array;
  dims: readonly number[];
}

interface OrtRuntime {
  env: {
    wasm: {
      wasmPaths: string;
      numThreads: number;
      proxy: boolean;
    };
  };
  Tensor: new (
    type: "float32",
    data: Float32Array,
    dims: readonly number[],
  ) => OrtTensor;
  InferenceSession: {
    create(
      model: string,
      options: { executionProviders: string[]; graphOptimizationLevel: "all" },
    ): Promise<OrtSession>;
  };
}

interface PreparedFrame {
  tensorData: Float32Array;
  sourceWidth: number;
  sourceHeight: number;
  gain: number;
  padX: number;
  padY: number;
}

declare global {
  interface Window {
    ort?: OrtRuntime;
  }
}

let detectorPromise: Promise<LearnedBasketballDetector> | null = null;
let runtimePromise: Promise<OrtRuntime> | null = null;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function supportsWebGpu(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

function loadRuntime(): Promise<OrtRuntime> {
  if (window.ort) return Promise.resolve(window.ort);
  if (runtimePromise) return runtimePromise;
  const created = new Promise<OrtRuntime>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${RUNTIME_URL}"]`);
    const script = existing ?? document.createElement("script");
    const timeout = window.setTimeout(
      () => reject(new Error("The on-device basketball detector took too long to load.")),
      30_000,
    );
    const cleanup = () => {
      window.clearTimeout(timeout);
      script.removeEventListener("load", handleLoad);
      script.removeEventListener("error", handleError);
    };
    const handleLoad = () => {
      cleanup();
      if (window.ort) resolve(window.ort);
      else reject(new Error("The on-device basketball detector could not start."));
    };
    const handleError = () => {
      cleanup();
      reject(new Error("The on-device basketball detector could not be downloaded."));
    };
    script.addEventListener("load", handleLoad);
    script.addEventListener("error", handleError);
    if (!existing) {
      script.src = RUNTIME_URL;
      script.async = true;
      document.head.appendChild(script);
    }
  }).catch((error) => {
    runtimePromise = null;
    throw error;
  });
  runtimePromise = created;
  return created;
}

function sourceSize(source: CanvasImageSource): { width: number; height: number } {
  if (source instanceof HTMLVideoElement) {
    return { width: source.videoWidth, height: source.videoHeight };
  }
  if (source instanceof HTMLImageElement) {
    return { width: source.naturalWidth || source.width, height: source.naturalHeight || source.height };
  }
  if (
    source instanceof HTMLCanvasElement ||
    (typeof OffscreenCanvas !== "undefined" && source instanceof OffscreenCanvas)
  ) {
    return { width: source.width, height: source.height };
  }
  if (typeof ImageBitmap !== "undefined" && source instanceof ImageBitmap) {
    return { width: source.width, height: source.height };
  }
  throw new Error("This browser cannot prepare video frames for the basketball detector.");
}

function prepareFrame(
  source: CanvasImageSource,
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  tensorData: Float32Array,
): PreparedFrame {
  const size = sourceSize(source);
  const sourceWidth = Math.max(1, size.width);
  const sourceHeight = Math.max(1, size.height);
  const gain = Math.min(MODEL_SIZE / sourceWidth, MODEL_SIZE / sourceHeight);
  const drawnWidth = Math.max(1, Math.round(sourceWidth * gain));
  const drawnHeight = Math.max(1, Math.round(sourceHeight * gain));
  const padX = Math.floor((MODEL_SIZE - drawnWidth) / 2);
  const padY = Math.floor((MODEL_SIZE - drawnHeight) / 2);
  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  context.drawImage(source, padX, padY, drawnWidth, drawnHeight);
  const pixels = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const planeSize = MODEL_SIZE * MODEL_SIZE;
  for (let index = 0; index < planeSize; index += 1) {
    const sourceOffset = index * 4;
    tensorData[index] = (pixels[sourceOffset] ?? 0) / 255;
    tensorData[planeSize + index] = (pixels[sourceOffset + 1] ?? 0) / 255;
    tensorData[planeSize * 2 + index] = (pixels[sourceOffset + 2] ?? 0) / 255;
  }
  return { tensorData, sourceWidth, sourceHeight, gain, padX, padY };
}

function intersectionOverUnion(
  left: LearnedObjectDetection,
  right: LearnedObjectDetection,
): number {
  const intersectionWidth = Math.max(0, Math.min(left.right, right.right) - Math.max(left.left, right.left));
  const intersectionHeight = Math.max(0, Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top));
  const intersection = intersectionWidth * intersectionHeight;
  const leftArea = Math.max(0, left.right - left.left) * Math.max(0, left.bottom - left.top);
  const rightArea = Math.max(0, right.right - right.left) * Math.max(0, right.bottom - right.top);
  return intersection / Math.max(1, leftArea + rightArea - intersection);
}

function nonMaximumSuppression(detections: LearnedObjectDetection[]): LearnedObjectDetection[] {
  const selected: LearnedObjectDetection[] = [];
  const sorted = [...detections].sort((left, right) => right.confidence - left.confidence);
  for (const detection of sorted) {
    const duplicate = selected.some((existing) =>
      existing.classId === detection.classId &&
      intersectionOverUnion(existing, detection) >= NMS_IOU
    );
    if (!duplicate) selected.push(detection);
  }
  return selected;
}

function decodeDetections(
  output: OrtTensorResult,
  prepared: PreparedFrame,
): LearnedObjectDetection[] {
  const channels = output.dims.at(-2) ?? 0;
  const anchors = output.dims.at(-1) ?? 0;
  if (channels < 4 + LABELS.length || anchors <= 0) {
    throw new Error("The basketball detector returned an unsupported result.");
  }
  const data = output.data;
  const detections: LearnedObjectDetection[] = [];
  for (let anchor = 0; anchor < anchors; anchor += 1) {
    let classId = 0;
    let confidence = 0;
    for (let candidateClass = 0; candidateClass < LABELS.length; candidateClass += 1) {
      const score = Number(data[(4 + candidateClass) * anchors + anchor] ?? 0);
      if (score > confidence) {
        confidence = score;
        classId = candidateClass;
      }
    }
    const label = LABELS[classId] ?? "ball";
    if (confidence < CLASS_CONFIDENCE[label]) continue;
    const centerX = Number(data[anchor] ?? 0);
    const centerY = Number(data[anchors + anchor] ?? 0);
    const width = Number(data[anchors * 2 + anchor] ?? 0);
    const height = Number(data[anchors * 3 + anchor] ?? 0);
    const left = clamp((centerX - width / 2 - prepared.padX) / prepared.gain, 0, prepared.sourceWidth);
    const top = clamp((centerY - height / 2 - prepared.padY) / prepared.gain, 0, prepared.sourceHeight);
    const right = clamp((centerX + width / 2 - prepared.padX) / prepared.gain, 0, prepared.sourceWidth);
    const bottom = clamp((centerY + height / 2 - prepared.padY) / prepared.gain, 0, prepared.sourceHeight);
    if (right - left < 2 || bottom - top < 2) continue;
    detections.push({
      classId,
      label,
      confidence,
      left,
      top,
      right,
      bottom,
    });
  }
  return nonMaximumSuppression(detections);
}

async function createSession(
  runtime: OrtRuntime,
  backend: "webgpu" | "wasm",
): Promise<OrtSession> {
  return runtime.InferenceSession.create(MODEL_URL, {
    executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
  });
}

async function createDetector(): Promise<LearnedBasketballDetector> {
  const runtime = await loadRuntime();
  const hardwareThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
  runtime.env.wasm.wasmPaths = WASM_PATH;
  runtime.env.wasm.numThreads = crossOriginIsolated ? hardwareThreads : 1;
  runtime.env.wasm.proxy = false;

  let backend: "webgpu" | "wasm" = supportsWebGpu() ? "webgpu" : "wasm";
  let session: OrtSession;
  try {
    session = await createSession(runtime, backend);
  } catch (error) {
    if (backend !== "webgpu") throw error;
    backend = "wasm";
    session = await createSession(runtime, backend);
  }

  const preprocessCanvas = document.createElement("canvas");
  preprocessCanvas.width = MODEL_SIZE;
  preprocessCanvas.height = MODEL_SIZE;
  const preprocessContext = preprocessCanvas.getContext("2d", { willReadFrequently: true });
  if (!preprocessContext) throw new Error("This browser cannot prepare detector frames.");
  const tensorData = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  const inputName = session.inputNames[0] ?? "images";
  const outputName = session.outputNames[0] ?? "output0";

  return {
    backend,
    async detect(source) {
      const prepared = prepareFrame(source, preprocessCanvas, preprocessContext, tensorData);
      const input = new runtime.Tensor(
        "float32",
        prepared.tensorData,
        [1, 3, MODEL_SIZE, MODEL_SIZE],
      );
      const outputs = await session.run({ [inputName]: input });
      const output = outputs[outputName];
      if (!output) throw new Error("The basketball detector produced no result.");
      const detections = decodeDetections(output, prepared);
      return {
        objects: detections,
        basketballs: detections.filter((detection) => detection.label === "ball"),
        hoops: detections.filter((detection) => detection.label === "hoop"),
        players: [],
      };
    },
  };
}

export function loadLearnedBasketballDetector(): Promise<LearnedBasketballDetector> {
  if (!detectorPromise) {
    detectorPromise = createDetector().catch((error) => {
      detectorPromise = null;
      throw error;
    });
  }
  return detectorPromise;
}
