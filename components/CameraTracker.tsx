"use no memo";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  AppState,
  LayoutChangeEvent,
  PanResponder,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Svg, { Circle, Polyline, Rect, Text as SvgText } from "react-native-svg";
import {
  useCameraDevice,
  useCameraPermission,
  useVideoOutput,
  type Recorder,
  type TargetCameraPosition,
} from "react-native-vision-camera";
import { SkiaCamera } from "react-native-vision-camera-skia";
import { useResizer, type GPUFrame } from "react-native-vision-camera-resizer";
import { createSynchronizable, scheduleOnRN } from "react-native-worklets";
import { Mat } from "react-native-fast-opencv";

import { colors, monoFont } from "../constants/theme";
import { persistRecording } from "../src/files/recordings";
import {
  createTrackerEngineState,
  MIN_AUTOMATIC_DECISION_CONFIDENCE,
  stepTracker,
} from "../src/tracking/engine";
import {
  detectOrangeBallCandidates,
} from "../src/vision/ballDetector";
import { createVisionTrackState, selectTrackedBall } from "../src/vision/ballTracker";
import { createPlayerMotionState, trackMovingPlayer } from "../src/vision/playerTracker";
import type {
  BallDetection,
  PlayerDetection,
  RimCalibration,
  ShotKind,
  ShotMethod,
} from "../types/tracking";

const ANALYSIS_WIDTH = 240;
const ANALYSIS_HEIGHT = 135;
const DEFAULT_RIM: RimCalibration = { x: 0.69, y: 0.25, width: 0.18, height: 0.065 };

interface CameraTrackerProps {
  sessionActive: boolean;
  onShot: (kind: ShotKind, method: ShotMethod) => void;
  onBeginSession: () => void;
  onEndSession: (recordingUri: string | null) => Promise<void> | void;
}

interface StageSize {
  width: number;
  height: number;
}

export function CameraTracker({
  sessionActive,
  onShot,
  onBeginSession,
  onEndSession,
}: CameraTrackerProps) {
  const recorderRef = useRef<Recorder | null>(null);
  const stopResolverRef = useRef<((uri: string | null) => void) | null>(null);
  const lastRecordingUriRef = useRef<string | null>(null);
  const endingSessionRef = useRef(false);
  const engineRef = useRef(createTrackerEngineState());
  const rimRef = useRef<RimCalibration>(DEFAULT_RIM);
  const [demoProgress] = useState(() => new Animated.Value(0));

  const [position, setPosition] = useState<TargetCameraPosition>("back");
  const [cameraActive, setCameraActive] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [recording, setRecording] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [reviewNotice, setReviewNotice] = useState("");
  const [rim, setRim] = useState<RimCalibration>(DEFAULT_RIM);
  const [draftRim, setDraftRim] = useState<RimCalibration | null>(null);
  const [calibrated, setCalibrated] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [ball, setBall] = useState<BallDetection | null>(null);
  const [player, setPlayer] = useState<PlayerDetection | null>(null);
  const [trail, setTrail] = useState<BallDetection[]>([]);
  const [stageSize, setStageSize] = useState<StageSize>({ width: 0, height: 0 });
  const [endingSession, setEndingSession] = useState(false);
  const [appActive, setAppActive] = useState(AppState.currentState === "active");

  const device = useCameraDevice(position);
  const { hasPermission, requestPermission } = useCameraPermission();
  const videoOutput = useVideoOutput({ enableAudio: false, fileType: "mov" });
  const cameraOutputs = useMemo(() => [videoOutput], [videoOutput]);
  const lastAnalysisAt = useMemo(() => createSynchronizable(0), []);
  const visionTrack = useMemo(
    () => createSynchronizable(createVisionTrackState()),
    [],
  );
  const playerMotion = useMemo(
    () => createSynchronizable(createPlayerMotionState()),
    [],
  );
  const { resizer } = useResizer({
    width: ANALYSIS_WIDTH,
    height: ANALYSIS_HEIGHT,
    channelOrder: "bgr",
    dataType: "uint8",
    pixelLayout: "interleaved",
    scaleMode: "contain",
  });

  useEffect(() => {
    rimRef.current = rim;
  }, [rim]);

  useEffect(() => {
    if (sessionActive) {
      engineRef.current = createTrackerEngineState();
      visionTrack.setBlocking(createVisionTrackState());
      playerMotion.setBlocking(createPlayerMotionState());
    }
  }, [playerMotion, sessionActive, visionTrack]);

  useEffect(() => {
    const subscription = AppState.addEventListener("change", (nextState) => {
      const active = nextState === "active";
      setAppActive(active);
      if (!active) {
        setBall(null);
        setPlayer(null);
        setTrail([]);
        visionTrack.setBlocking(createVisionTrackState());
        playerMotion.setBlocking(createPlayerMotionState());
      }
    });
    return () => subscription.remove();
  }, [playerMotion, visionTrack]);

  const handleDetection = useCallback(
    (detection: BallDetection | null, playerDetection: PlayerDetection | null) => {
      const now = Date.now();
      setBall(detection);
      setPlayer(playerDetection);
      if (detection) {
        setTrail((current) => [...current.slice(-11), detection]);
      } else {
        setTrail((current) => {
          const latest = current[current.length - 1];
          return latest && now - latest.at > 500 ? [] : current;
        });
      }

      if (!appActive || !sessionActive || endingSessionRef.current) return;
      const result = stepTracker(engineRef.current, detection, rimRef.current, now);
      engineRef.current = result.state;
      if (result.shot && result.confidence >= MIN_AUTOMATIC_DECISION_CONFIDENCE) {
        setReviewNotice("");
        onShot(result.shot, "tracked");
      } else if (result.shot) {
        setReviewNotice("Shot unclear — use + MAKE or + MISS to record it.");
      }
    },
    [appActive, onShot, sessionActive],
  );

  const processFrame = useCallback(
    (frame: Parameters<React.ComponentProps<typeof SkiaCamera>["onFrame"]>[0], render: Parameters<React.ComponentProps<typeof SkiaCamera>["onFrame"]>[1]) => {
      "worklet";
      render(({ canvas, frameTexture }) => {
        canvas.drawImage(frameTexture, 0, 0);
      });

      const timestamp = Date.now();
      const shouldAnalyze = timestamp - lastAnalysisAt.getBlocking() >= 83;
      if (!resizer || !shouldAnalyze) {
        frame.dispose();
        return;
      }
      lastAnalysisAt.setBlocking(timestamp);

      let resized: GPUFrame | null = null;
      let source: ReturnType<typeof Mat.create> | null = null;

      try {
        resized = resizer.resize(frame) as GPUFrame;
        const pixels = new Uint8Array(resized.getPixelBuffer());
        const playerResult = trackMovingPlayer(
          pixels,
          ANALYSIS_WIDTH,
          ANALYSIS_HEIGHT,
          playerMotion.getBlocking(),
          timestamp,
        );
        playerMotion.setBlocking(playerResult.state);
        source = Mat.createFromBuffer(
          "uint8",
          ANALYSIS_HEIGHT,
          ANALYSIS_WIDTH,
          3,
          pixels,
        );
        const rawCandidates = detectOrangeBallCandidates(
          source,
          ANALYSIS_WIDTH,
          ANALYSIS_HEIGHT,
          timestamp,
        );
        const candidates = frame.isMirrored
          ? rawCandidates.map((candidate) => ({ ...candidate, x: 1 - candidate.x }))
          : rawCandidates;
        const selection = selectTrackedBall(
          candidates,
          visionTrack.getBlocking(),
          rim,
          timestamp,
        );
        visionTrack.setBlocking(selection.state);
        const playerDetection = playerResult.detection
          ? frame.isMirrored
            ? {
                ...playerResult.detection,
                x: 1 - playerResult.detection.x - playerResult.detection.width,
              }
            : playerResult.detection
          : null;
        scheduleOnRN(handleDetection, selection.detection, playerDetection);
      } finally {
        resized?.dispose();
        source?.release();
        frame.dispose();
      }
    },
    [handleDetection, lastAnalysisAt, playerMotion, resizer, rim, visionTrack],
  );

  useEffect(() => {
    if (!demoMode) return;
    demoProgress.setValue(0);
    const animation = Animated.loop(
      Animated.timing(demoProgress, {
        toValue: 1,
        duration: 3_300,
        useNativeDriver: true,
      }),
    );
    animation.start();
    return () => animation.stop();
  }, [demoMode, demoProgress]);

  useEffect(() => {
    if (!sessionActive || !demoMode) return;
    let index = 0;
    const results: ShotKind[] = ["make", "make", "miss", "make"];
    const timer = setInterval(() => {
      const result = results[index % results.length];
      if (result) onShot(result, "demo");
      index += 1;
    }, 3_300);
    return () => clearInterval(timer);
  }, [demoMode, onShot, sessionActive]);

  const startCamera = async () => {
    setCameraError("");
    let granted = hasPermission;
    if (!granted) granted = await requestPermission();
    if (!granted) {
      setCameraError("Camera permission is required for live shot tracking.");
      return;
    }
    if (!device) {
      setCameraError("No compatible camera is available on this device.");
      return;
    }

    setDemoMode(false);
    setCameraActive(true);
    setCalibrated(false);
    setCalibrating(true);
    setBall(null);
    setPlayer(null);
    setTrail([]);
    visionTrack.setBlocking(createVisionTrackState());
    playerMotion.setBlocking(createPlayerMotionState());
  };

  const startDemo = () => {
    setCameraActive(false);
    setDemoMode(true);
    setCalibrated(true);
    setCalibrating(false);
    setCameraError("");
    setPlayer(null);
  };

  const switchCamera = () => {
    setPosition((current) => (current === "back" ? "front" : "back"));
    setCalibrated(false);
    setCalibrating(true);
    setBall(null);
    setPlayer(null);
    setTrail([]);
    visionTrack.setBlocking(createVisionTrackState());
    playerMotion.setBlocking(createPlayerMotionState());
  };

  const completeRecording = useCallback(async (temporaryPath: string) => {
    let uri: string | null = null;
    try {
      uri = await persistRecording(temporaryPath);
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : "The recorded clip could not be saved.");
    }
    lastRecordingUriRef.current = uri;
    recorderRef.current = null;
    setRecording(false);
    stopResolverRef.current?.(uri);
    stopResolverRef.current = null;
  }, []);

  const startRecording = useCallback(async () => {
    if (!cameraActive || demoMode) return;
    lastRecordingUriRef.current = null;
    try {
      const recorder = await videoOutput.createRecorder({});
      recorderRef.current = recorder;
      await recorder.startRecording(
        (filePath) => {
          void completeRecording(filePath);
        },
        (error) => {
          setCameraError(error.message);
          recorderRef.current = null;
          setRecording(false);
          stopResolverRef.current?.(null);
          stopResolverRef.current = null;
        },
      );
      setRecording(true);
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : "Recording could not start.");
      setRecording(false);
    }
  }, [cameraActive, completeRecording, demoMode, videoOutput]);

  const stopRecording = useCallback(async (): Promise<string | null> => {
    const recorder = recorderRef.current;
    if (!recorder || !recorder.isRecording) return lastRecordingUriRef.current;
    return new Promise((resolve) => {
      stopResolverRef.current = resolve;
      recorder.stopRecording().catch((error: unknown) => {
        setCameraError(error instanceof Error ? error.message : "Recording could not stop.");
        setRecording(false);
        stopResolverRef.current = null;
        resolve(null);
      });
    });
  }, []);

  const beginSession = useCallback(async () => {
    endingSessionRef.current = false;
    setReviewNotice("");
    engineRef.current = createTrackerEngineState();
    visionTrack.setBlocking(createVisionTrackState());
    playerMotion.setBlocking(createPlayerMotionState());
    setTrail([]);
    onBeginSession();
    await startRecording();
  }, [onBeginSession, playerMotion, startRecording, visionTrack]);

  const endSession = useCallback(async () => {
    endingSessionRef.current = true;
    setEndingSession(true);
    try {
      const uri = demoMode ? null : await stopRecording();
      await onEndSession(uri);
    } finally {
      endingSessionRef.current = false;
      setEndingSession(false);
    }
  }, [demoMode, onEndSession, stopRecording]);

  const updateStageSize = useCallback((event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setStageSize({ width, height });
  }, []);

  const calibrationResponder = useMemo(
    () => {
      // PanResponder returns stable native gesture handlers; it does not expose a render-time React ref.
      // eslint-disable-next-line react-hooks/refs
      return PanResponder.create({
        onStartShouldSetPanResponder: () => calibrating,
        onMoveShouldSetPanResponder: () => calibrating,
        onPanResponderGrant: () => {
          setDraftRim(null);
        },
        onPanResponderMove: (event, gesture) => {
          if (stageSize.width === 0 || stageSize.height === 0) return;
          const currentX = Math.max(0, Math.min(stageSize.width, event.nativeEvent.locationX));
          const currentY = Math.max(0, Math.min(stageSize.height, event.nativeEvent.locationY));
          const startX = Math.max(0, Math.min(stageSize.width, currentX - gesture.dx));
          const startY = Math.max(0, Math.min(stageSize.height, currentY - gesture.dy));
          setDraftRim({
            x: Math.min(startX, currentX) / stageSize.width,
            y: Math.min(startY, currentY) / stageSize.height,
            width: Math.abs(currentX - startX) / stageSize.width,
            height: Math.abs(currentY - startY) / stageSize.height,
          });
        },
        onPanResponderRelease: (event, gesture) => {
          if (stageSize.width === 0 || stageSize.height === 0) return;
          const currentX = Math.max(0, Math.min(stageSize.width, event.nativeEvent.locationX));
          const currentY = Math.max(0, Math.min(stageSize.height, event.nativeEvent.locationY));
          const startX = Math.max(0, Math.min(stageSize.width, currentX - gesture.dx));
          const startY = Math.max(0, Math.min(stageSize.height, currentY - gesture.dy));
          const width = Math.max(0.08, Math.abs(currentX - startX) / stageSize.width);
          const height = Math.max(0.025, Math.abs(currentY - startY) / stageSize.height);
          const next: RimCalibration = {
            x: Math.min(Math.min(startX, currentX) / stageSize.width, 1 - width),
            y: Math.min(Math.min(startY, currentY) / stageSize.height, 1 - height),
            width,
            height,
          };
          setDraftRim(null);
          setRim(next);
          rimRef.current = next;
          setCalibrated(true);
          setCalibrating(false);
        },
        onPanResponderTerminate: () => {
          setDraftRim(null);
        },
      });
    },
    [calibrating, stageSize.height, stageSize.width],
  );

  const activeRim = draftRim ?? rim;
  const ballVisible = Boolean(ball);
  const status = !cameraActive && !demoMode
    ? "Camera idle"
    : !appActive
      ? "Camera paused"
    : calibrating
      ? "Mark the rim"
        : demoMode
          ? "Demo ball locked"
        : ball
          ? player
            ? `Ball ${Math.round(ball.confidence * 100)}% · Player ${Math.round(player.confidence * 100)}%`
            : `Ball locked · ${Math.round(ball.confidence * 100)}%`
          : "Scanning for ball";

  const demoX = demoProgress.interpolate({
    inputRange: [0, 0.48, 0.68, 1],
    outputRange: [0, stageSize.width * 0.42, stageSize.width * 0.58, stageSize.width * 0.72],
  });
  const demoY = demoProgress.interpolate({
    inputRange: [0, 0.48, 0.68, 1],
    outputRange: [stageSize.height * 0.5, -stageSize.height * 0.38, -stageSize.height * 0.19, stageSize.height * 0.3],
  });
  const demoRotation = demoProgress.interpolate({
    inputRange: [0, 1],
    outputRange: ["0deg", "900deg"],
  });

  return (
    <View style={styles.card}>
      <View style={styles.toolbar}>
        <View style={[styles.scanIcon, ballVisible && styles.scanIconLocked]} accessibilityElementsHidden>
          <View style={styles.scanHorizontal} />
          <View style={styles.scanVertical} />
        </View>
        <View style={styles.statusCopy}>
          <Text style={styles.statusEyebrow}>ACTIVE VISION</Text>
          <Text style={styles.statusText} numberOfLines={1}>{status}</Text>
        </View>
        {recording ? (
          <View style={styles.recordingBadge}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>REC</Text>
          </View>
        ) : null}
        {cameraActive && !sessionActive ? (
          <Pressable
            style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}
            onPress={switchCamera}
            accessibilityRole="button"
            accessibilityLabel="Switch camera"
          >
            <Text style={styles.iconButtonText}>↻</Text>
          </Pressable>
        ) : null}
      </View>

      <View style={styles.stage} onLayout={updateStageSize}>
        {cameraActive && device ? (
          <SkiaCamera
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={cameraActive && appActive}
            outputs={cameraOutputs}
            orientationSource="interface"
            pixelFormat="yuv"
            enablePhysicalBufferRotation
            enablePreviewSizedOutputBuffers
            onFrame={processFrame}
          />
        ) : null}

        {!cameraActive && !demoMode ? <EmptyCamera /> : null}

        {demoMode ? (
          <View style={styles.demoStage}>
            <View style={styles.demoBackboard} />
            <View style={styles.demoRim} />
            <View style={styles.demoArc} />
            <Animated.View
              style={[
                styles.demoBall,
                { transform: [{ translateX: demoX }, { translateY: demoY }, { rotate: demoRotation }] },
              ]}
            >
              <View style={styles.demoBallSeamVertical} />
              <View style={styles.demoBallSeamHorizontal} />
            </Animated.View>
            <Text style={styles.demoLabel}>DEMO FEED</Text>
          </View>
        ) : null}

        {stageSize.width > 0 && (cameraActive || demoMode) ? (
          <Svg
            style={[StyleSheet.absoluteFill, styles.nonInteractive]}
            width={stageSize.width}
            height={stageSize.height}
          >
            <Rect
              x={activeRim.x * stageSize.width}
              y={activeRim.y * stageSize.height}
              width={activeRim.width * stageSize.width}
              height={activeRim.height * stageSize.height}
              fill="transparent"
              stroke={calibrating ? colors.orange : colors.acid}
              strokeWidth={3}
              strokeDasharray={calibrating ? "10 7" : undefined}
            />
            {player && cameraActive ? (
              <>
                <Rect
                  x={player.x * stageSize.width}
                  y={player.y * stageSize.height}
                  width={player.width * stageSize.width}
                  height={player.height * stageSize.height}
                  fill="rgba(69,226,137,0.04)"
                  stroke={colors.green}
                  strokeWidth={2}
                  strokeDasharray="8 5"
                />
                <SvgText
                  x={player.x * stageSize.width + 5}
                  y={Math.max(12, player.y * stageSize.height - 6)}
                  fill={colors.green}
                  fontSize={10}
                  fontWeight="800"
                >
                  {`PLAYER ${Math.round(player.confidence * 100)}%`}
                </SvgText>
              </>
            ) : null}
            {trail.length > 1 ? (
              <Polyline
                points={trail.map((point) => `${point.x * stageSize.width},${point.y * stageSize.height}`).join(" ")}
                fill="none"
                stroke="rgba(255,90,31,0.72)"
                strokeWidth={4}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ) : null}
            {ball ? (
              <Circle
                cx={ball.x * stageSize.width}
                cy={ball.y * stageSize.height}
                r={Math.max(12, ball.width * stageSize.width * 0.7)}
                fill="transparent"
                stroke={colors.orange}
                strokeWidth={4}
              />
            ) : null}
          </Svg>
        ) : null}

        <View
          style={[
            StyleSheet.absoluteFill,
            calibrating ? styles.interactive : styles.nonInteractive,
          ]}
          {...calibrationResponder.panHandlers}
        />

        {calibrating && cameraActive ? (
          <View style={[styles.calibrationHint, styles.nonInteractive]}>
            <Text style={styles.calibrationTitle}>MARK THE RIM</Text>
            <Text style={styles.calibrationText}>Drag a tight box across the opening</Text>
          </View>
        ) : null}

        <FrameCorners />
      </View>

      {cameraError ? (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{cameraError}</Text>
        </View>
      ) : null}

      {reviewNotice ? (
        <View style={styles.reviewBanner}>
          <Text style={styles.reviewBannerText}>{reviewNotice}</Text>
        </View>
      ) : null}

      <View style={styles.controls}>
        {!cameraActive && !demoMode ? (
          <>
            <ActionButton label="START CAMERA" primary onPress={startCamera} />
            <ActionButton label="TRY DEMO" onPress={startDemo} />
          </>
        ) : !sessionActive ? (
          <>
            {demoMode ? (
              <ActionButton label="USE CAMERA" onPress={startCamera} />
            ) : (
              <ActionButton
                label={calibrated ? "ADJUST RIM" : "MARK RIM"}
                selected={calibrating}
                onPress={() => setCalibrating(true)}
              />
            )}
            <ActionButton
              label="START SESSION"
              primary
              disabled={!demoMode && !calibrated}
              onPress={beginSession}
            />
          </>
        ) : (
          <>
            <ActionButton
              label="+ MAKE"
              compact
              tone="make"
              onPress={() => {
                setReviewNotice("");
                onShot("make", "manual");
              }}
            />
            <ActionButton
              label={endingSession ? "SAVING..." : "END SESSION"}
              light
              disabled={endingSession}
              onPress={endSession}
            />
            <ActionButton
              label="+ MISS"
              compact
              tone="miss"
              onPress={() => {
                setReviewNotice("");
                onShot("miss", "manual");
              }}
            />
          </>
        )}
      </View>
    </View>
  );
}

interface ActionButtonProps {
  label: string;
  onPress: () => void;
  primary?: boolean;
  light?: boolean;
  compact?: boolean;
  selected?: boolean;
  disabled?: boolean;
  tone?: "make" | "miss";
}

function ActionButton({
  label,
  onPress,
  primary = false,
  light = false,
  compact = false,
  selected = false,
  disabled = false,
  tone,
}: ActionButtonProps) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
      style={({ pressed }) => [
        styles.actionButton,
        primary && styles.actionPrimary,
        light && styles.actionLight,
        compact && styles.actionCompact,
        selected && styles.actionSelected,
        disabled && styles.actionDisabled,
        pressed && !disabled && styles.pressed,
      ]}
    >
      <Text
        style={[
          styles.actionText,
          light && styles.actionLightText,
          tone === "make" && styles.actionMakeText,
          tone === "miss" && styles.actionMissText,
          disabled && styles.actionDisabledText,
        ]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

function EmptyCamera() {
  return (
    <View style={styles.emptyStage}>
      <View style={styles.previewRim} />
      <View style={styles.previewArc} />
      <View style={styles.emptyBall}>
        <View style={styles.emptyBallVertical} />
        <View style={styles.emptyBallHorizontal} />
      </View>
      <Text style={styles.emptyTitle}>Frame the hoop + shooter</Text>
      <Text style={styles.emptyBody}>Keep the full ball flight in view. The app is locked to landscape for tracking accuracy.</Text>
    </View>
  );
}

function FrameCorners() {
  return (
    <View style={[styles.frameCorners, styles.nonInteractive]} accessibilityElementsHidden>
      <View style={[styles.corner, styles.cornerTopLeft]} />
      <View style={[styles.corner, styles.cornerTopRight]} />
      <View style={[styles.corner, styles.cornerBottomLeft]} />
      <View style={[styles.corner, styles.cornerBottomRight]} />
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    boxShadow: `7px 7px 0 ${colors.orange}`,
    elevation: 10,
  },
  interactive: { pointerEvents: "auto" },
  nonInteractive: { pointerEvents: "none" },
  toolbar: {
    minHeight: 57,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  scanIcon: {
    width: 28,
    height: 28,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    marginRight: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  scanIconLocked: {
    borderColor: colors.acid,
    backgroundColor: "rgba(223,255,79,0.06)",
  },
  scanHorizontal: {
    position: "absolute",
    width: 12,
    height: 1,
    backgroundColor: colors.acid,
  },
  scanVertical: {
    position: "absolute",
    width: 1,
    height: 12,
    backgroundColor: colors.acid,
  },
  statusCopy: {
    flex: 1,
    minWidth: 0,
  },
  statusEyebrow: {
    color: colors.muted,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "800",
    letterSpacing: 1.3,
  },
  statusText: {
    color: colors.text,
    fontSize: 11,
    fontWeight: "600",
    marginTop: 2,
  },
  recordingBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.orangeSoft,
    paddingHorizontal: 9,
    paddingVertical: 7,
    marginRight: 8,
  },
  recordingDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.orange,
    marginRight: 5,
  },
  recordingText: {
    color: colors.red,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 1,
  },
  iconButton: {
    width: 36,
    height: 36,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: colors.surfaceRaised,
    borderWidth: 1,
    borderColor: colors.line,
  },
  iconButtonText: {
    color: colors.acid,
    fontSize: 20,
  },
  stage: {
    width: "100%",
    aspectRatio: 16 / 9,
    backgroundColor: colors.black,
    overflow: "hidden",
  },
  emptyStage: {
    ...StyleSheet.absoluteFill,
    backgroundColor: "#0A0F0B",
    justifyContent: "center",
    alignItems: "center",
  },
  previewRim: {
    position: "absolute",
    top: "25%",
    right: "16%",
    width: "18%",
    height: "6%",
    borderWidth: 3,
    borderColor: colors.orange,
    borderRadius: 999,
    opacity: 0.5,
  },
  previewArc: {
    position: "absolute",
    left: "29%",
    top: "25%",
    width: "41%",
    height: "40%",
    borderTopWidth: 2,
    borderTopColor: "rgba(223,255,79,0.35)",
    borderRadius: 999,
    transform: [{ rotate: "-8deg" }],
  },
  emptyBall: {
    width: 49,
    height: 49,
    borderRadius: 25,
    borderWidth: 2,
    borderColor: colors.orange,
    overflow: "hidden",
    marginBottom: 12,
  },
  emptyBallVertical: {
    position: "absolute",
    left: 22,
    width: 2,
    height: 52,
    backgroundColor: colors.orange,
  },
  emptyBallHorizontal: {
    position: "absolute",
    top: 22,
    width: 52,
    height: 2,
    backgroundColor: colors.orange,
  },
  emptyTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "800",
  },
  emptyBody: {
    color: colors.muted,
    maxWidth: 340,
    textAlign: "center",
    fontSize: 10,
    lineHeight: 15,
    marginTop: 5,
  },
  demoStage: {
    ...StyleSheet.absoluteFill,
    backgroundColor: "#0A100B",
  },
  demoBackboard: {
    position: "absolute",
    top: "11%",
    right: "12%",
    width: "24%",
    height: "32%",
    borderWidth: 3,
    borderColor: "rgba(255,255,255,0.65)",
  },
  demoRim: {
    position: "absolute",
    top: "39%",
    right: "19%",
    width: "17%",
    height: "5%",
    borderWidth: 3,
    borderColor: colors.orange,
    borderRadius: 999,
  },
  demoArc: {
    position: "absolute",
    left: "20%",
    top: "18%",
    width: "55%",
    height: "60%",
    borderTopWidth: 2,
    borderTopColor: "rgba(223,255,79,0.52)",
    borderRadius: 999,
    transform: [{ rotate: "-6deg" }],
  },
  demoBall: {
    position: "absolute",
    left: "15%",
    top: "71%",
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: colors.orange,
    borderWidth: 2,
    borderColor: "#6F2409",
    overflow: "hidden",
  },
  demoBallSeamVertical: {
    position: "absolute",
    left: 15,
    width: 2,
    height: 36,
    backgroundColor: "#6F2409",
  },
  demoBallSeamHorizontal: {
    position: "absolute",
    top: 15,
    width: 36,
    height: 2,
    backgroundColor: "#6F2409",
  },
  demoLabel: {
    position: "absolute",
    left: 20,
    bottom: 17,
    color: colors.acid,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "700",
    letterSpacing: 1.2,
  },
  calibrationHint: {
    position: "absolute",
    bottom: 15,
    alignSelf: "center",
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.orange,
    paddingHorizontal: 11,
    paddingVertical: 8,
  },
  calibrationTitle: {
    color: colors.white,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
    marginRight: 9,
  },
  calibrationText: {
    color: colors.white,
    fontSize: 9,
    opacity: 0.85,
  },
  frameCorners: {
    ...StyleSheet.absoluteFill,
    margin: 13,
  },
  corner: {
    position: "absolute",
    width: 21,
    height: 21,
    borderColor: "rgba(255,255,255,0.58)",
  },
  cornerTopLeft: {
    left: 0,
    top: 0,
    borderLeftWidth: 1,
    borderTopWidth: 1,
  },
  cornerTopRight: {
    right: 0,
    top: 0,
    borderRightWidth: 1,
    borderTopWidth: 1,
  },
  cornerBottomLeft: {
    left: 0,
    bottom: 0,
    borderLeftWidth: 1,
    borderBottomWidth: 1,
  },
  cornerBottomRight: {
    right: 0,
    bottom: 0,
    borderRightWidth: 1,
    borderBottomWidth: 1,
  },
  errorBanner: {
    backgroundColor: "#4F190F",
    borderTopWidth: 1,
    borderTopColor: colors.red,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  errorText: {
    color: "#FFD5C7",
    fontSize: 10,
  },
  reviewBanner: {
    backgroundColor: "#3D3512",
    borderTopWidth: 1,
    borderTopColor: colors.yellow,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  reviewBannerText: {
    color: "#FFF0A6",
    fontSize: 10,
  },
  controls: {
    minHeight: 68,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  actionButton: {
    minWidth: 124,
    minHeight: 43,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: colors.surfaceRaised,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    marginHorizontal: 5,
    paddingHorizontal: 16,
  },
  actionPrimary: {
    backgroundColor: colors.orange,
    borderColor: colors.orange,
  },
  actionLight: {
    backgroundColor: colors.text,
    borderColor: colors.text,
  },
  actionCompact: {
    minWidth: 86,
    paddingHorizontal: 10,
  },
  actionSelected: {
    borderColor: colors.acid,
  },
  actionDisabled: {
    backgroundColor: colors.surfaceSoft,
    borderColor: colors.line,
  },
  actionText: {
    color: colors.text,
    fontFamily: monoFont,
    fontSize: 10,
    fontWeight: "900",
    letterSpacing: 0.4,
  },
  actionLightText: {
    color: colors.background,
  },
  actionMakeText: {
    color: colors.green,
  },
  actionMissText: {
    color: colors.red,
  },
  actionDisabledText: {
    color: colors.faint,
  },
  pressed: {
    opacity: 0.65,
  },
});
