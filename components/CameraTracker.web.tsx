import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  LayoutChangeEvent,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Svg, { Circle, Polyline, Rect, Text as SvgText } from "react-native-svg";

import { colors, monoFont } from "../constants/theme";
import {
  startLiveBrowserPipeline,
  type LiveBrowserPipeline,
} from "../src/vision/liveBrowserPipeline.web";
import type {
  BallDetection,
  PlayerDetection,
  RimCalibration,
  ShotKind,
  ShotMethod,
} from "../types/tracking";

interface CameraTrackerProps {
  sessionActive: boolean;
  onShot: (kind: ShotKind, method: ShotMethod) => void;
  onBeginSession: () => void;
  onEndSession: (recordingUri: string | null) => Promise<void> | void;
}

type BrowserMode = "idle" | "camera" | "demo";

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
  const previewHostRef = useRef<HTMLElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const pipelineRef = useRef<LiveBrowserPipeline | null>(null);
  const generationRef = useRef(0);
  const onShotRef = useRef(onShot);

  const [mode, setMode] = useState<BrowserMode>("idle");
  const [loading, setLoading] = useState(false);
  const [ending, setEnding] = useState(false);
  const [error, setError] = useState("");
  const [reviewNotice, setReviewNotice] = useState("");
  const [rim, setRim] = useState<RimCalibration | null>(null);
  const [rimConfidence, setRimConfidence] = useState(0);
  const [ball, setBall] = useState<BallDetection | null>(null);
  const [player, setPlayer] = useState<PlayerDetection | null>(null);
  const [trail, setTrail] = useState<BallDetection[]>([]);
  const [backend, setBackend] = useState<"webgpu" | "wasm" | null>(null);
  const [stageSize, setStageSize] = useState<StageSize>({ width: 0, height: 0 });
  const [frameAspectRatio, setFrameAspectRatio] = useState(16 / 9);
  const [progress] = useState(() => new Animated.Value(0));

  useEffect(() => {
    onShotRef.current = onShot;
  }, [onShot]);

  useEffect(() => {
    pipelineRef.current?.setSessionActive(sessionActive);
  }, [sessionActive]);

  useEffect(() => {
    if (mode !== "demo") return;
    progress.setValue(0);
    const animation = Animated.loop(
      Animated.timing(progress, {
        toValue: 1,
        duration: 3_300,
        useNativeDriver: true,
      }),
    );
    animation.start();
    return () => animation.stop();
  }, [mode, progress]);

  useEffect(() => {
    if (!sessionActive || mode !== "demo") return;
    let index = 0;
    const outcomes: ShotKind[] = ["make", "make", "miss", "make"];
    const timer = window.setInterval(() => {
      const outcome = outcomes[index % outcomes.length];
      if (outcome) onShotRef.current(outcome, "demo");
      index += 1;
    }, 3_300);
    return () => window.clearInterval(timer);
  }, [mode, sessionActive]);

  const stopCamera = useCallback(() => {
    generationRef.current += 1;
    pipelineRef.current?.stop();
    pipelineRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.remove();
    }
    videoRef.current = null;
    setLoading(false);
    setRim(null);
    setRimConfidence(0);
    setBall(null);
    setPlayer(null);
    setTrail([]);
    setBackend(null);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const attachPreviewHost = useCallback((node: View | null) => {
    const element = node as unknown as HTMLElement | null;
    previewHostRef.current = element;
    if (element && videoRef.current && videoRef.current.parentElement !== element) {
      element.appendChild(videoRef.current);
    }
  }, []);

  const startCamera = useCallback(async () => {
    stopCamera();
    const generation = generationRef.current;
    setMode("camera");
    setLoading(true);
    setError("");
    setReviewNotice("");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not provide secure camera access. Use the HTTPS site in Safari or Chrome.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 60 },
        },
      });
      if (generation !== generationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      const video = document.createElement("video");
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.setAttribute("playsinline", "true");
      video.setAttribute("aria-label", "Live basketball camera preview");
      video.style.cssText =
        "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;";
      video.srcObject = stream;
      streamRef.current = stream;
      videoRef.current = video;
      previewHostRef.current?.appendChild(video);
      await video.play();
      setFrameAspectRatio(video.videoWidth / Math.max(1, video.videoHeight));

      const pipeline = await startLiveBrowserPipeline(video, {
        onFrame(frame) {
          if (generation !== generationRef.current) return;
          setRim(frame.rim);
          setRimConfidence(frame.rimConfidence);
          setBall(frame.ball);
          setPlayer(frame.player);
          setBackend(frame.backend);
          setTrail((current) => {
            if (frame.ball) return [...current.slice(-13), frame.ball];
            const latest = current.at(-1);
            return latest && Date.now() - latest.at <= 520 ? current : [];
          });
        },
        onShot(kind) {
          if (generation !== generationRef.current) return;
          setReviewNotice("");
          onShotRef.current(kind, "tracked");
        },
        onReview() {
          if (generation !== generationRef.current) return;
          setReviewNotice("Shot needs review — use + MAKE or + MISS to record it.");
        },
        onError(message) {
          if (generation !== generationRef.current) return;
          setError(message);
        },
      });
      if (generation !== generationRef.current) {
        pipeline.stop();
        return;
      }
      pipelineRef.current = pipeline;
      pipeline.setSessionActive(sessionActive);
      setBackend((current) => current ?? "wasm");
      setLoading(false);
    } catch (caught) {
      if (generation !== generationRef.current) return;
      setError(caught instanceof Error ? caught.message : "The live camera could not start.");
      stopCamera();
      setMode("idle");
    }
  }, [sessionActive, stopCamera]);

  const startDemo = useCallback(() => {
    stopCamera();
    setMode("demo");
    setError("");
    setReviewNotice("");
  }, [stopCamera]);

  const beginSession = useCallback(() => {
    setReviewNotice("");
    pipelineRef.current?.setSessionActive(true);
    onBeginSession();
  }, [onBeginSession]);

  const endSession = useCallback(async () => {
    setEnding(true);
    pipelineRef.current?.setSessionActive(false);
    try {
      await onEndSession(null);
    } finally {
      setEnding(false);
    }
  }, [onEndSession]);

  const relockHoop = useCallback(() => {
    setRim(null);
    setRimConfidence(0);
    setBall(null);
    setTrail([]);
    setReviewNotice("");
    pipelineRef.current?.relockHoop();
  }, []);

  const projection = useMemo(() => {
    const stageAspect = stageSize.width / Math.max(1, stageSize.height);
    if (frameAspectRatio >= stageAspect) {
      const height = stageSize.width / Math.max(0.01, frameAspectRatio);
      return { x: 0, y: (stageSize.height - height) / 2, width: stageSize.width, height };
    }
    const width = stageSize.height * frameAspectRatio;
    return { x: (stageSize.width - width) / 2, y: 0, width, height: stageSize.height };
  }, [frameAspectRatio, stageSize.height, stageSize.width]);

  const demoX = progress.interpolate({
    inputRange: [0, 0.48, 0.68, 1],
    outputRange: [0, stageSize.width * 0.42, stageSize.width * 0.58, stageSize.width * 0.72],
  });
  const demoY = progress.interpolate({
    inputRange: [0, 0.48, 0.68, 1],
    outputRange: [stageSize.height * 0.5, -stageSize.height * 0.38, -stageSize.height * 0.19, stageSize.height * 0.3],
  });
  const rotation = progress.interpolate({
    inputRange: [0, 1],
    outputRange: ["0deg", "900deg"],
  });

  const status = mode === "idle"
    ? "Ready for live basketball AI"
    : mode === "demo"
      ? sessionActive ? "Simulated trajectory active" : "Simulation ready"
      : loading
        ? "Loading detector + finding hoop"
        : rim
          ? ball
            ? `Hoop locked · Ball ${Math.round(ball.confidence * 100)}%`
            : `Hoop locked · ${Math.round(rimConfidence * 100)}%`
          : "Scanning every frame for the hoop";

  return (
    <View style={styles.card}>
      <View style={styles.toolbar}>
        <View style={[styles.scanIcon, rim && styles.scanIconReady]}>
          <View style={styles.scanHorizontal} />
          <View style={styles.scanVertical} />
        </View>
        <View style={styles.statusCopy}>
          <Text style={styles.statusEyebrow}>
            {mode === "camera" ? "LIVE BASKETBALL AI" : mode === "demo" ? "BROWSER SIMULATION" : "ACTIVE VISION"}
          </Text>
          <Text style={styles.statusText} numberOfLines={1}>{status}</Text>
        </View>
        {mode === "camera" && backend ? (
          <View style={styles.engineBadge}>
            <Text style={styles.engineBadgeText}>{backend.toUpperCase()} · BYTETRACK</Text>
          </View>
        ) : null}
        {sessionActive ? (
          <View style={styles.recordingBadge}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>{mode === "demo" ? "DEMO" : "TRACKING"}</Text>
          </View>
        ) : null}
      </View>

      <View
        style={styles.stage}
        onLayout={(event: LayoutChangeEvent) => {
          const { width, height } = event.nativeEvent.layout;
          setStageSize({ width, height });
        }}
      >
        <View ref={attachPreviewHost} style={[StyleSheet.absoluteFill, styles.nonInteractive]} />

        {mode === "idle" ? <EmptyCamera /> : null}
        {mode === "demo" ? (
          <View style={styles.demoStage}>
            <View style={styles.backboard} />
            <View style={styles.demoRim} />
            <View style={styles.arc} />
            <Animated.View
              style={[
                styles.demoBall,
                { transform: [{ translateX: demoX }, { translateY: demoY }, { rotate: rotation }] },
              ]}
            >
              <View style={styles.ballSeamVertical} />
              <View style={styles.ballSeamHorizontal} />
            </Animated.View>
          </View>
        ) : null}

        {mode === "camera" && stageSize.width > 0 ? (
          <Svg style={[StyleSheet.absoluteFill, styles.nonInteractive]} width={stageSize.width} height={stageSize.height}>
            {rim ? (
              <>
                <Rect
                  x={projection.x + rim.x * projection.width}
                  y={projection.y + rim.y * projection.height}
                  width={rim.width * projection.width}
                  height={rim.height * projection.height}
                  fill="rgba(223,255,79,0.05)"
                  stroke={colors.acid}
                  strokeWidth={3}
                />
                <SvgText
                  x={projection.x + rim.x * projection.width}
                  y={Math.max(13, projection.y + rim.y * projection.height - 7)}
                  fill={colors.acid}
                  fontSize={10}
                  fontWeight="900"
                >
                  {`HOOP LOCK ${Math.round(rimConfidence * 100)}%`}
                </SvgText>
              </>
            ) : null}
            {player ? (
              <Rect
                x={projection.x + player.x * projection.width}
                y={projection.y + player.y * projection.height}
                width={player.width * projection.width}
                height={player.height * projection.height}
                fill="rgba(68,218,138,0.04)"
                stroke={colors.green}
                strokeWidth={2}
                strokeDasharray="8 5"
              />
            ) : null}
            {trail.length > 1 ? (
              <Polyline
                points={trail.map((point) =>
                  `${projection.x + point.x * projection.width},${projection.y + point.y * projection.height}`
                ).join(" ")}
                fill="none"
                stroke="rgba(255,90,31,0.82)"
                strokeWidth={4}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ) : null}
            {ball ? (
              <Circle
                cx={projection.x + ball.x * projection.width}
                cy={projection.y + ball.y * projection.height}
                r={Math.max(10, ball.width * projection.width * 0.7)}
                fill="transparent"
                stroke={colors.orange}
                strokeWidth={4}
              />
            ) : null}
          </Svg>
        ) : null}

        <View style={[styles.pipelineNotice, styles.nonInteractive]}>
          <Text style={styles.pipelineNoticeTitle}>
            {mode === "demo" ? "SIMULATED EVENTS" : mode === "camera" ? "REAL CAMERA ANALYSIS" : "FIXED CAMERA REQUIRED"}
          </Text>
          <Text style={styles.pipelineNoticeBody}>
            {mode === "camera" ? "Attalla YOLO · detector tracking · rim-crossing scoring" : "Keep the hoop and complete flight visible"}
          </Text>
        </View>
        <FrameCorners />
      </View>

      {error ? (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : null}
      {reviewNotice ? (
        <View style={styles.reviewBanner}>
          <Text style={styles.reviewText}>{reviewNotice}</Text>
        </View>
      ) : null}

      <View style={styles.controls}>
        {mode === "idle" ? (
          <>
            <ActionButton label="START LIVE CAMERA" primary onPress={() => void startCamera()} />
            <ActionButton label="TRY SIMULATION" onPress={startDemo} />
          </>
        ) : !sessionActive ? mode === "camera" ? (
          <>
            <ActionButton label="STOP CAMERA" onPress={() => { stopCamera(); setMode("idle"); }} />
            <ActionButton label={rim ? "RELOCK HOOP" : "FINDING HOOP..."} disabled={loading} onPress={relockHoop} />
            <ActionButton label="START SESSION" primary disabled={loading || !rim} onPress={beginSession} />
          </>
        ) : (
          <>
            <ActionButton label="USE LIVE CAMERA" onPress={() => void startCamera()} />
            <ActionButton label="START DEMO SESSION" primary onPress={beginSession} />
          </>
        ) : (
          <>
            <ActionButton label="+ MAKE" tone="make" onPress={() => onShotRef.current("make", "manual")} />
            <ActionButton label={ending ? "SAVING..." : "END SESSION"} light disabled={ending} onPress={() => void endSession()} />
            <ActionButton label="+ MISS" tone="miss" onPress={() => onShotRef.current("miss", "manual")} />
          </>
        )}
      </View>
    </View>
  );
}

function ActionButton({
  label,
  onPress,
  primary = false,
  light = false,
  disabled = false,
  tone,
}: {
  label: string;
  onPress: () => void;
  primary?: boolean;
  light?: boolean;
  disabled?: boolean;
  tone?: "make" | "miss";
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [
        styles.actionButton,
        primary && styles.actionPrimary,
        light && styles.actionLight,
        disabled && styles.actionDisabled,
        pressed && !disabled && styles.pressed,
      ]}
    >
      <Text style={[
        styles.actionText,
        light && styles.actionLightText,
        tone === "make" && styles.actionMakeText,
        tone === "miss" && styles.actionMissText,
      ]}>
        {label}
      </Text>
    </Pressable>
  );
}

function EmptyCamera() {
  return (
    <View style={styles.emptyStage}>
      <View style={styles.emptyBall}>
        <View style={styles.ballSeamVertical} />
        <View style={styles.ballSeamHorizontal} />
      </View>
      <Text style={styles.emptyTitle}>Frame the hoop + shooter</Text>
      <Text style={styles.emptyBody}>
        Start the real browser camera. ActiveTrack will find the hoop automatically, lock its identity, and track the ball and player on this device.
      </Text>
    </View>
  );
}

function FrameCorners() {
  return (
    <View style={[styles.frameCorners, styles.nonInteractive]}>
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
  },
  nonInteractive: { pointerEvents: "none" },
  toolbar: {
    minHeight: 59,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  scanIcon: {
    width: 30,
    height: 30,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    marginRight: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  scanIconReady: { borderColor: colors.acid, backgroundColor: "rgba(223,255,79,0.08)" },
  scanHorizontal: { position: "absolute", width: 13, height: 1, backgroundColor: colors.acid },
  scanVertical: { position: "absolute", width: 1, height: 13, backgroundColor: colors.acid },
  statusCopy: { flex: 1, minWidth: 0 },
  statusEyebrow: { color: colors.muted, fontFamily: monoFont, fontSize: 8, fontWeight: "900", letterSpacing: 1.3 },
  statusText: { color: colors.text, fontSize: 11, fontWeight: "700", marginTop: 2 },
  engineBadge: { borderWidth: 1, borderColor: colors.acid, paddingHorizontal: 9, paddingVertical: 6, marginLeft: 8 },
  engineBadgeText: { color: colors.acid, fontFamily: monoFont, fontSize: 7, fontWeight: "900" },
  recordingBadge: { flexDirection: "row", alignItems: "center", backgroundColor: colors.orangeSoft, paddingHorizontal: 9, paddingVertical: 7, marginLeft: 8 },
  recordingDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.orange, marginRight: 5 },
  recordingText: { color: colors.red, fontFamily: monoFont, fontSize: 8, fontWeight: "900" },
  stage: { width: "100%", aspectRatio: 16 / 9, backgroundColor: colors.black, overflow: "hidden" },
  emptyStage: { ...StyleSheet.absoluteFill, backgroundColor: "#0A0F0B", justifyContent: "center", alignItems: "center" },
  emptyBall: { width: 52, height: 52, borderRadius: 26, backgroundColor: colors.orange, overflow: "hidden", marginBottom: 13 },
  emptyTitle: { color: colors.text, fontSize: 19, fontWeight: "900" },
  emptyBody: { color: colors.muted, maxWidth: 520, textAlign: "center", fontSize: 11, lineHeight: 17, marginTop: 7, paddingHorizontal: 22 },
  demoStage: { ...StyleSheet.absoluteFill, backgroundColor: "#0A100B" },
  backboard: { position: "absolute", top: "11%", right: "12%", width: "24%", height: "32%", borderWidth: 3, borderColor: "rgba(255,255,255,0.65)" },
  demoRim: { position: "absolute", top: "39%", right: "19%", width: "17%", height: "5%", borderWidth: 3, borderColor: colors.orange, borderRadius: 999 },
  arc: { position: "absolute", left: "20%", top: "18%", width: "55%", height: "60%", borderTopWidth: 2, borderTopColor: "rgba(223,255,79,0.52)", borderRadius: 999, transform: [{ rotate: "-6deg" }] },
  demoBall: { position: "absolute", left: "15%", top: "19%", width: 34, height: 34, borderRadius: 17, backgroundColor: colors.orange, borderWidth: 2, borderColor: "#6F2409", overflow: "hidden" },
  ballSeamVertical: { position: "absolute", left: "48%", width: 2, height: "100%", backgroundColor: "#6F2409" },
  ballSeamHorizontal: { position: "absolute", top: "48%", width: "100%", height: 2, backgroundColor: "#6F2409" },
  pipelineNotice: { position: "absolute", right: 16, bottom: 14, backgroundColor: "rgba(7,10,8,0.9)", borderWidth: 1, borderColor: colors.lineStrong, paddingHorizontal: 10, paddingVertical: 7 },
  pipelineNoticeTitle: { color: colors.orange, fontFamily: monoFont, fontSize: 8, fontWeight: "900" },
  pipelineNoticeBody: { color: colors.muted, fontSize: 8, marginTop: 2 },
  frameCorners: { ...StyleSheet.absoluteFill, margin: 13 },
  corner: { position: "absolute", width: 21, height: 21, borderColor: "rgba(255,255,255,0.58)" },
  cornerTopLeft: { left: 0, top: 0, borderLeftWidth: 1, borderTopWidth: 1 },
  cornerTopRight: { right: 0, top: 0, borderRightWidth: 1, borderTopWidth: 1 },
  cornerBottomLeft: { left: 0, bottom: 0, borderLeftWidth: 1, borderBottomWidth: 1 },
  cornerBottomRight: { right: 0, bottom: 0, borderRightWidth: 1, borderBottomWidth: 1 },
  controls: { minHeight: 72, flexDirection: "row", flexWrap: "wrap", justifyContent: "center", alignItems: "center", paddingHorizontal: 10, paddingVertical: 9 },
  actionButton: { minWidth: 128, minHeight: 45, justifyContent: "center", alignItems: "center", backgroundColor: colors.surfaceRaised, borderWidth: 1, borderColor: colors.lineStrong, marginHorizontal: 5, marginVertical: 3, paddingHorizontal: 16 },
  actionPrimary: { backgroundColor: colors.orange, borderColor: colors.orange },
  actionLight: { backgroundColor: colors.text, borderColor: colors.text },
  actionDisabled: { opacity: 0.38 },
  actionText: { color: colors.text, fontFamily: monoFont, fontSize: 9, fontWeight: "900", letterSpacing: 0.4 },
  actionLightText: { color: colors.background },
  actionMakeText: { color: colors.green },
  actionMissText: { color: colors.red },
  errorBanner: { backgroundColor: "#4F190F", borderTopWidth: 1, borderTopColor: colors.red, paddingHorizontal: 13, paddingVertical: 9 },
  errorText: { color: "#FFD5C7", fontSize: 9 },
  reviewBanner: { backgroundColor: "#3D3512", borderTopWidth: 1, borderTopColor: colors.yellow, paddingHorizontal: 13, paddingVertical: 9 },
  reviewText: { color: "#FFF0A6", fontSize: 9 },
  pressed: { opacity: 0.65 },
});
