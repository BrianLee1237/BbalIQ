import * as ImagePicker from "expo-image-picker";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Brand } from "../components/Brand";
import { VideoRimCalibration } from "../components/VideoRimCalibration";
import { VideoReviewPlayer } from "../components/VideoReviewPlayer";
import { colors, monoFont } from "../constants/theme";
import {
  analyzeBasketballVideo,
  createVideoPreview,
  MAX_IMPORT_DURATION_SECONDS,
  releaseVideoPreview,
  type VideoPreview,
} from "../src/vision/videoAnalyzer";
import { validateVideoRimCalibration } from "../src/vision/videoAnalysisPolicy";
import type {
  RimCalibration,
  ShotKind,
  VideoAnalysisResult,
} from "../types/tracking";
import { formatDuration } from "../utils/format";

interface SelectedVideo {
  uri: string;
  fileName: string;
}

export default function AnalyzeVideoScreen() {
  const cancelledRef = useRef(false);
  const analysisCancelledRef = useRef(false);
  const [video, setVideo] = useState<SelectedVideo | null>(null);
  const [preview, setPreview] = useState<VideoPreview | null>(null);
  const [rim, setRim] = useState<RimCalibration | null>(null);
  const [rimSource, setRimSource] = useState<"automatic" | "manual" | null>(null);
  const [calibrating, setCalibrating] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [result, setResult] = useState<VideoAnalysisResult | null>(null);
  const [reviewRequest, setReviewRequest] = useState<{
    id: string;
    atSeconds: number;
  } | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    cancelledRef.current = false;
    return () => {
      cancelledRef.current = true;
      analysisCancelledRef.current = true;
    };
  }, []);

  useEffect(() => {
    const previewUri = preview?.uri;
    return () => {
      if (previewUri) {
        void releaseVideoPreview(previewUri);
      }
    };
  }, [preview?.uri]);

  const chooseVideo = useCallback(async () => {
    setError("");
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert(
        "Video access required",
        "Allow video-library access to analyze an existing basketball recording.",
      );
      return;
    }

    const selection = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["videos"],
      allowsEditing: false,
      quality: 1,
      shouldDownloadFromNetwork: true,
    });
    if (selection.canceled) return;
    const asset = selection.assets[0];
    if (!asset || asset.type !== "video") {
      setError("Choose a standard video file from the device library.");
      return;
    }

    const reportedDuration = (asset.duration ?? 0) / 1_000;
    if (reportedDuration > MAX_IMPORT_DURATION_SECONDS) {
      setError(`Choose a video that is ${MAX_IMPORT_DURATION_SECONDS / 60} minutes or shorter.`);
      return;
    }

    setPreparing(true);
    setResult(null);
    setReviewRequest(null);
    setRim(null);
    setRimSource(null);
    setPreview(null);
    setVideo({ uri: asset.uri, fileName: asset.fileName ?? "Basketball video" });
    try {
      const nextPreview = await createVideoPreview(
        asset.uri,
        reportedDuration > 0 ? Math.min(1, reportedDuration * 0.08) : 0.25,
      );
      if (nextPreview.durationSeconds > MAX_IMPORT_DURATION_SECONDS) {
        await releaseVideoPreview(nextPreview.uri);
        throw new Error(`Choose a video that is ${MAX_IMPORT_DURATION_SECONDS / 60} minutes or shorter.`);
      }
      if (!cancelledRef.current) {
        setPreview(nextPreview);
        setRim(nextPreview.automaticRim);
        setRimSource(nextPreview.automaticRim ? "automatic" : null);
        setCalibrating(!nextPreview.automaticRim);
        if (!nextPreview.automaticRim) {
          setError("Automatic hoop lock could not identify a clear target in the sampled frames. Draw the rim only as a fallback.");
        }
      } else {
        await releaseVideoPreview(nextPreview.uri);
      }
    } catch (caught) {
      if (!cancelledRef.current) {
        setError(caught instanceof Error ? caught.message : "The video preview could not be prepared.");
      }
    } finally {
      if (!cancelledRef.current) setPreparing(false);
    }
  }, []);

  const shiftCalibrationFrame = useCallback(async (offsetSeconds: number) => {
    if (!video || !preview || preparing || analyzing) return;
    const targetSeconds = Math.max(
      0,
      Math.min(preview.durationSeconds - 0.05, preview.atSeconds + offsetSeconds),
    );
    if (Math.abs(targetSeconds - preview.atSeconds) < 0.02) return;

    setError("");
    setPreparing(true);
    try {
      const nextPreview = await createVideoPreview(video.uri, targetSeconds);
      if (cancelledRef.current) {
        await releaseVideoPreview(nextPreview.uri);
      } else {
        setPreview(nextPreview);
        setRim(nextPreview.automaticRim);
        setRimSource(nextPreview.automaticRim ? "automatic" : null);
        setCalibrating(!nextPreview.automaticRim);
        if (!nextPreview.automaticRim) {
          setError("No confident hoop was found on nearby frames. Draw the rim to continue.");
        }
      }
    } catch (caught) {
      if (!cancelledRef.current) {
        setError(caught instanceof Error ? caught.message : "That calibration frame could not be loaded.");
      }
    } finally {
      if (!cancelledRef.current) setPreparing(false);
    }
  }, [analyzing, preparing, preview, video]);

  const rimValidationError = useMemo(
    () => rim && preview
      ? validateVideoRimCalibration(rim, preview.width / Math.max(1, preview.height), preview.sourceWidth ?? preview.width)
      : null,
    [preview, rim],
  );

  const analyze = useCallback(async () => {
    if (!video || !preview || !rim) return;
    const calibrationError = validateVideoRimCalibration(
      rim,
      preview.width / Math.max(1, preview.height),
      preview.sourceWidth ?? preview.width,
    );
    if (calibrationError) {
      setError(calibrationError);
      setCalibrating(true);
      return;
    }
    analysisCancelledRef.current = false;
    setError("");
    setAnalyzing(true);
    setCalibrating(false);
    setResult(null);
    setReviewRequest(null);
    setProgress({ completed: 0, total: 0 });
    try {
      const analysis = await analyzeBasketballVideo(video.uri, rim, {
        durationSeconds: preview.durationSeconds,
        rimCalibrationTimeSeconds: preview.atSeconds,
        isCancelled: () => cancelledRef.current || analysisCancelledRef.current,
        onProgress: (completed, total) => {
          if (!cancelledRef.current) setProgress({ completed, total });
        },
      });
      if (!cancelledRef.current) setResult(analysis);
    } catch (caught) {
      if (!cancelledRef.current) {
        setError(
          analysisCancelledRef.current
            ? "Analysis cancelled. Your video and rim calibration are still ready."
            : caught instanceof Error
              ? caught.message
              : "The video could not be analyzed.",
        );
      }
    } finally {
      if (!cancelledRef.current) setAnalyzing(false);
    }
  }, [preview, rim, video]);

  const cancelAnalysis = useCallback(() => {
    analysisCancelledRef.current = true;
  }, []);

  const resolveDecision = useCallback((id: string, kind: ShotKind) => {
    setResult((current) => current
      ? {
          ...current,
          decisions: current.decisions.map((decision) =>
            decision.id === id ? { ...decision, finalKind: kind } : decision,
          ),
        }
      : current);
  }, []);

  const summary = useMemo(() => {
    const decisions = result?.decisions ?? [];
    return {
      makes: decisions.filter((decision) => decision.finalKind === "make").length,
      misses: decisions.filter((decision) => decision.finalKind === "miss").length,
      reviews: decisions.filter((decision) => decision.finalKind === null).length,
    };
  }, [result]);
  const attempts = summary.makes + summary.misses;
  const accuracy = attempts ? Math.round((summary.makes / attempts) * 100) : 0;
  const progressPercent = progress.total
    ? Math.round((progress.completed / progress.total) * 100)
    : 0;
  const calibrationFrameStep = Math.max(1, Math.min(5, (preview?.durationSeconds ?? 0) * 0.05));

  return (
    <SafeAreaView style={styles.safeArea} edges={["left", "right", "bottom"]}>
      <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
        <View style={styles.modeHeader}>
          <Brand />
          <View style={styles.modeActions}>
            <View style={styles.videoModeBadge}>
              <Text style={styles.videoModeText}>VIDEO ANALYZER · V23 BETA</Text>
            </View>
          </View>
        </View>

        <Text style={styles.eyebrow}>RECORDED VIDEO ANALYSIS</Text>
        <Text style={styles.title}>Upload. Auto-lock. Analyze every shot.</Text>
        <Text style={styles.body}>
          Choose a recording with a steady camera and a clear view of the rim. The analyzer uses detail from the original video to follow the ball near the basket. Check uncertain results using the review controls. Processing stays on this device.
        </Text>

        <View style={styles.pipelineStrip}>
          <View style={styles.pipelineStep}>
            <Text style={styles.pipelineStepNumber}>01</Text>
            <Text style={styles.pipelineStepTitle}>LOCK + DETECT</Text>
            <Text style={styles.pipelineStepBody}>Full frame · rim scale · ball scale</Text>
          </View>
          <View style={styles.pipelineStep}>
            <Text style={styles.pipelineStepNumber}>02</Text>
            <Text style={styles.pipelineStepTitle}>RECOVERY TRACK</Text>
            <Text style={styles.pipelineStepBody}>ByteTrack ID · predicted motion · occlusion</Text>
          </View>
          <View style={styles.pipelineStep}>
            <Text style={styles.pipelineStepNumber}>03</Text>
            <Text style={styles.pipelineStepTitle}>EVIDENCE FUSION</Text>
            <Text style={styles.pipelineStepBody}>Trajectory · basket occupancy · exit</Text>
          </View>
          <View style={styles.fixedCameraStep}>
            <Text style={styles.fixedCameraTitle}>FIXED CAMERA REQUIRED</Text>
            <Text style={styles.fixedCameraBody}>Keep the device and hoop still for the complete recording.</Text>
          </View>
        </View>

        <View style={styles.actionsRow}>
          <ActionButton
            label={video ? "CHOOSE ANOTHER VIDEO" : "CHOOSE VIDEO"}
            primary={!video}
            disabled={preparing || analyzing}
            onPress={chooseVideo}
          />
          {preview ? (
            <ActionButton
              label={calibrating ? "DRAW A BOX ON THE RIM" : "ADJUST RIM"}
              selected={calibrating}
              disabled={analyzing}
              onPress={() => {
                setCalibrating(true);
                setRimSource("manual");
                setResult(null);
                setReviewRequest(null);
              }}
            />
          ) : null}
          {preview && !analyzing ? (
            <>
              <ActionButton
                label={`FRAME -${calibrationFrameStep.toFixed(0)}s`}
                disabled={preparing || preview.atSeconds <= 0.05}
                onPress={() => void shiftCalibrationFrame(-calibrationFrameStep)}
              />
              <ActionButton
                label={`FRAME +${calibrationFrameStep.toFixed(0)}s`}
                disabled={preparing || preview.atSeconds >= preview.durationSeconds - 0.08}
                onPress={() => void shiftCalibrationFrame(calibrationFrameStep)}
              />
            </>
          ) : null}
          {preview && rim ? (
            <ActionButton
              label={analyzing ? `ANALYZING ${progressPercent}%` : "ANALYZE VIDEO"}
              primary
              disabled={analyzing || calibrating || Boolean(rimValidationError)}
              onPress={analyze}
            />
          ) : null}
          {analyzing ? (
            <ActionButton label="CANCEL ANALYSIS" onPress={cancelAnalysis} />
          ) : null}
        </View>

        {preparing ? <Text style={styles.status}>PREPARING VIDEO · FINDING HOOP...</Text> : null}
        {video ? (
          <View style={styles.fileRow}>
            <Text style={styles.fileName} numberOfLines={1}>{video.fileName}</Text>
            {preview ? (
              <Text style={styles.fileMeta}>
                {formatDuration(Math.round(preview.durationSeconds))} · calibration frame {formatDuration(Math.round(preview.atSeconds))}
              </Text>
            ) : null}
          </View>
        ) : null}

        {preview ? (
          <>
            <VideoRimCalibration
              imageUri={preview.uri}
              imageWidth={preview.width}
              imageHeight={preview.height}
              rim={rim}
              calibrating={calibrating}
              onRimChange={(nextRim) => {
                setRim(nextRim);
                setRimSource("manual");
                setCalibrating(false);
                setResult(null);
                setReviewRequest(null);
                setError("");
              }}
            />
            {rim ? (
              <View style={[styles.calibrationStatus, rimValidationError && styles.calibrationStatusError]}>
                <Text style={[styles.calibrationStatusText, rimValidationError && styles.calibrationStatusTextError]}>
                  {rimValidationError ?? (rimSource === "automatic"
                    ? `HOOP CANDIDATE · ${Math.round((preview.automaticRimConfidence ?? 0) * 100)}% INITIAL CONFIDENCE · VERIFY THE RIM OPENING`
                    : "MANUAL RIM READY · TRACKING EACH ANALYSIS SAMPLE")}
                </Text>
                <Text style={styles.calibrationStatusText}>
                  Before analyzing, check that the box surrounds only the rim opening—not the support arm, backboard, or entire net. Use Adjust rim if needed.
                </Text>
              </View>
            ) : null}
          </>
        ) : (
          <View style={styles.emptyCard}>
            <Text style={styles.emptyTitle}>No video selected</Text>
            <Text style={styles.emptyBody}>
              Select a fixed-camera landscape recording up to {MAX_IMPORT_DURATION_SECONDS / 60} minutes long. Keep the device still and the rim plus complete ball flight visible for every shot.
            </Text>
          </View>
        )}

        {analyzing ? (
          <View style={styles.progressCard}>
            <View style={styles.progressHeader}>
              <Text style={styles.progressLabel}>RUNNING DUAL TRACK + BASKET EVIDENCE</Text>
              <Text style={styles.progressValue}>{progressPercent}%</Text>
            </View>
            <View style={styles.progressTrack}>
              <View style={[styles.progressFill, { width: `${progressPercent}%` }]} />
            </View>
            <Text style={styles.progressBody}>
              {progress.completed} of {progress.total} frames checked
            </Text>
          </View>
        ) : null}

        {error ? (
          <View style={styles.errorCard}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        ) : null}

        {result ? (
          <>
            <VideoReviewPlayer
              uri={video?.uri ?? ""}
              aspectRatio={(preview?.width ?? 16) / Math.max(1, preview?.height ?? 9)}
              request={reviewRequest}
            />
            <View style={styles.results}>
            <Text style={styles.resultsEyebrow}>ANALYSIS COMPLETE</Text>
            <View style={styles.statsRow}>
              <ResultStat label="MAKES" value={String(summary.makes)} color={colors.green} />
              <ResultStat label="MISSES" value={String(summary.misses)} color={colors.orange} />
              <ResultStat label="ACCURACY" value={`${accuracy}%`} color={colors.acid} />
              <ResultStat label="REVIEW" value={String(summary.reviews)} color={colors.yellow} />
            </View>
            <Text style={styles.reviewExplainer}>
              Low-confidence events are left uncounted instead of being guessed. Every result can be replayed and corrected.
            </Text>

            {result.diagnostics.warnings.length > 0 ? (
              <View style={styles.qualityWarning}>
                <Text style={styles.qualityWarningTitle}>VIDEO QUALITY REVIEW REQUIRED</Text>
                {result.diagnostics.warnings.map((warning) => (
                  <Text style={styles.qualityWarningText} key={warning}>{warning}</Text>
                ))}
              </View>
            ) : null}
            <Text style={styles.qualityReady}>
              {result.framesAnalyzed} VIDEO SAMPLES · TARGET {result.diagnostics.analysisFps} FPS
            </Text>
            <Text style={styles.qualityReady}>
              HOOP LOCK {Math.round(
                (result.diagnostics.rimTrackedFrames /
                  Math.max(
                    1,
                    result.diagnostics.rimTrackedFrames + result.diagnostics.rimTrackingLostFrames,
                  )) * 100,
              )}% · CAMERA RELOCKS {result.diagnostics.rimGlobalReacquisitions} · BALL TRACKED IN {result.diagnostics.ballTrackedFrames} FRAMES
            </Text>
            <Text style={styles.qualityReady}>
              LEARNED DETECTOR {result.diagnostics.learnedDetectorBackend.toUpperCase()} · HOOP FOUND IN {result.diagnostics.learnedHoopDetectionFrames} FRAMES · BALL FOUND IN {result.diagnostics.learnedBallDetectionFrames} FRAMES
            </Text>

            <Text style={styles.qualityReady}>
              PLAYER DETECTOR {result.diagnostics.learnedPlayerDetectionFrames} FRAMES · BYTETRACK PLAYER LOCK {result.diagnostics.playerTrackedFrames} FRAMES
            </Text>

            {result.decisions.length === 0 ? (
              <View style={styles.noShots}>
                <Text style={styles.noShotsTitle}>No complete shot trajectories found</Text>
                <Text style={styles.noShotsBody}>
                  No complete near-rim and below-net sequence was found. Confirm the camera stayed fixed, review the automatic hoop box, and use Adjust rim only if it is visibly off target.
                </Text>
              </View>
            ) : (
              <View style={styles.decisionList}>
                {result.decisions.map((decision) => (
                  <View style={styles.decisionRow} key={decision.id}>
                    <View style={styles.decisionCopy}>
                      <Text style={styles.decisionTime}>{formatDuration(Math.round(decision.atSeconds))}</Text>
                      <Text style={styles.decisionKind}>
                        {decision.finalKind
                          ? decision.finalKind.toUpperCase()
                          : `REVIEW · LIKELY ${decision.suggestedKind.toUpperCase()}`}
                      </Text>
                      <Text style={styles.decisionConfidence}>
                        {Math.round(decision.confidence * 100)}% trajectory confidence · {formatDecisionReason(decision.reason)}
                      </Text>
                    </View>
                    <View style={styles.reviewActions}>
                      <SmallButton
                        label="VIEW"
                        onPress={() => setReviewRequest({
                          id: `${decision.id}-${Date.now()}`,
                          atSeconds: decision.atSeconds,
                        })}
                      />
                      <SmallButton
                        label="MAKE"
                        selected={decision.finalKind === "make"}
                        onPress={() => resolveDecision(decision.id, "make")}
                      />
                      <SmallButton
                        label="MISS"
                        selected={decision.finalKind === "miss"}
                        onPress={() => resolveDecision(decision.id, "miss")}
                      />
                    </View>
                  </View>
                ))}
              </View>
            )}
            </View>
          </>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function ActionButton({
  label,
  onPress,
  primary = false,
  selected = false,
  disabled = false,
}: {
  label: string;
  onPress: () => void;
  primary?: boolean;
  selected?: boolean;
  disabled?: boolean;
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
        selected && styles.actionSelected,
        disabled && styles.disabled,
        pressed && !disabled && styles.pressed,
      ]}
    >
      <Text style={styles.actionText}>{label}</Text>
    </Pressable>
  );
}

function SmallButton({
  label,
  onPress,
  selected = false,
}: {
  label: string;
  onPress: () => void;
  selected?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label === "VIEW" ? "Review shot video" : `Mark as ${label.toLowerCase()}`}
      onPress={onPress}
      style={({ pressed }) => [
        styles.smallButton,
        selected && styles.smallButtonSelected,
        pressed && styles.pressed,
      ]}
    >
      <Text style={styles.smallButtonText}>{label}</Text>
    </Pressable>
  );
}

function formatDecisionReason(reason: string): string {
  switch (reason) {
    case "rim-crossing": return "detector track crossed the rim plane";
    case "rim-entry-exit": return "ball entered the rim and exited below the net";
    case "rim-proximity-exit": return "near-rim descent confirmed by centered net exit";
    case "rim-entry-lost": return "rim entry found; net exit needs review";
    case "semantic-basket": return "basket occupancy confirmed by centered net exit";
    case "evidence-fusion": return "independent shot signals agree";
    case "evidence-conflict": return "shot signals disagree; review required";
    case "airball": return "outside rim opening";
    case "lost": return "ball lost after release";
    case "timeout": return "incomplete trajectory";
    case "cooldown": return "duplicate suppressed";
    default: return "trajectory evidence";
  }
}

function ResultStat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={[styles.statValue, { color }]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background },
  screen: { flex: 1, backgroundColor: colors.background },
  content: { padding: 24, paddingBottom: 50 },
  modeHeader: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: colors.line,
    paddingBottom: 16,
    marginBottom: 24,
  },
  modeActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    marginHorizontal: -5,
  },
  videoModeBadge: {
    minHeight: 39,
    justifyContent: "center",
    borderWidth: 1,
    borderColor: colors.acid,
    backgroundColor: "rgba(223,255,79,0.06)",
    paddingHorizontal: 14,
    marginHorizontal: 5,
    marginVertical: 4,
  },
  videoModeText: {
    color: colors.acid,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 0.5,
  },
  eyebrow: {
    color: colors.orange,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 1.3,
  },
  title: {
    color: colors.text,
    fontSize: 36,
    lineHeight: 42,
    fontWeight: "900",
    letterSpacing: -2.2,
    marginTop: 7,
  },
  body: {
    color: colors.muted,
    maxWidth: 680,
    fontSize: 11,
    lineHeight: 17,
    marginTop: 7,
  },
  pipelineStrip: {
    flexDirection: "row",
    flexWrap: "wrap",
    borderWidth: 1,
    borderColor: colors.lineStrong,
    backgroundColor: colors.surface,
    marginTop: 18,
  },
  pipelineStep: {
    flex: 1,
    minWidth: 175,
    minHeight: 86,
    borderRightWidth: 1,
    borderRightColor: colors.line,
    paddingHorizontal: 13,
    paddingVertical: 12,
  },
  pipelineStepNumber: { color: colors.orange, fontFamily: monoFont, fontSize: 8, fontWeight: "900" },
  pipelineStepTitle: { color: colors.text, fontFamily: monoFont, fontSize: 9, fontWeight: "900", marginTop: 6 },
  pipelineStepBody: { color: colors.muted, fontSize: 8, lineHeight: 12, marginTop: 4 },
  fixedCameraStep: {
    flex: 1.15,
    minWidth: 210,
    minHeight: 86,
    justifyContent: "center",
    backgroundColor: colors.orangeSoft,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  fixedCameraTitle: { color: colors.orange, fontFamily: monoFont, fontSize: 9, fontWeight: "900" },
  fixedCameraBody: { color: colors.text, fontSize: 8, lineHeight: 12, marginTop: 5 },
  actionsRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginTop: 20,
    marginHorizontal: -5,
  },
  actionButton: {
    minHeight: 43,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.lineStrong,
    backgroundColor: colors.surfaceRaised,
    paddingHorizontal: 15,
    marginHorizontal: 5,
    marginBottom: 10,
  },
  actionPrimary: { backgroundColor: colors.orange, borderColor: colors.orange },
  actionSelected: { borderColor: colors.acid },
  actionText: {
    color: colors.text,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
  },
  status: { color: colors.acid, fontFamily: monoFont, fontSize: 9, marginBottom: 10 },
  fileRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.line,
    paddingHorizontal: 12,
    paddingVertical: 9,
    marginBottom: 10,
  },
  fileName: { flex: 1, color: colors.text, fontSize: 10, fontWeight: "700" },
  fileMeta: { color: colors.muted, fontFamily: monoFont, fontSize: 9, marginLeft: 10 },
  emptyCard: {
    minHeight: 230,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.line,
    padding: 28,
  },
  emptyTitle: { color: colors.text, fontSize: 18, fontWeight: "800" },
  emptyBody: { color: colors.muted, maxWidth: 480, fontSize: 10, lineHeight: 16, textAlign: "center", marginTop: 6 },
  calibrationStatus: {
    backgroundColor: "rgba(223,255,79,0.08)",
    borderWidth: 1,
    borderTopWidth: 0,
    borderColor: colors.acid,
    paddingHorizontal: 12,
    paddingVertical: 9,
  },
  calibrationStatusError: { backgroundColor: "#4F190F", borderColor: colors.red },
  calibrationStatusText: { color: colors.acid, fontFamily: monoFont, fontSize: 8, fontWeight: "900" },
  calibrationStatusTextError: { color: "#FFD5C7" },
  progressCard: { backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.line, padding: 15, marginTop: 16 },
  progressHeader: { flexDirection: "row", justifyContent: "space-between" },
  progressLabel: { color: colors.text, fontFamily: monoFont, fontSize: 9, fontWeight: "900" },
  progressValue: { color: colors.acid, fontFamily: monoFont, fontSize: 10, fontWeight: "900" },
  progressTrack: { height: 8, backgroundColor: colors.surfaceRaised, marginTop: 11, overflow: "hidden" },
  progressFill: { height: "100%", backgroundColor: colors.orange },
  progressBody: { color: colors.muted, fontSize: 9, marginTop: 7 },
  errorCard: { backgroundColor: "#4F190F", borderWidth: 1, borderColor: colors.red, padding: 12, marginTop: 14 },
  errorText: { color: "#FFD5C7", fontSize: 10 },
  results: { backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.lineStrong, padding: 16, marginTop: 18 },
  resultsEyebrow: { color: colors.acid, fontFamily: monoFont, fontSize: 9, fontWeight: "900", letterSpacing: 1.1 },
  statsRow: { flexDirection: "row", marginTop: 14 },
  stat: { flex: 1 },
  statLabel: { color: colors.muted, fontFamily: monoFont, fontSize: 8, fontWeight: "800" },
  statValue: { fontFamily: monoFont, fontSize: 31, lineHeight: 35, fontWeight: "900", marginTop: 3 },
  reviewExplainer: { color: colors.muted, fontSize: 9, lineHeight: 14, marginTop: 10 },
  qualityReady: { color: colors.green, fontFamily: monoFont, fontSize: 8, fontWeight: "800", marginTop: 10 },
  qualityWarning: { backgroundColor: "#3D3512", borderWidth: 1, borderColor: colors.yellow, padding: 11, marginTop: 11 },
  qualityWarningTitle: { color: colors.yellow, fontFamily: monoFont, fontSize: 9, fontWeight: "900" },
  qualityWarningText: { color: "#FFF0A6", fontSize: 9, lineHeight: 14, marginTop: 4 },
  noShots: { borderTopWidth: 1, borderTopColor: colors.line, paddingTop: 15, marginTop: 15 },
  noShotsTitle: { color: colors.text, fontSize: 14, fontWeight: "800" },
  noShotsBody: { color: colors.muted, fontSize: 9, lineHeight: 14, marginTop: 4 },
  decisionList: { borderTopWidth: 1, borderTopColor: colors.line, marginTop: 15 },
  decisionRow: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", borderBottomWidth: 1, borderBottomColor: colors.line, paddingVertical: 11 },
  decisionCopy: { flex: 1 },
  decisionTime: { color: colors.muted, fontFamily: monoFont, fontSize: 8 },
  decisionKind: { color: colors.text, fontFamily: monoFont, fontSize: 11, fontWeight: "900", marginTop: 2 },
  decisionConfidence: { color: colors.muted, fontSize: 8, marginTop: 2 },
  reviewActions: { flexDirection: "row", flexWrap: "wrap", marginLeft: 12, marginVertical: 4 },
  smallButton: { borderWidth: 1, borderColor: colors.acid, paddingHorizontal: 10, paddingVertical: 8, marginLeft: 7 },
  smallButtonSelected: { backgroundColor: "rgba(223,255,79,0.16)", borderWidth: 2, paddingHorizontal: 9, paddingVertical: 7 },
  smallButtonText: { color: colors.acid, fontFamily: monoFont, fontSize: 8, fontWeight: "900" },
  disabled: { opacity: 0.38 },
  pressed: { opacity: 0.65 },
});
