import { Link } from "expo-router";
import * as Haptics from "expo-haptics";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Brand } from "../components/Brand";
import { CameraTracker } from "../components/CameraTracker";
import { SessionStats } from "../components/SessionStats";
import { SetupStrip } from "../components/SetupStrip";
import { ShotFeed } from "../components/ShotFeed";
import { colors, monoFont } from "../constants/theme";
import { saveRecordingToLibrary, shareRecording } from "../src/files/recordings";
import { saveSession } from "../src/storage/sessions";
import type { SessionRecord, ShotEvent, ShotKind, ShotMethod } from "../types/tracking";
import { formatDuration } from "../utils/format";

export default function CourtScreen() {
  const { width } = useWindowDimensions();
  const compact = width < 790;
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionComplete, setSessionComplete] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [makes, setMakes] = useState(0);
  const [misses, setMisses] = useState(0);
  const [events, setEvents] = useState<ShotEvent[]>([]);
  const [recordingUri, setRecordingUri] = useState<string | null>(null);

  const elapsedRef = useRef(0);
  const makesRef = useRef(0);
  const missesRef = useRef(0);
  const sessionIdRef = useRef("");
  const startedAtRef = useRef("");

  useEffect(() => {
    elapsedRef.current = elapsedSeconds;
  }, [elapsedSeconds]);

  useEffect(() => {
    makesRef.current = makes;
  }, [makes]);

  useEffect(() => {
    missesRef.current = misses;
  }, [misses]);

  useEffect(() => {
    if (!sessionActive) return;
    const timer = setInterval(() => {
      const next = Math.max(
        0,
        Math.floor((Date.now() - Date.parse(startedAtRef.current)) / 1_000),
      );
      elapsedRef.current = next;
      setElapsedSeconds(next);
    }, 1_000);
    return () => clearInterval(timer);
  }, [sessionActive]);

  const handleShot = useCallback(
    (kind: ShotKind, method: ShotMethod) => {
      if (!sessionActive) return;
      const event: ShotEvent = {
        id: `${Date.now()}-${Math.random()}`,
        kind,
        method,
        elapsedSeconds: elapsedRef.current,
        createdAt: new Date().toISOString(),
      };

      if (kind === "make") {
        makesRef.current += 1;
        setMakes(makesRef.current);
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } else {
        missesRef.current += 1;
        setMisses(missesRef.current);
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      }
      setEvents((current) => [event, ...current].slice(0, 50));
    },
    [sessionActive],
  );

  const beginSession = useCallback(() => {
    const now = new Date().toISOString();
    sessionIdRef.current = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    startedAtRef.current = now;
    elapsedRef.current = 0;
    makesRef.current = 0;
    missesRef.current = 0;
    setElapsedSeconds(0);
    setMakes(0);
    setMisses(0);
    setEvents([]);
    setRecordingUri(null);
    setSessionComplete(false);
    setSessionActive(true);
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  }, []);

  const endSession = useCallback(async (uri: string | null) => {
    setSessionActive(false);
    const finalDuration = Math.max(
      elapsedRef.current,
      Math.floor((Date.now() - Date.parse(startedAtRef.current)) / 1_000),
    );
    elapsedRef.current = finalDuration;
    setElapsedSeconds(finalDuration);
    const attempts = makesRef.current + missesRef.current;
    const session: SessionRecord = {
      id: sessionIdRef.current,
      startedAt: startedAtRef.current,
      endedAt: new Date().toISOString(),
      durationSeconds: finalDuration,
      makes: makesRef.current,
      misses: missesRef.current,
      accuracy: attempts > 0 ? Math.round((makesRef.current / attempts) * 100) : 0,
      recordingUri: uri,
    };

    setRecordingUri(uri);
    setSessionComplete(true);
    try {
      await saveSession(session);
    } catch {
      Alert.alert("Session not saved", "The results are still visible, but they could not be added to history.");
    }
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }, []);

  const undoLast = useCallback(() => {
    const last = events[0];
    if (!last) return;
    if (last.kind === "make") {
      makesRef.current = Math.max(0, makesRef.current - 1);
      setMakes(makesRef.current);
    } else {
      missesRef.current = Math.max(0, missesRef.current - 1);
      setMisses(missesRef.current);
    }
    setEvents((current) => current.slice(1));
    void Haptics.selectionAsync();
  }, [events]);

  const attempts = makes + misses;
  const accuracy = attempts > 0 ? Math.round((makes / attempts) * 100) : 0;

  const handleShare = useCallback(async () => {
    if (!recordingUri) return;
    try {
      await shareRecording(recordingUri);
    } catch (error) {
      Alert.alert("Could not share clip", error instanceof Error ? error.message : "Please try again.");
    }
  }, [recordingUri]);

  const handleSave = useCallback(async () => {
    if (!recordingUri) return;
    try {
      await saveRecordingToLibrary(recordingUri);
      Alert.alert("Clip saved", "The session recording is now in your Photo Library.");
    } catch (error) {
      Alert.alert("Could not save clip", error instanceof Error ? error.message : "Please try again.");
    }
  }, [recordingUri]);

  return (
    <SafeAreaView style={styles.safeArea} edges={["top", "left", "right"]}>
      <ScrollView
        style={styles.screen}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.topbar}>
          <Brand />
          <View style={styles.topbarActions}>
            {!compact ? (
              <>
                <View style={styles.betaBadge}>
                  <Text style={styles.betaText}>ACTIVE BETA</Text>
                </View>
                <View style={styles.privacyBadge}>
                  <View style={styles.privacyDot} />
                  <Text style={styles.privacyText}>ON-DEVICE ANALYSIS</Text>
                </View>
              </>
            ) : null}
            {sessionActive ? (
              <Pressable
                disabled
                accessibilityRole="button"
                accessibilityLabel="End the current session before analyzing a video"
                style={[styles.historyButton, styles.historyButtonDisabled]}
              >
                <Text style={styles.historyText}>UPLOAD VIDEO</Text>
              </Pressable>
            ) : (
              <Link href="/analyze-video" asChild>
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel="Analyze a recorded basketball video"
                  style={({ pressed }) => [styles.historyButton, pressed && styles.pressed]}
                >
                  <Text style={styles.historyText}>UPLOAD VIDEO</Text>
                </Pressable>
              </Link>
            )}
            {sessionActive ? (
              <Pressable
                disabled
                accessibilityRole="button"
                accessibilityLabel="End the current session before opening history"
                style={[styles.historyButton, styles.historyButtonDisabled]}
              >
                <Text style={styles.historyText}>HISTORY</Text>
              </Pressable>
            ) : (
              <Link href="/history" asChild>
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel="Open session history"
                  style={({ pressed }) => [styles.historyButton, pressed && styles.pressed]}
                >
                  <Text style={styles.historyText}>HISTORY</Text>
                </Pressable>
              </Link>
            )}
          </View>
        </View>

        <View style={[styles.hero, compact && styles.heroCompact]}>
          <View style={styles.heroCopy}>
            <Text style={styles.eyebrow}>●  LIVE SHOT TRACKING</Text>
            <Text style={styles.heroTitle}>YOUR REPS.</Text>
            <Text style={styles.heroTitleOrange}>COUNTED.</Text>
            <Text style={styles.heroBody}>
              Set a fixed camera courtside. ActiveTrack locks onto the hoop, follows the ball and player, and confirms makes at the rim and below the net.
            </Text>
            <View style={styles.pipelineHighlights}>
              <Text style={styles.pipelineHighlight}>HOOP LOCK</Text>
              <Text style={styles.pipelineHighlight}>BALL + PLAYER TRACK</Text>
              <Text style={styles.pipelineHighlight}>ENTRY + NET EXIT</Text>
            </View>
          </View>
          <SessionStats
            makes={makes}
            misses={misses}
            elapsedSeconds={elapsedSeconds}
            active={sessionActive}
          />
        </View>

        <View style={[styles.workspace, compact && styles.workspaceCompact]}>
          <View style={styles.cameraColumn}>
            <CameraTracker
              sessionActive={sessionActive}
              onShot={handleShot}
              onBeginSession={beginSession}
              onEndSession={endSession}
            />
            <SetupStrip />
          </View>
          <View style={[styles.feedColumn, compact && styles.feedColumnCompact]}>
            <ShotFeed events={events} attempts={attempts} onUndo={undoLast} />

            {sessionComplete ? (
              <View style={styles.recap}>
                <Text style={styles.recapEyebrow}>SESSION SAVED</Text>
                <View style={styles.recapRow}>
                  <Text style={styles.recapScore}>{makes}/{attempts}</Text>
                  <Text style={styles.recapDetail}>{accuracy}% accuracy{"\n"}in {formatDuration(elapsedSeconds)}</Text>
                </View>
                {recordingUri ? (
                  <View style={styles.recapActions}>
                    <Pressable style={({ pressed }) => [styles.recapButton, pressed && styles.pressed]} onPress={handleShare}>
                      <Text style={styles.recapButtonText}>SHARE CLIP</Text>
                    </Pressable>
                    <Pressable style={({ pressed }) => [styles.recapButton, pressed && styles.pressed]} onPress={handleSave}>
                      <Text style={styles.recapButtonText}>SAVE CLIP</Text>
                    </Pressable>
                  </View>
                ) : null}
              </View>
            ) : null}
          </View>
        </View>

        {sessionActive ? (
          <View style={[styles.videoGateway, compact && styles.videoGatewayCompact, styles.videoGatewayDisabled]}>
            <View style={styles.videoGatewayCopy}>
              <Text style={styles.videoGatewayEyebrow}>SECOND ACTIVITY · AVAILABLE AFTER THIS SESSION</Text>
              <Text style={styles.videoGatewayTitle}>VIDEO UPLOAD ANALYZER</Text>
              <Text style={styles.videoGatewayBody}>End the live session to open recorded-video analysis.</Text>
            </View>
            <Text style={styles.videoGatewayArrow}>→</Text>
          </View>
        ) : (
          <Link href="/analyze-video" asChild>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Open the latest recorded video upload analyzer"
              style={({ pressed }) => [
                styles.videoGateway,
                compact && styles.videoGatewayCompact,
                pressed && styles.pressed,
              ]}
            >
              <View style={styles.videoGatewayCopy}>
                <Text style={styles.videoGatewayEyebrow}>FULL RECORDED-VIDEO ACTIVITY</Text>
                <Text style={styles.videoGatewayTitle}>UPLOAD VIDEO. AUTO-LOCK HOOP. ANALYZE EVERY SHOT.</Text>
                <Text style={styles.videoGatewayBody}>
                  Run the basketball detector and its above-rim to below-net crossing counter with dense 30 FPS recovery, replay, and manual review.
                </Text>
                <View style={styles.videoGatewayTags}>
                  <Text style={styles.videoGatewayTag}>ON-DEVICE</Text>
                  <Text style={styles.videoGatewayTag}>AUTO HOOP FINDER</Text>
                  <Text style={styles.videoGatewayTag}>MAKE / MISS REVIEW</Text>
                </View>
              </View>
              <View style={[styles.videoGatewayAction, compact && styles.videoGatewayActionCompact]}>
                <Text style={styles.videoGatewayActionLabel}>OPEN VIDEO UPLOAD</Text>
                <Text style={styles.videoGatewayArrow}>→</Text>
              </View>
            </Pressable>
          </Link>
        )}

        <View style={styles.footer}>
          <Text style={styles.footerBrand}>ACTIVETRACK / COURT VISION BETA</Text>
          <Text style={styles.footerPrivacy}>VIDEO IS ANALYZED ON THIS DEVICE. NOTHING IS UPLOADED.</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.background,
  },
  screen: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    paddingBottom: 28,
  },
  topbar: {
    minHeight: 61,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 22,
    borderBottomWidth: 1,
    borderBottomColor: colors.line,
  },
  topbarActions: {
    flexDirection: "row",
    alignItems: "center",
  },
  betaBadge: {
    borderWidth: 1,
    borderColor: "rgba(255,90,31,0.45)",
    borderRadius: 999,
    backgroundColor: colors.surface,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  betaText: {
    color: colors.orange,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "800",
    letterSpacing: 0.8,
  },
  privacyBadge: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 999,
    backgroundColor: colors.surface,
    paddingHorizontal: 10,
    paddingVertical: 6,
    marginLeft: 8,
  },
  privacyDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.green,
    marginRight: 6,
  },
  privacyText: {
    color: colors.text,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "700",
    letterSpacing: 0.6,
  },
  historyButton: {
    marginLeft: 8,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  historyText: {
    color: colors.text,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "800",
    letterSpacing: 0.8,
  },
  historyButtonDisabled: {
    opacity: 0.35,
  },
  hero: {
    flexDirection: "row",
    alignItems: "flex-end",
    justifyContent: "space-between",
    paddingHorizontal: 24,
    paddingTop: 30,
    paddingBottom: 25,
  },
  heroCompact: {
    flexDirection: "column",
    alignItems: "stretch",
  },
  heroCopy: {
    flex: 1,
    maxWidth: 590,
    marginRight: 30,
  },
  eyebrow: {
    color: colors.orange,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 1.3,
    marginBottom: 9,
  },
  heroTitle: {
    color: colors.text,
    fontSize: 47,
    lineHeight: 45,
    fontWeight: "900",
    letterSpacing: -3.6,
  },
  heroTitleOrange: {
    color: colors.orange,
    fontSize: 47,
    lineHeight: 45,
    fontWeight: "900",
    letterSpacing: -3.6,
  },
  heroBody: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 18,
    maxWidth: 520,
    marginTop: 14,
  },
  pipelineHighlights: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    marginTop: 15,
    marginHorizontal: -4,
  },
  pipelineHighlight: {
    color: colors.acid,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "900",
    letterSpacing: 0.8,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    backgroundColor: colors.surface,
    paddingHorizontal: 9,
    paddingVertical: 7,
    marginHorizontal: 4,
    marginBottom: 7,
  },
  workspace: {
    flexDirection: "row",
    alignItems: "flex-start",
    paddingHorizontal: 24,
  },
  workspaceCompact: {
    flexDirection: "column",
  },
  cameraColumn: {
    flex: 1.65,
    minWidth: 0,
  },
  feedColumn: {
    flex: 0.7,
    minWidth: 285,
    marginLeft: 24,
  },
  feedColumnCompact: {
    width: "100%",
    marginLeft: 0,
    marginTop: 34,
  },
  videoGateway: {
    minHeight: 142,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: colors.orange,
    borderWidth: 1,
    borderColor: colors.orange,
    marginHorizontal: 24,
    marginTop: 34,
    paddingHorizontal: 24,
    paddingVertical: 20,
    boxShadow: `7px 7px 0 ${colors.acid}`,
  },
  videoGatewayDisabled: { opacity: 0.42, boxShadow: "none" },
  videoGatewayCompact: { flexDirection: "column", alignItems: "stretch" },
  videoGatewayCopy: { flex: 1, maxWidth: 820 },
  videoGatewayEyebrow: {
    color: "rgba(255,255,255,0.82)",
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 1.2,
  },
  videoGatewayTitle: {
    color: colors.white,
    fontSize: 23,
    lineHeight: 27,
    fontWeight: "900",
    letterSpacing: -1.2,
    marginTop: 6,
  },
  videoGatewayBody: {
    color: "rgba(255,255,255,0.84)",
    fontSize: 10,
    lineHeight: 15,
    maxWidth: 760,
    marginTop: 6,
  },
  videoGatewayTags: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginTop: 10,
    marginHorizontal: -4,
  },
  videoGatewayTag: {
    color: colors.white,
    fontFamily: monoFont,
    fontSize: 7,
    fontWeight: "900",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.56)",
    paddingHorizontal: 8,
    paddingVertical: 5,
    marginHorizontal: 4,
    marginBottom: 4,
  },
  videoGatewayAction: { minWidth: 178, alignItems: "flex-end", marginLeft: 22 },
  videoGatewayActionCompact: { minWidth: 0, alignItems: "flex-start", marginLeft: 0, marginTop: 16 },
  videoGatewayActionLabel: {
    color: colors.white,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 0.8,
  },
  videoGatewayArrow: { color: colors.white, fontSize: 42, lineHeight: 44, marginTop: 5 },
  recap: {
    backgroundColor: colors.acid,
    borderWidth: 1,
    borderColor: colors.black,
    marginTop: 14,
    padding: 15,
  },
  recapEyebrow: {
    color: colors.background,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 1.2,
  },
  recapRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 7,
  },
  recapScore: {
    color: colors.background,
    fontFamily: monoFont,
    fontSize: 32,
    fontWeight: "900",
    letterSpacing: -2,
    marginRight: 12,
  },
  recapDetail: {
    color: colors.background,
    fontSize: 9,
    lineHeight: 13,
  },
  recapActions: {
    flexDirection: "row",
    marginTop: 11,
  },
  recapButton: {
    borderWidth: 1,
    borderColor: colors.background,
    paddingHorizontal: 10,
    paddingVertical: 7,
    marginRight: 8,
  },
  recapButtonText: {
    color: colors.background,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "900",
  },
  footer: {
    flexDirection: "row",
    justifyContent: "space-between",
    borderTopWidth: 1,
    borderTopColor: colors.line,
    marginTop: 40,
    marginHorizontal: 24,
    paddingTop: 18,
  },
  footerBrand: {
    color: colors.orange,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "800",
    letterSpacing: 0.7,
  },
  footerPrivacy: {
    color: colors.muted,
    fontFamily: monoFont,
    fontSize: 8,
    letterSpacing: 0.6,
  },
  pressed: {
    opacity: 0.65,
  },
});
