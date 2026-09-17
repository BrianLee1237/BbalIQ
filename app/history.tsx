import { useFocusEffect } from "expo-router";
import { useCallback, useState } from "react";
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { colors, monoFont } from "../constants/theme";
import { shareRecording } from "../src/files/recordings";
import { loadSessions } from "../src/storage/sessions";
import type { SessionRecord } from "../types/tracking";
import { formatDuration, formatSessionDate } from "../utils/format";

export default function HistoryScreen() {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      let mounted = true;
      setLoading(true);
      loadSessions()
        .then((value) => {
          if (mounted) setSessions(value);
        })
        .catch(() => {
          if (mounted) Alert.alert("History unavailable", "Saved sessions could not be loaded.");
        })
        .finally(() => {
          if (mounted) setLoading(false);
        });
      return () => {
        mounted = false;
      };
    }, []),
  );

  const handleShare = useCallback(async (uri: string) => {
    try {
      await shareRecording(uri);
    } catch (error) {
      Alert.alert("Could not share clip", error instanceof Error ? error.message : "Please try again.");
    }
  }, []);

  return (
    <SafeAreaView style={styles.safeArea} edges={["left", "right", "bottom"]}>
      <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
        <Text style={styles.eyebrow}>PERMANENT ON-DEVICE STATS</Text>
        <Text style={styles.title}>Session history</Text>
        <Text style={styles.body}>Completed results are saved on this device. Camera frames are never stored or uploaded.</Text>

        {loading ? (
          <Text style={styles.loading}>LOADING SESSIONS...</Text>
        ) : sessions.length === 0 ? (
          <View style={styles.emptyCard}>
            <Text style={styles.emptyTitle}>No saved sessions yet</Text>
            <Text style={styles.emptyBody}>Finish a court session and its makes, misses, accuracy, and duration will appear here.</Text>
          </View>
        ) : (
          <View style={styles.list}>
            {sessions.map((session) => {
              const attempts = session.makes + session.misses;
              return (
                <View style={styles.card} key={session.id}>
                  <View style={styles.cardHeader}>
                    <Text style={styles.date}>{formatSessionDate(session.startedAt)}</Text>
                    <Text style={styles.duration}>{formatDuration(session.durationSeconds)}</Text>
                  </View>
                  <View style={styles.statsRow}>
                    <HistoryStat label="MAKES" value={String(session.makes)} color={colors.green} />
                    <HistoryStat label="MISSES" value={String(session.misses)} color={colors.orange} />
                    <HistoryStat label="ATTEMPTS" value={String(attempts)} color={colors.text} />
                    <HistoryStat label="ACCURACY" value={`${session.accuracy}%`} color={colors.acid} />
                  </View>
                  {session.recordingUri ? (
                    <Pressable
                      onPress={() => handleShare(session.recordingUri!)}
                      style={({ pressed }) => [styles.shareButton, pressed && styles.pressed]}
                      accessibilityRole="button"
                      accessibilityLabel="Share saved session recording"
                    >
                      <Text style={styles.shareText}>SHARE SAVED CLIP</Text>
                    </Pressable>
                  ) : null}
                </View>
              );
            })}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function HistoryStat({ label, value, color }: { label: string; value: string; color: string }) {
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
  content: { padding: 24, paddingBottom: 40 },
  eyebrow: {
    color: colors.orange,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 1.3,
  },
  title: {
    color: colors.text,
    fontSize: 38,
    lineHeight: 43,
    fontWeight: "900",
    letterSpacing: -2.5,
    marginTop: 7,
  },
  body: {
    color: colors.muted,
    fontSize: 11,
    lineHeight: 17,
    maxWidth: 520,
    marginTop: 7,
  },
  loading: {
    color: colors.muted,
    fontFamily: monoFont,
    fontSize: 10,
    marginTop: 28,
  },
  emptyCard: {
    borderWidth: 1,
    borderColor: colors.line,
    backgroundColor: colors.surface,
    padding: 28,
    marginTop: 25,
  },
  emptyTitle: { color: colors.text, fontSize: 17, fontWeight: "800" },
  emptyBody: { color: colors.muted, fontSize: 10, lineHeight: 15, marginTop: 5 },
  list: { marginTop: 23 },
  card: {
    borderWidth: 1,
    borderColor: colors.line,
    backgroundColor: colors.surface,
    padding: 16,
    marginBottom: 13,
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderBottomWidth: 1,
    borderBottomColor: colors.line,
    paddingBottom: 10,
  },
  date: { color: colors.text, fontSize: 12, fontWeight: "700" },
  duration: { color: colors.muted, fontFamily: monoFont, fontSize: 10 },
  statsRow: { flexDirection: "row", marginTop: 13 },
  stat: { flex: 1 },
  statLabel: {
    color: colors.muted,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "700",
    letterSpacing: 0.8,
  },
  statValue: {
    fontFamily: monoFont,
    fontSize: 25,
    lineHeight: 29,
    fontWeight: "900",
    letterSpacing: -1.5,
    marginTop: 3,
  },
  shareButton: {
    alignSelf: "flex-start",
    borderWidth: 1,
    borderColor: colors.acid,
    paddingHorizontal: 11,
    paddingVertical: 7,
    marginTop: 13,
  },
  shareText: {
    color: colors.acid,
    fontFamily: monoFont,
    fontSize: 8,
    fontWeight: "900",
  },
  pressed: { opacity: 0.65 },
});
