import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import { colors, monoFont } from "../constants/theme";
import type { ShotEvent } from "../types/tracking";
import { formatDuration } from "../utils/format";

interface ShotFeedProps {
  events: ShotEvent[];
  attempts: number;
  onUndo: () => void;
}

export function ShotFeed({ events, attempts, onUndo }: ShotFeedProps) {
  return (
    <View style={styles.panel}>
      <View style={styles.headingRow}>
        <View>
          <Text style={styles.eyebrow}>SESSION FEED</Text>
          <Text style={styles.heading}>Every shot, live.</Text>
        </View>
        <Pressable
          onPress={onUndo}
          disabled={events.length === 0}
          accessibilityRole="button"
          accessibilityLabel="Undo last shot"
          style={({ pressed }) => [styles.undo, pressed && styles.pressed]}
        >
          <Text style={[styles.undoText, events.length === 0 && styles.disabledText]}>UNDO</Text>
        </Pressable>
      </View>

      <ScrollView style={styles.feed} contentContainerStyle={events.length === 0 && styles.feedEmptyContent}>
        {events.length === 0 ? (
          <View style={styles.emptyState}>
            <View style={styles.emptyNumberBox}>
              <Text style={styles.emptyNumber}>00</Text>
            </View>
            <Text style={styles.emptyTitle}>No attempts yet</Text>
            <Text style={styles.emptyBody}>Makes and misses will appear here while you shoot.</Text>
          </View>
        ) : (
          events.map((event, index) => (
            <View style={styles.event} key={event.id}>
              <Text style={styles.eventNumber}>{String(attempts - index).padStart(2, "0")}</Text>
              <View style={[styles.resultBox, event.kind === "make" ? styles.makeBox : styles.missBox]}>
                <Text style={[styles.resultSymbol, event.kind === "make" ? styles.makeText : styles.missText]}>
                  {event.kind === "make" ? "✓" : "×"}
                </Text>
              </View>
              <View style={styles.eventCopy}>
                <Text style={styles.eventTitle}>{event.kind === "make" ? "MAKE" : "MISS"}</Text>
                <Text style={styles.eventMethod}>
                  {event.method === "tracked"
                    ? "Auto detected"
                    : event.method === "demo"
                      ? "Demo detection"
                      : "Manual correction"}
                </Text>
              </View>
              <Text style={styles.eventTime}>{formatDuration(event.elapsedSeconds)}</Text>
            </View>
          ))
        )}
      </ScrollView>

      <View style={styles.tip}>
        <View style={styles.tipIcon}>
          <Text style={styles.tipIconText}>i</Text>
        </View>
        <View style={styles.tipCopy}>
          <Text style={styles.tipTitle}>BETA TRACKING TIP</Text>
          <Text style={styles.tipBody}>Use an orange ball, steady lighting, and a contrasting background.</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    flex: 1,
    minWidth: 285,
    borderTopWidth: 3,
    borderTopColor: colors.acid,
  },
  headingRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-end",
    borderBottomWidth: 1,
    borderBottomColor: colors.line,
    paddingVertical: 14,
  },
  eyebrow: {
    color: colors.orange,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 1.3,
    marginBottom: 4,
  },
  heading: {
    color: colors.text,
    fontSize: 25,
    lineHeight: 29,
    fontWeight: "800",
    letterSpacing: -1.4,
  },
  undo: {
    padding: 8,
  },
  undoText: {
    color: colors.muted,
    fontFamily: monoFont,
    fontSize: 10,
    textDecorationLine: "underline",
  },
  disabledText: {
    opacity: 0.3,
  },
  pressed: {
    opacity: 0.65,
  },
  feed: {
    minHeight: 196,
    maxHeight: 285,
    borderBottomWidth: 1,
    borderBottomColor: colors.line,
  },
  feedEmptyContent: {
    flexGrow: 1,
  },
  emptyState: {
    flex: 1,
    minHeight: 196,
    justifyContent: "center",
    alignItems: "center",
    padding: 18,
  },
  emptyNumberBox: {
    width: 42,
    height: 42,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 10,
  },
  emptyNumber: {
    color: colors.faint,
    fontFamily: monoFont,
    fontSize: 10,
  },
  emptyTitle: {
    color: colors.text,
    fontSize: 13,
    fontWeight: "700",
  },
  emptyBody: {
    color: colors.muted,
    fontSize: 10,
    lineHeight: 15,
    textAlign: "center",
    maxWidth: 210,
    marginTop: 4,
  },
  event: {
    minHeight: 57,
    flexDirection: "row",
    alignItems: "center",
    borderBottomWidth: 1,
    borderBottomColor: colors.line,
    paddingVertical: 9,
  },
  eventNumber: {
    width: 27,
    color: colors.faint,
    fontFamily: monoFont,
    fontSize: 9,
  },
  resultBox: {
    width: 34,
    height: 34,
    borderWidth: 1,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 10,
  },
  makeBox: {
    borderColor: colors.green,
    backgroundColor: "rgba(68,218,138,0.08)",
  },
  missBox: {
    borderColor: colors.orange,
    backgroundColor: colors.orangeSoft,
  },
  resultSymbol: {
    fontSize: 18,
    fontWeight: "900",
  },
  makeText: {
    color: colors.green,
  },
  missText: {
    color: colors.orange,
  },
  eventCopy: {
    flex: 1,
  },
  eventTitle: {
    color: colors.text,
    fontFamily: monoFont,
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 0.8,
  },
  eventMethod: {
    color: colors.muted,
    fontSize: 9,
    marginTop: 2,
  },
  eventTime: {
    color: colors.muted,
    fontFamily: monoFont,
    fontSize: 9,
  },
  tip: {
    flexDirection: "row",
    backgroundColor: colors.surfaceRaised,
    borderWidth: 1,
    borderColor: colors.line,
    borderLeftWidth: 3,
    borderLeftColor: colors.orange,
    padding: 13,
    marginTop: 14,
  },
  tipIcon: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.text,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 10,
  },
  tipIconText: {
    color: colors.text,
    fontFamily: monoFont,
    fontSize: 10,
  },
  tipCopy: {
    flex: 1,
  },
  tipTitle: {
    color: colors.text,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "800",
    marginBottom: 3,
  },
  tipBody: {
    color: colors.muted,
    fontSize: 9,
    lineHeight: 13,
  },
});
