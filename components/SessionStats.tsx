import { StyleSheet, Text, View } from "react-native";

import { colors, monoFont } from "../constants/theme";
import { formatDuration } from "../utils/format";

interface SessionStatsProps {
  makes: number;
  misses: number;
  elapsedSeconds: number;
  active: boolean;
}

export function SessionStats({
  makes,
  misses,
  elapsedSeconds,
  active,
}: SessionStatsProps) {
  const attempts = makes + misses;
  const accuracy = attempts > 0 ? Math.round((makes / attempts) * 100) : 0;

  return (
    <View style={styles.wrapper} accessibilityLabel="Current session statistics">
      <View style={styles.timerRow}>
        <View style={[styles.liveDot, active && styles.liveDotActive]} />
        <Text style={styles.timer}>{formatDuration(elapsedSeconds)}</Text>
      </View>
      <View style={styles.card}>
        <Stat label="MAKES" value={String(makes).padStart(2, "0")} color={colors.green} />
        <Stat label="MISSES" value={String(misses).padStart(2, "0")} color={colors.orange} />
        <Stat label="ACCURACY" value={`${accuracy}%`} color={colors.text} last />
      </View>
    </View>
  );
}

interface StatProps {
  label: string;
  value: string;
  color: string;
  last?: boolean;
}

function Stat({ label, value, color, last = false }: StatProps) {
  return (
    <View style={[styles.stat, !last && styles.statBorder]}>
      <Text style={styles.label}>{label}</Text>
      <Text style={[styles.value, { color }]} numberOfLines={1} adjustsFontSizeToFit>
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    minWidth: 345,
  },
  timerRow: {
    flexDirection: "row",
    justifyContent: "flex-end",
    alignItems: "center",
    marginBottom: 7,
  },
  liveDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: colors.faint,
    marginRight: 7,
  },
  liveDotActive: {
    backgroundColor: colors.orange,
  },
  timer: {
    color: colors.text,
    fontFamily: monoFont,
    fontSize: 11,
    fontWeight: "700",
  },
  card: {
    flexDirection: "row",
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    boxShadow: `6px 6px 0 ${colors.black}`,
    elevation: 8,
  },
  stat: {
    flex: 1,
    minWidth: 100,
    paddingHorizontal: 15,
    paddingVertical: 13,
  },
  statBorder: {
    borderRightWidth: 1,
    borderRightColor: colors.line,
  },
  label: {
    color: colors.muted,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "700",
    letterSpacing: 1.2,
  },
  value: {
    fontFamily: monoFont,
    fontSize: 38,
    lineHeight: 43,
    fontWeight: "800",
    letterSpacing: -3,
    marginTop: 3,
  },
});
