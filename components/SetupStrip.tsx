import { StyleSheet, Text, View } from "react-native";

import { colors, monoFont } from "../constants/theme";

const steps = [
  { number: "01", title: "Lock phone in place", detail: "Landscape, no pan or zoom" },
  { number: "02", title: "Frame player + rim", detail: "Full body and flight visible" },
  { number: "03", title: "Mark the rim", detail: "Drag tightly over opening" },
];

export function SetupStrip() {
  return (
    <View style={styles.container}>
      {steps.map((step, index) => (
        <View style={styles.group} key={step.number}>
          <View style={styles.step}>
            <View style={styles.numberBox}>
              <Text style={styles.number}>{step.number}</Text>
            </View>
            <View style={styles.copy}>
              <Text style={styles.title} numberOfLines={1}>{step.title}</Text>
              <Text style={styles.detail} numberOfLines={1}>{step.detail}</Text>
            </View>
          </View>
          {index < steps.length - 1 ? <Text style={styles.arrow}>→</Text> : null}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.line,
    paddingHorizontal: 14,
    paddingVertical: 12,
    marginTop: 17,
  },
  group: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
  },
  step: {
    flex: 1,
    minWidth: 0,
    flexDirection: "row",
    alignItems: "center",
  },
  numberBox: {
    width: 28,
    height: 28,
    borderWidth: 1,
    borderColor: colors.orange,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 9,
  },
  number: {
    color: colors.orange,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "700",
  },
  copy: {
    flex: 1,
    minWidth: 0,
  },
  title: {
    color: colors.text,
    fontSize: 10,
    fontWeight: "700",
  },
  detail: {
    color: colors.muted,
    fontSize: 8,
    marginTop: 2,
  },
  arrow: {
    color: colors.faint,
    fontFamily: monoFont,
    fontSize: 13,
    marginHorizontal: 10,
  },
});
