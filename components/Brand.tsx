import { StyleSheet, Text, View } from "react-native";

import { colors } from "../constants/theme";

export function Brand() {
  return (
    <View style={styles.container} accessibilityLabel="ActiveTrack">
      <View style={styles.ball} accessibilityElementsHidden>
        <View style={[styles.seam, styles.seamVertical]} />
        <View style={[styles.seam, styles.seamHorizontal]} />
        <View style={[styles.seam, styles.seamDiagonal]} />
      </View>
      <Text style={styles.active}>ACTIVE</Text>
      <Text style={styles.track}>TRACK</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "center",
  },
  ball: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: colors.orange,
    marginRight: 9,
    overflow: "hidden",
    borderWidth: 2,
    borderColor: colors.black,
  },
  seam: {
    position: "absolute",
    backgroundColor: colors.black,
  },
  seamVertical: {
    width: 2,
    height: 34,
    left: 13,
    top: -3,
    transform: [{ rotate: "18deg" }],
  },
  seamHorizontal: {
    width: 34,
    height: 2,
    left: -3,
    top: 13,
    transform: [{ rotate: "-10deg" }],
  },
  seamDiagonal: {
    width: 38,
    height: 2,
    left: -5,
    top: 13,
    transform: [{ rotate: "45deg" }],
  },
  active: {
    color: colors.text,
    fontSize: 17,
    lineHeight: 20,
    fontWeight: "900",
    letterSpacing: -1,
  },
  track: {
    color: colors.orange,
    fontSize: 17,
    lineHeight: 20,
    fontWeight: "900",
    letterSpacing: -1,
  },
});
