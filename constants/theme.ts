import { Platform } from "react-native";

export const colors = {
  background: "#070A08",
  surface: "#101612",
  surfaceRaised: "#161D18",
  surfaceSoft: "#1B241D",
  line: "#2B352E",
  lineStrong: "#465249",
  text: "#F3F6F1",
  muted: "#8D998F",
  faint: "#5A655D",
  orange: "#FF5A1F",
  orangeSoft: "rgba(255,90,31,0.14)",
  acid: "#DFFF4F",
  green: "#44DA8A",
  yellow: "#F2C94C",
  red: "#FF805A",
  black: "#000000",
  white: "#FFFFFF",
} as const;

export const monoFont = Platform.select({
  ios: "SFMono-Regular",
  android: "monospace",
  default: "monospace",
});

export const radii = {
  small: 6,
  medium: 10,
  large: 16,
  pill: 999,
} as const;
