import "react-native-gesture-handler";

import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";

import { colors } from "../constants/theme";

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          contentStyle: { backgroundColor: colors.background },
          headerStyle: { backgroundColor: colors.background },
          headerTintColor: colors.text,
          headerShadowVisible: false,
          headerBackTitle: "Court",
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="analyze-video" options={{ title: "Analyze video" }} />
        <Stack.Screen name="history" options={{ title: "Session history" }} />
      </Stack>
    </>
  );
}
