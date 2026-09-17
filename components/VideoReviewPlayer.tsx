import { useEffect } from "react";
import { StyleSheet, Text, View } from "react-native";
import { useVideoPlayer, VideoView } from "expo-video";

import { colors, monoFont } from "../constants/theme";

interface ReviewRequest {
  id: string;
  atSeconds: number;
}

interface VideoReviewPlayerProps {
  uri: string;
  aspectRatio: number;
  request: ReviewRequest | null;
}

export function VideoReviewPlayer({
  uri,
  aspectRatio,
  request,
}: VideoReviewPlayerProps) {
  const player = useVideoPlayer(uri, (createdPlayer) => {
    createdPlayer.loop = false;
    createdPlayer.muted = true;
  });

  useEffect(() => {
    if (!request) return;
    // Expo's VideoPlayer is an imperative native shared object; direct time
    // assignment is its frame-accurate seek API.
    // eslint-disable-next-line react-hooks/immutability
    player.currentTime = Math.max(0, request.atSeconds - 1.25);
    player.play();
  }, [player, request]);

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.eyebrow}>SHOT REVIEW PLAYER</Text>
        <Text style={styles.hint}>Playback starts 1.25 seconds before the decision.</Text>
      </View>
      <VideoView
        player={player}
        nativeControls
        contentFit="contain"
        style={[styles.video, { aspectRatio: Math.max(0.5, aspectRatio) }]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.black,
    borderWidth: 1,
    borderColor: colors.lineStrong,
    marginTop: 18,
  },
  header: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: colors.surface,
    paddingHorizontal: 12,
    paddingVertical: 9,
  },
  eyebrow: {
    color: colors.acid,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
  },
  hint: {
    color: colors.muted,
    fontSize: 8,
  },
  video: {
    width: "100%",
    maxHeight: 520,
    backgroundColor: colors.black,
  },
});
