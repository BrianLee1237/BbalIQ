import { useMemo, useState } from "react";
import {
  Image,
  LayoutChangeEvent,
  PanResponder,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Svg, { Line, Rect } from "react-native-svg";

import { colors, monoFont } from "../constants/theme";
import type { RimCalibration } from "../types/tracking";

interface VideoRimCalibrationProps {
  imageUri: string;
  imageWidth: number;
  imageHeight: number;
  rim: RimCalibration | null;
  calibrating: boolean;
  onRimChange: (rim: RimCalibration) => void;
}

interface StageSize {
  width: number;
  height: number;
}

export function VideoRimCalibration({
  imageUri,
  imageWidth,
  imageHeight,
  rim,
  calibrating,
  onRimChange,
}: VideoRimCalibrationProps) {
  const [containerWidth, setContainerWidth] = useState(0);
  const [stageSize, setStageSize] = useState<StageSize>({ width: 0, height: 0 });
  const [draftRim, setDraftRim] = useState<RimCalibration | null>(null);
  const aspectRatio = imageWidth / Math.max(1, imageHeight);
  const displayWidth = containerWidth > 0
    ? Math.min(containerWidth, 600 * aspectRatio)
    : 0;
  const displayHeight = displayWidth / Math.max(0.01, aspectRatio);

  const onStageLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setStageSize({ width, height });
  };

  const responder = useMemo(
    () => PanResponder.create({
      onStartShouldSetPanResponder: () => calibrating,
      onMoveShouldSetPanResponder: () => calibrating,
      onPanResponderGrant: () => setDraftRim(null),
      onPanResponderMove: (event, gesture) => {
        if (!stageSize.width || !stageSize.height) return;
        const currentX = Math.max(0, Math.min(stageSize.width, event.nativeEvent.locationX));
        const currentY = Math.max(0, Math.min(stageSize.height, event.nativeEvent.locationY));
        const startX = Math.max(0, Math.min(stageSize.width, currentX - gesture.dx));
        const startY = Math.max(0, Math.min(stageSize.height, currentY - gesture.dy));
        setDraftRim({
          x: Math.min(startX, currentX) / stageSize.width,
          y: Math.min(startY, currentY) / stageSize.height,
          width: Math.abs(currentX - startX) / stageSize.width,
          height: Math.abs(currentY - startY) / stageSize.height,
        });
      },
      onPanResponderRelease: (event, gesture) => {
        if (!stageSize.width || !stageSize.height) return;
        const currentX = Math.max(0, Math.min(stageSize.width, event.nativeEvent.locationX));
        const currentY = Math.max(0, Math.min(stageSize.height, event.nativeEvent.locationY));
        const startX = Math.max(0, Math.min(stageSize.width, currentX - gesture.dx));
        const startY = Math.max(0, Math.min(stageSize.height, currentY - gesture.dy));
        const width = Math.abs(currentX - startX) / stageSize.width;
        const height = Math.abs(currentY - startY) / stageSize.height;
        // Preserve the user's geometry. Validation can explain a small box;
        // silently widening it changes the scoring plane for distant hoops.
        if (width * stageSize.width < 2 || height * stageSize.height < 2) return;
        const nextRim = {
          x: Math.min(Math.min(startX, currentX) / stageSize.width, 1 - width),
          y: Math.min(Math.min(startY, currentY) / stageSize.height, 1 - height),
          width,
          height,
        };
        setDraftRim(null);
        onRimChange(nextRim);
      },
      onPanResponderTerminate: () => setDraftRim(null),
    }),
    [calibrating, onRimChange, stageSize.height, stageSize.width],
  );

  const activeRim = draftRim ?? rim;

  return (
    <View
      style={styles.shell}
      onLayout={(event) => setContainerWidth(event.nativeEvent.layout.width)}
    >
      {displayWidth > 0 ? (
        <View
          style={[styles.stage, { width: displayWidth, height: displayHeight }]}
          onLayout={onStageLayout}
        >
          <Image source={{ uri: imageUri }} style={StyleSheet.absoluteFill} resizeMode="stretch" />
          {activeRim && stageSize.width > 0 ? (
            <Svg
              style={[StyleSheet.absoluteFill, styles.nonInteractive]}
              width={stageSize.width}
              height={stageSize.height}
            >
              <Rect
                x={activeRim.x * stageSize.width}
                y={activeRim.y * stageSize.height}
                width={activeRim.width * stageSize.width}
                height={activeRim.height * stageSize.height}
                fill="rgba(255,90,31,0.08)"
                stroke={calibrating ? colors.orange : colors.acid}
                strokeWidth={3}
                strokeDasharray={calibrating ? "10 7" : undefined}
              />
              <Line
                x1={activeRim.x * stageSize.width}
                x2={(activeRim.x + activeRim.width) * stageSize.width}
                y1={(activeRim.y + activeRim.height * 0.48) * stageSize.height}
                y2={(activeRim.y + activeRim.height * 0.48) * stageSize.height}
                stroke={colors.acid}
                strokeWidth={1.5}
              />
            </Svg>
          ) : null}
          <View
            style={[
              StyleSheet.absoluteFill,
              calibrating ? styles.interactive : styles.nonInteractive,
            ]}
            {...responder.panHandlers}
          />
          {calibrating ? (
            <View style={[styles.hint, styles.nonInteractive]}>
              <Text style={styles.hintTitle}>MARK THE RIM OPENING</Text>
              <Text style={styles.hintBody}>Drag tightly around the inside edge—not the net or backboard</Text>
            </View>
          ) : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  interactive: { pointerEvents: "auto" },
  nonInteractive: { pointerEvents: "none" },
  shell: {
    width: "100%",
    alignItems: "center",
    backgroundColor: colors.black,
    borderWidth: 1,
    borderColor: colors.lineStrong,
  },
  stage: {
    backgroundColor: colors.black,
    overflow: "hidden",
  },
  hint: {
    position: "absolute",
    bottom: 14,
    alignSelf: "center",
    alignItems: "center",
    backgroundColor: colors.orange,
    paddingHorizontal: 12,
    paddingVertical: 9,
  },
  hintTitle: {
    color: colors.white,
    fontFamily: monoFont,
    fontSize: 9,
    fontWeight: "900",
    marginBottom: 3,
  },
  hintBody: {
    color: colors.white,
    fontSize: 8,
    textAlign: "center",
  },
});
