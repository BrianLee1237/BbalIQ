import * as FileSystem from "expo-file-system/legacy";
import * as MediaLibrary from "expo-media-library";
import * as Sharing from "expo-sharing";

function toFileUri(path: string): string {
  return path.startsWith("file://") ? path : `file://${path}`;
}

export async function persistRecording(temporaryPath: string): Promise<string> {
  if (!FileSystem.documentDirectory) {
    throw new Error("A persistent app documents directory is not available.");
  }

  const directory = `${FileSystem.documentDirectory}recordings/`;
  await FileSystem.makeDirectoryAsync(directory, { intermediates: true });
  const sourceUri = toFileUri(temporaryPath);
  const extension = temporaryPath.toLowerCase().endsWith(".mp4") ? "mp4" : "mov";
  const destination = `${directory}activetrack-${Date.now()}.${extension}`;
  await FileSystem.copyAsync({ from: sourceUri, to: destination });
  return destination;
}

export async function shareRecording(uri: string): Promise<void> {
  const available = await Sharing.isAvailableAsync();
  if (!available) throw new Error("Sharing is not available on this device.");
  await Sharing.shareAsync(uri, {
    mimeType: uri.toLowerCase().endsWith(".mp4") ? "video/mp4" : "video/quicktime",
    dialogTitle: "Share ActiveTrack session",
    UTI: uri.toLowerCase().endsWith(".mp4") ? "public.mpeg-4" : "com.apple.quicktime-movie",
  });
}

export async function saveRecordingToLibrary(uri: string): Promise<void> {
  const permission = await MediaLibrary.requestPermissionsAsync(true, ["video"]);
  if (!permission.granted) {
    throw new Error("Photo Library permission is required to save the clip.");
  }
  await MediaLibrary.createAssetAsync(uri);
}
