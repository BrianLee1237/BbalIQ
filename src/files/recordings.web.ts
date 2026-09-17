export async function persistRecording(): Promise<string> {
  throw new Error("Live recording is available in the native app build.");
}

export async function shareRecording(uri: string): Promise<void> {
  if (navigator.share) {
    await navigator.share({ title: "ActiveTrack session", url: uri });
    return;
  }
  window.open(uri, "_blank", "noopener,noreferrer");
}

export async function saveRecordingToLibrary(): Promise<void> {
  throw new Error("Saving a live camera recording is available in the native app build.");
}
