# ChatGPT project context

This directory is a local mirror of the ChatGPT project “activetrack”.

- Treat every file under `sources/` as read-only reference material.
- Do not edit, rename, move, or delete synced project files under `sources/`.

# ActiveTrack Expo architecture

- This repository is an Expo SDK 57 React Native app using the New Architecture and TypeScript strict mode.
- Route files live only in `app/` and use `expo-router`. Shared UI belongs in `components/`; tracking logic belongs in `src/tracking/`; device services belong in `src/storage/` and `src/files/`.
- Use React Native primitives in shared and native UI. Never add HTML tags, CSS files, or browser storage. DOM and canvas APIs are allowed only inside explicitly named `.web.ts`/`.web.tsx` platform adapters.
- Every rendered string must be inside a native `Text` component.
- Use `StyleSheet.create`, Flexbox, and native responsive measurements. Do not add Tailwind or web-only style properties.
- Use `onPress` and native gesture responders instead of web click, pointer, or mouse events.
- Persist session data with AsyncStorage. Use SecureStore only for future secrets or credentials.
- The camera pipeline is VisionCamera 5 -> SkiaCamera -> native GPU resizer -> Fast OpenCV plus the dominant-player motion tracker -> react-native-worklets -> the pure shot state machine. Keep frame processing off the React JS thread.
- Real-time tracking requires an Expo development build or App Store build. Do not claim Expo Go support.
- Keep video processing on-device. Do not upload frames or recordings without an explicit product requirement and privacy review.
- Preserve cooldown, adjacent-airball handling, rim calibration, corrections, recording, and demo mode when changing camera code.
- Keep live camera and imported-video analysis on the same detector, motion tracker, confidence threshold, and shot state machine. Never auto-count a low-confidence decision; route it to review.
- Keep player motion separate from orange-ball detection. Reject camera-wide movement, preserve short occlusions, and never use the player's clothing or skin color as a basketball signal.
- Imported-video analysis assumes a fixed camera and a manually calibrated rim. Use actual decoded-frame timestamps, skip repeated frames, gate unreliable timing or camera movement to manual review, keep processing on-device, and release sampled native images promptly.
- Keep localhost compatibility isolated in `.web.ts`/`.web.tsx` adapters. Never weaken or replace the native VisionCamera/OpenCV implementation to make web bundling easier.
- Label the localhost camera simulation clearly. Browser demo events must never be presented as live computer-vision detections.
- Add pure tests for tracking-state changes and run `npm run typecheck`, `npm test`, and `npm run doctor` before handoff.
- iOS configuration is authoritative in `app.json`; release profiles are in `eas.json`. Replace the placeholder bundle identifier before submission.
