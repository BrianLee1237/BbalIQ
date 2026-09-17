# ActiveTrack Native

ActiveTrack is an Expo SDK 57 basketball camera app that counts makes and misses in real time. Camera frames stay on the device and are processed by VisionCamera 5, Skia, GPU resizing, worklets, and Fast OpenCV. The live view tracks the ball's predicted trajectory and the dominant moving player independently, so player appearance is never used as a basketball signal.

The browser ActiveTrack camera and recorded-video route use a basketball/hoop YOLOv8n detector with Joseph Attalla's multi-ball/multi-hoop shot pipeline. The video route accepts a device-library recording up to five minutes long, automatically selects a clear hoop frame, and requests a dense 30 FPS pass while rejecting repeated decoded timestamps. V21 keeps periodic full-frame, rim-scale, and predicted-ball inference; scores against the approved calibrated rim plane rather than a variable detector box; preserves ByteTrack identities when another ball is already in flight; explicitly samples both video boundaries; safely arms a shot already descending at clip start; and lets track-guided motion bridge a learned-detector gap or direction reversal inside the rim-contact zone. A narrow near-rim miss remains deferred while a possible rattling make settles. Frame-level one-to-one association still isolates the airborne shot from a stationary ball rack. Disagreement and low-confidence events remain uncounted for review.

V22 adds original-resolution video crops before detector resizing, full-frame evidence on every upload sample, fresh-track priority over stale below-net observations, guarded recovery of weak net exits, tracker-ID teleport rejection, and duplicate/out-of-order timestamp guards. Continuously observed rim bounces remain eligible while long unseen crossings go to review. Small manually drawn rims are preserved rather than silently widened. The calibration screen explicitly asks users to verify the opening rather than the support arm or net.

For best tracking, keep the camera fixed and the rim and ball clearly visible, avoid hard cuts or extreme zoom changes, and use even court lighting. No vision system can promise zero mistakes on arbitrary footage; this is a beta and its thresholds still need validation against a larger labeled-video set. See [V22 validation notes](docs/v22-validation.md) for actual checks and known limitations.

## Run locally

V23 adds a conservative temporal flight guide after reviewing HanaFEKI and nvan21.
It fits observed x/y against timestamps, with short-horizon and outlier guards,
to help search/reacquire a ball near a curved path. Existing accepted associations
take priority; predictions never become detections or automatic shot results.
This is not 3D reconstruction or an all-angle accuracy guarantee. See the
[repository review and validation](docs/v23-repository-review.md).

For the latest local video analyzer, run `npm run web` and open `http://localhost:8081/analyze-video`. The browser supports real camera analysis on secure origins, a separately labeled simulation, history, manual corrections, automatic hoop lock, and local recorded-video analysis. Camera permission and model support vary by browser. V23 is validated locally; the older deployed site has not been updated by this change.

The real-time camera pipeline uses native frame processors and requires an Expo development build. Expo Go cannot load the tracking modules.

1. Install dependencies with `npm install`.
2. Generate native projects with `npx expo prebuild`.
3. On macOS, build the iOS development client with `npx expo run:ios --device`; from Windows, use an EAS development build.
4. Start Metro with `npx expo start --dev-client`.

## Validate

- `npm run typecheck`
- `npm run lint`
- `npm test`
- `npm run doctor`

Before App Store submission, replace the placeholder bundle identifier in `app.json` and connect the project to the correct Apple Developer account with EAS.

## Model and tracking credits

The browser analyzer uses the MIT-licensed `josephattalla/Basketball-Shot-Detection` model and algorithm, the MIT-licensed `byte-track-ts` ball/hoop association tracker, and ONNX Runtime Web. Full attribution, hashes, and license copies are in `THIRD_PARTY_NOTICES.md` and `LICENSES/`.
