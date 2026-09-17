# V22 validation — 2026-09-07

## What changed

- Crop the original decoded source pixels, not an already reduced 640-pixel canvas. Map crop detections back to the common scoring coordinates.
- Run full-frame inference on every uploaded-video sample alongside focused inference. This trades processing speed for approach coverage.
- Prefer a recently observed moving ball over a stale below-net position during geometric association.
- Permit weak exit detections only when continuing an armed, spatially plausible ball track.
- Reject implausible reused track IDs and invalid or repeated input timestamps.
- Measure gaps between consecutive observations when assessing rim-rattle uncertainty, not the entire time spent around the rim.
- Preserve small manual rim boxes and clearly label automatic boxes as candidates requiring verification.

## Evidence and limits

108 pure tests passed, including the existing three-make detection fixture and new cases for scaled/mirrored geometry, crop-coordinate transforms, weak exits, stale-track competition, continuous rim rattles, concurrent shots, and invalid timestamps. Synthetic coordinate transforms are not proof of real-world multi-angle accuracy.

Seven basketball-named recordings in the supplied Captures folder were decoded locally with OpenCV and the application's Attalla ONNX weights. The resulting observations were replayed through the production TypeScript ByteTrack adapter and detector-crossing state machine. No recordings were uploaded. Local diagnostic outputs are ignored under `work/recording-audit/`.

This audit compares full-frame-only inference against full-frame plus original-pixel crops. It is **not** the complete browser pipeline: decoding/timestamp handling, dynamic rim tracking, predicted-ball crops, motion recovery, and semantic/trajectory evidence fusion differ or are omitted. Counts below are diagnostic detector outputs, not independently verified ground truth.

| Local clip | Full-frame makes/misses/review | With original-pixel crops |
| --- | --- | --- |
| Outdoor free throws, Aug 30 | 3 / 0 / 0 | 3 / 0 / 0 |
| HOOPERS short, Aug 23 | 0 / 0 / 0 | 3 / 0 / 0 |
| Guinness rapid drill, Aug 15 | 7 / 1 / 0 | 12 / 1 / 0 |
| Street clothes, 19:44:35 | 1 / 0 / 0 | 1 / 0 / 0 |
| Street clothes, 19:44:53 | 1 / 0 / 0 | 1 / 0 / 0 |
| Camera shooting clip, Aug 13 | 3 / 0 / 0 | 3 / 0 / 0 |
| Curry screen recording, Aug 11 | 0 / 2 / 1 | 5 / 3 / 1 |

The crop initially lost the 19:44:53 clip's full-frame event. Diagnostics showed a stale below-net track stealing the new exit; recency-aware association restored the event, and a pure regression test now covers that failure.

The Curry recording's automatically inferred box visibly includes the red support arm. Its table counts are unreliable and must not be marketed as an accuracy result. Correct rim calibration and manually labeled event timestamps are needed before judging that clip. Edited/accelerated recordings and balls hidden by racks or nets remain difficult cases.

The browser still seeks a video element and uses `currentTime` for its timing checks. That value does not prove unique decoded-frame presentation timestamps; the UI no longer claims verified unique frames. A future exact-frame decoder evaluation is needed before claiming source-frame coverage, especially for variable frame rates or videos above 30 FPS.

## Reproduce the optional local detector audit

Install OpenCV, NumPy, and ONNX Runtime in a separate Python environment. Run from the repository root:

```text
python scripts/evaluate-recordings.py "C:\Users\Owner\OneDrive\Videos\Captures"
npx tsx scripts/replay-recordings.ts
```

The extractor caches each completed clip; use a new `--output` directory to force fresh inference. The replay recalculates classifications from cached detections after state-machine changes. Do not modify expected counts to fit outputs.

## Repository review

The existing detector and scoring implementation remain credited in `THIRD_PARTY_NOTICES.md`. V22 does not claim to embed SwishAI's complete stack. Its shooting and ball-in-basket classes require its corresponding trained model; transplanting those class checks into a two-class model would not work. Its cooldowns also do not establish universal rapid-shot or multi-angle accuracy. Avishah3's above/below-rim geometry is useful context, but its fixed pixel margins are not an angle-independent guarantee.

- https://github.com/sPappalard/SwishAI/blob/master/BE/app.py
- https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker/blob/master/utils.py
- https://docs.ultralytics.com/modes/track/

This release improves specific failure modes; it is not a perfected all-angle counter or a validated native-camera release.
