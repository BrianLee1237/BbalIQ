# V23: repository review and bounded trajectory integration

Reviewed 2026-09-09. Changes preserve the existing local Expo interface, detector
weights, counters, confidence threshold, correction workflow and dependencies.

## HanaFEKI/AI_BasketBall_Analysis_v1

Revision: `791365cefdda25455df14f2168c642db4bd709cc`.

Relevant inspected sources: `trackers/ball_tracker.py`, `tactical_view/homography.py`,
`tactical_view/tactical_view.py`, `Court_keypoint_detection/court_keypoint_detection.py`,
`main.py`, model configuration and speed/distance code.

- Court keypoints and RANSAC homography map **player foot positions on the floor**
  into court coordinates. This is useful for tactical maps, not a direct 3D
  reconstruction of an airborne shot. Using that floor transform on the ball
  would produce misleading spatial information.
- Ball cleaning rejects large displacements using a fixed pixels-per-frame
  limit. Its ball tracker selects the highest-confidence ball per frame rather
  than maintaining multiple ball identities. Wholesale replacement would lose
  our rack/multi-ball protections.
- Interpolation/backfilling can fill missing detections across long gaps. That
  is not evidence a ball went through the net, so it is not used for counting.
- Reprojection validation and outlier rejection are useful design principles.
  V23 applies analogous residual/consensus checks to short temporal ball fits,
  not the repo's court transform.

Source: https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1/tree/791365cefdda25455df14f2168c642db4bd709cc

## nvan21/Basketball-Shot-Detection

Revision: `2127f6d94ecde39f66ed45bc24a979244b497bc9`.

Inspected `basketball_detection.py`, `main.py`, `utils.py`, data configuration and README.

- Computes velocity from recent positions and draws a simulated future arc;
  also includes a quadratic y=f(x) fit. The README explicitly says it does not
  confirm whether a shot went in, and describes 2D false predictions.
- The main loop hard-codes 60 FPS. The backboard extractor assumes white markings,
  width/height ranges and fixed rim offsets. Its own TODO calls out angle dependence.
- We retain temporal regression as an idea, but independently fit **x(t), y(t)**
  using timestamps. This avoids the singularity of a near-vertical y=f(x) path
  and does not assume known gravity in image pixels or a front-facing backboard.

Source: https://github.com/nvan21/Basketball-Shot-Detection/tree/2127f6d94ecde39f66ed45bc24a979244b497bc9

No source/model from either repository was copied into the application; no root
code license was found in the reviewed trees. No new package was installed.

## Implemented

`src/tracking/flightPrediction.ts` fits a short local quadratic trajectory using
up to ten actual detections. It requires at least five ordered samples, checks
fit residuals in ball-size units per axis, permits at most one older outlier,
and rejects recent discontinuities, ID changes, large size shifts and long gaps.
Prediction is limited to 150 ms and 60% of the observed time span.

Integration points:

1. Shared detector tracker: curved association is a fallback after all eligible
   existing associations. Both browser live and upload use this shared tracker.
2. Shared vision selection: retain actual observation history; preserve any
   successful existing selection; only reacquire an unambiguous real candidate.
3. Upload crop: retain the complete old linear crop and extend it when the
   supported curved path would fall outside. Never replace it with only an arc.
4. Track-guided pixel recovery: allow a well-supported curved path while retaining
   motion, appearance and size checks. History moves with calibrated rim shifts.

No fitted point is sent to a shot state machine as a detection. Make/miss geometry
and confidence thresholds are unchanged. Long unseen net passages remain uncertain.

## Validation and remaining limitations

- 120 pure tests pass, including the previously working three-make fixture.
- New cases cover vertical/mirrored/local perspective projections, independent
  axis scaling, one outlier, ID switches, impacts, timestamp errors, ambiguity,
  detector recovery and preservation of existing selections/crop coverage.
- Replayed 3,699 cached frames across seven local videos. Detector-crossing results
  match V22. The guide comparison also preserves every accepted observation and
  trajectory event in those cached inputs. It forms supported predictions but
  records **zero additional recoveries** there: this is non-regression evidence,
  **not a measured real-video accuracy gain**.
- These replays do not rerun source video decoding, fresh model inference,
  enlarged crops, pixel recovery, or the complete browser evidence-fusion loop.
- This is a local image-space approximation, not a true perspective calibration,
  court map or 3D shot model. It cannot resolve front/behind-hoop ambiguity from
  missing visual evidence. No new all-angle accuracy claim is warranted.
- Final validation on 2026-09-12: typecheck, lint, web export and diff whitespace
  checks pass. Expo doctor passes 19/20 checks and flags 14 existing package
  patch-version mismatches; dependencies are intentionally not upgraded here.

Final safety review tightened curved-only association: known conflicting ball
IDs are rejected, and recovery must have a mutually unique observation/track
match. Tests cover both two balls competing for one observation and two
observations competing for one ball. Existing successful association remains
the first choice. The seven cached detector replays retain their V22 counts
after these safeguards.

### Fresh browser check (2026-09-12)

Opened the local `/analyze-video` route and confirmed the `V23 BETA` label.
Selected `How To Shoot A Basketball Better With A Camera - YouTube - Google
Chrome 2026-08-13 18-39-08.mp4` from the user's Captures directory through the
normal file chooser. Visually verified the automatic rim candidate and ran
Analyze video without editing the counters or calibration.

The complete browser pipeline returned **3 makes, 0 misses, 0 review events**
at approximately 7, 12 and 19 seconds, matching the previously confirmed
three-make clip. Diagnostics: 658 requested video samples at a 30 FPS target,
100% reported hoop lock, zero camera relocks, 162 tracked-ball frames, and
WebGPU detector observations in 644 frames. No browser error logs were captured.
Processing took several minutes for the 22-second clip on this machine.

This adds one fresh end-to-end regression check; it does not establish all-angle
accuracy or exact source-frame coverage. No counters were manually corrected.

Reproduce cached checks with `npx tsx scripts/replay-recordings.ts` and
`npx tsx scripts/evaluate-flight-guide.ts`. Output remains local under ignored
`work/recording-audit/`. Better real-world angle coverage still requires labeled
recordings and full inference evaluation, not relaxed counters or invented paths.
