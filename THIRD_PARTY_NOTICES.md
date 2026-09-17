# Third-party notices

ActiveTrack's browser video analyzer includes the following third-party components:

## Joseph Attalla basketball shot detector

- Source: https://github.com/josephattalla/Basketball-Shot-Detection
- Revision: `e320817d0f87eceb4093c871de6d29d1adca0006`
- Model: `bball_model.pt`, exported to ONNX for local browser inference
- Author: Joseph Y Attalla
- License: MIT; full text in `LICENSES/attalla-basketball-shot-detection-MIT.txt`
- Original SHA-256: `40F3E596652A427BA290B3F72384E49AED12CAF1A8AE41BEAEF4A8FFFCF09FA3`
- ONNX SHA-256: `5D2E8C0F39EAB69C98D371333ABF5E74F06A2245B5A7889D12352A620EF541A9`

The model detects basketballs and hoops. ActiveTrack ports the repository's
multi-ball/multi-hoop association and above-rim to below-net line-crossing
classifier to strict TypeScript. The port fixes the upstream constructor typo,
vertical-line division failure, frame-rate-dependent track expiry, and duplicate
event handling while retaining the published detector geometry. ActiveTrack V20
adds its own multi-scale inference, boundary-frame sampling, context-sensitive
rim-zone threshold, rim-bounce settling, persistent detector identity, and
per-attempt reset.

## byte-track-ts

- Source: https://github.com/billmyplate/byte-track-ts
- Revision: `ac2439c47175307d559daf423c8a8a425abb67bb`
- License: MIT

## ONNX Runtime Web

- Source: https://github.com/microsoft/onnxruntime
- Version: 1.27.0
- License: MIT
- Distribution: pinned jsDelivr package assets are loaded at runtime

## Systems evaluated as design references

Ultralytics' tracking documentation, the ByteTrack paper/implementation,
OC-SORT, BoT-SORT, and OpenCV optical-flow/Kalman documentation were evaluated
to choose the tracking design. ActiveTrack uses its existing MIT-licensed
TypeScript ByteTrack implementation; it does not bundle Ultralytics, OC-SORT,
BoT-SORT, or Python/OpenCV tracking source.

No source or model weights from the unlicensed `avishah3`,
`AggieSportsAnalytics`, or `srz08` repositories are incorporated. The `chonyy`
project includes a noncommercial OpenPose license and is not incorporated.
`iamyb/shotcut` was reviewed but publishes a Windows binary rather than an
integrable detector implementation.

`sPappalard/SwishAI` is cited as an architectural reference for treating
"ball in basket" as independent semantic evidence. ActiveTrack independently
implements basket-occupancy evidence from its existing detector and calibrated
rim; SwishAI source and weights are not incorporated. SwishAI is AGPLv3.

`HaiderAbasi/OpenCV_Basketball_Shot_Counter` is cited as a design reference for
retaining lower-confidence basketball observations near a known hoop and for
ending an attempt immediately after its crossing is classified. ActiveTrack
independently implements stricter rim-normalized shape and trajectory gates;
that repository's source and weights are not incorporated.

`HanaFEKI/AI_BasketBall_Analysis_v1` (revision
`791365cefdda25455df14f2168c642db4bd709cc`) and
`nvan21/Basketball-Shot-Detection` (revision
`2127f6d94ecde39f66ed45bc24a979244b497bc9`) were reviewed for
motion plausibility, outlier handling, court-plane perspective mapping and
trajectory regression. No root code license was found in either reviewed
tree; their source and weights are not bundled. ActiveTrack's short-horizon
temporal quadratic regression and one-outlier consensus implementation is
independent, uses actual observations only, and does not implement their
court homography or predicted-score classifier. See `docs/v23-repository-review.md`.
