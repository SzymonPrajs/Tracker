# Validation and adversarial audit

## Evidence ladder

Use these labels consistently:

1. **specified** — stated by a pinned primary source;
2. **derived** — arithmetic from specified values;
3. **host-smoke** — local code path executes on synthetic or bounded data;
4. **host-validated** — declared real-data tests and metrics pass;
5. **quantized-host-validated** — ESP-PPQ simulated INT8 passes;
6. **board-unit-validated** — embedded gold tensors and isolated hardware stages pass;
7. **board-end-to-end-validated** — sustained physical camera-to-centroid tests pass.

Never promote evidence across these levels without the corresponding test.

## Stage acceptance matrix

| Area | Required adversarial checks |
|---|---|
| downloads | URL/version/license/checksum/archive-type verification; safe extraction; resumability |
| annotations | native-count reconciliation, coordinate convention, orientation, clipping, zero area, duplicate IDs, ignore flags |
| splits | immutable publisher split, exact hash, perceptual hash, embedding-neighbour review; internal video/movie/camera/location grouping |
| negatives | exact MID/class coverage and thresholds, human audit, hard-negative review, no inference from missing labels; NightOwls background is not `no_human` |
| targets | conformance fixtures plus sealed audit gold; semantic-specific empty/single/crowded/edge/truncated/overlap cases; same-cell collision report |
| augmentation | identity bypass, monotonicity, sector Jacobian and corner-ray seam tests, no crop, round trip, label alignment, replay, mixture frequencies |
| training | one-batch, tiny overfit, finite loss, resume equivalence, deterministic replay, no test leakage |
| search | fixed budget, validation-only selection, Pareto report, multiple finalist seeds |
| export | supported static operators, shape inference, PyTorch/ONNX parity, separated output semantics |
| quantization | representative training-only calibration, separate stress pack, exact preprocessing, PTQ/AutoQuant/TQT/QAT ablations, task metrics, saturation, multiple gold tensors |
| firmware | format negotiation, buffer alignment/ownership, reference differential tests, build-profile independence |
| board | sustained capture, stage latency, end-to-end latency, FPS, drops, corruption, heaps, PSRAM contention, accuracy |

## Model metrics

Report at minimum:

- head AP/recall and centroid error;
- normalized centroid error by head width or diagonal;
- visible-head and full-head metrics separately;
- face/person auxiliary metrics separately;
- small/medium/large head strata at actual model resolution;
- indoor/outdoor, ordinary room, crowd, occlusion, edge/truncation;
- daylight, low light, severe noise;
- flat, native fisheye, and synthetic fisheye;
- false positives per certified-negative image and per target-camera hour;
- temporal jitter, loss-of-track, and incorrect reacquisition on video;
- float versus quantized deltas at validation-selected candidate thresholds,
  threshold-free curves, and a common-threshold calibration diagnostic.

Keep one hand-checked gold set out of parser development, augmentation tuning,
architecture search, QAT, calibration, and threshold selection.

## Hardware metrics

For each camera/color/input candidate, record:

- sensor opportunity count and captured/completed/dropped/corrupt frames;
- integrated CSI+ISP frame-completion throughput/latency, observable PPA,
  input-map, inference, decode, and end-to-end p50/p95/max; isolated ISP latency
  only when a documented standalone ISP DMA harness measures it;
- current-frame age and discarded stale work;
- full and small buffer addresses, sizes, alignment, queue depth, and ownership;
- free/minimum/largest internal and PSRAM blocks after each allocation;
- model parameters, activation memory, static-planner placement, and load time;
- PSRAM traffic/contention and effects of two versus three capture buffers,
  labelling every traffic value as arithmetic estimate, hardware-counter
  measurement, or contention proxy;
- sustained results for at least a declared 60-second stress run, followed by a
  longer soak before production claims.

## Final acceptance contract

Freeze budgets at the boundary where they become decision criteria:

- before dataset work: target scenes, minimum head size at model input,
  recall/localization and negative false-positive objectives, hard model/memory
  ceilings, p95 camera-to-centroid latency, and sustained FPS target;
- before quantization experiments: maximum allowed float→INT8 degradation;
- before final physical validation: maximum dropped/corrupt frames, heap drift,
  and sustained/soak-run limits.

Those values are product decisions and are intentionally not invented in this
planning pass. No value may be created or relaxed after observing the
evaluation it governs. Twenty to twenty-five FPS remains a target until the
physical system meets the complete contract.

Float baselines, architecture search, quantization-method selection, threshold
selection, and ablations use validation data only. The final test remains
sealed until the quantization recipe, model, preprocessing, and
candidate-specific threshold are frozen. Then evaluate the selected INT8 model
and its matched float parent together exactly once.

## Plan-order audit

The authoritative dependency order is the stage map in the root README. Any
supporting-document change that conflicts with it is invalid and must update
the controlling plan rather than creating a second order here.
