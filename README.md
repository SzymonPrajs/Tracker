# Tracker rebuild plan

This is the single controlling plan for rebuilding Tracker from scratch. The
active repository contains planning documents only. The old implementation and
old numbered documentation are preserved under `previous/`; they are evidence
and reference material, not the new baseline.

The intended result is a small, fast ESP32-P4 system that receives frames from
an OV5647 Raspberry-Pi-style MIPI-CSI camera, locates heads reliably across
ordinary, dark, noisy, crowded, distant, flat, and fisheye-like scenes, and
returns the position needed by a controller. Nothing is called fast, accurate,
or production-ready until it passes the physical-board gates in Stage 11.

## Non-negotiable rules

1. Complete one stage and its gate before beginning the next stage.
2. Keep head, face, visible body, full body, pose, and negative labels distinct.
3. Never infer that an image is negative merely because an annotation is absent.
4. Preserve source, license, checksum, annotation meaning, and split group for
   every image.
5. Geometric augmentation transforms image and vector labels together; targets
   are generated afterwards.
6. Fisheye augmentation keeps the complete canvas. It does not zoom or replace
   the scene with a crop.
7. RGB versus luminance is an experiment. Training supports both; deployment
   selects one only after accuracy and board-cost measurements.
8. Quantization is ESP32-P4-specific INT8, not generic “INT8-compatible” export.
9. Start with clear portable code. Optimize only measured hot paths, and keep a
   tested reference implementation for every optimized kernel.
10. Debug, profile, and release behavior are compile-time profiles. Release hot
    paths do not carry runtime debug branches.
11. A host build, export, simulated INT8 run, or firmware build is not a board
    FPS or accuracy result.

## Fixed boundary and open decisions

Fixed now:

- Board family: Waveshare `ESP32-P4-Module-DEV-KIT`, planned conservatively as
  the 360 MHz `ESP32-P4NRW32` product with 32 MB PSRAM and 16 MB flash.
- Camera assumption: OV5647 MIPI-CSI module using the supported RAW modes. The
  physical module, cable, sensor ID, and chip revision still require inspection.
- Training framework: PyTorch with static-shape export to ESP-DL `.espdl`.
- Deployment batch: one.
- Primary deployment target: head localization; face, body, mask, and pose data
  may provide auxiliary supervision but do not redefine a face box as a head.

Deliberately open:

- RGB or one-channel luminance model.
- Camera mode and model input dimensions.
- Output stride and final model topology.
- PTQ, AutoQuant-selected PTQ, TQT refinement, or QAT as the selected INT8
  result.
- Exact capture/model buffer count.
- Whether any custom assembly is justified.

## Entry gate

Before work starts, record a provisional product envelope: target scene types,
minimum required head size at model input, false-positive objective, hard model
and memory ceilings, and latency/FPS target. Freeze the allowed float-to-INT8
degradation before quantization runs and the sustained-run limits before final
physical validation. A limit cannot be invented or relaxed after seeing the
evaluation it governs.

## Stage map

| Stage | Work | Required output | Gate |
|---:|---|---|---|
| 1 | Dataset acquisition | source/rights manifests, verified archives, native annotations, class-specific negatives | admitted archives and annotations reconcile with their sources |
| 2 | Canonical corpus and targets | semantic-preserving labels, immutable source splits, leak-free internal splits, target generator, shards | conformance fixtures and sealed audit cases pass |
| 3 | Augmentation | full-canvas warp, physically ordered provisional low-light/noise model, exact label transforms | identity, no-crop, geometry, representability, and distribution tests pass |
| 4 | Parameterized PyTorch engine | one typed training system producing static C1 and C3 models | forward/backward, overfit, resume, and replay tests pass |
| 5 | Hardware input characterization | disposable board harness, proven camera/ISP/PPA formats and shapes, measured input bytes | at least one C1 and one C3 candidate input contract are reproducible |
| 6 | Float baseline runs | controlled RGB/luminance results using hardware-exact candidate contracts | comparable validation reports and reproducible checkpoints exist |
| 7 | Architecture and hyperparameter optimization | bounded search, multi-seed Pareto finalists, frozen float parents | finalists satisfy declared validation, size, and estimated-cost budgets |
| 8 | Quantization pipeline | pinned P4 export, representative calibration, PTQ artifact, parity harness | valid `.espdl` and simulated INT8 evaluation pass |
| 9 | Quantization-oriented runs | separate PTQ, AutoQuant, TQT, and QAT comparisons | recipe, preprocessing, model, and operating threshold are frozen |
| 10 | Firmware integration | camera-to-centroid firmware with build profiles and telemetry | golden tensors match and every observable stage is measurable |
| 11 | Physical and final validation | one final test evaluation plus sustained board evidence | every predeclared accuracy, memory, latency, FPS, and robustness limit passes |

## Stage 1 — acquire the dataset

The first implementation task is data acquisition. Download annotations first,
record access terms, pixel and annotation rights, checksums, and then download
pixels source by source. Primary head-detection research candidates are
CrowdHuman, RPEE-Heads, JRDB-Pose, and SCUT-HEAD. Open Images is conditional on
per-image licence allowlisting and exact class-coverage parsing. COCO is
auxiliary. JHU-CROWD++, WIDER FACE, DARK FACE, CrowdPose, OCHuman, FishEye8K,
and every other unresolved source remain gated until their precise role and
pixel rights are recorded. Research-only inputs do not form a commercial data
lineage.

Negatives are class-specific evidence, never a generic empty-annotation claim.
Open Images negatives name the exact class MID and coverage status. NightOwls
may support `pedestrian_above_50px_verified_negative`, not `no_human`. Only a
complete manual check against the frozen human ontology may create
`no_human_verified`. Target-camera negatives are captured and certified during
the hardware-characterization stage, then admitted before float training.

Deliverables and the complete source portfolio are in [Data](docs/data.md).

## Stage 2 — canonical labels, splits, and targets

Convert native annotations into a provenance-preserving vector schema. Keep
`head_full`, `head_visible`, `head_point`, `face_visible`, `person_full`,
`person_visible`, native masks, and pose separate, with geometry quality
recorded as exact, approximate, derived, or weak. Face/body labels never create
head positives; a pose record may create only a declared weak head-centre target
when a sufficient visible-head-keypoint rule passes.

Preserve the publisher's split as immutable `source_split`; never promote its
validation/test images into training. Within publisher training data, group by
sequence, movie, camera, location, source URL, or duplicate cluster before any
internal split. Generate centre heatmaps, offsets, sizes, per-semantic validity
masks, and optional auxiliary targets only after final run geometry. A box can
create a box proxy, but never a claimed ground-truth segmentation mask.

The stage ends only after counts reconcile, cross-split duplicates are absent,
and empty/single/crowded/edge/occluded gold examples match both numerically and
visually.

## Stage 3 — build augmentation

Augmentation is a separately testable system. It contains:

- an exact identity/clean branch;
- flat images with photometric variation only;
- mild and medium full-canvas radial warps;
- a bounded strong fisheye-like branch;
- exposure loss, tone response, white balance, vignetting, shot noise, read noise,
  mild blur, and optional fixed-pattern noise.

There is no random resized crop and no telephoto zoom substitute. Head size may
balance small-object sampling or condition a declared nonphysical curriculum;
it does not become a false metric-depth or per-person lens model. Native
fisheye and target-camera images remain available as held-out evidence.

The exact warp and tests are in
[Targets and augmentation](docs/targets-and-augmentation.md).

## Stage 4 — build the parameterized training engine

Build one typed, validated PyTorch training system. It creates separate static
models for one-channel and three-channel inputs; there is no runtime channel
branch in the deployed graph. A resolved run configuration records dataset and
split hashes, input conversion, target definition, architecture, optimizer,
schedule, augmentation, seed, checkpoints, evaluation cadence, and future
quantization target.

Before real training, require a one-batch forward/backward check, a tiny-subset
overfit, deterministic augmentation replay, checkpoint/resume equivalence, and
CPU/accelerator agreement within tolerance.

See [Training](docs/training.md).

## Stage 5 — characterize the hardware input contract

Use a disposable, bounded board harness—not production firmware—to identify the
delivered module, cable, sensor ID, and ESP32-P4 revision; enumerate supported
OV5647 modes; capture representative RAW/ISP frames and certified empty-scene
negatives; prove candidate RAW10→YUV420 and RGB paths; measure CSI+ISP frame
completion; and benchmark exact PPA full-frame output shapes on its 1/16 scale
grid. Record byte layout, stride, range, colour matrix, buffering, image quality,
  and minimum useful head size. Preserve a non-GRAY build until silicon revision
support is known.

This stage freezes candidate hardware-exact C1 and C3 input contracts for
training, fits the provisional augmentation ranges from captured frames, and
admits the target-camera negatives. It does not implement the final
camera-to-model stack.

## Stage 6 — run controlled float baselines

The RGB/luminance comparison begins from the same decoded sample, sampler order,
split, seed, geometric transform, exposure, and photon/read-noise realization.
Only at the declared conversion boundary does it fork into exact RGB bytes or
the exact firmware luminance conversion. The post-input architecture, targets,
optimizer, and schedule remain the same; only the first convolution's input
channels differ, and that parameter/MAC difference is reported.

Compare validation AP/precision-recall, distant-head recall, candidate-specific
thresholds chosen under one certified-negative false-positive policy, stress
strata, model work, activation memory, and predicted board input cost. Then run
a second resource-matched comparison under the same measured latency and peak
memory envelope. The final test remains sealed.

## Stage 7 — optimize architecture and training

Run a bounded architecture and hyperparameter search on validation data. Freeze
a small Pareto set and retrain each finalist from scratch with multiple declared
seeds. Auxiliary face/person/pose outputs are optional ablations, not mandatory
baseline complexity. Freeze the float parents without opening the final test.

## Stage 8 — build the P4 quantization pipeline

Pin current compatible ESP-DL and ESP-PPQ versions, confirm operator support,
export static batch-one ONNX, and prove PyTorch/ONNX parity. Freeze exact
camera-byte preprocessing from the completed hardware input contract. Build a
deterministic training-only calibration manifest sampled in deployment-
representative proportions. Keep a separate adversarial quantization pack for
rare, dark, noisy, warped, crowded, distant, edge, and negative cases.

The first required result is a plain P4 INT8 PTQ baseline using symmetric
power-of-two quantization, current P4 per-channel Conv/Gemm rules, per-tensor
rules elsewhere, and round-half-even behavior. Export `.espdl`, `.info`, `.json`,
hashes, metadata, and several explicit golden inputs.

See [Quantization](docs/quantization.md).

## Stage 9 — run quantization-oriented experiments

Compare separate candidates:

1. fixed default all-INT8 PTQ;
2. AutoQuant-selected all-INT8 PTQ, with evaluation on quantization-validation
   data rather than calibration or final-test data;
3. PTQ initialized with TQT threshold/weight refinement;
4. QAT warm-started from the reproducible frozen float checkpoint;
5. explicitly named combinations only when standalone results justify them.

Do not stack methods without an ablation and do not select by layer SNR or file
size alone. Inspect export dispatch and reject unintended INT16/float fallbacks.
Choose each candidate's operating threshold on validation under the same policy,
report threshold-free curves plus a common-threshold diagnostic, then freeze the
model, recipe, preprocessing, and threshold. Only after that freeze may the
selected INT8 model and its matched float parent be evaluated on final test,
together and once.

## Stage 10 — integrate the smallest clear firmware

Use the camera and preprocessing contract already proved by the bounded
characterization harness. The candidate low-copy luminance path is OV5647 RAW10
→ ISP YUV420 → PPA full-frame scale to a
small GRAY8 buffer → exact unsigned-to-signed input conversion. It avoids a
full RGB888 framebuffer but is not accepted until chip revision, format
negotiation, range, image correctness, and timing pass. The comparable RGB path
must use the same capture contract where possible.

PPA scaling is hardware-accelerated but not in-place, so “downscale directly in
the buffer" means writing a separate small output buffer. ISP ROI cropping is
optional and disabled initially. The initial PPA source block is the complete
negotiated frame; block selection may become a separate compile-time geometry
choice only after its field-of-view effect is measured. Camera mode, input
shape, channel count, buffer count, and diagnostics are compile-time profiles.

Start in portable C/C++ and use ESP-DL's optimized operators. Tiny measured leaf
sequences may use guarded inline assembly; substantial loops or reusable kernels
belong in separate `.S` files. Every optimized path retains a reference-C
differential test.

See [Deployment](docs/deployment.md).

## Stage 11 — physical and final validation

Validate the full chain on the delivered board: sensor → CSI → ISP → PPA → input
mapping → ESP-DL → decoder. Measure camera modes, frame corruption, dropped
frames, CSI+ISP frame-completion latency and throughput, p50/p95 observable-stage
and end-to-end latency, PSRAM traffic with evidence type, peak heaps, largest
free block, model memory, output parity, false positives, localization error,
and sustained camera-to-centroid FPS. Isolated ISP latency is reported only from
a documented standalone DMA harness and never presented as a decomposition of
the live streaming path.

The acceptance matrix is in [Validation](docs/validation.md). Until it passes,
20–25 fps remains a target rather than a result.

## Document map

- [Data](docs/data.md): portfolio, licensing, negatives, canonical schema, and acquisition gates.
- [Targets and augmentation](docs/targets-and-augmentation.md): heatmaps, masks, radial warp, noise, and property tests.
- [Training](docs/training.md): configuration, RGB/luminance experiments, loop, and optimization runs.
- [Quantization](docs/quantization.md): P4 INT8 semantics, PTQ/TQT/QAT ladder, artifacts, and parity.
- [Deployment](docs/deployment.md): OV5647/ISP/PPA path, buffers, compile profiles, and assembly policy.
- [Validation](docs/validation.md): adversarial audit and stage acceptance matrix.
- [Sources](docs/sources.md): current primary sources and verification boundary.

## Historical material

The complete prior implementation is in [previous](previous/README.md). The old
numbered research documents are in `previous/documentation-archive/`. They may
inform a stage, but no old source file, model, dataset result, or document number
is automatically carried into the rebuild.
