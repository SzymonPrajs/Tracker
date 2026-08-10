# Parameterized PyTorch training

## One engine, static exported profiles

Build one training codepath with a validated typed configuration. It may create
an RGB model (`in_channels=3`) or luminance model (`in_channels=1`), but each
export is static. Do not put a runtime color-mode branch inside the deployed
graph.

The input contract must name the exact bytes and conversion, for example:

- `rgb8`: channel order, full/limited range, integer-to-float mapping;
- `gray8`: direct GRAY8 range and mapping;
- `y_from_yuv420`: Y-plane layout and range;
- `rgb_to_y_bt601`: exact integer coefficients matching firmware.

Training grayscale from an arbitrary library conversion is invalid if firmware
uses different coefficients, range, clipping, or rounding. Python and firmware
must share byte-for-byte golden preprocessing vectors.

## Resolved run configuration

Each run records:

```text
data:
  manifest hashes, split hashes, source/scale mixture, negative policy
input:
  mode, channels, width, height, byte range, conversion, normalization
targets:
  enabled tasks, output stride, Gaussian/radius rule, ignore policy
augmentation:
  exact probabilities, parameter ranges, seed policy
model:
  topology, width/depth, activation, output branches
optimization:
  optimizer, LR schedule, epochs, batch/accumulation, AMP, clipping
loss:
  heatmap, offset, size, auxiliary, ignore weights
runtime:
  device, workers, deterministic/debug flags, evaluation cadence
deployment:
  intended ESP-DL/ESP-PPQ versions and operator contract
```

Write the resolved configuration, code revision, environment, dataset hashes,
checkpoint hashes, history, and metrics into each run directory. Command-line
overrides must appear in the resolved file; there are no hidden defaults.

## Loop requirements

- source- and scale-balanced sampler with explicit negative fraction;
- deterministic validation/test order;
- resumable model, optimizer, scheduler, scaler, epoch, sampler, and RNG state;
- best and last checkpoints with declared selection metric;
- mixed precision only where verified for the training device;
- gradient clipping and non-finite loss/gradient checks;
- throughput timing separated into load, augment, transfer, forward, backward,
  and evaluation;
- per-stratum metrics, not only aggregate loss;
- no final-test access during model or hyperparameter selection.

## Build gate before real runs

1. Validate the configuration and reject unknown/incompatible fields.
2. Run one forward/backward batch on CPU and the selected accelerator.
3. Overfit a tiny real subset, including a class-certified negative.
4. Prove deterministic sample and augmentation replay.
5. Prove uninterrupted versus checkpoint/resume equivalence within tolerance.
6. Prove preprocessing and target golden cases.
7. Confirm every planned model operator has a supported static ESP-DL form
   before an expensive search.

## Float baseline stage

The controlled comparison begins from the same decoded source sample and uses
the same:

- source and split hashes;
- sampler order;
- seeds;
- geometric transform, exposure, and photon/read-noise realization;
- post-input architecture and output contract;
- optimizer budget;
- evaluation threshold selection procedure.

The pipeline forks only at the declared input conversion: one branch creates
the exact RGB bytes and the other applies the exact firmware luminance
conversion to the same augmented scene. Only the first convolution's channel
count differs, and its parameter/MAC difference is reported. The initial
baseline is one declared head semantic with head heatmap, offset, and size;
face, person, mask, and pose heads enter later only as explicit optimization
ablations.

Compare head recall/AP, centroid error, small/distant heads, face/body auxiliary
effects, certified-negative false positives, low-light, flat/native/synthetic fisheye,
MACs, model bytes, peak activation estimates, and later board preprocessing
cost. Operating thresholds are selected per candidate on validation under the
same policy, such as maximum recall subject to a fixed certified-negative
false-positive limit. Report threshold-free curves and a common-threshold
diagnostic as well. Luminance wins only if end-to-end evidence wins; a smaller input tensor
does not automatically remove the camera capture buffer.

After the controlled comparison, separately optimize RGB and luminance under
the same measured board latency and peak-memory envelope. This resource-matched
comparison, rather than nominal parameter equality, makes the deployment
decision.

## Optimization stage

Start search only after baseline and data gates pass. Search a bounded space:

- camera-compatible input shape;
- output stride and collision behavior;
- model width/depth and supported block types;
- optimizer and schedule;
- source/negative mixture;
- loss weights and heatmap radius;
- bounded augmentation probabilities/severity.

Rank a Pareto frontier rather than one scalar: task accuracy and false positives,
stress robustness, MACs, parameter bytes, and peak activations. Select a small
finalist set, freeze every configuration, retrain from scratch with multiple
seeds, and report validation variation. Do not open the final test set. It is
evaluated only after the quantization recipe, model, preprocessing, and
candidate-specific threshold are frozen, when the matched float parent and
selected INT8 model are tested together once.

Keep decoder thresholding, peak selection, and box/centroid post-processing out
of the model graph where practical. Separate output branches with different
numeric ranges unless quantization evidence supports concatenation.
