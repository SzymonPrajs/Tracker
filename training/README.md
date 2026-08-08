# Local Mac training and quantization

This directory trains the fixed-shape HC-DS31 head-centroid model on Apple Silicon.
Synthetic scenes are included to verify that data, targets, losses, checkpoints, export,
and evaluation work before real annotations exist. Synthetic success is not evidence of
face accuracy.

## Backend decision

PyTorch on MPS is the deployment-authoritative path:

```text
float32 PyTorch/MPS training
  -> fixed batch-1 ONNX, opset 18
  -> ESP-PPQ calibration and P4 INT8 quantization
  -> target-specific .espdl
  -> physical-board correctness and latency tests
```

MLX-native training is possible and is useful for Apple-Silicon experiments, but MLX
does not provide a supported direct path into ESP-PPQ. The optional MLX implementation
therefore has an isomorphic PyTorch mirror and a strict weight/output parity gate. A
checkpoint that fails that gate must not be exported.

## Mac quick start

The setup script requires Apple Silicon, macOS, and Python 3.11 or 3.12. It deliberately
creates separate environments because ESP-PPQ pins an older ONNX package than a general
training environment may otherwise select.

```sh
make training-setup
make training-check
make training-smoke
make training-export
make training-onnx-check
make training-calibration-smoke
make training-quantize-smoke
make training-quantized-compare
```

`training-smoke` performs real float32 optimization on MPS using deterministic,
procedurally generated multi-head scenes. The final four commands are explicitly a
converter smoke test: their two calibration tensors are not a substitute for a
representative 128--512-frame calibration set or a disjoint held-out test manifest.

To exercise only the MLX/PyTorch graph and checkpoint bridge:

```sh
PYTHONPATH=training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .tools/tracker-train/bin/python -m pytest -q training/tests/test_mlx_bridge.py
```

The complete training environment includes MLX, but the shipped trainer uses PyTorch
MPS so its checkpoint is directly exportable. A future MLX-native optimizer can use the
same model and bridge only if the strict layer-by-layer parity tests remain green.

## Resolution decision

The student network always receives `160 x 288` RGB. Do not train this student at a
larger tensor size and simply reduce it at export: that changes object scale, feature
statistics, receptive-field behavior, compute, and often the graph itself.

Keep original images and annotations at their native resolution. For every training
sample, choose a full-frame view or a source-resolution crop, apply augmentation, then
resize/letterbox to `160 x 288`. Vary crop scale and difficulty, not the network tensor
shape. A larger teacher can be trained separately and distilled into HC-DS31 later.

## Multi-face representation

The physical output is `40 x 72 x 16`:

| Channel | Meaning |
|---:|---|
| 0 | shared heatmap logit containing one Gaussian peak per head |
| 1--2 | x/y offset from the output-cell centre |
| 3--4 | normalized head width/height |
| 5--15 | zero-trained physical padding for the HWC16 target layout |

Multiple Gaussian targets are combined using elementwise maximum, never addition. A
local-maxima decoder can therefore return several faces from one heatmap.

There is one hard representational limit: two centers that fall in the same stride-4
cell cannot both own different offset and size vectors. The target encoder records these
collisions, chooses one regression owner deterministically, and excludes the other
regression target. Collision rate must be measured on real annotations before deciding
whether stride 2 or a multi-slot head is necessary.

## Target geometry

For a center `(cx, cy)` in model-input pixels and output stride `s=4`:

```text
u = cx / s
v = cy / s
ix = floor(u)
iy = floor(v)
dx = u - (ix + 0.5)
dy = v - (iy + 0.5)
```

The Gaussian is anchored at integer cell `(ix, iy)`. Offsets are therefore in
`[-0.5, 0.5)` and agree with the existing C decoder, which first computes the local
heatmap moment at cell centers and then adds the learned offset.

## Initial loss

```text
total = 1.00 * modified_focal(heatmap)
      + 1.00 * smooth_l1(offset at owned centers)
      + 0.15 * smooth_l1(normalized size at owned centers)
      + 0.00 * C-decoder-consistency loss initially
      + 0.10 * mean_square(padding channels)
      + 0.01 * mean_square(background offset/size channels)
      + 0.01 * encoded-range saturation penalty
```

The initial implementation leaves decoder consistency disabled. A floating-point
ground-truth-anchor surrogate can disagree with the predicted-argmax and later INT8
decoder. Introduce it only with exact output fake quantization and a parity-tested peak
selection rule. Empty images still contribute heatmap negatives. Regression losses are
normalized by representable owned centers, not by image pixels.

The first ablations should change one item at a time: Gaussian radius, heatmap focal
versus BCE, L1 versus Smooth-L1 offsets, decoder consistency, size branch, and finally
output stride. Use at least three seeds once real data exists.

## Training curriculum

1. Target-centered crops with large, easy synthetic/real heads and mild color changes.
2. Mixed target crops and full frames, multiple heads, crop jitter, blur, and occlusion.
3. Tracker-like crops perturbed by position/scale error, partial exits, hard negatives,
   and full-frame reacquisition samples.
4. Final epochs using the exact deployed crop, resize, RGB order, and normalization.

The float preprocessing contract is `(RGB_uint8 - 128) / 128`, matching the firmware's
signed input transform with an input quantization exponent of `-7`.

## Quantization sequence

1. Train and select the float32 checkpoint using centroid metrics, not training loss.
2. Export static NCHW batch-1 ONNX with only Conv/depthwise Conv, ReLU, Add, and nearest
   Resize in the compute graph.
3. Check PyTorch versus ONNX Runtime outputs on identical tensors.
4. Calibrate P4 per-channel INT8 Conv/GEMM using deterministic, representative batch-1
   samples. Start with 128--512 real deployment-like frames; the synthetic smoke uses
   fewer only to exercise the code.
5. Compare float and ESP-PPQ-simulated decoded centroids on a held-out set.
6. Try TQT/AutoQuant only after a measured PTQ failure. Use INT16 only for a measured
   sensitive fused block, not all early layers by assumption.

ESP32-P4 uses symmetric power-of-two quantization, per-channel Conv/GEMM, per-tensor
other operators, and round-half-even behavior. `.espdl` output is target-specific.

### Physical HWC16 output encoding

The float training model keeps ordinary semantic units. Export creates a non-mutating
clone and folds channel gains `[2, 16, 16, 16, 16, 0, ..., 0]` into the final Conv,
then clamps its physical output to `[-16, 15.5]`. With the required shared ESP-DL
output exponent `-3` (scale `0.125`), the INT8 tensor has this contract:

```text
channel 0:     q = round(16 * heatmap_logit)      (Q4)
channels 1-2: q = round(128 * x/y offset)        (Q7)
channels 3-4: q = round(128 * normalized size)   (Q7)
channels 5-15: exactly zero
```

This preserves sub-pixel offsets despite ESP-DL assigning a single per-tensor scale to
the output. The ONNX file records the encoding and required exponent as metadata. The
quantizer verifies both metadata and the exponent selected by ESP-PPQ, and refuses to
publish artifacts if either differs. The comparison tool converts encoded tensors back
to semantic units before decoding and measuring centroid drift.

The clamp becomes an ONNX `Clip` node. ESP-DL supports INT8 `Clip`; it still requires
physical-board profiling because support does not establish its latency or memory cost.

## Honest validation

Synthetic data can validate learning, deterministic targets, same-cell collisions,
decode/matching code, checkpointing, ONNX parity, and converter execution. It cannot
establish human-head accuracy, real confidence calibration, INT8 robustness on camera
images, memory use, latency, or FPS.

Primary real-data metrics will be center AP at 0.05/0.10/0.20 head diagonals, recall,
false positives, collision rate, and p50/p90/p95/p99 centroid error in source pixels and
as a fraction of head diagonal. Split real sequences by person/session/camera; never put
near-adjacent frames from one sequence in different partitions.

## References

- [MLX examples](https://github.com/ml-explore/mlx-examples)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [ESP-DL](https://github.com/espressif/esp-dl)
- [ESP-DL quantization guide](https://github.com/espressif/esp-dl/blob/master/docs/en/tutorials/how_to_quantize_model.rst)
- [Objects as Points (CenterNet)](https://arxiv.org/abs/1904.07850)
