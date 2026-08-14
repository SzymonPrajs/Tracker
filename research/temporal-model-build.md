# Concrete build: motion-first temporal head tracker

## Conclusion

The supplied TMC-DTA proposal is coherent enough to build, but it contains one
important ordering error and several estimates that must not be presented as
measurements. The corrected first model is:

```text
full-resolution luminance Y
+ half-resolution positive/negative motion surfaces
+ stride-4 previous-owner prior
+ ordinary stride-4/stride-8 depthwise CNN
+ two-pole state on 8 of the 64 stride-8 channels
-> heatmap, sub-cell offset, displacement to previous centre
```

The host model, motion surfaces, readable configuration, and an inspection
script now exist. They establish exact tensor shapes and arithmetic; they do not
establish accuracy, quantized parity, board memory use, or frame rate.

## What the audit changed

1. The proposal's table placed the temporal adapter after the seven stride-8
   blocks, while its equations say later blocks mix the recurrent signal. The
   implementation puts the adapter immediately after the stride-8 downsample and
   before those seven blocks.
2. `39,565` is the number of deployed convolution coefficients when BatchNorm is
   folded and every convolution has an output bias. It is not the PyTorch
   trainable-parameter count. This implementation has `40,815` trainable values:
   convolution weights/biases, BatchNorm scale/offset, 24 prior scales, 16 mixing
   values, and two shared poles.
3. The proposal's convolution totals are correct: `15.451M`, `38.419M`, and
   `60.030M` MAC at 200x100, 320x160, and 400x200.
4. Persistent INT8 state totals are also correct at the two larger shapes:
   `54.4 kB` at 320x160 and `85.0 kB` at 400x200. These totals include previous
   half-resolution luminance, P/N surfaces, the owner prior, and both recurrent
   maps.
5. The claimed `~230 kB` activation arena and `~5.23 MB` logical bytes moved at
   320x160 remain planning estimates. Only an exported ESP-DL allocator plan and
   the physical board can establish peak memory and real memory traffic.
6. The initial deployment should expose recurrent state as named graph inputs and
   outputs. ESP-DL supports multiple named inputs and ordinary Add/Sub/Mul ops.
   Its automatic streaming conversion creates caches for temporal convolution;
   it does not directly express this arbitrary two-pole recurrence [3][4].

## The code boundary

The design is split by responsibility, without adding another package hierarchy:

| File | One responsibility |
|---|---|
| `config/temporal.toml` | Shapes, widths, pole starts, and motion constants |
| `python/common/motion.py` | Construct and update the three half-resolution frame-derived state planes |
| `python/common/temporal_model.py` | Define the neural graph and its two recurrent feature maps |
| `python/build_model.py` | Instantiate the model and recalculate shapes, parameters, MACs, and state bytes |

Run the inspection from the repository root:

```bash
python3 python/build_model.py
```

It performs a two-frame synthetic forward pass at every configured resolution.
It does not download data, train, export, or claim a board result.

Do not create one package per ablation. The later executable phases should remain
simple top-level scripts:

```text
python/train.py           still-image spatial control
python/train_temporal.py  synthetic pairs and contiguous clips
python/track_video.py     host decoder and ownership behaviour
python/quantize.py        selected graph only
```

Only add those files when their phase is implemented. Sequence loading and target
construction may become `python/common/sequences.py` once both training and host
evaluation share it. Ownership logic becomes `python/common/tracker.py` only when
both host evaluation and firmware need the same specified behaviour.

## Exact tensor contract

All neural tensors are NCHW in the host implementation. At the normal 320x160
input, one step consumes and produces:

| Name | Shape | Owner |
|---|---:|---|
| current luminance | `1x1x160x320` | camera/preprocessing |
| positive and negative surfaces | `1x2x80x160` | motion update |
| gated and warped previous-owner heatmap | `1x1x40x80` | tracker state machine |
| previous fast state | `1x8x20x40` | neural recurrence |
| previous slow state | `1x8x20x40` | neural recurrence |
| five output planes | `1x5x40x80` | neural graph |
| next fast state | `1x8x20x40` | neural recurrence |
| next slow state | `1x8x20x40` | neural recurrence |

The five output channels are fixed in this order:

```text
0  head-centre logit
1  sub-cell x offset
2  sub-cell y offset
3  x displacement to the previous centre
4  y displacement to the previous centre
```

At a stride-4 grid point `g`, decode the current centre as
`c(t) = 4 * (g + offset(t))`. The displacement target is in stride-4 cells:
`delta(t) = p(t-1) - p(t)`. Association compares
`c(t) + 4 * delta(t)` with the stored previous centre, following CenterTrack's
point-tracking interface [1][2].

The heatmap describes every visible head. The deterministic state machine, not a
special neural output, decides which confirmed subject owns the tracker.
The prior input is a bounded `[0, 1]` owner-centre map rendered by that state
machine, not the unfiltered previous all-head logit plane.

## Frame-derived motion

`motion.py` first averages luminance to half resolution. For each later frame it:

1. subtracts the previous half-resolution luminance;
2. estimates one global exposure shift as the median of tile means;
3. applies a deadband and clips positive and negative changes separately;
4. normalizes instantaneous changes to `[0, 1]`;
5. takes the maximum of each new change and its decayed old surface;
6. saves the current half-resolution luminance.

The first frame initializes the previous plane and empty surfaces, so starting a
sequence cannot create a false full-frame event. Motion must be generated after
temporally consistent geometry and sensor augmentation. Independent frame warps
would manufacture motion that never occurred.

The current `decay_shift = 4`, clip, deadband, tile size, and average-pooling rule
are experiment values, not camera-calibrated constants. Before firmware freezes
them, compare the floating reference with exact unsigned/signed integer versions
on recorded OV5647 clips containing exposure changes, dark noise, lighting
flicker, stationary people, and genuine motion.

## Neural graph, in execution order

| Stage | Operation | Output at 320x160 | Conv MAC |
|---|---|---:|---:|
| stem | Y 3x3 s2 `1->8`; P/N 1x1 `2->8`; add/ReLU | `8x80x160` | `1.126M` |
| localization | DS s2 `8->16`; DS `16->24`; residual DS `24` | `24x40x80` | `4.864M` |
| prior | zero-initialized learned scale per channel | `24x40x80` | none |
| downsample | DS s2 `24->64` | `64x20x40` | `1.402M` |
| temporal | two poles on the first 8 channels | `64x20x40` | none |
| trunk | seven residual DS `64` blocks | `64x20x40` | `26.163M` |
| decoder | top `64->16`, nearest resize, lateral `24->16`, two DS `16` | `16x40x80` | `4.608M` |
| head | 1x1 `16->5` | `5x40x80` | `0.256M` |

The prior scales and recurrent mixing values start at zero. Therefore the model
starts without trusting its own previous prediction and the temporal adapter is
initially a pass-through. This is essential for warm-starting and avoids an
untrained self-confirming track.

For recurrent input `U(t)` and old states `F` and `S`, the implemented adapter is:

```text
M(t) = U(t) + q_fast * (U(t) - F) + q_slow * (F - S)
F'   = a_fast * F + (1 - a_fast) * U(t)
S'   = a_slow * S + (1 - a_slow) * U(t)
```

`M(t)` is calculated from the old states before `F'` and `S'` are committed. The
shared poles start at `1 - 2^-3 = 0.875` and `1 - 2^-6 = 0.984375`; their logits
are learned in float while keeping each pole in `(0, 1)`. Skipped-frame handling
uses `a^elapsed_frames`. QAT must later decide whether learned poles earn their
multiplies or should snap back to shift-compatible values.

Before export, `prepare_for_export()` materializes the learned poles as constants.
This removes training-only Sigmoid work from every inference step. The exported
one-frame graph also omits exponentiation; elapsed-frame compensation belongs in
training or the later fused runtime state update.

## Training sequence

The current still-image cache can train appearance and synthetic pairs, but it
contains no real temporal identity. It cannot validate the displacement head or
long-lived state by itself.

Build training in controlled steps:

1. Train the same five-output graph on still images with luminance only, zero
   P/N, zero prior, and reset states. Only heatmap and offsets have valid loss.
2. Create synthetic frame pairs from one labelled scene. Keep one shared camera
   transform, move identified head instances independently where appropriate,
   render P/N after both frames exist, and supervise displacement only for heads
   present in both frames.
3. Add contiguous annotated video. Each sample must carry a sequence ID, frame
   index/timestamp, scene-cut flag, track ID, current box, and previous box.
4. Train 8-frame clips with state reset only at sequence boundaries. First use a
   ground-truth previous heatmap; then progressively replace it with detached
   predictions corrupted by jitter, missed peaks, false peaks, delay, blur,
   scale, and complete dropout.
5. Extend to longer clips with truncated backpropagation. Include stationary
   pauses, occlusion, crossings, global exposure changes, frame drops, duplicated
   frames, and variable elapsed-frame counts.
6. Compare the same backbone with no state, one pole, two poles, and online
   temporal shift. Select by accuracy per persistent byte and measured board
   latency, not by training loss.
7. Run PTQ first. Add recurrent fake quantization/QAT only if state saturation or
   a measured float-to-INT8 loss justifies it.

Primary loss terms are focal heatmap loss, masked Smooth L1 offset loss, and
masked Smooth L1 displacement loss. Normalize each regression loss by its own
valid tracked-point count. Do not add consistency or saturation penalties until
the three direct targets have been independently verified.

## Ownership remains outside the network

The host runtime should later implement only five readable states:

```text
SEARCH -> CANDIDATE -> LOCKED -> OCCLUDED -> LOST
```

The tracker gates and warps the prior before passing it to the model. `SEARCH`
uses an empty prior; `LOCKED` uses a bounded prior around the predicted owner;
`OCCLUDED` decays it; reset and scene cuts clear all motion, prior, and recurrent
state. A later, higher-confidence person must not steal a locked track.

This geometry-only interface cannot guarantee identity through an ambiguous full
occlusion or exact crossing. First measure that failure. Only then consider a
small embedding or external appearance template.

## ESP-DL deployment path

The first quantized reference should keep one ESP-DL graph with explicit inputs
`Y`, `P/N`, `prior`, `F`, and `S`, and explicit outputs `O`, `F'`, and `S'`.
ESP-DL exposes named input/output maps and supports the Conv, Add, Sub, Mul, ReLU,
and Resize operations used here [3][5]. This path may allocate avoidable
intermediates, but it gives one graph to compare against ESP-PPQ simulation.

After exact parity, profile three implementation choices:

1. leave the standard graph dense;
2. register a custom ESP-DL module that fuses the two-pole update, and later the
   prior/event additions [6];
3. split the graph around custom C only if the extra arenas and boundary copies
   are demonstrably cheaper.

Do not use ESP-DL's automatic streaming cache as if it implemented TMC-DTA. That
facility rewrites temporal convolutions and their causal windows [4]. A custom
state equation still needs explicit graph operations or a registered module.

The firmware eventually owns half-resolution integer motion updates, state
reset, prior warp/gating, decoding, the ownership state machine, and camera-motion
compensation. Start in clear C and preserve byte-for-byte host reference vectors.
Fuse or use PIE only after the complete board profile identifies the traffic or
loop as a bottleneck.

## Evidence required before selecting the design

- static-head accuracy must not collapse when motion surfaces decay;
- motion must improve acquisition at a fixed false-positive rate;
- corrupted predicted priors must not create self-lock;
- the two-pole model must beat simpler state controls at a justified byte cost;
- 320x160 must be compared with 400x200 on small heads;
- ESP-PPQ and board tensors must match fixed sequence vectors across resets;
- peak internal memory, PSRAM traffic, and camera-to-point p95 latency must be
  measured on the ESP32-P4-Module-DEV-KIT with the OV5647.

Until those tests exist, TMC-DTA is a precise and runnable research hypothesis,
not the selected production model and not a demonstrated novel deployment.

## Primary sources

1. Zhou, Koltun, and Krähenbühl, [Tracking Objects as Points](https://arxiv.org/abs/2004.01177), 2020.
2. Zhou et al., [official CenterTrack implementation](https://github.com/xingyizhou/CenterTrack).
3. Espressif, [ESP-DL operator support](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md).
4. Espressif, [deploying streaming models](https://github.com/espressif/esp-dl/blob/master/docs/en/tutorials/how_to_deploy_streaming_model.rst).
5. Espressif, [running models and named inputs/outputs](https://github.com/espressif/esp-dl/blob/master/docs/en/tutorials/how_to_run_model.rst).
6. Espressif, [creating a custom ESP-DL module](https://github.com/espressif/esp-dl/blob/master/docs/en/tutorials/how_to_add_a_new_module%28operator%29.rst).
