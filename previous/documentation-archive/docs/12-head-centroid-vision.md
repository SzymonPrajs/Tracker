# Head-centroid vision at 20--25 frames per second

Last verified: 2026-08-08.

Status after the 2026-08-10 workspace reset: this is a design and acceptance
reference. The earlier firmware/model implementation is archived under
`../previous/`, and no physical-board camera-to-centroid measurement has yet
validated the 20--25 fps target or application accuracy.

## Decision

The selected `ESP32-P4-Module-DEV-KIT` can plausibly track a face or head centre at 20--25 frames per second. It cannot run a useful learned pixel-to-pixel segmenter over a full 1920 x 1080 frame at that cadence.

The correct design is:

```text
camera RAW frame
  -> MIPI CSI and hardware ISP
  -> hardware PPA crop, scale, and format conversion
  -> small fixed-shape INT8 centre or landmark network
  -> confidence/centre/size decode
  -> temporal filter and forward prediction
  -> camera-framing controller
```

The neural output should be the location required by the controller, not a high-resolution mask that is immediately reduced to a location.

## Exact selected product boundary

Amazon UK ASIN `B0FPG45999` is the Waveshare `ESP32-P4-Module-DEV-KIT` base development board. The listing does not state that a camera is included. The board provides a two-lane MIPI-CSI connector compatible with cameras such as OV5647, so a matched camera and 15-pin cable remain separate system components.

The board vendor specifies:

| Resource | Selected-board value |
|---|---:|
| HP CPU | two 32-bit RISC-V cores, up to 360 MHz |
| LP CPU | one RISC-V core, up to 40 MHz |
| External mutable memory | 32 MiB in-package PSRAM |
| Flash | 16 MiB NOR on the module |
| HP L2MEM pool | 768 KiB, shared between L2 cache and L2 RAM |
| Scratchpad | 8 KiB |
| Camera path | two-lane MIPI CSI, ISP, PPA, JPEG, H.264 |
| Radio | ESP32-C6 companion over SDIO; not a vision accelerator |

This is a 360-MHz `ESP32-P4NRW32` product boundary. Do not substitute the 400-MHz P4X ceilings without reading the delivered eFuse revision and confirming the actual clock.

## Arithmetic ceilings and real model budget

At 360 MHz, the PIE packed-MAC arithmetic ceilings are:

| Mode | Both HP cores | Multiply and add counted separately |
|---|---:|---:|
| INT8, 16 MAC lanes/core/cycle | 11.52 GMAC/s | 23.04 GOPS |
| INT16, 8 MAC lanes/core/cycle | 5.76 GMAC/s | 11.52 GOPS |

The corresponding ideal per-frame allowances are 576 MMAC at 20 fps and 460.8 MMAC at 25 fps. Those are architectural ceilings, not usable model sizes.

ESP-Detection reports these complete P4 examples:

| Model input | Reported work | Latency | Derived effective rate |
|---|---:|---:|---:|
| 224 x 224 | 0.17 GFLOP | 51.4 ms | about 1.65 GMAC/s |
| 160 x 288 | 0.16 GFLOP | 45.9 ms | about 1.74 GMAC/s |

The derived rates assume the common model-reporting convention that a MAC is two FLOPs. They are useful empirical anchors, not proof that every topology sustains the same rate. ESP-DL's P4 operator measurements range widely with convolution shape, depthwise operation, reuse, and cache behaviour.

Use this envelope until the selected model is profiled on the delivered board:

| Application target | Frame period | Model-inference allocation | Design model budget |
|---|---:|---:|---:|
| 25 fps | 40 ms | 32--35 ms | 45--55 MMAC/frame |
| 20 fps | 50 ms | 40--43 ms | 60--70 MMAC/frame |

The remainder covers preprocessing, output decoding, queueing, control, and variance. PPA work can overlap CPU inference only after measurement proves that the buffer ownership and memory traffic permit it.

## Why learned 1080p segmentation is outside the envelope

One 3 x 3 convolution from three input channels to 16 output channels at 1920 x 1080 costs:

```text
1920 x 1080 x 3 x 16 x 3 x 3 = 895,795,200 MAC
```

That single layer is more than 13 times the conservative 20-fps model budget. Its `1920 x 1080 x 16` INT8 output is about 31.6 MiB, consuming essentially all physical PSRAM before camera buffers, weights, firmware, or another layer.

Even the same convolution at 480 x 270 costs about 56 MMAC. Therefore the ISP/PPA must reduce the image before learned processing, and the decoder must stay at low resolution.

## First baseline: existing face landmarks

For a visible face, begin with ESP-DL's existing `MSR_S8_V1 + MNP_S8_V1` two-stage detector. It returns a face box plus left eye, right eye, nose, left mouth, and right mouth landmarks.

Published P4 timing is:

```text
first stage = 1.3 ms preprocess + 13.1 ms model + 0.1 ms postprocess
second stage = 0.3 ms preprocess + 2.4 ms model per candidate
```

Therefore:

| Second-stage candidates | Derived total | Derived maximum cadence |
|---:|---:|---:|
| 1 | 17.2 ms | 58.1 fps |
| 5 | 28.0 ms | 35.7 fps |
| 9 | 38.8 ms | 25.8 fps |

The two P4 `.espdl` files total 191,216 bytes. This is the fastest credible bring-up baseline for a one-person framing system. It detects visible faces rather than arbitrary head orientation; side views, back-of-head tracking, severe occlusion, and a task-specific optical centre require custom training.

## Custom centre-heatmap model

For general head tracking, train a direct centre model rather than a semantic mask. Use one low-resolution heatmap plus four regression channels:

```text
confidence, offset_x, offset_y, width, height
```

For a single target, the heatmap can be decoded by spatial moments. For multiple people, find local peaks and choose the candidate nearest the temporally predicted target.

### Reference architecture

Use fixed `160 x 288 x 3` input in height-width-channel order:

| Stage | Output shape | Structure |
|---|---:|---|
| Input | 160 x 288 x 3 | PPA-produced RGB or supported model input format |
| Stem | 80 x 144 x 16 | 3 x 3 stride-2 Conv + ReLU |
| Stage 1 | 40 x 72 x 24 | two expansion-2 depthwise-separable blocks |
| Stage 2 | 20 x 36 x 32 | two expansion-2 blocks |
| Stage 3 | 10 x 18 x 48 | four expansion-2 blocks |
| Stage 4 | 5 x 9 x 64 | three expansion-2 blocks |
| Feature pyramid | 40 x 72 x 24 | small Resize, 1 x 1 lateral, Add, and depthwise refinement path |
| Output | 40 x 72 x 5 | 1 x 1 prediction head |

The calculated convolutional envelope is:

| Quantity | Value |
|---|---:|
| MAC/frame | 48.694 million |
| FLOP/frame under two-ops-per-MAC convention | 0.0974 billion |
| Convolution weights | 104,440 |
| Raw INT8 weight bytes | about 102 KiB |
| Largest individual expansion tensor | about 360 KiB |

Resize, Add, quantization, tensor layout, decode, and scheduling are not included in the MAC count. They must be included in measured latency. This network is a starting envelope, not a trained accuracy claim.

### Layer rules

Prefer:

- 3 x 3 Conv, 1 x 1 Conv, and 3 x 3 depthwise Conv;
- fused Conv + ReLU where available;
- channel counts aligned to 16 or 32 where accuracy permits;
- groups equal to one or to the input-channel count;
- Add instead of wide concatenation when the accuracy result is equivalent;
- a small INT8 Resize/FPN path and output-stride-four head;
- batch one and fixed shapes.

Avoid:

- full-resolution decoder activations and broad U-Net skip tensors;
- arbitrary grouped convolution, because current ESP-DL Conv supports group one or depthwise grouping;
- ConvTranspose, which ESP-DL implements through zero insertion plus Conv;
- attention, transformers, large LayerNorm surfaces, and large float Softmax outputs;
- RGB888 full-frame CPU scans or copies.

## Precision without full-resolution segmentation

Full-frame `288 x 160` preprocessing maps approximately 6.67 source pixels to one model pixel horizontally and 6.75 vertically at 1080p. An output-stride-four heatmap cell maps to roughly 27 x 27 source pixels, but a learned offset or heatmap moment is not restricted to the cell centre. A numerical error of 0.1 cell would map to about 2.7 source pixels; this arithmetic is not an accuracy guarantee.

For better precision at the same inference cost:

1. Reacquire globally at 5--10 Hz.
2. Crop a region around the predicted head from the current full-resolution frame.
3. Use PPA to resize the crop to 192 x 192 or 224 x 224.
4. Run a small landmark or centre-refinement network at 25 Hz.
5. Fall back to global acquisition when confidence, innovation, or crop-boundary tests fail.

A 512 x 512 source crop resized to 192 x 192 maps only about 2.67 source pixels to one model pixel. This improves coordinate resolution without applying convolutions to the complete 1080p image.

Use an alpha-beta or Kalman filter after the neural result. Estimate velocity, measure the full sensor-to-actuator latency, and predict the centre forward by that delay. Always consume the newest completed frame; discard stale inference work instead of allowing a queue to accumulate latency.

## Quantization decision

Start fully INT8:

- current ESP-DL and ESP-PPQ versions that support P4 per-channel Conv/Gemm;
- symmetric power-of-two quantization with P4 rounding behaviour;
- representative calibration covering lighting, gain, motion blur, pose, occlusion, scale, and background;
- PTQ as the first reproducible baseline;
- TQT, AutoQuant, or hardware-aligned QAT only after measuring layerwise error;
- distillation from a larger external teacher where it improves a small student's centre error.

Do not assume that early layers should be INT16. Early tensors are the largest and account for much of the model's spatial compute. INT16 halves the packed-MAC lane ceiling and doubles activation bytes. If layerwise analysis proves INT8 inadequate, prefer making only a small sensitive layer or late regression head INT16, then measure the conversion and memory cost. Mixed precision is an evidence-led exception, not the starting architecture.

Evaluate quantization using the controller's metric: centre error and temporal stability. Detection mAP alone does not establish a stable optical centre.

## Memory allocation target

The following is a design allocation rather than a claim about the final heap:

| Allocation | Approximate size |
|---|---:|
| two 1080p RGB565 capture frames | 7.91 MiB |
| or two 1080p YUV420 capture frames | 5.93 MiB |
| two 160 x 288 x 3 byte AI inputs | 270 KiB |
| custom model weights and metadata | target below 0.25 MiB |
| ESP-DL activation arena | target at or below 1.5 MiB |
| JPEG/encoded-output reserve | initial 1 MiB; test noisy high-detail scenes |
| stacks, queues, networking, filesystem | budget 2--4 MiB |
| uncommitted PSRAM safety margin | at least 8 MiB |

Keep normal planned PSRAM allocation below roughly 18--20 MiB until boot-time measurements prove more is safely contiguous. Record total free, minimum free, and largest free PSRAM block after every major pipeline allocation. Model parameters fit easily; compute and traffic, not nominal capacity, are the binding constraints.

## Training target and validation

Define the target point before annotating data. Possible definitions include face-box centre, head-box centre, midpoint between eyes with a learned vertical offset, or a photographer-defined framing point. Mixing definitions creates irreducible label noise.

Report at least:

- absolute centre error in source-image pixels;
- centre error normalized by head-box width or diagonal;
- median, p95, and worst-case error by pose and head size;
- frame-to-frame stationary-target jitter;
- loss-of-track and incorrect-reacquisition rate;
- error under blur, occlusion, backlight, high sensor gain, and partial frame exit;
- quantized-versus-float delta on exactly the same test frames.

## Board acceptance gates

The design is accepted at 25 fps only when all of these hold simultaneously:

- delivered silicon revision and 360-MHz clock are recorded;
- the selected camera, cable, sensor PID, mode, and negotiated buffer sizes are recorded;
- capture, PPA, inference, decode, filter, and controller timestamps use the same monotonic clock;
- p95 end-to-end vision latency is at or below 40 ms, with no unbounded queue;
- inference model work is at or below the profiled capacity, not merely the calculated MAC budget;
- the newest-frame policy and every dropped/stale frame are counted;
- largest-free PSRAM block and minimum-free internal/PSRAM heaps remain stable;
- centre accuracy and jitter pass a separately stated optical requirement;
- a 60-second capture test explains every difference from the sensor's expected frame opportunities.

If the stock face-landmark detector passes these gates, keep it as the first implementation. Replace it only when a measured failure mode requires the custom centre model.
