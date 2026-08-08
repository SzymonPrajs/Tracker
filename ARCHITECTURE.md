# Tracker implementation architecture

This repository is the bring-up scaffold for a low-latency head-centroid tracker on the
Waveshare ESP32-P4 module board. The first target is a fresh centroid at 25 frames per
second, not a full-resolution semantic mask.

## Evidence boundary

- Host tests establish portable C correctness, memory safety, deterministic fixtures,
  and reporting behavior.
- ESP-IDF cross-builds establish target syntax, linking, ABI, and emitted instructions.
- Only a physical-board camera-to-centroid run may establish ESP32-P4 frame rate.

The benchmark schema enforces this distinction. A host run cannot set
`device_fps_claim=true`.

## Runtime pipeline

```text
MIPI CSI -> ISP -> PPA crop/resize/convert -> INT8 network -> centroid decode
         -> alpha-beta prediction -> camera controller
```

Camera buffers and the model arena belong in PSRAM. Small hot tensors, task stacks,
mailbox state, DMA descriptors, and control state should preferentially remain in
internal memory. The pipeline has a one-frame mailbox: a newly completed frame replaces
an older unconsumed frame so latency cannot grow into a queue.

## Code ownership

- `src/core/*.c`: portable, testable scalar reference code.
- `firmware/main/src/rv32/*.S`: standalone target assembly; no inline assembly.
- ESP-DL: production convolution/PIE backend once a quantized `.espdl` model is added.
- `bench/` and `tools/`: fixtures, collection, summaries, and acceptance gates.

The custom assembly boundary is deliberately narrow. ESP-DL already owns optimized
convolution kernels; replacing them without device profiles would make the system less
verifiable. The first assembly routines provide cycle/instruction counters and a signed
INT8 HWC16 argmax baseline.

## HC-DS31 model envelope

The initial custom model envelope is fixed-shape NHWC INT8:

| Property | Budget |
|---|---:|
| Input | 160 x 288 x 3 |
| Physical output | 40 x 72 x 16 |
| Logical output channels | confidence, x/y offset, width, height |
| Convolution/depthwise layers | 36 |
| Convolution MAC per frame | 31,256,640 |
| INT8 convolution weights | 66,224 bytes |
| Raw weights plus bias/shift estimate | about 73.3 KiB |
| Largest activation | 180 KiB |
| Analytical hot live set | about 315 KiB |

The output is padded to 16 channels to match the target layout. Residual/FPN additions
and three nearest-neighbor resize operations have no convolution MACs but still consume
memory bandwidth and time. A 22--30 ms inference interval is an analytical planning
range, not a measured result.

Start with per-channel symmetric power-of-two INT8 quantization and INT32 accumulation.
Do not move early layers to INT16 by default: those layers contain the largest spatial
tensors, INT16 doubles their bytes, and packed throughput is lower. Mixed precision is a
fallback for a layer whose measured quantization error justifies its cost.

## Acceptance gate

A 25 FPS claim requires a physical-device run of at least 60 seconds with:

- at least 25 fresh centroids in every measured second/window;
- p95 CSI-complete-to-centroid latency at or below 40 ms and p99 at or below 50 ms;
- inference queue depth no greater than one;
- reconciled captured, processed, replaced, and dropped frames;
- zero lost telemetry and passing correctness records;
- stable free and largest-free internal/PSRAM heap after warm-up.

Until that run exists, the repository reports cross-build and host evidence only.
