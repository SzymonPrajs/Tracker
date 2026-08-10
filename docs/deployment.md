# ESP32-P4 deployment plan

## Hardware contract

The hardware is the Waveshare `ESP32-P4-Module-DEV-KIT`: 360 MHz
`ESP32-P4NRW32`, 32 MB in-package PSRAM, 16 MB flash, and two-lane MIPI-CSI.
The camera is the OV5647, the familiar Raspberry-Pi-style 5 MP sensor.
Current Espressif driver modes include RAW10 1920×1080 at 30 fps and lower or
binned RAW8/RAW10 modes. Runtime enumeration still records sensor ID, cable
path, clock, and chip revision as measured facts.

1080p30 is a sensor mode, not a required inference mode or measured tracker
rate. Lower native modes may reduce memory traffic enough to outperform 1080p.

## Candidate preprocessing graphs

Characterization compares three contracts:

```text
OV5647 RAW10
→ MIPI-CSI + ISP
   ├→ bypass/RAW capture → Bayer-aware unpack/reduce → exact INT8 → Bayer C1
   └→ ISP YUV420 capture buffer
       ├→ PPA full-frame scale → small GRAY8 → exact INT8 → luminance C1
       └→ PPA scale/color → small RGB888 → exact INT8 → RGB C3
```

All three avoid requiring a full-resolution RGB888 intermediate unless the RGB
path's negotiated implementation actually creates one. The chosen graph is
measurement-dependent:

- current source and documentation disagree in one RAW8/RAW10 conversion note;
- RAW10→YUV420 must be proved by format enumeration, negotiation, and frames;
- PPA GRAY8 depends on physical ESP32-P4 v3.0-or-later support and its build
  configuration;
- the board listing does not prove the physical chip revision;
- YUV range, color standard, layout, and image correctness must be measured;
- packed RAW10 Bayer phase, stride, black/white levels, unpacking cost, and
  available downscale path must be measured;
- current PPA SRM formats do not include RAW Bayer, so the direct-RAW branch
  needs ISP conversion or a measured custom reduction before the INT8 tensor;
- PPA scaling is bilinear and scale factors are truncated to 1/16 steps.

PPA GRAY8 requires both physical ESP32-P4 revision v3.0+ and a build with
`CONFIG_ESP32P4_REV_MIN=3`; this compiles the v3-only path and causes
incompatible earlier chips to be rejected by the bootloader. Preserve a
non-GRAY characterization build until the delivered revision is known.

## Buffer and bandwidth reasoning

One 1920×1080 frame is approximately:

| Format | Bytes |
|---|---:|
| packed RAW10 at exactly 10 bits/pixel | 2,592,000 |
| YUV420 | 3,110,400 |
| RAW10 unpacked to 16-bit, YUV422, or RGB565 | 4,147,200 |
| RGB888 | 6,220,800 |

At 30 fps, packed RAW10 is at least 77.8 MB/s and one full YUV420 write stream
is about 93.3 MB/s before PPA reads/writes, mapping, model activations/weights,
display/network, or other PSRAM users. Actual stride and padding can be larger.
Capacity alone does not prove performance.

PPA scale/rotate/mirror requires different input and output buffers. Begin with
two full capture buffers and one or two reusable small model-input buffers, then
measure two versus three capture buffers and queue depth. Use a newest-frame
policy; do not let inference build an unbounded stale queue. Maintain a
per-frame ledger for every DMA/CPU read, write, and copy and delete avoidable
full-frame copies before micro-optimizing arithmetic.

Measure native OV5647 modes as preprocessing candidates, including 1080p30,
1280×960@45, 800×640@50, and 800×800@50 where the current driver and delivered
sensor confirm them. Select model dimensions from exact supported PPA geometry,
measured camera coverage, and the current resource search.

## Characterization handoff

Before model finalist training, the reproducible setup and disposable benchmark
harness in [Hardware characterization](hardware-characterization.md) prove the
RAW-derived, luminance, and RGB input contracts plus the memory/resource
envelope. Production integration begins only after that evidence fixes
candidate byte layouts, ranges, and exact geometry.

## Firmware sequence

1. Record hardware and pin ESP-IDF, ESP-DL, ESP-PPQ, esp-video, sensor-driver,
   and board-support versions.
2. Re-run the selected camera and preprocessing contract as an integration
   smoke test.
3. Load the selected `.espdl` and pass multiple embedded golden tensors.
4. Add centroid decoding outside the model graph.
5. Connect camera → ISP → PPA → input mapping → ESP-DL → decode.
6. Add timing, heap, dropped-frame, and corruption telemetry.
7. Emit the per-frame byte-movement ledger and compare it with characterization.
8. Profile the complete loop; if the model underuses or exceeds the sustained
   envelope, return through the controlling resource-feedback loop to input
   shape/representation, topology, training, quantization, or buffer placement.
9. Optimize only measured bottlenecks.

ISP ROI cropping is optional and disabled initially. The initial PPA source
block is the complete negotiated frame. PPA block selection is a distinct
compile-time geometry choice and may be enabled only after its field-of-view
consequence is measured. A configuration header/Kconfig profile selects sensor
mode, input format, dimensions, optional ROI/block geometry, diagnostics, and
profiling so unused branches disappear at compile time. Keep conditional logic
in configuration/factory boundaries rather than scattering `#if` through hot
loops.

## Build profiles

| Profile | Intent | Typical properties |
|---|---|---|
| debug | correctness and debugger use | `-Og`, assertions, verbose logs, coredump, golden tests, OpenOCD/GDB |
| profile | realistic timing with observability | optimized build, stage timers, counters, limited logs |
| release | validated production deployment | measured speed/size choice between `-O2` and `-Os`, low log level, debug/test paths compiled out |

Use named ordered `sdkconfig.defaults` files and CMake targets. Do not assume
`-Os` is faster or smaller for the whole linked image without measuring it.

## C and assembly policy

Start with clean C/C++ orchestration and ESP-DL's existing P4 kernels, static
memory planner, and current dual-core Conv2D/DepthwiseConv2D scheduling. For
every local optimization:

1. write and test the portable reference;
2. profile the real board and inspect linked disassembly;
3. use compiler intrinsics or an existing ESP-DL/ESP-DSP primitive first;
4. use short guarded inline assembly only for a tiny measured leaf sequence
   where compiler-managed operands are an advantage;
5. use a separate `.S` file for a substantial loop, reusable kernel, explicit
   register schedule, or assembler-specific structure;
6. differential-test every optimized path against reference C and benchmark
   both.

Large inline-assembly blocks are not cleaner merely because they sit beside C.
They obscure constraints, ABI behavior, scheduling, and test boundaries. The
clean design allows small inline instructions but keeps real kernels modular.

## Deployment gate

- Actual chip revision, clock, memory, sensor ID, cable, and negotiated formats
  are recorded.
- Sustained capture shows no unexplained corruption or drops.
- RAW-derived Bayer, luminance, and RGB preprocessing costs are measured on the
  same board.
- Python and firmware input tensors are byte-identical.
- PPA geometry, alignment, cache ownership, and buffer counts are verified.
- Multiple `.espdl` gold tensors match.
- Reference and optimized decoders match adversarial inputs.
- Debug/profile/release profiles build independently.
- Stage telemetry can attribute end-to-end latency before optimization starts.
- Every full-frame and model-input read/write/copy is explained; no avoidable
  full-frame copy remains in release.
- Live capture reports integrated CSI+ISP frame-completion latency/throughput.
  An isolated ISP number appears only from a documented standalone ISP DMA
  harness and is labelled as such; PSRAM traffic is labelled as arithmetic
  estimate, hardware-counter measurement, or contention proxy.
