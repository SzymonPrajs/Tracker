# Register and peripheral reference

## Counting the register space

For algorithm engineering, use a reproducible census rather than a misleading single total:

- The HP core chapter of TRM v0.6 numbers 137 register descriptions across core CSRs, CLIC/CLINT, PMP/PMA, cache controls, trace, debug, and related CPU-local blocks.
- Many descriptions are parameterized arrays. For example, PMP entries and interrupt attributes instantiate multiple addresses.
- ESP-IDF v6.0.2 has 104 ESP32-P4 `*_reg.h` hardware-register headers. Of those, 102 contain 5,936 named register-definition comments in the v3 include tree; `io_mux_reg.h` and `wdev_reg.h` add no entries under this counting pattern.
- That 5,936 count includes channel/instance expansions, indexed registers, aliases, and ECO overlays. It is a software-definition census, not a claim that there are exactly 5,936 unique physical addresses.

The count was made against ESP-IDF commit `7101770dc6db2667b3c477cc31365dd1acd6db4e`. Recompute it after upgrading IDF rather than copying the number forward.

## Always use the matching generated headers

Peripheral programming should include symbolic SoC headers from the selected ESP-IDF release. Avoid literal base addresses and hand-copied bit positions:

```c
#include "soc/gpio_reg.h"
#include "soc/gpio_struct.h"
#include "hal/gpio_ll.h"
```

Prefer, in descending order:

1. Public ESP-IDF driver API.
2. HAL/LL API when driver overhead or missing functionality is measured.
3. Generated register macros for a narrowly documented operation.
4. Raw literal MMIO only for bring-up experiments, never long-lived code.

Driver APIs encode clocks, resets, DMA ownership, interrupt clearing, revision workarounds, locking, and power management. A direct register write can be faster in isolation while making the complete system incorrect.

## MMIO access semantics

Use `volatile` MMIO definitions supplied by IDF. `volatile` forces accesses to occur from the compiler's point of view; it does not provide multicore mutual exclusion, a cache flush, or all device-order guarantees.

For read-modify-write registers:

- determine whether set/clear aliases exist;
- determine whether status bits are write-one-to-clear;
- mask reserved bits exactly as the TRM requires;
- protect shared register updates against the other core/interrupt;
- do not read registers documented as write-only or with read side effects;
- account for synchronization delays between clock domains.

Reset and clock-gate a peripheral through the supported system API. Writes to an unclocked block may disappear or fault, and reset values can vary by revision.

## High-volume register-header groups

The largest v3 header groups in the IDF snapshot illustrate why a flat count is not useful:

| Header group | Named definitions | Why it is large |
|---|---:|---|
| GPIO | 380 | pin status/control, interrupt and matrix fields |
| H.264 DMA | 271 | channels, descriptors, status, interrupts |
| eFuse | 247 | many one-time-programmable words and controls |
| Cache | 245 | ways, regions, preload/lock/maintenance/status |
| AHB DMA | 188 | repeated channels and descriptor controls |
| 2D-DMA | 187 | channels, geometry, conversion/reordering |
| AXI DMA | 183 | repeated channels and AXI transaction controls |
| ISP | 157 | image stages, coefficients, windows/statistics |
| Per-core interrupt controller | 145 each | source configuration and status |
| DW-GDMA | 141 | repeated transfer/channel controls |
| SoC ETM | 138 | event/task routing matrix |
| PMU | 137 | power/sleep/clock-domain state |

These figures are code-navigation aids. The public driver/TRM remains authoritative for access sequencing.

## Peripherals most relevant to algorithm pipelines

### DMA engines

| Engine | Channels/shape | Design notes |
|---|---|---|
| GDMA-AHB | 3 TX + 3 RX | INCR4/8/16; descriptors 4-byte aligned |
| GDMA-AXI | 3 TX + 3 RX | up to 8 out-of-order and 8 outstanding; descriptors 8-byte aligned |
| VDMA | 4 unidirectional | two AXI masters; 64-deep FIFO/channel |
| 2D-DMA | 4 M2P + 3 P2M | unaligned starts; image macroblock reorder and color conversion |

DMA descriptors, payload alignment, burst size, cache policy, encryption, and target peripheral all impose constraints. Use the owning driver's descriptor type and helpers.

### Media/image

- ISP: RAW8/10/12, maximum 1920 x 1080; processing stages and statistics are controlled through a large coefficient/window register set.
- JPEG: baseline encoder/decoder; up to 4K still and stated dynamic-image rates.
- H.264: baseline YUV420 encoder up to 1080p30 aggregate.
- PPA: rotate/scale/mirror/blend/fill/alpha/color-key.
- MIPI CSI/DSI: two lanes each at up to 1.5 Gbit/s/lane.

Treat stride, crop, chroma subsampling, line padding, DMA descriptor boundaries, and pixel format as one pipeline contract. Reformatting a frame on the CPU can cost more than the accelerated stage.

### Streaming and storage

| Peripheral | Rate/capacity fact | Hardware buffer fact |
|---|---|---|
| USB HS OTG | 480 Mbit/s link | 4 KiB shared dynamic FIFO RAM |
| USB FS OTG | 12 Mbit/s link | 1 KiB shared dynamic FIFO RAM |
| Ethernet MAC | 10/100 Mbit/s | 256 B TX + 256 B RX FIFO |
| SD/MMC | card/clock-mode dependent | 2 KiB FIFO |
| GP-SPI2/3 | mode/clock dependent | 64 B data register bank/controller |
| HP I2S x3 | audio/sample-mode dependent | 256 B TX + 256 B RX/controller |
| HP UART x5 | baud dependent | 260 B RAM shared across all five TX/RX pairs |

The tiny hardware FIFO is not the driver ring buffer. Software should use DMA/ring buffers large enough to absorb scheduling latency, while keeping end-to-end latency and PSRAM traffic under control.

### Control/field buses

- HP I2C x2: 32-byte TX and 32-byte RX RAM each.
- LP I2C: 16-byte TX and 16-byte RX.
- TWAI x3: 64-byte RX FIFO, 13-byte TX buffer, 13-byte receive window each.
- RMT: 384 x 32-bit words shared by eight channels.
- PARLIO, I3C, PCNT, LEDC, MCPWM, and ETM can offload timing-sensitive GPIO/control work from CPUs.

Use ETM to connect hardware events/tasks when it eliminates ISR latency and jitter. Document which channel/event resources the pipeline owns so independently developed modules do not collide.

### BitScrambler

BitScrambler is a programmable DMA-side bit-manipulation engine with eight 257-bit instruction words and up to 32 output bits per DMA clock. It is a candidate for packing, unpacking, bit permutation, protocol transforms, and simple coding that would otherwise touch every word on a CPU. Benchmark setup cost; short records may remain faster in a scalar/Zb loop.

## Interrupt design

Each HP core has 32 external CLIC interrupt inputs plus CLINT sources. Optimization rules:

- acknowledge/clear the exact source according to its documented semantics;
- keep ISR work bounded and move bulk processing to a pinned task;
- place required ISR code/data in internal IRAM/DRAM-safe memory;
- avoid floating-point/PIE work in interrupts unless context support is explicitly verified;
- use DMA thresholds and batching to reduce interrupt rate;
- record worst-case service latency, not only average throughput.

High-rate per-byte/per-sample interrupts defeat the chip's DMA architecture.

## Register audit recipe

When writing a low-level driver, create a small register ledger:

| Field | Record |
|---|---|
| Source | TRM version/page/section and IDF header commit |
| Address | symbolic macro and resolved address |
| Width/access | RO, WO, RW, W1C, self-clearing, latch rules |
| Reset | reset value and required clock/reset sequence |
| Ownership | task/core/ISR that may write it |
| Ordering | barrier, poll, delay, or clock-domain sync |
| Revision | affected errata and workaround |
| Test | readback/status/logic-analyzer or loopback evidence |

This is more useful than a pasted register dump and survives code review.
