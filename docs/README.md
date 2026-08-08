# ESP32-P4 high-efficiency engineering reference

This folder is a design reference for algorithms targeting the 32 MB ESP32-P4, specifically the current `ESP32-P4NRW32X` system-in-package part. It is written for C, compiler-assisted optimization, hand-written RISC-V assembly, DMA-heavy pipelines, and the ESP32-P4 Processor Instruction Extension (PIE).

Last source verification: 2026-08-08.

## Read this first

The `32 MB` is in-package PSRAM, not on-chip SRAM and not flash. The part has 32 MiB (33,554,432 bytes) of 16-bit DDR PSRAM in the package. Firmware still requires separate off-package flash. The SoC supports up to 64 MB flash, but flash capacity is a board/product choice.

The current part is:

- `ESP32-P4NRW32X`: 32 MB OPI/HPI PSRAM, 1.8 V, -40 to 85 C, chip revision v3.x.
- Prefer revision v3.2 when choosing hardware. It fixes the v3.0 PSRAM/DMA coherency defects and v3.1 secure-download defect. Revision v3.2 still has one documented flash-power-on erratum.
- ESP32-P4 has no integrated Wi-Fi or Bluetooth radio. Designs that need radio networking normally add a companion chip such as ESP32-C6.

## Document map

1. [Target and architecture](01-target-and-architecture.md) - exact part, CPU/memory design, buses, accelerators, and address map.
2. [Memory, cache, DMA, and buffers](02-memory-cache-dma.md) - every memory tier, usable-RAM caveats, alignment, coherence, and FIFO/buffer census.
3. [CPU registers and instruction sets](03-cpu-registers-and-isa.md) - architectural registers, ABI, standard RV32 extensions, hardware loops, and PIE.
4. [Throughput and operations per minute](04-throughput-model.md) - defensible ceilings, what an "operation" means, and benchmark methodology.
5. [Efficient C](05-efficient-c.md) - allocation, layout, compiler, multicore, cache, and accelerator guidance.
6. [Direct assembly](06-direct-assembly.md) - ABI rules, `.S` integration, cycle counting, atomics, hardware-loop and PIE constraints.
7. [Register and peripheral reference](07-registers-and-peripherals.md) - register census, access strategy, major peripheral limits, and generated-header caveats.
8. [Silicon revisions and errata](08-errata-and-document-gaps.md) - defects that change algorithm or buffer design, plus conflicts in the official manuals.
9. [Camera development boards](09-camera-development-boards.md) - camera-equipped 32 MB boards, optical modules, driver modes, real demo behavior, and revision traps.
10. [Camera pipeline and bandwidth](10-camera-pipeline-bandwidth.md) - CSI/ISP/PPA/codec compute split, frame and lane rates, buffer budgets, C/assembly strategy, and acceptance tests.
11. [Purchasable board and camera kit](11-buyable-board-camera-kit.md) - the exact ESP32-P4-NANO-KIT-A recommendation, Amazon selection checks, OV5647 1080p30 mode, bandwidth, buffers, compute budget, and bring-up route.
12. [PIE mnemonic inventory](appendix-pie-mnemonics.md) - the 360 unique custom mnemonics present in ESP-IDF v6.0.2's assembler/decode test corpus.
13. [Register census](appendix-register-census.md) - all 137 numbered HP schemas and the complete per-header MMIO-definition count.
14. [Sources](sources.md) - primary documents, versions, exact code snapshot, and provenance rules.

## Five rules that dominate performance

1. Treat 32 MB PSRAM as high-bandwidth external memory, not zero-latency SRAM. The advertised 1 GB/s is a signaling ceiling, not application throughput.
2. Keep the hot loop and hottest working set in L2 RAM/SPM or locked/preloaded cache; stream large data through aligned tiles.
3. Allocate by capability (`MALLOC_CAP_SIMD`, `MALLOC_CAP_DMA`, `MALLOC_CAP_SPIRAM`) and keep cache-line ownership explicit across CPU/DMA handoffs.
4. Use both 400 MHz HP cores only for genuinely independent work. Shared cache, PSRAM, locks, and DMA can make two cores slower than one.
5. Prefer dedicated hardware (JPEG, H.264, ISP, PPA, 2D-DMA, BitScrambler, crypto) and PIE over scalar C when the algorithm maps to it.

## Status vocabulary

- **Specified**: stated by Espressif's current datasheet/TRM/errata or by a ratified RISC-V specification.
- **Derived ceiling**: arithmetic from specified widths and clocks; not a benchmark.
- **Toolchain-observed**: present in ESP-IDF v6.0.2 source or its assembler test corpus, but not necessarily fully documented by the TRM.
- **Must benchmark**: affected by cache state, placement, contention, compiler, silicon revision, or incomplete public documentation.
