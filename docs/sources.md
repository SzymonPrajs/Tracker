# Sources and provenance

Last verified: 2026-08-08.

This reference prefers current primary sources: Espressif's datasheet, TRM, errata, ESP-IDF programming guide/source, and ratified RISC-V specifications. Secondary board-shop descriptions and unsourced benchmark posts were not used as authorities.

## Pinned source matrix

| Source | Pinned version/snapshot | Used for |
|---|---|---|
| [ESP32-P4 Series Datasheet](https://www.espressif.com/sites/default/files/documentation/esp32-p4_datasheet_en.pdf) | v0.7, July 2026 | part variants, clocks, memory/cache geometry, buses, DMA, accelerators, peripheral/FIFO capacities, electrical/package facts |
| [ESP32-P4 Technical Reference Manual](https://www.espressif.com/sites/default/files/documentation/esp32-p4_technical_reference_manual_en.pdf) | v0.6, July 2026, pre-release | processor pipeline, ISA/core registers, address map, peripheral behavior, register descriptions |
| [ESP32-P4 Chip Revision Errata](https://docs.espressif.com/projects/esp-chip-errata/en/latest/esp32p4/esp-chip-errata-en-master-esp32p4.pdf) | v1.3 | affected revisions, defects, fixes/workarounds |
| [ESP-IDF stable programming guide for ESP32-P4](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/) | v6.0.2 stable at verification | supported APIs, cache/external-RAM/allocation behavior, target configuration |
| [ESP-IDF repository](https://github.com/espressif/esp-idf/tree/7101770dc6db2667b3c477cc31365dd1acd6db4e) | tag v6.0.2, commit `7101770dc6db2667b3c477cc31365dd1acd6db4e` | actual architecture flags, target capabilities, linker regions, register headers, PIE assembler corpus |
| [Espressif TRM source repository](https://github.com/espressif/esp-technical-reference-manual-latex/tree/cf74bb3227bcab3dd4391153d3e563cc13e36f1e) | commit `cf74bb3227bcab3dd4391153d3e563cc13e36f1e` | searchable cross-check of the generated TRM |
| [RISC-V Unprivileged ISA](https://docs.riscv.org/reference/isa/unpriv/unpriv-index.html) | official ratified/specification set current at verification | RV32I/M/A/F/C, Zifencei/Zicsr and extension semantics |
| [RISC-V Bit-Manipulation ISA](https://docs.riscv.org/reference/isa/unpriv/b-st-ext.html) | official specification | Zba/Zbb/Zbs instruction semantics |
| [RISC-V Code-Size Reduction ISA](https://docs.riscv.org/reference/isa/unpriv/zc.html) | official specification | Zcb/Zcmp/Zcmt instruction semantics |
| [RISC-V psABI](https://docs.riscv.org/reference/abi/_attachments/riscv-abi.pdf) | official ABI document current at verification | integer/floating calling convention, stack alignment, ELF ABI |
| [ESP-Video-Components repository](https://github.com/espressif/esp-video-components/tree/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e) | commit `6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e` | current OV2710, OV5647, SC2336 and SC202CS/SC2356 mode tables; V4L2 devices; H.264 controls |
| [ESP-Dev-Kits repository](https://github.com/espressif/esp-dev-kits/tree/ddb440c409da7adb0a4ff7902b0133fb3fb6cfa3) | commit `ddb440c409da7adb0a4ff7902b0133fb3fb6cfa3` | P4X board guides and source-level audit of the P4-EYE factory-demo buffers, formats and recording path |
| [ESP32-P4X-EYE guide](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32p4/esp32-p4x-eye/user_guide.html) | latest at verification | board contents, camera/display/storage interfaces, P4X revision statement |
| [ESP32-P4X-Function-EV-Board guide](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32p4/esp32-p4x-function-ev-board/user_guide.html) | latest at verification | optional camera/LCD kit, ports and P4X revision statement |
| [P4X-EYE OV2710 module drawing](https://dl.espressif.com/AE/esp-dev-kits/HDF2710-47-MIPI-V2.0.pdf) and [OV2710 brief](https://dl.espressif.com/AE/esp-dev-kits/ov2710pbv1.1web.pdf) | module drawing dated 2024-12; sensor brief v1.1 | optics, connector, native array and sensor-level maximum modes |
| [Function EV camera module](https://dl.espressif.com/dl/schematics/camera_datasheet.pdf) and [adapter schematic](https://dl.espressif.com/dl/schematics/esp32-p4-function-ev-board-camera-subboard-schematics.pdf) | module A0, 2024-07-11; adapter rev 1.1 | unnamed module limits, two-lane wiring, rails, SCCB level shift and 24 MHz clock |
| [Waveshare ESP32-P4-WIFI6-Touch-LCD-3.5](https://www.waveshare.com/esp32-p4-wifi6-touch-lcd-3.5.htm) | vendor page current at verification | supplied OV5647, advertised sensor modes, optics and board memory/interface configuration |
| [M5Stack Tab5](https://docs.m5stack.com/en/core/Tab5) | vendor documentation current at verification | integrated SC2356, display, memory, clocks and camera pin routing |
| [Waveshare ESP32-P4-NANO product](https://www.waveshare.com/esp32-p4-nano.htm), [documentation](https://docs.waveshare.com/ESP32-P4-NANO) and [schematic](https://files.waveshare.com/wiki/ESP32-P4-NANO/ESP32-P4-NANO-schematic.pdf) | product/docs/schematic current at verification | SKU and kit contents, 360-MHz processor, 32-MB PSRAM/16-MB flash, interfaces, dimensions, Pi-style two-lane camera connector and SCCB pins |
| [Waveshare RPi Camera (B)](https://www.waveshare.com/rpi-camera-b.htm) | vendor page current at verification | OV5647 sensor, optics, manual focus and no-night-vision statement; cross-check exposed a conflicting FOV value on the kit page |
| [Waveshare ESP32-P4 Platform examples](https://github.com/waveshareteam/ESP32-P4-Platform/tree/028473b3bac120d38589e1c18f8ea90daccc090c) | commit `028473b3bac120d38589e1c18f8ea90daccc090c` | board check, camera/display and simple-video-server routes; supported-board matrix; ESP-IDF version guidance; default OV5647 mode and SCCB pins |
| [Waveshare ESP32-P4-NANO BSP](https://components.espressif.com/components/waveshare/esp32_p4_nano/versions/3.0.0/readme?language=en) | component v3.0.0 | current board support and explicit rev-v1.3/rev-v3.x compatibility boundary |
| [Amazon ESP32-P4-NANO camera bundle](https://www.amazon.com/dp/B0DKT7ZP48) | ASIN/title and package statement verified 2026-08-08 | confirms a direct Amazon listing whose title and bullet specify the four-item camera bundle; not used as technical authority or proof of UK stock/price |

## Useful ESP-IDF subsections

- [Memory types](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/memory-types.html): executable/data memory behavior, external-memory caveats, IRAM/DRAM rules.
- [External RAM](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/external-ram.html): PSRAM integration, allocation and XIP configuration.
- [Heap memory allocation](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/system/mem_alloc.html): capability allocator and runtime heap metrics.
- [Performance overview](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/performance/index.html): build/runtime performance controls.
- [Speed optimization](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/performance/speed.html): compiler and placement guidance.
- [Cache library](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/system/mm_sync.html): cache synchronization and cache-line ownership APIs.
- [Asynchronous memcpy](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/system/async_memcpy.html): DMA-backed copy and backlog/burst configuration.
- [Chip revision](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/system/chip_revision.html): revision identification and app compatibility controls.
- [Camera controller](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/camera_driver.html): CSI/DVP configuration, transactions, callback context and buffer allocation.
- [ESP-Video-Components overview](https://docs.espressif.com/projects/esp-video-components/en/latest/esp32p4/Get_Started/index.html): sensor/SCCB/IPA/V4L2 software architecture.
- [ESP-Vision camera pipeline](https://docs.espressif.com/projects/esp-vision/en/latest/esp32p4/concepts/camera-pipeline.html): double buffering and PPA-backed scale/color conversion.

If a deep link changes, start from the stable ESP32-P4 programming-guide root pinned above and search its local version. Do not substitute another target's guide without checking target capability guards.

## Source hierarchy for conflicts

Use this decision order:

1. Current errata for known silicon defects and official workarounds.
2. Current datasheet for marketed part/revision capabilities and headline limits.
3. Current TRM for architectural/register behavior, while respecting its pre-release status.
4. Matching ESP-IDF release source for what the build, driver, linker, and toolchain actually do.
5. Ratified RISC-V specifications/psABI for standard instructions and calling convention.
6. Measurement on the exact silicon/configuration for latency, sustained bandwidth, contention, and undocumented custom-instruction behavior.

When sources disagree, this reference records the disagreement rather than silently choosing the most favorable value. See [document gaps](08-errata-and-document-gaps.md).

## Derived quantities

The following are calculations, not directly quoted performance claims:

- 32 MiB = `32 x 1,048,576 = 33,554,432 bytes`.
- One HP core at 400 MHz = `24 billion cycles/minute`; two = `48 billion`.
- PSRAM signaling = `16 bits x 2 DDR edges x 250 MHz = 8 Gbit/s = 1 GB/s`.
- PIE 8-bit dual-core = `2 cores x 400M cycles/s x 16 MAC/cycle = 12.8 GMAC/s`.
- PIE 16-bit dual-core = `2 x 400M x 8 = 6.4 GMAC/s`.
- If one MAC is counted as two primitive arithmetic operations, the reported primitive-op rate is twice the MAC rate.
- 6.92 CoreMark/MHz x 400 MHz = 2,768 CoreMark, assuming the stated dual-core result scales linearly to the maximum current clock.

Every derived ceiling is labeled as such. It excludes stalls, setup, tails, protocol overhead, arbitration, refresh, synchronization, and thermal/power constraints.

## Reproducible source-code censuses

The PIE list was extracted from:

```text
components/esp_gdbstub/test_gdbstub_host/rv_decode/xesppie.S
```

The register-definition census used the ESP32-P4 v3 register-header tree and counted named register declaration comments across `*_reg.h`. It found 104 header files, with 5,936 definitions occurring in 102 of them. This method intentionally counts the software surface, including aliases/arrays, and is not a unique-address census.

The HP-core TRM census counted headings `Register 2.1` through `Register 2.137`. Some headings define an indexed family.

To refresh the research:

```sh
git -C esp-idf describe --tags --always
git -C esp-idf rev-parse HEAD
rg -o 'esp\.[A-Za-z0-9_.]+' \
  esp-idf/components/esp_gdbstub/test_gdbstub_host/rv_decode/xesppie.S \
  | sort -u
```

Record the command/method with any new count; otherwise two “register totals” may be incomparable.

## Claims deliberately not made

This documentation does not claim:

- sustained 1 GB/s application PSRAM bandwidth;
- 48 billion useful scalar operations per minute;
- sustained peak PIE throughput;
- a complete, stable public PIE programming specification;
- 640 KiB of freely allocatable internal RAM;
- that the “32 MB version” includes 32 MB flash;
- a production result without measurement on the intended board/revision.
