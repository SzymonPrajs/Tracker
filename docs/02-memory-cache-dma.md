# Memory, cache, DMA, and buffers

## Memory hierarchy

| Memory | Physical capacity | Clock/access | Executable | Typical use |
|---|---:|---|---|---|
| HP ROM | 128 KiB | 200 MHz, through L2 cache or direct alias | Yes | boot and ROM functions |
| HP L2MEM pool | 768 KiB | 200 MHz | Yes | divided between L2 cache and L2 RAM |
| HP SPM | 8 KiB | 400 MHz clock domain | Yes | deterministic hot loop/data |
| LP ROM | 16 KiB | 40 MHz, zero-wait from LP | Yes | LP boot/runtime |
| LP SRAM | 32 KiB | 40 MHz, zero-wait from LP | Yes | retained LP code/data |
| In-package PSRAM | 32 MiB | 16-bit DDR, up to 250 MHz | Yes through mapping | large mutable data, models, frames, optional XIP |
| Off-package flash | board-specific, up to 64 MB | SPI/Dual/Quad/QPI, up to 120 MHz | Yes through XIP | firmware and constants |
| eFuse | 4,096 bits | OTP | No | security/config; 1,792 user bits |

The PSRAM signaling ceiling is `16 bits x 2 edges x 250 MHz = 8 Gbit/s = 1 GB/s`. Protocol, refresh, turn-around, cache misses, arbitration, encryption, and access pattern reduce sustained throughput.

## L2 cache versus L2 RAM

L2 cache and L2 RAM consume the same 768 KiB L2MEM pool:

| L2 cache | L2 RAM | L2 line |
|---:|---:|---:|
| 512 KiB | 256 KiB | 128 B |
| 256 KiB | 512 KiB | 64 or 128 B |
| 128 KiB | 640 KiB | 64 or 128 B |
| 0 KiB | 768 KiB | hardware supports it; ESP-IDF v6.0.2 Kconfig does not expose it |

ESP-IDF v6.0.2 defaults to 128 KiB L2 cache with a 64-byte line. That leaves 640 KiB of hardware L2 RAM, but not 640 KiB of application heap. On v3.x, the IDF linker exposes `0x4FF20000..0x4FFADFBF`, 581,568 bytes (about 568 KiB), before `.text`, `.data`, `.bss`, stacks, runtime allocations, and other reservations.

Always measure the built image and boot-time heap:

```sh
idf.py size
idf.py size-components
idf.py size-files
```

At runtime use `heap_caps_get_free_size()`, `heap_caps_get_largest_free_block()`, and `heap_caps_get_minimum_free_size()` for each relevant capability.

## Cache geometry

| Level | Size | Line/block | Associativity | Notes |
|---|---:|---:|---:|---|
| L1 instruction | 16 KiB | 64 B | 4-way | shared external/internal cached instruction path |
| L1 data | 64 KiB | 64 B | 2-way | write-through or write-back |
| L2 unified | 128/256/512 KiB | 64/128 B | 8-way | carved out of L2MEM |

Supported operations include write-back, clean, invalidate, manual/automatic preload, and lock/unlock. A locked cache is not absolute: if every way is locked, replacement proceeds as if unlocked.

For high-efficiency kernels:

- Align hot arrays to at least 64 bytes; use 128 bytes if selecting the 512 KiB L2 configuration.
- Make the innermost tile fit comfortably below the effective cache capacity after code, stack, DMA, and the other core's traffic.
- Prefer sequential lines and reuse each line before advancing.
- Avoid array-of-struct layouts when a kernel touches only a subset of fields; structure-of-arrays often lowers fetched bytes.
- Separate producer and consumer control words onto different cache lines to avoid false sharing.

## SPM latency caveat

Current public sources conflict:

- TRM v0.6 says an HP SPM access finishes in two cycles.
- Current ESP-IDF documentation says one cycle with parity disabled and four cycles with parity enabled.
- The datasheet describes SPM as 400 MHz but does not settle effective load-to-use latency.

Treat SPM as deterministic and very small, but benchmark its exact latency on the selected revision and parity configuration. Do not hard-code a one-cycle scheduling assumption from documentation alone.

## Address map

| Range | Size | Target | Access |
|---|---:|---|---|
| `0x30100000..0x30101FFF` | 8 KiB | HP SPM | direct |
| `0x3FF00000..0x3FF1FFFF` | 128 KiB | HP CPU peripherals | MMIO |
| `0x40000000..0x43FFFFFF` | 64 MiB virtual | external flash | cacheable/PMA-selected |
| `0x48000000..0x4BFFFFFF` | 64 MiB virtual | external RAM | cacheable/PMA-selected |
| `0x4FC00000..0x4FC1FFFF` | 128 KiB | HP ROM | cached/PMA-selected |
| `0x4FF00000..0x4FFBFFFF` | 768 KiB | HP L2MEM | cached/PMA-selected |
| `0x50000000..0x500FFFFF` | 1 MiB | HP peripherals | MMIO |
| `0x50100000..0x50103FFF` | 16 KiB | LP ROM | direct |
| `0x50108000..0x5010FFFF` | 32 KiB | LP SRAM | direct |
| `0x50110000..0x5012FFFF` | 128 KiB | LP peripherals | MMIO |
| `0x80000000..0x83FFFFFF` | 64 MiB virtual | external flash | direct/uncached, slower |
| `0x88000000..0x8BFFFFFF` | 64 MiB virtual | external RAM | direct/uncached, slower |
| `0x8FC00000..0x8FC1FFFF` | 128 KiB | HP ROM | direct/uncached |
| `0x8FF00000..0x8FFBFFFF` | 768 KiB | HP L2MEM | direct/uncached |

External flash/RAM MMU mapping uses 64 KiB blocks and provides a 64 MiB instruction/data virtual window.

## Capability-based allocation

Use intent-bearing allocation instead of plain `malloc()` for important buffers:

```c
#include "esp_heap_caps.h"

void *hot = heap_caps_aligned_alloc(64, bytes,
                                    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
void *simd = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SIMD);
void *large = heap_caps_aligned_alloc(64, bytes,
                                      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
void *edma = heap_caps_aligned_alloc(64, bytes,
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA);
```

- `MALLOC_CAP_SIMD` provides PIE-accessible memory with preferred 16-byte alignment.
- `MALLOC_CAP_DMA` selects DMA-capable memory. On ESP32-P4 it can be combined with `MALLOC_CAP_SPIRAM` for EDMA-capable external buffers.
- `MALLOC_CAP_32BIT` may return IRAM and permits only aligned 32-bit accesses. Byte or half-word access can fault.
- `MALLOC_CAP_SPM` requests SPM.
- Check every result and capture largest-free-block metrics; 32 MiB total does not imply a 32 MiB contiguous allocation.

## CPU/DMA coherence protocol

ESP32-P4 is not hardware coherent. Use `esp_cache_msync()` or non-cacheable memory.

CPU produces, DMA consumes:

1. CPU fills the buffer.
2. `esp_cache_msync(buf, len, ESP_CACHE_MSYNC_FLAG_DIR_C2M | ESP_CACHE_MSYNC_FLAG_TYPE_DATA)`.
3. Publish the descriptor/start DMA.
4. Do not modify the buffer until completion.

DMA produces, CPU consumes:

1. Ensure old dirty CPU data cannot overwrite the DMA result.
2. Start DMA and wait for completion/ownership.
3. `esp_cache_msync(buf, len, ESP_CACHE_MSYNC_FLAG_DIR_M2C | ESP_CACHE_MSYNC_FLAG_TYPE_DATA)`.
4. CPU reads the result.

Align both address and length to the line size returned by `esp_cache_get_line_size_by_addr()`. The `UNALIGNED` override can invalidate adjoining dirty data and silently corrupt memory.

## DMA descriptor alignment

- GDMA-AHB descriptor: one word (4-byte) aligned.
- GDMA-AXI descriptor: two words (8-byte) aligned.
- Internal or unencrypted external data address/length: no hardware alignment requirement stated by the current datasheet.
- Encrypted external data address and length: 16-byte aligned.
- v3.0 PSRAM erratum: DMA reads with 1- or 2-byte bursts that are not 4-byte aligned can return old data after an overlapping write. Use at least 4-byte alignment even when targeting later revisions if portable binaries matter.
- Peripheral drivers may impose stricter alignment. Let `heap_caps_aligned_alloc()` or the driver helper satisfy the maximum of cache, DMA, encryption, SIMD, and peripheral requirements.

## Known hardware FIFOs and buffers

This table covers explicitly sized storage in the current datasheet/TRM. Driver ring buffers and application queues are separate software allocations.

| Block | Hardware storage |
|---|---:|
| Trace encoder, per core | 128 x 8-bit FIFO |
| VDMA, per channel | 64-deep data FIFO; element width is not stated in the overview |
| VAD | 2 KiB, four frames; each frame 256 x 16-bit samples |
| HP UART group | 260 x 8-bit RAM shared among TX/RX FIFOs of five UARTs |
| LP UART | 20 x 8-bit RAM shared by TX/RX |
| GP-SPI2/3 | 16 x 32-bit data registers = 64 bytes per controller |
| LP SPI | 16 x 32-bit data registers = 64 bytes |
| HP I2C, per controller | 32-byte TX RAM and 32-byte RX RAM |
| LP I2C | 16-byte TX and 16-byte RX FIFO/RAM |
| HP I2S, per controller | 64 x 32-bit TX plus 64 x 32-bit RX = 256 bytes each direction |
| LP I2S | 64 x 32-bit RX FIFO plus separate 16-bit circular memory described by TRM |
| USB HS OTG | 4 KiB shared dynamic FIFO RAM |
| USB FS OTG | 1 KiB shared dynamic FIFO RAM |
| Ethernet MAC | 256-byte TX plus 256-byte RX FIFO |
| TWAI, per controller | 64-byte receive FIFO; 13-byte TX buffer; 13-byte receive window |
| SD/MMC | 512 x 32-bit FIFO = 2 KiB |
| RMT | 384 x 32-bit RAM shared by 8 channels = 1,536 bytes |
| BitScrambler | eight 257-bit instruction words; data path includes 64 bits of input state |

## Buffer sizing strategy

- Double-buffer when one engine produces while another consumes; triple-buffer only when queuing jitter justifies the extra PSRAM/cache pressure.
- Size streaming tiles as multiples of cache line, DMA burst, pixel/audio frame, and encryption block.
- For PIE kernels, start with 16-byte vectors and tile working sets to L1D/L2. Avoid crossing pages or lines in the inner loop unless the fused unaligned loads prove faster on hardware.
- For large copies, use AXI async memcpy to overlap transfer and compute. The async driver defaults to a backlog of four requests; tune backlog and burst size using measurements.
- Keep descriptors and ownership flags internal and aligned even when payloads live in PSRAM.
