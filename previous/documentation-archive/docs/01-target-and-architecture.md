# Target and architecture

## Exact 32 MB target

| Item | Value |
|---|---:|
| Part | `ESP32-P4NRW32X` |
| Package | QFN104, 10 x 10 mm |
| In-package PSRAM | 32 MB OPI/HPI, 1.8 V |
| PSRAM usable bytes | 33,554,432 bytes, before allocator/metadata use |
| Current part revision family | v3.x |
| Recommended known revision | v3.2 |
| Off-package flash | Required; up to 64 MB supported; not included in the part number |
| Native radio | None |

`ESP32-P4NRW32X` and `ESP32-P4NRW16X` are the two current datasheet variants. The `W32` portion denotes 32 MB in-package 16-line PSRAM. Board listings that say "32 MB" without identifying PSRAM and flash independently are underspecified.

## Compute complex

### High-performance subsystem

- Two 32-bit RISC-V HP cores, up to 400 MHz on v3.x silicon.
- Five-stage, in-order, scalar pipeline per core.
- Standard ISA hardware: RV32I, M, A, F, C, Zc, and Zb.
- Custom hardware loops and 128-bit PIE SIMD/DSP/AI extension.
- Machine and User privilege modes; no Supervisor mode.
- 32 PMP regions and 16 PMA regions on current v3.x documentation.
- 32 external CLIC interrupts plus two CLINT sources per core.
- Up to three hardware breakpoints/watchpoints per HP core.
- Branch predictor per core: 512 x 16-bit BHT, 16-entry BTB, four-entry return-address stack.
- Two CoreMark result: 6.92 CoreMark/MHz, or a nominal 2,768 CoreMark at 400 MHz if scaled linearly. CoreMark is not an instruction rate.

The cores are scalar and in-order even though PIE performs packed 128-bit work. ESP32-P4 does **not** implement the standard RISC-V `V` vector extension: it has eight custom `q0`-`q7` vector registers, not 32 standard `v0`-`v31` registers.

### Low-power subsystem

- One 32-bit RISC-V LP core at up to 40 MHz.
- Two-stage, in-order, scalar pipeline.
- RV32IMAC ISA.
- 18 vectored interrupts and two hardware breakpoints/watchpoints.
- LP SRAM and LP ROM remain available while the HP system sleeps; LP code can wake HP.
- LP access to its own ROM/SRAM is documented as zero-wait. LP access to HP L2 RAM is approximately 20 LP cycles per access.

Use the LP core for always-on sensing, thresholds, housekeeping, and wake decisions. Do not move a bandwidth-heavy algorithm to it merely to free an HP core.

## Memory-side design

The HP data bus is 128 bits wide. Other buses are 32 bits. The 768 KB L2MEM runs at 200 MHz, half the HP CPU rate, and is shared between L2 cache and directly usable L2 RAM. The 8 KB SPM sits close to HP and is intended for deterministic hot code/data.

The memory system is little-endian. Both instruction and data buses share the same 32-bit address space. Addresses beginning `0x4...` can be cacheable or non-cacheable under PMA control; corresponding `0x8...` aliases provide direct, uncached access and are slower.

## Interconnect and DMA

The architecture has both AHB and AXI fabrics:

- GDMA-AHB: three TX and three RX channels, INCR4/8/16 memory bursts, one-word-aligned descriptors.
- GDMA-AXI: three TX and three RX channels, up to eight out-of-order and eight outstanding transactions, two-word-aligned descriptors.
- VDMA: four unidirectional channels, two AXI master interfaces, 64-deep data FIFO per channel.
- 2D-DMA: four memory-to-peripheral and three peripheral-to-memory channels, unaligned starting-address support, macroblock reordering, and color-space conversion.
- Many high-throughput peripherals have their own DMA path, including USB, trace, H.264, MIPI-related blocks, SD/MMC, and Ethernet.

There is no hardware cache-coherent interconnect. CPU and DMA ownership transitions require non-cacheable mappings or explicit cache synchronization.

## Dedicated algorithm accelerators

| Block | Useful work and specified ceiling |
|---|---|
| PIE | 128-bit packed integer/complex operations; 16 x 8-bit or 8 x 16-bit MAC elements per cycle |
| JPEG | Baseline encode/decode; up to 4K stills; 1080p40 or 720p70 dynamic images excluding header work |
| H.264 | Baseline encoder; YUV420 up to 1080p30; I/P frames, CAVLC, rate control, dual stream within same aggregate ceiling |
| ISP | RAW8/10/12 input, up to 1920 x 1080; demosaic, BLC, DPC, LSC, CCM, gamma, statistics, color conversion/crop |
| PPA | Rotation, scale, mirror, blend, fill, alpha and color key across common RGB/YUV/gray formats |
| 2D-DMA | Image macroblock reorder and color-space conversion during transfer |
| BitScrambler | Programmable bit transforms, up to 32 output bits per DMA clock; eight 257-bit instructions |
| Crypto | AES, SHA, RSA, ECC, HMAC, ECDSA/RSA digital signature, XTS-AES, TRNG, key manager |

For an image, audio, crypto, or format-conversion pipeline, first check whether the operation can be fused into an accelerator or DMA stage. A scalar loop that touches every byte is often the wrong baseline.

## High-level peripheral inventory

- 55 GPIOs: 39 HP and 16 LP.
- Five HP UARTs plus LP UART.
- GP-SPI2, GP-SPI3, flash MSPI, PSRAM MSPI, and LP SPI.
- Two HP I2C plus one LP I2C, internal analog I2C, and I3C.
- Three HP I2S plus LP I2S.
- USB 2.0 high-speed OTG (480 Mbit/s), full-speed OTG (12 Mbit/s), and USB Serial/JTAG.
- 10/100 Mbit/s Ethernet MAC through external PHY.
- Three TWAI/CAN 2.0 controllers, SD/MMC host, PCNT, LEDC, two MCPWM units, RMT, PARLIO.
- MIPI CSI and DSI, each two lanes at up to 1.5 Gbit/s per lane.

## Important design implication

The chip is best viewed as a small heterogeneous media/edge-compute SoC, not simply a fast microcontroller. Peak performance comes from scheduling data movement and fixed-function engines, while the two HP cores run control, residual kernels, and tasks that do not map to accelerators.
