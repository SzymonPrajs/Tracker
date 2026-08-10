# Throughput and operations per minute

## There is no single “operations per minute” rating

An integer add, cache miss, fused multiply-accumulate, JPEG macroblock, DMA byte, and floating-point divide are not equivalent operations. The useful answer is a set of ceilings with the counting convention stated.

All figures below assume current v3.x silicon at 400 MHz for both HP cores and 40 MHz for the LP core. They are **derived ceilings**, not benchmark results.

## Clock and scalar issue ceilings

| Resource | Calculation | Ceiling |
|---|---:|---:|
| One HP core cycles/second | 400 MHz | 400 million |
| One HP core cycles/minute | 400M x 60 | 24 billion |
| Two HP cores cycles/minute | 2 x 400M x 60 | 48 billion |
| LP core cycles/minute | 40M x 60 | 2.4 billion |

Because each HP CPU is a scalar five-stage in-order core, a useful best-case scalar model is at most one issued/retired instruction per core cycle. Thus 48 billion instructions/minute is an ideal dual-core ceiling for one-cycle instructions with no stalls. It is not 48 billion useful algorithm operations: loads, stores, branches, address calculations, synchronization, cache misses, and multi-cycle operations all consume instruction slots or time.

At 400 MHz:

- One cycle is 2.5 ns.
- A documented two-cycle multiply has 5 ns latency.
- A documented 1-19-cycle divide has 2.5-47.5 ns latency before surrounding work.
- L2MEM runs at 200 MHz, so the CPU can request work faster than the shared memory system clocks.

## PIE packed ceilings

The datasheet specifies 16 x 8-bit or 8 x 16-bit MAC elements per cycle. If one MAC is counted as one compound operation:

| Mode | Per HP core | Both HP cores | Both cores per minute |
|---|---:|---:|---:|
| 8-bit | 6.4 GMAC/s | 12.8 GMAC/s | 768 billion MAC/min |
| 16-bit | 3.2 GMAC/s | 6.4 GMAC/s | 384 billion MAC/min |

If the conventional accelerator metric counts a multiply and add separately, double those numbers:

| Mode | Both cores | Both cores per minute |
|---|---:|---:|
| 8-bit | 25.6 GOPS | 1.536 trillion primitive ops/min |
| 16-bit | 12.8 GOPS | 768 billion primitive ops/min |

A 16-lane 8-bit packed add has the same arithmetic lane ceiling—12.8 billion element-adds/s across two cores—if a suitable instruction issues every cycle.

These limits require all operands ready, a suitable instruction every cycle, no prologue/tail cost, no saturation/format overhead, and no contention. The public TRM does not provide the full PIE latency/throughput table, so sustained percentages must be measured.

## Memory signaling ceilings

| Path | Calculation | Signaling ceiling |
|---|---:|---:|
| 32 MB PSRAM | 16 bits x DDR x 250 MHz | 8 Gbit/s = 1 GB/s = 60 GB/min |
| USB HS | specified link rate | 480 Mbit/s = 60 MB/s = 3.6 GB/min |
| Ethernet | specified link rate | 100 Mbit/s = 12.5 MB/s = 750 MB/min |
| MIPI, two lanes | 2 x 1.5 Gbit/s | 3 Gbit/s = 375 MB/s = 22.5 GB/min |

These mix bus payloads and link rates and are not directly comparable. Encoding, packet framing, refresh, protocol commands, turn-around, arbitration, cache-line fetches, encryption, PHY behavior, and software reduce payload throughput. Full-duplex links also need separate direction accounting.

### Arithmetic intensity required to feed PIE

The 8-bit dual-core ceiling is 12.8 GMAC/s while the PSRAM signaling ceiling is 1 GB/s. Even before overhead, sustaining that compute ceiling directly from PSRAM requires at least `12.8 MAC/byte` of external traffic. A naïve dot product that reads two fresh 8-bit operands per MAC offers only 0.5 MAC/byte, so it is bandwidth-bound by more than 25x. Reuse weights/tiles in L1/L2/SPM or the nominal PIE rate is irrelevant.

Example: if an 8-bit matrix tile loads 4 KiB of A, 4 KiB of B, and writes 4 KiB of output but performs 1 million MACs, its external intensity is about 81 MAC/byte. That is potentially compute-bound. If it performs only 4,096 MACs for the same 12 KiB traffic, it is memory-bound.

## Fixed-function ceilings

These are useful workload ratings, not generic operations:

- JPEG: 1080p at 40 frames/s or 720p at 70 frames/s for dynamic images, excluding header processing; up to 4K still images.
- H.264 encoder: YUV420 up to 1080p30 aggregate.
- ISP: RAW8/10/12 input up to 1920 x 1080.
- Ethernet MAC: 10/100 Mbit/s.
- USB high-speed OTG: 480 Mbit/s.

For supported formats, the accelerator can outperform any reasonable C implementation while freeing both CPUs. Include conversion, DMA, cache synchronization, header/control work, and queue latency in an end-to-end benchmark.

## CoreMark context

Espressif reports 6.92 CoreMark/MHz for the two HP cores. Linear multiplication by 400 MHz gives 2,768 CoreMark for the stated configuration. This is a benchmark score, not CoreMark “operations,” instructions per second, or a guarantee for a particular algorithm.

## A model for a real kernel

For one tile, estimate:

```text
T_compute = scalar_cycles / 400 MHz
         or packed_operations / measured_PIE_rate

T_memory  = bytes_moved / measured_sustained_bandwidth

T_tile >= max(T_compute, T_memory) + synchronization + setup + tail
```

When stages cannot overlap, sum their times. When DMA and compute overlap perfectly with double buffering, the slower stage dominates after pipeline fill. Measure both; do not assume overlap merely because APIs are asynchronous.

## Benchmark protocol

### Control the experiment

Record all of the following with each result:

- exact part and silicon revision from `esp_chip_info()`/eFuse;
- ESP-IDF commit/version, compiler version, full `-march`, `-mabi`, and optimization flags;
- CPU and memory clocks, power-management locks, PSRAM mode/frequency;
- cache size, line size, write policy, placement of code/data/stack;
- one or two cores active, core affinity, interrupts, competing DMA/peripherals;
- warm-cache, cold-cache, and streaming results separately;
- alignment, tile size, stride, input distribution, and tail length;
- median plus p95/p99 or min/max—not only the fastest iteration.

### Measure cycles and retired work

Use a serialization strategy appropriate to the code under test, read `mcycle` around a repeated kernel, and subtract harness overhead. Also record `minstret`; cycles alone cannot distinguish instruction bloat from memory stalls. For branch-heavy work, program the available HPM counters for conditional branches and mispredictions.

```c
static inline uint64_t rdcycle64(void) {
    uint32_t hi0, lo, hi1;
    do {
        __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi0));
        __asm__ volatile ("csrr %0, mcycle"  : "=r"(lo));
        __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi1));
    } while (hi0 != hi1);
    return ((uint64_t)hi1 << 32) | lo;
}
```

Confirm that the execution environment permits these CSR reads. Pin the benchmark task, suppress dynamic frequency changes with the appropriate ESP-IDF power-management lock, and run enough inner iterations that timer overhead is negligible.

### Memory matrix to run on the actual board

For each SPM, L2 RAM, warm-cache PSRAM, streaming PSRAM, and uncached alias, measure:

- sequential read, write, copy, and read-modify-write;
- 8/16/32/128-bit access patterns where supported;
- aligned and deliberately misaligned starts;
- working sets spanning L1D and each possible L2 size;
- strides 1, 2, 4, 8, 16, 64, 128, one page, and a relatively prime stride;
- CPU alone, each DMA engine alone, and overlap/contended cases;
- one core and two cores.

The resulting surface—not the 1 GB/s signaling number—is the input to algorithm tiling.

## Report both rate and efficiency

For PIE kernels:

```text
MAC/s              = useful_MACs / seconds
PIE utilization    = measured_MAC/s / architectural_MAC_ceiling
external bandwidth = external_bytes / seconds
arithmetic intensity = useful_MACs / external_bytes
```

For scalar kernels report cycles/element, instructions/element, bytes/element, and branch misses/element. Those quantities remain meaningful when clock frequency changes.
