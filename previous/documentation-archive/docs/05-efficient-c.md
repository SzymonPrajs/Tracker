# Efficient C on ESP32-P4

## Optimization order

Use this order unless measurements show a different bottleneck:

1. Choose the fixed-function accelerator or DMA path that removes the most CPU work.
2. Minimize bytes moved and conversions between stages.
3. Choose memory placement and tile size.
4. Establish a correct scalar reference implementation and test vectors.
5. Compile with optimization, inspect the disassembly, and measure counters.
6. Add PIE/intrinsics or a narrow assembly kernel only where the profile justifies it.
7. Parallelize across two cores only after single-core locality is good.

Hand assembly cannot rescue an algorithm that streams too many bytes from PSRAM or performs in C work that the ISP/PPA/JPEG/H.264/crypto block already implements.

## Build modes and compiler flags

Use the ESP-IDF build system so the silicon-revision workarounds and custom extensions match the selected target. For release benchmarking, start with `-O2` or `-O3`; compare `-Os` when instruction-cache pressure matters. Keep a correctness build with assertions/sanitizing checks where possible.

Useful per-function attributes include:

```c
__attribute__((hot, noinline))
void kernel(...);

__attribute__((aligned(64)))
static int16_t coefficients[...];
```

`noinline` makes microbenchmark boundaries stable but may hurt production optimization. Re-evaluate without it. Avoid global `-ffast-math` unless the algorithm explicitly permits changed NaN, infinity, signed-zero, reassociation, and rounding behavior. Apply relaxed math locally and verify numerical error.

Always retain the exact compiler invocation and inspect the final binary:

```sh
idf.py build
riscv32-esp-elf-objdump -drwC -S build/*.elf > build/app.disasm
idf.py size-components
```

## Make aliasing and bounds visible to the compiler

For non-overlapping arrays, use `restrict` and explicit lengths:

```c
void mix_i16(int16_t *restrict dst,
             const int16_t *restrict a,
             const int16_t *restrict b,
             size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (int16_t)(a[i] + b[i]);
    }
}
```

Only use `restrict` when the promise is true for the entire lifetime of each access in the block. Violating it is undefined behavior, not a slower fallback.

Prefer counted loops with a separate vector body and scalar tail. Keep signed overflow out of control/address arithmetic. Unsigned fixed-width types make wraparound explicit; wider accumulators prevent unintended overflow.

## Data layout and alignment

- Use structure-of-arrays when a kernel consumes only one or two fields.
- Pack external/storage formats for bandwidth, but decode into aligned compute tiles if packed fields cause repeated shifts or unaligned loads.
- Align PIE data to 16 bytes and streaming/cache-owned buffers to 64 bytes; use 128 bytes if the chosen L2 line is 128 bytes.
- Pad row strides to a cache-line/burst multiple when the extra bytes cost less than conflict/misalignment overhead.
- Put immutable, repeatedly used tables together; keep frequently written state away from them.
- Give each core its own accumulator, queue indices, and scratch lines. Reduce once per tile rather than atomically updating a shared scalar per element.

`sizeof(struct)` and member offsets are part of the performance model. Assert critical formats:

```c
_Static_assert(sizeof(struct dma_desc) % 8 == 0, "AXI descriptor alignment");
_Static_assert((TILE_BYTES % 64) == 0, "cache-line complete tile");
```

## Deliberate memory placement

Use the capability allocator and make ownership part of the type/API contract:

```c
void *tile = heap_caps_aligned_alloc(64, tile_bytes,
    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
void *model = heap_caps_aligned_alloc(64, model_bytes,
    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
void *pie = heap_caps_aligned_alloc(16, vector_bytes, MALLOC_CAP_SIMD);
```

- SPM: tiny deterministic kernel, lookup, or scratch state after benchmarking its parity-dependent latency.
- L2 RAM: stacks, descriptors, hot tiles, queues, and code/data whose latency matters.
- PSRAM: frames, models, history, bulk input/output, and cold objects.
- Flash: immutable code/constants, with XIP cache behavior included in measurement.

Do not place latency-critical task stacks in PSRAM. External RAM may be inaccessible while the flash/cache path is disabled unless the selected XIP PSRAM configuration explicitly supports the use case. Interrupt-time code and data must obey ESP-IDF's IRAM/DRAM-safe rules.

## Tile rather than stream blindly

For a transform or matrix kernel:

1. Read a cache-line-aligned input tile from PSRAM.
2. Reuse it in L1/L2 for as many outputs as possible.
3. Accumulate in registers or an internal scratch tile.
4. Write each output line once.
5. Hand off complete cache lines to DMA.

Pick tile dimensions from measured working-set behavior. Nominal cache capacity overstates usable capacity because code, stacks, the other core, associativity, and DMA compete for it.

## Cache and DMA ownership in APIs

Represent state transitions explicitly:

```text
CPU_WRITABLE -> CPU_CLEAN -> DMA_OWNED -> DMA_COMPLETE
             -> CPU_INVALIDATED -> CPU_READABLE
```

The producer must clean/write back before DMA reads; the consumer must invalidate after DMA writes. Synchronize complete cache lines. Do not put an unrelated mutable object in the same line as a DMA buffer edge.

Use memory barriers for ordering and `esp_cache_msync()` for cache data movement. A mutex or C atomic does not clean a dirty cache line for a non-coherent DMA engine.

## Double-buffered pipeline shape

```c
for (unsigned t = 0; t < tiles; ++t) {
    unsigned cur = t & 1u;
    unsigned next = cur ^ 1u;

    wait_input_dma(cur);
    sync_dma_to_cpu(input[cur], tile_bytes);

    if (t + 1 < tiles) {
        start_input_dma(next, source_for(t + 1));
    }

    kernel(output[cur], input[cur]);

    sync_cpu_to_dma(output[cur], tile_bytes);
    wait_output_slot(cur);
    start_output_dma(cur, destination_for(t));
}
```

This is a shape, not a drop-in API. A real implementation must handle the first fill, final drain, driver ownership, errors, and cache-line-safe allocation. Measure whether input DMA, compute, and output DMA genuinely overlap on the selected interconnect.

## Two-core strategy

Good partitions minimize shared writes:

- contiguous independent image bands with padded boundaries;
- independent channels, batches, tiles, or pipeline stages;
- static partitions when workload is uniform;
- coarse work stealing when variability outweighs queue contention.

Pin tasks during benchmarking. Start with one worker per core and one queue per producer/consumer pair. Avoid a single fine-grained global queue or an atomic counter per element. Place queue heads/tails on separate cache lines and batch publications.

The second core can decrease throughput when both cores saturate PSRAM or thrash the shared L2. Compare 1-core and 2-core bytes/s and cycles/item; speedup below 1.5x often signals a shared-resource limit worth fixing before further parallel tuning.

## Atomics and synchronization

Use C11 atomics or ESP-IDF synchronization primitives for shared CPU state. Select the weakest memory order that is demonstrably correct, but optimize only after a correct acquire/release design exists.

Batch work to amortize locks. Atomics in an inner numeric loop are usually a design smell. `LR.W`/`SC.W` may retry because the reservation can be cleared by ordinary events on this core; it is inappropriate for long critical sections.

## Branches and code layout

The BHT/BTB/RAS help predictable flow, but the machine is in-order and misprediction bubbles are visible.

- Hoist invariant validation outside the inner loop.
- Split common and exceptional paths.
- Replace unpredictable per-element branches with masks/min/max/saturation only when the resulting semantics and instruction count are better.
- Keep hot functions compact; compressed Zc/C code can improve I-cache residency.
- Avoid giant inline functions that duplicate code across call sites.

Use the branch and misprediction counters to settle the branch-versus-branchless choice.

## Fixed point, floating point, and quantization

The HP cores have hardware single-precision F but no hardware double-precision extension. `double` operations may require library sequences and should not be assumed fast. If the algorithm tolerates quantization, 8/16-bit PIE work offers dramatically more lanes and lower memory traffic.

For fixed point, document:

- Q format for every input/output;
- accumulator width and maximum sum;
- rounding mode;
- saturation versus wraparound;
- scaling at every stage;
- error bounds against the scalar/high-precision reference.

Random and adversarial vectors must include extrema, negative values, zero, ties, overflow, misalignment, and non-multiple vector lengths.

## Prefer engines over CPU loops

- ISP/PPA/2D-DMA for crop, rotate, scale, blend, color conversion, and reordering.
- JPEG/H.264 for supported codec paths.
- AES/SHA/RSA/ECC/HMAC/digital-signature blocks for cryptography.
- BitScrambler for programmable bit rearrangement.
- AXI async memcpy for large copies that can overlap useful work.

Validate input/output format, stride, alignment, supported resolution, and accelerator contention. “Hardware accelerated” does not remove staging overhead.

## Optimization acceptance checklist

An optimized kernel is ready only if:

- it matches the reference across normal, boundary, random, and adversarial inputs;
- it handles zero length, short tails, unaligned caller inputs, and aliasing as documented;
- its memory capability/alignment assumptions are asserted or checked;
- it is tested on the exact silicon revision and IDF/toolchain;
- warm, cold, and contended results are recorded;
- speed is reported with cycles/item and bytes/item, not only wall time;
- disassembly confirms the intended instructions and no unexpected helper calls;
- DMA/cache ownership is race-free under stress;
- the scalar fallback remains available when custom-ISA/toolchain conditions are not met.
