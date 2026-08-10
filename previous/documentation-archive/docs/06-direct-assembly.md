# Direct assembly

## When assembly is justified

Assembly is appropriate for a small, stable kernel when compiler output has a measured deficiency, when PIE/hardware-loop functionality is not expressible by supported intrinsics, or when exact instruction scheduling is required. Keep allocation, DMA, error handling, tails, and accelerator orchestration in C.

The public PIE documentation is incomplete. Standard RV32 assembly is portable within the declared ISA; `esp.*` assembly is tied to Espressif's GCC/binutils snapshot and silicon support.

## Use preprocessed `.S` files

Upper-case `.S` enables the C preprocessor, allowing target guards and shared constants:

```asm
    .section .text.kernel_add_u32, "ax", @progbits
    .balign 4
    .global kernel_add_u32
    .type kernel_add_u32, @function

/* void kernel_add_u32(uint32_t *dst, const uint32_t *src, size_t n); */
kernel_add_u32:
    beqz    a2, 2f
1:
    lw      t0, 0(a0)
    lw      t1, 0(a1)
    add     t0, t0, t1
    sw      t0, 0(a0)
    addi    a0, a0, 4
    addi    a1, a1, 4
    addi    a2, a2, -1
    bnez    a2, 1b
2:
    ret

    .size kernel_add_u32, .-kernel_add_u32
```

This leaf uses only argument/temporary registers and therefore needs no frame. It is an ABI example, not an optimized PIE implementation.

Add it to an ESP-IDF component with CMake source registration. Do not invoke a generic host assembler; let IDF pass the target architecture and silicon workarounds.

## Non-leaf frame pattern

```asm
    addi    sp, sp, -16
    sw      ra, 12(sp)
    sw      s0, 8(sp)
    mv      s0, a0

    call    helper

    mv      a0, s0
    lw      s0, 8(sp)
    lw      ra, 12(sp)
    addi    sp, sp, 16
    ret
```

Maintain 16-byte stack alignment at every call boundary. Save every callee-saved integer/F register that the function changes. Emit unwind/debug CFI if stack traces or exceptions must cross the routine.

## Inline assembly rules

Prefer a `.S` function for more than a few instructions. For inline assembly:

- describe every input, output, early clobber, and modified register;
- use `+r` for read/write operands;
- use `&` when an output cannot overlap an input;
- include `"memory"` only when the asm has memory effects invisible through operands;
- use `volatile` only when the operation must not be removed/reordered as ordinary pure computation;
- never hard-code an ABI register unless the instruction requires it and the constraints make that fact safe.

```c
static inline uint32_t rotate_right_7(uint32_t x) {
    uint32_t out;
    __asm__("rori %0, %1, 7" : "=r"(out) : "r"(x));
    return out;
}
```

This requires Zbb assembler/CPU support. Gate it with a build-time feature check and keep a defined C fallback. IDF v6.0.2 does not advertise Zb in its default P4 `-march`, despite the hardware manual.

## Ordering is not cache coherence

- `FENCE` orders specified memory/I/O observations.
- `FENCE.I` makes later instruction fetches observe prior code writes as defined by RISC-V.
- C compiler barriers prevent compiler reordering.
- Cache clean/write-back/invalidate moves or discards cache data.

These are different operations. A `fence` alone does not make a dirty CPU cache line visible to a DMA engine. Use the ESP-IDF cache synchronization API for CPU/DMA handoff.

## Atomics

Use toolchain atomics or C11 primitives unless a special primitive is proven necessary. A minimal exchange loop conceptually looks like:

```asm
1:  lr.w.aq t0, (a0)
    bne     t0, a1, 2f
    sc.w.rl t1, a2, (a0)
    bnez    t1, 1b
2:
```

On ESP32-P4, many events clear the reservation, and the `.aq`/`.rl` encoding bits are documented as ignored because ordering is already guaranteed by the core. Keep the suffixes for portable intent. Do no loads, stores, calls, long branches, or unrelated work between `lr.w` and `sc.w`; expose retry/failure semantics at the C boundary.

## Hardware loops

The custom loop family can remove loop-control instructions from tight counted loops. The official sources use two naming layers: conceptual `lp.*` names in the TRM and `esp.lp.*` mnemonics in assembler tests.

Before using it:

1. Compile a minimal assembler probe with the installed GCC/binutils.
2. Disassemble it and confirm the encoding.
3. Test zero, one, maximum, nested, interrupted, and context-switched iterations.
4. Confirm the matching IDF revision handles its documented HP loop-state workaround.

The `lp.setup.beqz` form is listed by the TRM but was not present in the IDF decoder corpus examined for this research. That discrepancy is a reason to probe, not to guess syntax.

## PIE assembly guardrails

PIE offers eight 128-bit `q` registers plus custom accumulators and state. The IDF decoder test proves hundreds of accepted `esp.*` spellings, including load/store, arithmetic, shifts, compares, pack/unpack, multiply/MAC, accumulator, FFT/complex, activation, and quantization families. It does not provide a public stable ABI or full semantic/latency specification.

For every PIE kernel:

- build only with the Espressif GCC toolchain version it was tested against;
- keep custom state within one wrapper and declare/preserve clobbers according to that toolchain's documented convention;
- align normal vectors to 16 bytes even if an unaligned instruction exists;
- separate a vector-aligned body from scalar prefix/tail handling;
- prove signedness, widening, rounding, saturation, accumulator reset, and final extraction;
- compare bit-for-bit against a scalar model where exact arithmetic is intended;
- test interruption/context switching and both HP cores;
- retain a scalar C fallback selected at build time.

Do not invent an `.insn` encoding from a mnemonic list. Without an authoritative encoding/semantics source, that is not maintainable assembly.

## Scheduling an in-order core

Independent instructions can cover some latency; dependent chains cannot. A scalar dot-product skeleton should use multiple accumulators so a multiply result is not immediately consumed:

```text
load A0/B0 -> multiply P0
load A1/B1 -> multiply P1
accumulate P0 into S0
load A2/B2 -> multiply P2
accumulate P1 into S1
...
```

Unroll only until dependency coverage and branch reduction outweigh code size, register pressure, and I-cache effects. Thirty-one writable integer registers includes ABI pointers/counters/temporaries; spilling to the stack can erase the benefit of aggressive unrolling.

The branch predictor has finite BHT/BTB/RAS structures. Keep the hot loop compact and use hardware loops when validated. Measure `mcycle`, `minstret`, branch count, and mispredictions.

## Code and data placement

Assembly source does not by itself guarantee SPM/L2/IRAM placement. Use the ESP-IDF linker-fragment mechanism and inspect the map file:

```sh
rg 'kernel_name|\.text|\.literal' build/*.map
riscv32-esp-elf-objdump -h build/*.elf
```

Any routine required while flash/cache is unavailable must satisfy all ESP-IDF IRAM-safe rules, including every transitive callee and referenced constant/data object. A single flash-resident literal or helper defeats the placement.

## Verification harness

Each assembly kernel should have:

- a simple scalar C oracle compiled in a separate translation unit;
- exhaustive tests for small domains where feasible;
- seeded random and adversarial tests;
- canaries before/after every buffer;
- tests for every supported alignment and length modulo vector width;
- overlap tests if overlap is permitted, or assertions if forbidden;
- cycle and retired-instruction benchmarks with empty-harness subtraction;
- `objdump` checks for intended instructions and unexpected calls/spills;
- tests on every supported silicon revision—not only an emulator.

Store the compiler version, ELF attributes, disassembly, map excerpt, and benchmark metadata with the result. Custom assembly is a binary/toolchain contract and should be reviewed as such.
