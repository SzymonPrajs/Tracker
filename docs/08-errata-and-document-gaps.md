# Silicon revisions, errata, and document gaps

## Revision policy

Target `ESP32-P4NRW32X` v3.x and prefer v3.2 for new hardware. Read the actual package/eFuse revision at runtime and log it with benchmark and fault reports. Do not infer silicon revision only from the board product name.

The current public errata is v1.3. The table below focuses on algorithm, memory, security, DMA, and bring-up consequences; consult the errata PDF for exact affected revisions and official workarounds.

## Current errata ledger

| ID | Affected | Consequence | Design action |
|---|---|---|---|
| RMT-176 | v0.0/v1.0 | RMT behavior defect | Use IDF workaround or v3.x; do not generalize old-board timing results |
| I2C-308 | v0.0/v1.0 | I2C behavior defect | Use IDF workaround or v3.x |
| APM-560 | through v3.0 | access-permission behavior defect | Prefer v3.1+; use official security/access workaround |
| MSPI-749 | v3.0 | first flash/PSRAM access after boot/deep sleep may fail | Prefer v3.1+; retain official initialization workaround for v3.0 |
| MSPI-750 | v3.0 | overlapping unaligned 1-/2-byte DMA read may return old external-RAM data | Align DMA accesses to at least 4 bytes and synchronize ownership; prefer v3.1+ |
| MSPI-751 | v3.0 | certain flash/PSRAM clock-ratio overlapping access can return old data | Prefer v3.1+; use supported clocks/IDF workaround |
| ROM-764 | v3.0 | secure-boot buffer handling defect | Prefer v3.1+ and follow official secure-boot workaround |
| Analog-765 | v3.0 | analog defect | Prefer v3.1+; consult errata for affected use |
| DMA-767 | v3.0 | channel-0 ID overlap | Prefer v3.1+ or use official channel/workaround constraints |
| ROM-770 | v3.1 | secure-download flash power handling defect | Prefer v3.2 or use official sequence/workaround |
| ROM-816 | v3.2 | repeated flash power-on sequence can hang | Still open on v3.2; ensure software never issues the prohibited double sequence |
| ECDSA-836 | v3.0-v3.2 | ECDSA behavior/security defect | Use official software/workflow mitigation; do not certify a design from raw accelerator tests |
| ECDSA-837 | older revisions | ECDSA behavior/security defect | Follow official mitigation on affected older silicon |

Security errata should be reviewed from the source document during threat modeling; the short descriptions here are navigation aids, not a security approval.

## ESP-IDF capability-level workarounds

ESP-IDF v6.0.2's ESP32-P4 capability/build logic exposes additional constraints that are not presented as a simple public errata row:

- Zcmp compiler workarounds `-mno-cm-push-reverse` and `-mno-cm-popret`.
- An HP hardware-loop state bug flag.
- An FPU extension illegal-instruction bug flag.
- PIE coprocessor assembly support marked GCC-only.

This is why direct assembly should be built through the target IDF release. Reconstructing `-march` and workaround flags manually is brittle.

## Official-document conflicts and omissions

### PIE is not fully documented

TRM v0.6 includes a PIE overview but says the full PIE chapter will be added later. It does not publicly provide a complete stable instruction encoding, semantics, latency table, state-saving ABI, or intrinsic catalogue. The 360-mnemonic ESP-IDF assembler corpus is useful evidence of tool support, not a substitute for that missing specification.

Consequence: production PIE kernels must be tied to a tested Espressif GCC version, disassembled, exhaustively compared with a scalar oracle, and retested after upgrades.

### Custom-extension naming differs

Current materials use both `XespLoop`/`XespV` and older/internal `Xhwlp`/`Xai` terminology. Assembler spellings use `esp.lp.*` and `esp.*`. Preserve all names in searches; use the build's accepted spelling in code.

### Zb hardware versus default compiler flags

The datasheet/TRM list Zba, Zbb, and Zbs. ESP-IDF v6.0.2's base target flags and P4 capability header do not add Zb to the default `-march`. Therefore:

- hardware availability is specified;
- automatic compiler emission is not enabled by that default configuration;
- explicit enablement must be validated against assembler, linker, actual silicon, and IDF workarounds.

### SPM latency conflict

TRM v0.6 says HP SPM access finishes in two cycles. Current ESP-IDF programming documentation says one cycle without parity and four cycles with parity. The datasheet gives the 400 MHz domain but not a decisive load-to-use figure. Benchmark both parity modes on the selected revision before scheduling by hand.

### HPM CSR address typo

TRM v0.6 labels `mhpmevent8` inconsistently: a table uses `0x308`, while the register heading uses `0x328`, the standard slot expected for event 8. Use symbolic CSR support and disassembly rather than the contradictory table cell.

### Hardware-loop operation mismatch

The TRM describes seven operations including `lp.setup.beqz`; the inspected ESP-IDF decoder test corpus contains six `esp.lp.*` families and omits that form. Compile a probe with the installed toolchain before using it.

### L2 “available” versus application-usable RAM

With 128 KiB L2 cache, hardware arithmetic leaves 640 KiB L2 RAM. The current v3 linker region exposes 581,568 bytes before application sections and runtime allocations. Never budget from 640 KiB as if it were free heap.

## Pre-release status

The public TRM v0.6 identifies itself as a pre-release document. The datasheet, TRM, IDF, and errata can evolve independently. Every algorithm design record should pin:

- datasheet and TRM versions/dates;
- errata version;
- IDF version and commit;
- compiler/binutils version;
- part number and silicon revision;
- configuration affecting CPU, PSRAM, cache, parity, and power management.

## Upgrade checklist

When upgrading ESP-IDF/toolchain or silicon:

1. Diff the target `soc_caps.h`, project include flags, linker memory layout, Kconfig cache options, and relevant `*_reg.h` files.
2. Re-read current errata and resolved/new issue rows.
3. Rebuild assembler probes for Zb, Zc, hardware loops, and every PIE mnemonic used.
4. Compare ELF attributes and disassembly.
5. Re-run correctness vectors, DMA/cache stress, interruption/context-switch tests, and performance matrix.
6. Update the source matrix in [sources](sources.md) and retain old results with their original provenance.

Do not silently merge benchmark results across revisions. A fixed erratum, new compiler scheduler, cache default, or linker reservation can change both correctness and speed.
