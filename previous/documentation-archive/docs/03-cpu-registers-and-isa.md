# CPU registers and instruction sets

## What “number of registers” means

There is no single honest register count for this SoC. Four different counts are useful:

| Register class | Count | Scope |
|---|---:|---|
| Integer architectural registers | 32 per RISC-V core | `x0`-`x31`; `x0` is hard-wired zero |
| Floating-point architectural registers | 32 per HP core | `f0`-`f31`, 32 bits each |
| PIE vector registers | 8 per HP core | `q0`-`q7`, 128 bits each |
| HP core/CLIC/CLINT/PMP/PMA/debug register schemas in TRM | 137 numbered descriptions | Some schemas describe arrays, so this is not the number of instantiated registers |
| ESP-IDF v6.0.2 ESP32-P4 MMIO definition comments | 5,936 definitions in 102 of 104 `*_reg.h` files | Reproducible header census; includes indexed registers and ECO aliases, not 5,936 unique physical addresses |

The last two figures answer different questions. The TRM number covers the processor complex. The IDF header census covers the whole SoC. Peripheral blocks often have indexed channels, repeated instances, set/clear aliases, and revision overlays, so a “unique hardware register” total depends on the counting rule.

## Integer register file and ABI

All three CPUs use RV32 integer registers. The HP application ABI is `ilp32f`: integers, longs, and pointers are 32 bits; single-precision floating-point arguments/results may use F registers.

| ABI name | Register | Caller/callee rule | Conventional role |
|---|---:|---|---|
| `zero` | `x0` | fixed | constant zero |
| `ra` | `x1` | caller-saved | return address |
| `sp` | `x2` | callee-saved | stack pointer; keep 16-byte aligned at calls |
| `gp` | `x3` | unallocatable | global pointer |
| `tp` | `x4` | unallocatable | thread pointer |
| `t0`-`t2` | `x5`-`x7` | caller-saved | temporaries |
| `s0`/`fp`, `s1` | `x8`-`x9` | callee-saved | saved/frame pointer |
| `a0`-`a7` | `x10`-`x17` | caller-saved | arguments; `a0`/`a1` return values |
| `s2`-`s11` | `x18`-`x27` | callee-saved | saved registers |
| `t3`-`t6` | `x28`-`x31` | caller-saved | temporaries |

Assembly that calls C must preserve `sp`, `s0`-`s11`, `gp`, and `tp`, and must obey the 16-byte stack alignment rule. A leaf routine can avoid a stack frame entirely if it uses only caller-saved registers and makes no call.

The HP floating-point register convention is `ft0`-`ft7` (`f0`-`f7`), `fs0`-`fs1`, `fa0`-`fa7`, `fs2`-`fs11`, then `ft8`-`ft11`. Under `ilp32f`, `fs0`-`fs11` are callee-saved only for values no wider than the ABI floating-point width.

PIE `q` registers and its accumulator state are custom. Do not assume the standard RISC-V vector calling convention, and do not preserve `q` state by analogy with `v0`-`v31`. Keep PIE kernels in compiler-provided intrinsics/assembly wrappers whose clobber behavior is verified against the installed Espressif toolchain.

## Supported HP instruction architecture

The hardware documentation describes:

```text
RV32I + M + A + F + C + Zicsr + Zifencei
      + Zba + Zbb + Zbs
      + Zcb + Zcmp + Zcmt
      + XespLoop + XespV/PIE
```

This is **not** RV32G, because there is no double-precision `D` extension. It is also **not** the standard `V` vector extension.

ESP-IDF v6.0.2's baseline GCC architecture string is
`rv32imafc_zicsr_zifencei_zaamo_zalrsc`; current ESP32-P4 capability logic adds Zc, `xesploop`, and `xespv`. The same release does not add Zba/Zbb/Zbs to the default P4 compile flags even though the current hardware manuals list them. Treat this as a toolchain-configuration gap: inspect the final compiler command and disassembly before relying on automatic Zb emission.

The LP core implements RV32IMAC. Code compiled for HP PIE, F, Zc, or Zb must not be run on the LP core unless the specific instruction belongs to the LP core's ISA.

## Exact standard instruction inventory

Pseudo-instructions such as `li`, `mv`, `nop`, `ret`, and `call` are assembler conveniences and expand to one or more real instructions below.

### RV32I base integer

- Upper immediate/control: `LUI`, `AUIPC`, `JAL`, `JALR`.
- Conditional branches: `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`.
- Loads: `LB`, `LH`, `LW`, `LBU`, `LHU`.
- Stores: `SB`, `SH`, `SW`.
- Immediate ALU: `ADDI`, `SLTI`, `SLTIU`, `XORI`, `ORI`, `ANDI`, `SLLI`, `SRLI`, `SRAI`.
- Register ALU: `ADD`, `SUB`, `SLL`, `SLT`, `SLTU`, `XOR`, `SRL`, `SRA`, `OR`, `AND`.
- Ordering/environment: `FENCE`, `ECALL`, `EBREAK`.
- Separate standard extensions: `FENCE.I`; `CSRRW`, `CSRRS`, `CSRRC`, `CSRRWI`, `CSRRSI`, `CSRRCI`.

### M multiply/divide

`MUL`, `MULH`, `MULHSU`, `MULHU`, `DIV`, `DIVU`, `REM`, `REMU`.

The TRM states that multiply takes two cycles and divide takes 1-19 cycles with no data hazard. Interpret that as core latency/behavior, not proof that an arbitrary multiply loop retires one result every two cycles: loads, dependencies, branches, and issue constraints still matter.

### A atomics

`LR.W`, `SC.W`, and `.W` forms of `AMOSWAP`, `AMOADD`, `AMOXOR`, `AMOAND`, `AMOOR`, `AMOMIN`, `AMOMAX`, `AMOMINU`, and `AMOMAXU`.

Each atomic may carry `.aq`, `.rl`, or `.aqrl`. The HP-core TRM says the hardware already guarantees ordering and ignores the encoded acquire/release bits. Keep them in portable source because they communicate the language memory order and matter on other RISC-V implementations.

The ESP32-P4 reservation is deliberately fragile. Interrupts, exceptions, debug entry, another load/store, selected control-flow/system instructions, and other events can clear it; a reserved section must not cross a 64-byte critical region. Keep `LR.W`/`SC.W` sequences minimal and retry in a bounded, contention-aware loop.

### F single-precision floating point

- Memory: `FLW`, `FSW`.
- Fused: `FMADD.S`, `FMSUB.S`, `FNMSUB.S`, `FNMADD.S`.
- Arithmetic: `FADD.S`, `FSUB.S`, `FMUL.S`, `FDIV.S`, `FSQRT.S`.
- Sign/min/max: `FSGNJ.S`, `FSGNJN.S`, `FSGNJX.S`, `FMIN.S`, `FMAX.S`.
- Integer conversions: `FCVT.W.S`, `FCVT.WU.S`, `FCVT.S.W`, `FCVT.S.WU`.
- Compare/classify: `FEQ.S`, `FLT.S`, `FLE.S`, `FCLASS.S`.
- Bit moves: `FMV.X.W`, `FMV.W.X`.

The rounding mode comes from the instruction or `frm` in `fcsr`; exception flags accumulate in `fflags`. The current IDF capability header records a silicon/toolchain workaround for an illegal-instruction issue involving FPU extension state. Do not manually alter extension state around normal compiled C without reading the matching IDF startup/context-switch code.

### C compressed instructions

The applicable RV32 compressed forms are `C.ADDI4SPN`, `C.LW`, `C.SW`, `C.NOP`, `C.ADDI`, `C.JAL`, `C.LI`, `C.ADDI16SP`, `C.LUI`, `C.SRLI`, `C.SRAI`, `C.ANDI`, `C.SUB`, `C.XOR`, `C.OR`, `C.AND`, `C.J`, `C.BEQZ`, `C.BNEZ`, `C.SLLI`, `C.LWSP`, `C.JR`, `C.MV`, `C.EBREAK`, `C.JALR`, `C.ADD`, and `C.SWSP`.

Compressed code reduces flash/cache traffic and often helps instruction-cache residency. It does not imply that two 16-bit instructions execute in one cycle.

### Zb bit manipulation

- Zba address generation: `SH1ADD`, `SH2ADD`, `SH3ADD`.
- Zbb basic bit operations: `ANDN`, `ORN`, `XNOR`, `CLZ`, `CTZ`, `CPOP`, `MAX`, `MAXU`, `MIN`, `MINU`, `SEXT.B`, `SEXT.H`, `ZEXT.H`, `ROL`, `ROR`, `RORI`, `ORC.B`, `REV8`.
- Zbs single-bit operations: `BCLR`, `BCLRI`, `BEXT`, `BEXTI`, `BINV`, `BINVI`, `BSET`, `BSETI`.

These replace multi-instruction masks, shifts, counts, byte swaps, and address calculations. Confirm they appear in the linked binary because IDF v6.0.2's default P4 architecture flags do not currently advertise Zb.

### Zc code-size instructions

- Zcb: `C.LBU`, `C.LHU`, `C.LH`, `C.SB`, `C.SH`, `C.ZEXT.B`, `C.SEXT.B`, `C.ZEXT.H`, `C.SEXT.H`, `C.NOT`, `C.MUL`.
- Zcmp: `CM.PUSH`, `CM.POP`, `CM.POPRET`, `CM.POPRETZ`, `CM.MVA01S`, `CM.MVA01` as spelled by the ESP32-P4 TRM. The last spelling differs from the ratified Zcmp mnemonic `CM.MVSA01`; probe the installed assembler rather than assuming either spelling is accepted.
- Zcmt: `CM.JT`, `CM.JALT`.

ESP-IDF adds workarounds named `-mno-cm-push-reverse` and `-mno-cm-popret` for ESP32-P4. Let the shipped build system select these; hand-authored Zcmp prologues/epilogues require silicon-specific validation.

## Custom hardware-loop instructions

The TRM describes seven loop-control operations: `lp.setupi`, `lp.setup`, `lp.setup.beqz`, `lp.starti`, `lp.endi`, `lp.count`, and `lp.counti`. Espressif assembler tests spell the custom prefix as `esp.lp.*`.

Hardware loops remove the decrement/branch pair from a small counted loop. They are most useful when loop-control overhead is a significant fraction of the body. They do not make memory or dependency stalls disappear. The IDF capability header also identifies an HP hardware-loop state bug, so save/restore and interrupt behavior should be delegated to the matching ESP-IDF/toolchain support rather than invented locally.

## PIE register state and execution model

PIE is Espressif's 128-bit Processor Instruction Extension:

- Eight 128-bit vector registers, `q0`-`q7`.
- Lane views include 16 x 8-bit, 8 x 16-bit, or 4 x 32-bit elements.
- A 512-bit vector accumulator, `qacc`.
- A 40-bit scalar accumulator, `xacc`.
- Additional unaligned-load, rounding, shift, saturation, and miscellaneous state.
- Specified vector/scalar MAC capacity: 16 8-bit or 8 16-bit elements per cycle.

The public TRM v0.6 still marks the full PIE chapter as future content. The core overview is not an encoding-level programming manual. ESP-IDF v6.0.2 does, however, ship a GDB/decoder assembler corpus containing 360 unique `esp.*` mnemonics; see the [PIE inventory](appendix-pie-mnemonics.md). That inventory proves toolchain recognition in that snapshot, not semantics, latency, ABI stability, or support by LLVM.

## Control/status registers worth knowing

The complete CSR map belongs in the TRM because privilege, debug, CLIC, PMP/PMA, and custom ranges are extensive. For optimization and measurement, the most useful standard/custom-visible state is:

- `mcycle`/`mcycleh`: cycle counter.
- `minstret`/`minstreth`: retired instruction counter.
- `mhpmcounter8`: branch-misprediction counter when event `0x6` is selected.
- `mhpmcounter9`: conditional-branch counter when event `0x7` is selected.
- `mhpmcounter13`: store counter when event `0xB` is selected.
- `mhpmevent8`, `mhpmevent9`, `mhpmevent13`: event selectors.
- `mstatus`, `mie`, `mip`, `mtvec`, `mepc`, `mcause`, `mtval`: machine execution/interrupt/trap state.
- `pmpcfg*`, `pmpaddr*`: 32 protection regions on v3.x.
- PMA CSRs: memory attribute regions; current documentation specifies 16.
- `fcsr` (`frm` + `fflags`): floating-point control/status.

The TRM v0.6 has a contradictory address label around `mhpmevent8` (`0x308` in one table versus `0x328` in its heading). Use the installed header/toolchain symbolic name and verify emitted CSR numbers; do not transcribe that table blindly.

## Inspect what the compiler really used

```sh
riscv32-esp-elf-gcc -Q --help=target
riscv32-esp-elf-readelf -h -A build/app.elf
riscv32-esp-elf-objdump -drwC -S build/app.elf > build/app.disasm
rg '\b(esp\.|cm\.|sh[123]add|clz|ctz|cpop|rev8)\b' build/app.disasm
```

The ELF attributes, actual compile command, and final disassembly outrank an assumed `-march` string. Repeat this inspection after every IDF/toolchain upgrade.
