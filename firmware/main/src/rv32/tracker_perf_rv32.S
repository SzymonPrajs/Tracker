#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32P4

    .section .text.tracker_perf_cycles, "ax", @progbits
    .p2align 2
    .global tracker_perf_cycles
    .type tracker_perf_cycles, @function

/* uint64_t tracker_perf_cycles(void); */
tracker_perf_cycles:
1:
    csrr    a1, mcycleh
    csrr    a0, mcycle
    csrr    t0, mcycleh
    bne     a1, t0, 1b
    ret

    .size tracker_perf_cycles, .-tracker_perf_cycles

    .section .text.tracker_perf_instructions, "ax", @progbits
    .p2align 2
    .global tracker_perf_instructions
    .type tracker_perf_instructions, @function

/* uint64_t tracker_perf_instructions(void); */
tracker_perf_instructions:
1:
    csrr    a1, minstreth
    csrr    a0, minstret
    csrr    t0, minstreth
    bne     a1, t0, 1b
    ret

    .size tracker_perf_instructions, .-tracker_perf_instructions

    .section .note.GNU-stack, "", @progbits

#endif
