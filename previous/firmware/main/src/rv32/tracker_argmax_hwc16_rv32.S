#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32P4

    .section .text.tracker_argmax_hwc16, "ax", @progbits
    .p2align 2
    .global tracker_argmax_hwc16
    .type tracker_argmax_hwc16, @function

/* a0: HWC16 data, a1: pixel count, a2: maximum output. */
tracker_argmax_hwc16:
    lb      t0, 0(a0)
    li      t1, 0
    li      t2, 1
1:
    bgeu    t2, a1, 3f
    addi    a0, a0, 16
    lb      t3, 0(a0)
    ble     t3, t0, 2f
    mv      t0, t3
    mv      t1, t2
2:
    addi    t2, t2, 1
    j       1b
3:
    sb      t0, 0(a2)
    mv      a0, t1
    ret

    .size tracker_argmax_hwc16, .-tracker_argmax_hwc16
    .section .note.GNU-stack, "", @progbits

#endif
