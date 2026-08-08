#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32P4

    .section .text.tracker_argmax_hwc16_s8, "ax", @progbits
    .p2align 2
    .global tracker_argmax_hwc16_s8
    .type tracker_argmax_hwc16_s8, @function

/*
 * size_t tracker_argmax_hwc16_s8(
 *     const int8_t *tensor, size_t pixels, size_t channel, int8_t *max_value);
 *
 * a0 tensor / return index, a1 pixels, a2 channel, a3 optional max_value.
 * This scalar RV32 baseline intentionally avoids undocumented PIE semantics.
 */
tracker_argmax_hwc16_s8:
    beqz    a0, .Linvalid
    beqz    a1, .Linvalid
    li      t0, 16
    bgeu    a2, t0, .Linvalid

    add     t0, a0, a2
    lb      t1, 0(t0)
    li      t2, 0
    li      t3, 1

.Lscan:
    bgeu    t3, a1, .Ldone
    addi    t0, t0, 16
    lb      t4, 0(t0)
    ble     t4, t1, .Lnext
    mv      t1, t4
    mv      t2, t3

.Lnext:
    addi    t3, t3, 1
    j       .Lscan

.Ldone:
    beqz    a3, .Lreturn
    sb      t1, 0(a3)

.Lreturn:
    mv      a0, t2
    ret

.Linvalid:
    li      a0, -1
    ret

    .size tracker_argmax_hwc16_s8, .-tracker_argmax_hwc16_s8
    .section .note.GNU-stack, "", @progbits

#endif
