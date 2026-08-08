#ifndef TRACKER_TARGET_KERNELS_H
#define TRACKER_TARGET_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
uint64_t tracker_perf_cycles(void);
uint64_t tracker_perf_instructions(void);

/*
 * Return the first pixel whose selected signed INT8 channel is maximal.
 *
 * tensor must contain pixels HWC records, each exactly 16 bytes wide.
 * Returns SIZE_MAX without reading tensor when pixels is zero, tensor is NULL,
 * or channel is outside [0, 15]. max_value may be NULL.
 */
size_t tracker_argmax_hwc16_s8(
    const int8_t *tensor,
    size_t pixels,
    size_t channel,
    int8_t *max_value
);

#ifdef __cplusplus
}
#endif

#endif
