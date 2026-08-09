#ifndef TRACKER_TARGET_KERNELS_H
#define TRACKER_TARGET_KERNELS_H

#include <stddef.h>
#include <stdint.h>

uint64_t tracker_perf_cycles(void);
uint64_t tracker_perf_instructions(void);
size_t tracker_argmax_hwc16(const int8_t *head, size_t pixels, int8_t *maximum);

#endif
