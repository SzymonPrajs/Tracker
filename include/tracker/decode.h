#ifndef TRACKER_DECODE_H
#define TRACKER_DECODE_H

#include "tracker/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t index;
    uint32_t x;
    uint32_t y;
    int8_t score;
} tracker_peak_t;

typedef struct {
    int32_t x_q16;
    int32_t y_q16;
    tracker_peak_t peak;
    uint32_t local_weight_sum;
} tracker_centroid_t;

/* Finds channel zero's first row-major maximum in an S8 HWC16 tensor. */
bool tracker_argmax_channel0_hwc16(const tracker_tensor_view_t *head, tracker_peak_t *peak);

/*
 * Decodes a local 3x3 weighted centroid and peak offsets.
 *
 * Channels 1 and 2 are signed offsets relative to the heatmap-cell centre,
 * encoded with offset_fraction_bits fractional bits. The returned coordinates
 * are Q16.16 input-image pixels after multiplication by output_stride. Local
 * weights are scores above the 3x3 minimum; a flat neighborhood falls back to
 * the argmax cell. Coordinates outside Q16.16 int32 range saturate.
 */
bool tracker_decode_centroid_hwc16(const tracker_tensor_view_t *head,
                                   uint32_t output_stride,
                                   uint8_t offset_fraction_bits,
                                   tracker_centroid_t *centroid);

#ifdef __cplusplus
}
#endif

#endif
