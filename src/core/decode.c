#include "tracker/decode.h"

#include <limits.h>
#include <stdint.h>

static const int8_t *head_pixel(const tracker_tensor_view_t *head, uint32_t x, uint32_t y)
{
    const uint8_t *row = (const uint8_t *)head->data + ((size_t)y * head->row_stride_bytes);
    return (const int8_t *)(row + ((size_t)x * 16U));
}

static int32_t clamp_i64_to_i32(int64_t value)
{
    if (value > INT32_MAX) {
        return INT32_MAX;
    }
    if (value < INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)value;
}

static bool valid_head(const tracker_tensor_view_t *head)
{
    return tracker_tensor_view_is_valid(head) && (head->element_type == TRACKER_ELEMENT_S8) &&
           (head->channels == 16U) && (head->width <= UINT16_MAX) && (head->height <= UINT16_MAX);
}

bool tracker_argmax_channel0_hwc16(const tracker_tensor_view_t *head, tracker_peak_t *peak)
{
    uint32_t x;
    uint32_t y;
    tracker_peak_t best;

    if (!valid_head(head) || (peak == NULL)) {
        return false;
    }

    best.index = 0U;
    best.x = 0U;
    best.y = 0U;
    best.score = head_pixel(head, 0U, 0U)[0];
    for (y = 0U; y < head->height; ++y) {
        for (x = 0U; x < head->width; ++x) {
            const int8_t score = head_pixel(head, x, y)[0];
            if (score > best.score) {
                best.index = (y * head->width) + x;
                best.x = x;
                best.y = y;
                best.score = score;
            }
        }
    }

    *peak = best;
    return true;
}

bool tracker_decode_centroid_hwc16(const tracker_tensor_view_t *head,
                                   uint32_t output_stride,
                                   uint8_t offset_fraction_bits,
                                   tracker_centroid_t *centroid)
{
    tracker_peak_t peak;
    uint32_t x_begin;
    uint32_t x_end;
    uint32_t y_begin;
    uint32_t y_end;
    uint32_t x;
    uint32_t y;
    int16_t minimum = INT8_MAX;
    uint64_t weighted_x = 0U;
    uint64_t weighted_y = 0U;
    uint32_t weight_sum = 0U;
    int64_t x_q16;
    int64_t y_q16;
    const int8_t *peak_pixel;
    int64_t offset_scale_q16;

    if ((centroid == NULL) || (output_stride == 0U) || (output_stride > UINT16_MAX) ||
        (offset_fraction_bits > 15U) ||
        !tracker_argmax_channel0_hwc16(head, &peak)) {
        return false;
    }

    x_begin = (peak.x == 0U) ? 0U : peak.x - 1U;
    y_begin = (peak.y == 0U) ? 0U : peak.y - 1U;
    x_end = (peak.x + 1U < head->width) ? peak.x + 1U : head->width - 1U;
    y_end = (peak.y + 1U < head->height) ? peak.y + 1U : head->height - 1U;

    for (y = y_begin; y <= y_end; ++y) {
        for (x = x_begin; x <= x_end; ++x) {
            const int16_t score = head_pixel(head, x, y)[0];
            if (score < minimum) {
                minimum = score;
            }
        }
    }

    for (y = y_begin; y <= y_end; ++y) {
        for (x = x_begin; x <= x_end; ++x) {
            const int16_t score = head_pixel(head, x, y)[0];
            const uint32_t weight = (uint32_t)(score - minimum);
            weight_sum += weight;
            weighted_x += (uint64_t)weight * x;
            weighted_y += (uint64_t)weight * y;
        }
    }

    if (weight_sum == 0U) {
        /* A flat neighborhood has no sub-cell evidence. Retain the first argmax. */
        weighted_x = peak.x;
        weighted_y = peak.y;
        weight_sum = 1U;
    }
    x_q16 = (int64_t)(((weighted_x << 16U) + (weight_sum / 2U)) / weight_sum) + 32768;
    y_q16 = (int64_t)(((weighted_y << 16U) + (weight_sum / 2U)) / weight_sum) + 32768;

    peak_pixel = head_pixel(head, peak.x, peak.y);
    offset_scale_q16 = INT64_C(65536) / (INT64_C(1) << offset_fraction_bits);
    x_q16 += (int64_t)peak_pixel[1] * offset_scale_q16;
    y_q16 += (int64_t)peak_pixel[2] * offset_scale_q16;
    x_q16 *= output_stride;
    y_q16 *= output_stride;

    centroid->x_q16 = clamp_i64_to_i32(x_q16);
    centroid->y_q16 = clamp_i64_to_i32(y_q16);
    centroid->peak = peak;
    centroid->local_weight_sum = weight_sum;
    return true;
}
