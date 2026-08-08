#include "tracker/preprocess.h"

#include <stdint.h>

void tracker_rgb888_center_s8_scalar(int8_t *dst, const uint8_t *src, size_t bytes)
{
    size_t index;
    const uintptr_t dst_address = (uintptr_t)dst;
    const uintptr_t src_address = (uintptr_t)src;

    if ((dst == NULL) || (src == NULL)) {
        return;
    }

    if ((dst_address > src_address) && ((dst_address - src_address) < bytes)) {
        for (index = bytes; index > 0U; --index) {
            dst[index - 1U] = (int8_t)((int16_t)src[index - 1U] - 128);
        }
    } else {
        for (index = 0U; index < bytes; ++index) {
            dst[index] = (int8_t)((int16_t)src[index] - 128);
        }
    }
}

static bool ranges_overlap(const void *left, size_t left_bytes, const void *right, size_t right_bytes)
{
    const uintptr_t left_address = (uintptr_t)left;
    const uintptr_t right_address = (uintptr_t)right;

    if ((left_bytes == 0U) || (right_bytes == 0U)) {
        return false;
    }
    if (left_address <= right_address) {
        return (right_address - left_address) < left_bytes;
    }
    return (left_address - right_address) < right_bytes;
}

bool tracker_rgb888_to_centered_s8(const tracker_tensor_view_t *src, tracker_tensor_view_t *dst)
{
    size_t active_row_bytes;
    size_t src_span_bytes;
    size_t dst_span_bytes;
    uint32_t row;
    const uint8_t *src_bytes;
    int8_t *dst_bytes;

    if (!tracker_tensor_view_is_valid(src) || !tracker_tensor_view_is_valid(dst) ||
        (src->element_type != TRACKER_ELEMENT_U8) || (dst->element_type != TRACKER_ELEMENT_S8) ||
        (src->height != dst->height) || (src->width != dst->width) || (src->channels != 3U) ||
        (dst->channels != 3U)) {
        return false;
    }

    active_row_bytes = (size_t)src->width * 3U;
    src_span_bytes = ((size_t)src->height - 1U) * src->row_stride_bytes + active_row_bytes;
    dst_span_bytes = ((size_t)dst->height - 1U) * dst->row_stride_bytes + active_row_bytes;
    if (src->data == dst->data) {
        if (src->row_stride_bytes != dst->row_stride_bytes) {
            return false;
        }
    } else if (ranges_overlap(src->data, src_span_bytes, dst->data, dst_span_bytes)) {
        return false;
    }

    src_bytes = (const uint8_t *)src->data;
    dst_bytes = (int8_t *)dst->data;
    for (row = 0U; row < src->height; ++row) {
        tracker_rgb888_center_s8_scalar(dst_bytes + ((size_t)row * dst->row_stride_bytes),
                                        src_bytes + ((size_t)row * src->row_stride_bytes),
                                        active_row_bytes);
    }
    return true;
}
