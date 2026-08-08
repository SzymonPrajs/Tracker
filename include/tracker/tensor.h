#ifndef TRACKER_TENSOR_H
#define TRACKER_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRACKER_ELEMENT_U8 = 1,
    TRACKER_ELEMENT_S8 = 2,
    TRACKER_ELEMENT_S16 = 3,
    TRACKER_ELEMENT_S32 = 4,
} tracker_element_type_t;

typedef struct {
    void *data;
    size_t size_bytes;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    size_t row_stride_bytes;
    tracker_element_type_t element_type;
} tracker_tensor_view_t;

size_t tracker_element_size(tracker_element_type_t type);

/* Validates dimensions, row stride, and backing-store capacity. */
bool tracker_tensor_view_is_valid(const tracker_tensor_view_t *view);

#ifdef __cplusplus
}
#endif

#endif
