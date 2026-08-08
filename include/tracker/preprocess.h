#ifndef TRACKER_PREPROCESS_H
#define TRACKER_PREPROCESS_H

#include "tracker/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Exact scalar oracle for a model trained with q = pixel - 128.
 * All overlap patterns are supported, including exact in-place conversion.
 */
void tracker_rgb888_center_s8_scalar(int8_t *dst, const uint8_t *src, size_t bytes);

/*
 * Converts equally-sized HWC3/U8 and HWC3/S8 views, preserving row padding.
 * Views may be exactly in-place when their row strides match. Other overlap is
 * rejected because differing row geometry can overwrite a future source row.
 */
bool tracker_rgb888_to_centered_s8(const tracker_tensor_view_t *src, tracker_tensor_view_t *dst);

#ifdef __cplusplus
}
#endif

#endif
