#include "tracker/tensor.h"

#include <stdint.h>

size_t tracker_element_size(tracker_element_type_t type)
{
    switch (type) {
    case TRACKER_ELEMENT_U8:
    case TRACKER_ELEMENT_S8:
        return 1U;
    case TRACKER_ELEMENT_S16:
        return 2U;
    case TRACKER_ELEMENT_S32:
        return 4U;
    default:
        return 0U;
    }
}

static bool multiply_size(size_t left, size_t right, size_t *result)
{
    if ((left != 0U) && (right > (SIZE_MAX / left))) {
        return false;
    }
    *result = left * right;
    return true;
}

bool tracker_tensor_view_is_valid(const tracker_tensor_view_t *view)
{
    size_t element_bytes;
    size_t row_elements;
    size_t minimum_row_bytes;
    size_t preceding_rows;
    size_t required_bytes;

    if ((view == NULL) || (view->data == NULL) || (view->height == 0U) || (view->width == 0U) ||
        (view->channels == 0U)) {
        return false;
    }

    element_bytes = tracker_element_size(view->element_type);
    if ((element_bytes == 0U) || !multiply_size(view->width, view->channels, &row_elements) ||
        !multiply_size(row_elements, element_bytes, &minimum_row_bytes) ||
        (view->row_stride_bytes < minimum_row_bytes)) {
        return false;
    }

    if (!multiply_size((size_t)view->height - 1U, view->row_stride_bytes, &preceding_rows) ||
        (minimum_row_bytes > (SIZE_MAX - preceding_rows))) {
        return false;
    }
    required_bytes = preceding_rows + minimum_row_bytes;
    return view->size_bytes >= required_bytes;
}
