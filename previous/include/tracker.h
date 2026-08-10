#ifndef TRACKER_H
#define TRACKER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t index;
    int8_t score;
    int32_t x_q16;
    int32_t y_q16;
} tracker_result_t;

void tracker_center_rgb(const uint8_t *input, int8_t *output, size_t bytes);

/* HWC16 head and output stride four. */
void tracker_decode(
    const int8_t *head, size_t pixels, size_t width, unsigned offset_q,
    tracker_result_t *result
);

#endif
