#include "tracker.h"

#include <limits.h>

enum { CHANNELS = 16, OUTPUT_STRIDE = 4 };

void tracker_center_rgb(const uint8_t *input, int8_t *output, size_t bytes)
{
    for (size_t i = 0; i < bytes; ++i) {
        output[i] = (int8_t)(input[i] ^ 0x80U);
    }
}

void tracker_decode(
    const int8_t *head, size_t pixels, size_t width, unsigned offset_q,
    tracker_result_t *result
)
{
    size_t best = 0;
    for (size_t i = 1; i < pixels; ++i) {
        if (head[i * CHANNELS] > head[best * CHANNELS]) {
            best = i;
        }
    }

    const size_t height = pixels / width;
    const size_t cx = best % width;
    const size_t cy = best / width;
    const size_t x0 = cx ? cx - 1 : 0;
    const size_t y0 = cy ? cy - 1 : 0;
    const size_t x1 = cx + 1 < width ? cx + 1 : cx;
    const size_t y1 = cy + 1 < height ? cy + 1 : cy;
    int minimum = INT8_MAX;
    for (size_t y = y0; y <= y1; ++y) {
        for (size_t x = x0; x <= x1; ++x) {
            const int score = head[(y * width + x) * CHANNELS];
            if (score < minimum) minimum = score;
        }
    }

    uint32_t sum = 0;
    uint64_t sx = 0, sy = 0;
    for (size_t y = y0; y <= y1; ++y) {
        for (size_t x = x0; x <= x1; ++x) {
            const uint32_t weight = (uint32_t)(head[(y * width + x) * CHANNELS] - minimum);
            sum += weight;
            sx += weight * x;
            sy += weight * y;
        }
    }
    if (!sum) {
        sum = 1;
        sx = cx;
        sy = cy;
    }

    const int8_t *peak = head + best * CHANNELS;
    int64_t x = (int64_t)(((sx << 16) + sum / 2) / sum) + 32768;
    int64_t y = (int64_t)(((sy << 16) + sum / 2) / sum) + 32768;
    x += (int64_t)peak[1] * (65536 >> offset_q);
    y += (int64_t)peak[2] * (65536 >> offset_q);
    result->index = best;
    result->score = peak[0];
    result->x_q16 = (int32_t)(x * OUTPUT_STRIDE);
    result->y_q16 = (int32_t)(y * OUTPUT_STRIDE);
}
