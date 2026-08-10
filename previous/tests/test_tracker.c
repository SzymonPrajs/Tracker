#include "tracker.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    const uint8_t rgb[] = {0, 127, 128, 255};
    int8_t centered[4];
    tracker_center_rgb(rgb, centered, sizeof(rgb));
    assert(centered[0] == -128 && centered[1] == -1);
    assert(centered[2] == 0 && centered[3] == 127);

    int8_t head[3][3][16];
    memset(head, INT8_MIN, sizeof(head));
    head[1][1][0] = 100;
    head[1][1][1] = 32;
    head[1][1][2] = -16;
    tracker_result_t result;
    tracker_decode(&head[0][0][0], 9, 3, 7, &result);
    assert(result.index == 4 && result.score == 100);
    assert(result.x_q16 == 7 * 65536);
    assert(result.y_q16 == 5 * 65536 + 32768);
    puts("ok");
}
