#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tracker.h"
#include "tracker_target_kernels.h"

static int8_t head[3][3][16] __attribute__((aligned(16)));

void app_main(void)
{
    memset(head, INT8_MIN, sizeof(head));
    head[1][1][0] = 100;
    head[1][1][1] = 32;
    head[1][1][2] = -16;

    tracker_result_t result;
    tracker_decode(&head[0][0][0], 9, 3, 7, &result);

    int8_t maximum;
    const uint64_t cycles = tracker_perf_cycles();
    const uint64_t instructions = tracker_perf_instructions();
    const size_t index = tracker_argmax_hwc16(&head[0][0][0], 9, &maximum);

    printf(
        "index=%zu value=%d x=%" PRId32 " y=%" PRId32 " cycles=%" PRIu64
        " instructions=%" PRIu64 "\n",
        index, maximum, result.x_q16, result.y_q16,
        tracker_perf_cycles() - cycles,
        tracker_perf_instructions() - instructions
    );
}
