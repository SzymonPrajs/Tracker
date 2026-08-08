#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "sdkconfig.h"

#include "tracker/alpha_beta.h"
#include "tracker/decode.h"
#include "tracker/mailbox.h"
#include "tracker/preprocess.h"
#include "tracker/tensor.h"
#include "tracker_target_kernels.h"

enum {
    HWC16_CHANNELS = 16,
    SMOKE_PIXELS = 4,
    SMOKE_CHANNEL = 3,
    PORTABLE_HEAD_WIDTH = 3,
    PORTABLE_HEAD_HEIGHT = 3,
};

static int8_t s_smoke_tensor[SMOKE_PIXELS][HWC16_CHANNELS] __attribute__((aligned(16)));

typedef struct {
    tracker_centroid_t centroid;
    tracker_mailbox_stats_t mailbox;
    tracker_motion_t motion;
} portable_smoke_result_t;

static tracker_tensor_view_t tensor_view(void *data,
                                         size_t size_bytes,
                                         uint32_t height,
                                         uint32_t width,
                                         uint32_t channels,
                                         size_t row_stride_bytes,
                                         tracker_element_type_t element_type)
{
    const tracker_tensor_view_t view = {
        data,
        size_bytes,
        height,
        width,
        channels,
        row_stride_bytes,
        element_type,
    };
    return view;
}

static portable_smoke_result_t run_portable_core_smoke(void)
{
    uint8_t rgb[8] = {0U, 127U, 128U, 255U, 1U, 129U, 77U, 88U};
    int8_t centered[8] = {11, 11, 11, 11, 11, 11, 11, 11};
    const int8_t expected_centered[6] = {-128, -1, 0, 127, -127, 1};
    tracker_tensor_view_t rgb_view =
        tensor_view(rgb, sizeof(rgb), 1U, 2U, 3U, sizeof(rgb), TRACKER_ELEMENT_U8);
    tracker_tensor_view_t centered_view =
        tensor_view(centered, sizeof(centered), 1U, 2U, 3U, sizeof(centered), TRACKER_ELEMENT_S8);

    assert(tracker_rgb888_to_centered_s8(&rgb_view, &centered_view));
    assert(memcmp(centered, expected_centered, sizeof(expected_centered)) == 0);
    assert(centered[6] == 11);
    assert(centered[7] == 11);

    int8_t head_data[PORTABLE_HEAD_HEIGHT][PORTABLE_HEAD_WIDTH][HWC16_CHANNELS]
        __attribute__((aligned(16)));
    memset(head_data, INT8_MIN, sizeof(head_data));
    head_data[1][1][0] = 100;
    head_data[1][1][1] = 32;
    head_data[1][1][2] = -16;
    tracker_tensor_view_t head = tensor_view(
        head_data,
        sizeof(head_data),
        PORTABLE_HEAD_HEIGHT,
        PORTABLE_HEAD_WIDTH,
        HWC16_CHANNELS,
        sizeof(head_data[0]),
        TRACKER_ELEMENT_S8
    );

    portable_smoke_result_t result = {0};
    assert(tracker_decode_centroid_hwc16(&head, 4U, 7U, &result.centroid));
    assert(result.centroid.peak.index == 4U);
    assert(result.centroid.peak.score == 100);
    assert(result.centroid.local_weight_sum == 237U);
    assert(result.centroid.x_q16 == (7 * 65536));
    assert(result.centroid.y_q16 == ((5 * 65536) + 32768));

    int8_t assembly_maximum = 0;
    const size_t assembly_index = tracker_argmax_hwc16_s8(
        &head_data[0][0][0],
        PORTABLE_HEAD_WIDTH * PORTABLE_HEAD_HEIGHT,
        0U,
        &assembly_maximum
    );
    assert(assembly_index == result.centroid.peak.index);
    assert(assembly_maximum == result.centroid.peak.score);

    tracker_mailbox_t mailbox;
    tracker_frame_t first = {.sequence = 1U};
    tracker_frame_t second = {.sequence = 2U};
    tracker_frame_t latest = {0};
    tracker_mailbox_init(&mailbox, NULL, NULL);
    tracker_mailbox_publish(&mailbox, &first);
    tracker_mailbox_publish(&mailbox, &second);
    assert(tracker_mailbox_take_latest(&mailbox, &latest));
    assert(latest.sequence == 2U);
    result.mailbox = tracker_mailbox_get_stats(&mailbox);
    assert(result.mailbox.published == 2U);
    assert(result.mailbox.replaced == 1U);
    assert(result.mailbox.taken == 1U);
    tracker_mailbox_destroy(&mailbox);

    const tracker_alpha_beta_config_t filter_config = {16384U, 8192U, 1500000U};
    tracker_alpha_beta_t filter;
    assert(tracker_alpha_beta_init(&filter, &filter_config));
    assert(tracker_alpha_beta_update(&filter, 0, 0, 0, &result.motion));
    assert(tracker_alpha_beta_update(
        &filter, 10 * 65536, 4 * 65536, 1000000, &result.motion
    ));
    assert(result.motion.x_q16 == (5 * 65536));
    assert(result.motion.y_q16 == (2 * 65536));
    assert(result.motion.velocity_x_q16_per_s == ((2 * 65536) + 32768));
    assert(result.motion.velocity_y_q16_per_s == (1 * 65536));

    return result;
}

static void prepare_smoke_tensor(void)
{
    for (size_t pixel = 0; pixel < SMOKE_PIXELS; ++pixel) {
        for (size_t channel = 0; channel < HWC16_CHANNELS; ++channel) {
            s_smoke_tensor[pixel][channel] = INT8_MIN;
        }
    }

    s_smoke_tensor[0][SMOKE_CHANNEL] = -5;
    s_smoke_tensor[1][SMOKE_CHANNEL] = 27;
    s_smoke_tensor[2][SMOKE_CHANNEL] = 27;
    s_smoke_tensor[3][SMOKE_CHANNEL] = 1;
}

void app_main(void)
{
    esp_chip_info_t chip = {0};
    esp_chip_info(&chip);

    printf(
        "{\"event\":\"boot\",\"target\":\"esp32p4\","
        "\"chip_revision\":%" PRIu16 ",\"cores\":%u,\"cpu_mhz\":%d,"
        "\"internal_free\":%zu,\"internal_largest\":%zu,"
        "\"spiram_free\":%zu,\"spiram_largest\":%zu}\n",
        chip.revision,
        chip.cores,
        CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)
    );

    const portable_smoke_result_t portable = run_portable_core_smoke();
    printf(
        "{\"event\":\"portable_core_smoke\",\"peak_index\":%" PRIu32 ","
        "\"centroid_x_q16\":%" PRId32 ",\"centroid_y_q16\":%" PRId32 ","
        "\"mailbox_published\":%" PRIu64 ",\"mailbox_replaced\":%" PRIu64 ","
        "\"mailbox_taken\":%" PRIu64 ",\"motion_x_q16\":%" PRId32 ","
        "\"verified\":true}\n",
        portable.centroid.peak.index,
        portable.centroid.x_q16,
        portable.centroid.y_q16,
        portable.mailbox.published,
        portable.mailbox.replaced,
        portable.mailbox.taken,
        portable.motion.x_q16
    );

    prepare_smoke_tensor();

    int8_t maximum = 0;
    const uint64_t instructions_before = tracker_perf_instructions();
    const uint64_t cycles_before = tracker_perf_cycles();
    const size_t index = tracker_argmax_hwc16_s8(
        &s_smoke_tensor[0][0], SMOKE_PIXELS, SMOKE_CHANNEL, &maximum
    );
    const uint64_t cycles_after = tracker_perf_cycles();
    const uint64_t instructions_after = tracker_perf_instructions();

    assert(index == 1);
    assert(maximum == 27);

    printf(
        "{\"event\":\"benchmark\",\"kernel\":\"argmax_hwc16_s8_rv32\","
        "\"pixels\":%d,\"channel\":%d,\"index\":%zu,\"value\":%d,"
        "\"cycles\":%" PRIu64 ",\"instructions\":%" PRIu64 ","
        "\"verified\":true}\n",
        SMOKE_PIXELS,
        SMOKE_CHANNEL,
        index,
        maximum,
        cycles_after - cycles_before,
        instructions_after - instructions_before
    );
}
