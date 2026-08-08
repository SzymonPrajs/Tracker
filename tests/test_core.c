#include "tracker/alpha_beta.h"
#include "tracker/decode.h"
#include "tracker/mailbox.h"
#include "tracker/preprocess.h"
#include "tracker/tensor.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define Q16(value) ((int32_t)((value) * 65536))

static tracker_tensor_view_t view(void *data,
                                  size_t size,
                                  uint32_t height,
                                  uint32_t width,
                                  uint32_t channels,
                                  size_t stride,
                                  tracker_element_type_t type)
{
    const tracker_tensor_view_t result = {data, size, height, width, channels, stride, type};
    return result;
}

static void test_tensor_validation(void)
{
    int8_t data[16] = {0};
    tracker_tensor_view_t tensor = view(data, sizeof(data), 2U, 4U, 2U, 8U, TRACKER_ELEMENT_S8);

    assert(tracker_element_size(TRACKER_ELEMENT_S16) == 2U);
    assert(tracker_tensor_view_is_valid(&tensor));
    tensor.size_bytes = sizeof(data) - 1U;
    assert(!tracker_tensor_view_is_valid(&tensor));
    tensor.size_bytes = sizeof(data);
    tensor.row_stride_bytes = 7U;
    assert(!tracker_tensor_view_is_valid(&tensor));
}

static void test_preprocess(void)
{
    uint8_t src_data[8] = {0U, 1U, 127U, 128U, 129U, 255U, 77U, 88U};
    int8_t dst_data[8] = {11, 11, 11, 11, 11, 11, 11, 11};
    const int8_t expected[6] = {-128, -127, -1, 0, 1, 127};
    tracker_tensor_view_t src = view(src_data, sizeof(src_data), 1U, 2U, 3U, 8U, TRACKER_ELEMENT_U8);
    tracker_tensor_view_t dst = view(dst_data, sizeof(dst_data), 1U, 2U, 3U, 8U, TRACKER_ELEMENT_S8);

    assert(tracker_rgb888_to_centered_s8(&src, &dst));
    assert(memcmp(dst_data, expected, sizeof(expected)) == 0);
    assert(dst_data[6] == 11);
    assert(dst_data[7] == 11);

    tracker_rgb888_center_s8_scalar((int8_t *)src_data, src_data, 6U);
    assert(memcmp(src_data, expected, sizeof(expected)) == 0);

    {
        uint8_t overlap[6] = {0U, 1U, 2U, 3U, 4U, 5U};
        const int8_t shifted_expected[5] = {-128, -127, -126, -125, -124};
        tracker_rgb888_center_s8_scalar((int8_t *)(overlap + 1U), overlap, 5U);
        assert(memcmp(overlap + 1U, shifted_expected, sizeof(shifted_expected)) == 0);
    }

    {
        uint8_t overlap[8] = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};
        tracker_tensor_view_t overlap_src =
            view(overlap, sizeof(overlap), 1U, 2U, 3U, 8U, TRACKER_ELEMENT_U8);
        tracker_tensor_view_t overlap_dst =
            view(overlap + 1U, sizeof(overlap) - 1U, 1U, 2U, 3U, 7U, TRACKER_ELEMENT_S8);
        assert(!tracker_rgb888_to_centered_s8(&overlap_src, &overlap_dst));
    }
}

static void test_decode(void)
{
    int8_t head_data[5U * 5U * 16U];
    tracker_tensor_view_t head = view(head_data, sizeof(head_data), 5U, 5U, 16U, 5U * 16U, TRACKER_ELEMENT_S8);
    tracker_peak_t peak;
    tracker_centroid_t centroid;
    uint32_t x;
    uint32_t y;

    memset(head_data, -128, sizeof(head_data));
    for (y = 1U; y <= 3U; ++y) {
        for (x = 1U; x <= 3U; ++x) {
            head_data[((y * 5U + x) * 16U)] = 50;
        }
    }
    head_data[((2U * 5U + 2U) * 16U)] = 100;
    head_data[((2U * 5U + 2U) * 16U) + 1U] = 32;  /* +0.25 cell in Q7. */
    head_data[((2U * 5U + 2U) * 16U) + 2U] = -16; /* -0.125 cell in Q7. */

    assert(tracker_argmax_channel0_hwc16(&head, &peak));
    assert(peak.index == 12U);
    assert(peak.x == 2U);
    assert(peak.y == 2U);
    assert(peak.score == 100);
    assert(tracker_decode_centroid_hwc16(&head, 4U, 7U, &centroid));
    assert(centroid.x_q16 == Q16(11));
    assert(centroid.y_q16 == Q16(9) + Q16(1) / 2);
    assert(centroid.local_weight_sum == 50U);

    /* Ties retain the first row-major maximum. */
    head_data[((1U * 5U + 1U) * 16U)] = 100;
    assert(tracker_argmax_channel0_hwc16(&head, &peak));
    assert(peak.index == 6U);
}

static void test_decode_edge(void)
{
    int8_t storage[32];
    tracker_tensor_view_t head = view(storage, sizeof(storage), 1U, 1U, 16U, 32U, TRACKER_ELEMENT_S8);
    tracker_centroid_t centroid;

    memset(storage, 0, sizeof(storage));
    storage[1] = -64; /* -0.5 cell in Q7. */
    storage[2] = 64;  /* +0.5 cell in Q7. */
    assert(tracker_decode_centroid_hwc16(&head, 4U, 7U, &centroid));
    assert(centroid.x_q16 == 0);
    assert(centroid.y_q16 == Q16(4));
    assert(centroid.local_weight_sum == 1U);
    assert(storage[16] == 0); /* Row padding remains outside the tensor. */

    {
        int8_t flat[3U * 3U * 16U];
        tracker_tensor_view_t flat_head =
            view(flat, sizeof(flat), 3U, 3U, 16U, 3U * 16U, TRACKER_ELEMENT_S8);
        memset(flat, -128, sizeof(flat));
        flat[1] = 0;
        flat[2] = 0;
        assert(tracker_decode_centroid_hwc16(&flat_head, 4U, 7U, &centroid));
        assert(centroid.peak.index == 0U);
        assert(centroid.x_q16 == Q16(2));
        assert(centroid.y_q16 == Q16(2));
        assert(centroid.local_weight_sum == 1U);
    }
}

typedef struct {
    unsigned count;
    uint64_t last_sequence;
} release_log_t;

static void release_frame(void *context, tracker_frame_t *frame)
{
    release_log_t *log = (release_log_t *)context;
    ++log->count;
    log->last_sequence = frame->sequence;
}

static void test_mailbox(void)
{
    tracker_mailbox_t mailbox;
    tracker_mailbox_stats_t stats;
    release_log_t releases = {0U, 0U};
    tracker_frame_t first = {0};
    tracker_frame_t second = {0};
    tracker_frame_t output;

    first.sequence = 1U;
    second.sequence = 2U;
    tracker_mailbox_init(&mailbox, release_frame, &releases);
    tracker_mailbox_publish(&mailbox, &first);
    tracker_mailbox_publish(&mailbox, &second);
    assert(releases.count == 1U);
    assert(releases.last_sequence == 1U);
    assert(tracker_mailbox_take_latest(&mailbox, &output));
    assert(output.sequence == 2U);
    assert(!tracker_mailbox_take_latest(&mailbox, &output));

    stats = tracker_mailbox_get_stats(&mailbox);
    assert(stats.published == 2U);
    assert(stats.replaced == 1U);
    assert(stats.taken == 1U);
    tracker_mailbox_destroy(&mailbox);
    assert(releases.count == 1U);
}

static void test_alpha_beta(void)
{
    const tracker_alpha_beta_config_t config = {16384U, 8192U, 1500000U};
    tracker_alpha_beta_t filter;
    tracker_motion_t motion;
    tracker_motion_t before_rejected;

    assert(tracker_alpha_beta_init(&filter, &config));
    assert(!tracker_alpha_beta_update(&filter, Q16(0), Q16(0), -1, &motion));
    assert(tracker_alpha_beta_update(&filter, Q16(0), Q16(0), 0, &motion));
    assert(tracker_alpha_beta_update(&filter, Q16(10), Q16(4), 1000000, &motion));
    assert(motion.x_q16 == Q16(5));
    assert(motion.y_q16 == Q16(2));
    assert(motion.velocity_x_q16_per_s == Q16(2) + Q16(1) / 2);
    assert(motion.velocity_y_q16_per_s == Q16(1));

    assert(tracker_alpha_beta_predict(&filter, 2000000, &motion));
    assert(motion.x_q16 == Q16(7) + Q16(1) / 2);
    assert(motion.y_q16 == Q16(3));

    before_rejected = motion;
    assert(!tracker_alpha_beta_update(&filter, Q16(100), Q16(100), 1000000, &motion));
    assert(memcmp(&motion, &before_rejected, sizeof(motion)) == 0);

    /* A gap beyond reset_gap_us reacquires without retaining stale velocity. */
    assert(tracker_alpha_beta_update(&filter, Q16(20), Q16(30), 3000000, &motion));
    assert(motion.x_q16 == Q16(20));
    assert(motion.y_q16 == Q16(30));
    assert(motion.velocity_x_q16_per_s == 0);
    assert(motion.velocity_y_q16_per_s == 0);

    /* Combine in int64 before saturation when displacement alone exceeds int32. */
    filter.timestamp_us = 0;
    filter.x_q16 = -1000000000;
    filter.y_q16 = 1000000000;
    filter.velocity_x_q16_per_s = 1500000000;
    filter.velocity_y_q16_per_s = -1500000000;
    assert(tracker_alpha_beta_predict(&filter, 2000000, &motion));
    assert(motion.x_q16 == 2000000000);
    assert(motion.y_q16 == -2000000000);

    /* Extreme prediction horizons saturate instead of overflowing int64. */
    filter.velocity_x_q16_per_s = INT32_MAX;
    filter.velocity_y_q16_per_s = INT32_MIN;
    assert(tracker_alpha_beta_predict(&filter, INT64_MAX, &motion));
    assert(motion.x_q16 == INT32_MAX);
    assert(motion.y_q16 == INT32_MIN);
}

int main(void)
{
    test_tensor_validation();
    test_preprocess();
    test_decode();
    test_decode_edge();
    test_mailbox();
    test_alpha_beta();
    puts("tracker core tests passed");
    return 0;
}
