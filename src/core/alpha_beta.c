#include "tracker/alpha_beta.h"

#include <limits.h>
#include <stddef.h>

#define TRACKER_Q15_ONE UINT16_C(32768)
#define TRACKER_MICROSECONDS_PER_SECOND INT64_C(1000000)

static int32_t clamp_i64_to_i32(int64_t value)
{
    if (value > INT32_MAX) {
        return INT32_MAX;
    }
    if (value < INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)value;
}

static int32_t q15_scale(int32_t value, uint16_t gain_q15)
{
    return clamp_i64_to_i32(((int64_t)value * gain_q15) / TRACKER_Q15_ONE);
}

static int32_t advance_position(int32_t position_q16, int32_t velocity_q16_per_s, int64_t delta_us)
{
    int64_t displacement;

    if ((velocity_q16_per_s > 0) && (delta_us > (INT64_MAX / velocity_q16_per_s))) {
        return INT32_MAX;
    } else if ((velocity_q16_per_s < 0) &&
               (delta_us > (INT64_MAX / -(int64_t)velocity_q16_per_s))) {
        return INT32_MIN;
    } else {
        displacement = ((int64_t)velocity_q16_per_s * delta_us) / TRACKER_MICROSECONDS_PER_SECOND;
    }
    return clamp_i64_to_i32((int64_t)position_q16 + displacement);
}

static void write_motion(const tracker_alpha_beta_t *filter, tracker_motion_t *motion)
{
    if (motion != NULL) {
        motion->x_q16 = filter->x_q16;
        motion->y_q16 = filter->y_q16;
        motion->velocity_x_q16_per_s = filter->velocity_x_q16_per_s;
        motion->velocity_y_q16_per_s = filter->velocity_y_q16_per_s;
    }
}

bool tracker_alpha_beta_init(tracker_alpha_beta_t *filter,
                             const tracker_alpha_beta_config_t *config)
{
    if ((filter == NULL) || (config == NULL) || (config->alpha_q15 > TRACKER_Q15_ONE) ||
        (config->beta_q15 > TRACKER_Q15_ONE) || (config->reset_gap_us == 0U)) {
        return false;
    }

    filter->config = *config;
    filter->x_q16 = 0;
    filter->y_q16 = 0;
    filter->velocity_x_q16_per_s = 0;
    filter->velocity_y_q16_per_s = 0;
    filter->timestamp_us = 0;
    filter->initialized = false;
    return true;
}

bool tracker_alpha_beta_update(tracker_alpha_beta_t *filter,
                               int32_t measured_x_q16,
                               int32_t measured_y_q16,
                               int64_t timestamp_us,
                               tracker_motion_t *motion)
{
    int64_t delta_us;
    int32_t predicted_x;
    int32_t predicted_y;
    int32_t residual_x;
    int32_t residual_y;
    int32_t velocity_step_x;
    int32_t velocity_step_y;

    if ((filter == NULL) || (timestamp_us < 0)) {
        return false;
    }

    if (!filter->initialized) {
        filter->x_q16 = measured_x_q16;
        filter->y_q16 = measured_y_q16;
        filter->velocity_x_q16_per_s = 0;
        filter->velocity_y_q16_per_s = 0;
        filter->timestamp_us = timestamp_us;
        filter->initialized = true;
        write_motion(filter, motion);
        return true;
    }

    if (timestamp_us <= filter->timestamp_us) {
        return false;
    }
    delta_us = timestamp_us - filter->timestamp_us;
    if ((uint64_t)delta_us > filter->config.reset_gap_us) {
        filter->x_q16 = measured_x_q16;
        filter->y_q16 = measured_y_q16;
        filter->velocity_x_q16_per_s = 0;
        filter->velocity_y_q16_per_s = 0;
        filter->timestamp_us = timestamp_us;
        write_motion(filter, motion);
        return true;
    }

    predicted_x = advance_position(filter->x_q16, filter->velocity_x_q16_per_s, delta_us);
    predicted_y = advance_position(filter->y_q16, filter->velocity_y_q16_per_s, delta_us);
    residual_x = clamp_i64_to_i32((int64_t)measured_x_q16 - predicted_x);
    residual_y = clamp_i64_to_i32((int64_t)measured_y_q16 - predicted_y);

    filter->x_q16 = clamp_i64_to_i32((int64_t)predicted_x + q15_scale(residual_x, filter->config.alpha_q15));
    filter->y_q16 = clamp_i64_to_i32((int64_t)predicted_y + q15_scale(residual_y, filter->config.alpha_q15));

    velocity_step_x = q15_scale(residual_x, filter->config.beta_q15);
    velocity_step_y = q15_scale(residual_y, filter->config.beta_q15);
    velocity_step_x = clamp_i64_to_i32(((int64_t)velocity_step_x * TRACKER_MICROSECONDS_PER_SECOND) / delta_us);
    velocity_step_y = clamp_i64_to_i32(((int64_t)velocity_step_y * TRACKER_MICROSECONDS_PER_SECOND) / delta_us);
    filter->velocity_x_q16_per_s =
        clamp_i64_to_i32((int64_t)filter->velocity_x_q16_per_s + velocity_step_x);
    filter->velocity_y_q16_per_s =
        clamp_i64_to_i32((int64_t)filter->velocity_y_q16_per_s + velocity_step_y);
    filter->timestamp_us = timestamp_us;
    write_motion(filter, motion);
    return true;
}

bool tracker_alpha_beta_predict(const tracker_alpha_beta_t *filter,
                                int64_t timestamp_us,
                                tracker_motion_t *motion)
{
    int64_t delta_us;

    if ((filter == NULL) || (motion == NULL) || !filter->initialized || (timestamp_us < 0) ||
        (timestamp_us < filter->timestamp_us)) {
        return false;
    }
    delta_us = timestamp_us - filter->timestamp_us;
    motion->x_q16 = advance_position(filter->x_q16, filter->velocity_x_q16_per_s, delta_us);
    motion->y_q16 = advance_position(filter->y_q16, filter->velocity_y_q16_per_s, delta_us);
    motion->velocity_x_q16_per_s = filter->velocity_x_q16_per_s;
    motion->velocity_y_q16_per_s = filter->velocity_y_q16_per_s;
    return true;
}
