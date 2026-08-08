#ifndef TRACKER_ALPHA_BETA_H
#define TRACKER_ALPHA_BETA_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t alpha_q15;
    uint16_t beta_q15;
    uint32_t reset_gap_us;
} tracker_alpha_beta_config_t;

typedef struct {
    tracker_alpha_beta_config_t config;
    int32_t x_q16;
    int32_t y_q16;
    int32_t velocity_x_q16_per_s;
    int32_t velocity_y_q16_per_s;
    int64_t timestamp_us;
    bool initialized;
} tracker_alpha_beta_t;

typedef struct {
    int32_t x_q16;
    int32_t y_q16;
    int32_t velocity_x_q16_per_s;
    int32_t velocity_y_q16_per_s;
} tracker_motion_t;

/* Gains must be <= 32768 (1.0 in unsigned Q1.15). */
bool tracker_alpha_beta_init(tracker_alpha_beta_t *filter,
                             const tracker_alpha_beta_config_t *config);

/* Rejects a timestamp that does not advance, without modifying the state. */
bool tracker_alpha_beta_update(tracker_alpha_beta_t *filter,
                               int32_t measured_x_q16,
                               int32_t measured_y_q16,
                               int64_t timestamp_us,
                               tracker_motion_t *motion);

/* Predicts without modifying the filter. Timestamp must not precede the state. */
bool tracker_alpha_beta_predict(const tracker_alpha_beta_t *filter,
                                int64_t timestamp_us,
                                tracker_motion_t *motion);

#ifdef __cplusplus
}
#endif

#endif
