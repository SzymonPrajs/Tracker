#ifndef TRACKER_MODEL_RUNNER_H
#define TRACKER_MODEL_RUNNER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Runs ESP-DL's embedded golden-vector test, then prints memory and latency. */
void tracker_model_test_and_profile(void);

#ifdef __cplusplus
}
#endif

#endif
