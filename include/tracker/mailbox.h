#ifndef TRACKER_MAILBOX_H
#define TRACKER_MAILBOX_H

#include "tracker/tensor.h"

#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    tracker_tensor_view_t image;
    uint64_t sequence;
    int64_t capture_timestamp_us;
    void *owner;
} tracker_frame_t;

typedef void (*tracker_frame_release_fn)(void *context, tracker_frame_t *frame);

typedef struct {
    uint64_t published;
    uint64_t replaced;
    uint64_t taken;
} tracker_mailbox_stats_t;

/* Do not copy or move a mailbox after initialization. */
typedef struct {
    atomic_bool lock;
    bool has_frame;
    tracker_frame_t frame;
    tracker_frame_release_fn release;
    void *release_context;
    tracker_mailbox_stats_t stats;
} tracker_mailbox_t;

void tracker_mailbox_init(tracker_mailbox_t *mailbox,
                          tracker_frame_release_fn release,
                          void *release_context);

/* Transfers frame ownership to the mailbox and releases any older pending frame. */
void tracker_mailbox_publish(tracker_mailbox_t *mailbox, const tracker_frame_t *frame);

/* Transfers ownership of the newest pending frame to the caller. */
bool tracker_mailbox_take_latest(tracker_mailbox_t *mailbox, tracker_frame_t *frame);

tracker_mailbox_stats_t tracker_mailbox_get_stats(tracker_mailbox_t *mailbox);

/* Releases a pending frame, if any. The mailbox may be initialized again. */
void tracker_mailbox_destroy(tracker_mailbox_t *mailbox);

#ifdef __cplusplus
}
#endif

#endif
