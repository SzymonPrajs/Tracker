#include "tracker/mailbox.h"

static void mailbox_lock(tracker_mailbox_t *mailbox)
{
    while (atomic_exchange_explicit(&mailbox->lock, true, memory_order_acquire)) {
    }
}

static void mailbox_unlock(tracker_mailbox_t *mailbox)
{
    atomic_store_explicit(&mailbox->lock, false, memory_order_release);
}

void tracker_mailbox_init(tracker_mailbox_t *mailbox,
                          tracker_frame_release_fn release,
                          void *release_context)
{
    if (mailbox == NULL) {
        return;
    }
    atomic_init(&mailbox->lock, false);
    mailbox->has_frame = false;
    mailbox->frame = (tracker_frame_t){0};
    mailbox->release = release;
    mailbox->release_context = release_context;
    mailbox->stats = (tracker_mailbox_stats_t){0U, 0U, 0U};
}

void tracker_mailbox_publish(tracker_mailbox_t *mailbox, const tracker_frame_t *frame)
{
    tracker_frame_t replaced;
    bool had_replaced;

    if ((mailbox == NULL) || (frame == NULL)) {
        return;
    }

    mailbox_lock(mailbox);
    had_replaced = mailbox->has_frame;
    if (had_replaced) {
        replaced = mailbox->frame;
        ++mailbox->stats.replaced;
    }
    mailbox->frame = *frame;
    mailbox->has_frame = true;
    ++mailbox->stats.published;
    mailbox_unlock(mailbox);

    if (had_replaced && (mailbox->release != NULL)) {
        mailbox->release(mailbox->release_context, &replaced);
    }
}

bool tracker_mailbox_take_latest(tracker_mailbox_t *mailbox, tracker_frame_t *frame)
{
    if ((mailbox == NULL) || (frame == NULL)) {
        return false;
    }

    mailbox_lock(mailbox);
    if (!mailbox->has_frame) {
        mailbox_unlock(mailbox);
        return false;
    }
    *frame = mailbox->frame;
    mailbox->has_frame = false;
    ++mailbox->stats.taken;
    mailbox_unlock(mailbox);
    return true;
}

tracker_mailbox_stats_t tracker_mailbox_get_stats(tracker_mailbox_t *mailbox)
{
    tracker_mailbox_stats_t stats = {0U, 0U, 0U};

    if (mailbox == NULL) {
        return stats;
    }
    mailbox_lock(mailbox);
    stats = mailbox->stats;
    mailbox_unlock(mailbox);
    return stats;
}

void tracker_mailbox_destroy(tracker_mailbox_t *mailbox)
{
    tracker_frame_t pending;
    bool had_pending;

    if (mailbox == NULL) {
        return;
    }
    mailbox_lock(mailbox);
    had_pending = mailbox->has_frame;
    if (had_pending) {
        pending = mailbox->frame;
        mailbox->has_frame = false;
    }
    mailbox_unlock(mailbox);

    if (had_pending && (mailbox->release != NULL)) {
        mailbox->release(mailbox->release_context, &pending);
    }
}
