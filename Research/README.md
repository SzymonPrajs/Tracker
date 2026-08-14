# Research

This folder keeps one readable file per genuinely different research direction.
This README is only the index, conclusion summary, and progress record.

## Current directions

### 1. Motion-first head tracking

**Status:** initial research complete; temporal implementation not started.

Use motion to decide where and when to look, a tiny neural network to confirm that
the changed structure is a head, and a persistent tracker to retain ownership of
the first confirmed subject.

The first experiment uses two model inputs:

```text
current ISP luminance Y(t)
clipped signed change Y(t) - Y(t-1)
```

It reuses the existing head-centre model and still-image training. Fixed-camera
background subtraction comes first. Camera-motion compensation is a later,
separately measured extension because ordinary frame differencing fails when the
whole camera moves.

Read the complete evidence, hardware constraints, alternatives, datasets, and
experiment order in [motion-first-head-tracking.md](motion-first-head-tracking.md).

### 2. Temporal neural architectures

**Status:** broad model-family research and bounded novelty audit complete;
prototypes not started.

The deeper model investigation covers joint detection/tracking, recurrent feature
maps, online temporal shifts, diagonal state-space models, change-sparse inference,
streaming 3-D CNNs, Siamese trackers, and event/spiking models. It recommends two
different goals:

- a tiny CenterTrack-style network as the safest unusual ESP32 deployment;
- a deployment-specific dual-timescale motion-state detector as the more original
  experiment.

The second design keeps ordinary INT8 convolutions but gives a small subset of its
stride-8 features a learned fast and slow memory. Read the family comparison,
proposed equations, exact state sizes, novelty boundary, and experiment ladder in
[temporal-neural-architectures.md](temporal-neural-architectures.md).

## Progress

- [x] Define the motion, neural recognition, and tracking responsibilities.
- [x] Define first-subject ownership and stopped/occluded-subject behaviour.
- [x] Check the OV5647, ISP, PPA, ESP-DL, and ESP32-P4 memory constraints.
- [x] Select the smallest first temporal model and reject heavier starting points.
- [x] Identify suitable video evidence and tracking metrics.
- [x] Survey temporal neural-network families beyond ESP32 examples.
- [x] Perform a bounded public search for prior ESP32 implementations.
- [x] Define a deployment-specific recurrent prototype and its state budget.
- [ ] Record a small fixed-camera OV5647 room dataset.
- [ ] Add frame-pair loading and temporally consistent augmentation.
- [ ] Convert the trained RGB stem to luminance plus signed change.
- [ ] Implement the tiny CenterTrack interface as the temporal neural baseline.
- [ ] Implement the dual-timescale motion-state adapter as a separate ablation.
- [ ] Implement and test the ownership state machine on the host.
- [ ] Export, quantize, and benchmark the complete path on the ESP32-P4.
- [ ] Research and implement moving-camera compensation only after that baseline.

## Current decision

The next implementation step is still the fixed-camera frame-pair pipeline. The
first temporal network should then be the small CenterTrack-style baseline; the
dual-timescale adapter is compared against it, not silently mixed into it. No
temporal model, ownership tracker, background model, or mobile-camera compensation
has been implemented yet, and no motion-tracking performance has been measured on
the physical board.

If a future investigation reaches a genuinely different conclusion—for example,
an event-camera approach or a tracker with no semantic neural detector—it should
become another file here. Variants of this same motion-first design stay together
in the existing direction file.
