# Research

This folder keeps one readable file per genuinely different research direction.
This README is only the index, conclusion summary, and progress record.

## Current directions

### 1. Motion-first head tracking

**Status:** initial research complete; the selected float temporal foundation is
implemented, but sequence training and runtime tracking have not started.

Use motion to decide where and when to look, a tiny neural network to confirm that
the changed structure is a head, and a persistent tracker to retain ownership of
the first confirmed subject.

The selected experiment uses three image-derived planes:

```text
current ISP luminance Y(t)
decaying positive change P(t)
decaying negative change N(t)
```

It preserves the existing head-centre idea while adding displacement, a previous
owner prior, and a small recurrent state. Fixed-camera motion comes first.
Camera-motion compensation is a later, separately measured extension because
ordinary frame differencing fails when the whole camera moves.

Read the complete evidence, hardware constraints, alternatives, datasets, and
experiment order in [motion-first-head-tracking.md](motion-first-head-tracking.md).

### 2. Temporal neural architectures

**Status:** broad model-family research and bounded novelty audit complete; the
selected model and motion-state path now have a host implementation.

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

### 3. Concrete temporal model build

**Status:** proposal audited; float model, motion surfaces, configuration, and
shape/MAC/state inspection implemented. Sequence training, export, firmware, and
physical measurements have not started.

The selected TMC-DTA instantiation uses full-resolution luminance,
half-resolution positive/negative motion surfaces, a previous-owner prior, five
CenterTrack-style outputs, and an eight-channel two-pole state. The concrete
tensor contract, corrected architecture order, exact verified arithmetic, code
boundaries, training sequence, deployment boundary, and remaining questions are
in [temporal-model-build.md](temporal-model-build.md).

### 4. Temporal tracking data

**Status:** public-data, own-capture, and Mac-local annotation design complete;
the OV5647 pilot has not yet been recorded or labelled.

There is no single perfect dataset. The selected small combination is CAVIAR's
partially head-enriched indoor sequences after a coverage audit, JRDB-Pose for
broad real head trajectories, ChokePoint for fixed-camera portal walking, a
capped HT21 subset for hard association, and our own OV5647 recordings for the
actual sensor and room.
The document defines exactly what to annotate, a CVAT-based Mac workflow with
model pre-labelling, compact storage, capture scenarios, temporal augmentation,
split rules, and the first experiment.

Read [tracking-data.md](tracking-data.md).

## Progress

- [x] Define the motion, neural recognition, and tracking responsibilities.
- [x] Define first-subject ownership and stopped/occluded-subject behaviour.
- [x] Check the OV5647, ISP, PPA, ESP-DL, and ESP32-P4 memory constraints.
- [x] Select the smallest first temporal model and reject heavier starting points.
- [x] Identify suitable video evidence and tracking metrics.
- [x] Survey temporal neural-network families beyond ESP32 examples.
- [x] Perform a bounded public search for prior ESP32 implementations.
- [x] Define a deployment-specific recurrent prototype and its state budget.
- [x] Audit the concrete TMC-DTA proposal and separate exact results from estimates.
- [x] Implement the float motion surfaces and recurrent neural graph.
- [x] Recalculate shapes, convolution MACs, parameters, and persistent state.
- [x] Select a bounded public-video mixture and local annotation workflow.
- [x] Define the OV5647 pilot, track schema, split rules, and temporal augmentation.
- [ ] Record a small fixed-camera OV5647 room dataset.
- [ ] Add frame-pair loading and temporally consistent augmentation.
- [ ] Train the luminance-only five-output spatial control (`python/train.py` is that loop; it has not been run to convergence).
- [ ] Add synthetic pairs, tracked video, displacement targets, and prior corruption.
- [ ] Compare no state, one pole, two poles, and online temporal shift.
- [ ] Implement and test the ownership state machine on the host.
- [ ] Export, quantize, and benchmark the complete path on the ESP32-P4.
- [ ] Research and implement moving-camera compensation only after that baseline.

## Current decision

`python/train.py` now trains the five-output graph as spatial control (still
images, zero motion/prior/state). The two-pole code exists so its cost can be
inspected with `python/show_model.py`, but it is not yet evidence that recurrence
helps. It must be compared against the same graph with no temporal state. No
temporal model has been trained to convergence, and no ownership, camera-motion
compensation, quantized runtime, or physical motion-tracking result exists yet.

If a future investigation reaches a genuinely different conclusion—for example,
an event-camera approach or a tracker with no semantic neural detector—it should
become another file here. Broad alternatives stay in the architecture survey;
changes to the selected tensor graph stay in the concrete build file.
