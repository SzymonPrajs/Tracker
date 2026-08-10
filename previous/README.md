# Previous implementation snapshot

This directory contains the complete pre-reset implementation workspace moved
aside on 2026-08-10. It is preserved for reference and is not the active
baseline for new firmware work.

Downloaded and derived real-image dataset payloads were deleted before this
snapshot was archived. Real-image calibration tensors and the augmentation
preview were also removed because they embedded samples from those datasets.
Synthetic smoke-test fixtures, scripts, source code, toolchains, build outputs,
model checkpoints, metrics, and firmware artifacts remain.

Commands in this snapshot assume this directory is the working directory. Data
training or evaluation cannot run until the dataset is intentionally rebuilt
using the preserved scripts in `training/datasets/`.

The active design and research documentation remains in `../docs/`. The
physical ESP32-P4 camera-to-centroid path has not yet been validated at 20--25
fps or for application accuracy.
