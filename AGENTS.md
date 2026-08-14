# For agents working in this repo

This is a hobby project. The owner is a deep-learning developer learning
the ESP32-P4. They want to read and change the Python. They will
hand-optimize the firmware, including assembly.

## Shape

One script per job, run from the repo root:

- `python/download.py`
- `python/preprocess.py`
- `python/train.py`
- `python/show_model.py`
- `python/export.py`
- `firmware/main/check.c` (then later tracker C)

`python/datasets/` holds one parser per external dataset. `python/common/`
holds code that two scripts actually share. Do not add another package
layer. Do not add `train_temporal.py`, `track_video.py`, or a
`characterize/` tree.

## Training

The temporal graph in `python/common/model.py` is the model. The first
loop is still-image spatial control (zero motion, zero prior, reset
state). Grow `train.py` into pairs and clips. Do not start a second
trainer.

Export and INT8 belong in `python/export.py`. That is deploy, not a
research phase.

## Firmware

Keep C obvious. Assembly only for a measured hot path, next to a C
reference that produces the same bytes. The first app is the board +
camera check. Mac instructions live in `docs/hardware.md`.

## Docs

Write like a README a friend can follow on a Mac. Do not add acceptance
gates, evidence-level taxonomies, or process handbooks. Research stays
in `research/`. If the code map in a research file is wrong after a
rename, fix that table and leave the rest.

## Do not

- recreate `previous/`, `tools/`, or empty `build/` ceremony
- write docs for scripts that do not exist
- invent folders for future work
- bring back the old still-image-only network as a second model
