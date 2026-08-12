# Tracker

A small hobby project for training a head locator and running it on an
ESP32-P4-Module-DEV-KIT with an OV5647 camera.

## Repository

```text
config/            values changed between runs
docs/              one short document per project phase
python/
  download.py      download and compact the datasets
  preprocess.py    preview augmentation and training targets
  train.py         train and validate the PyTorch model
  datasets/        one adapter per external dataset
  common/          shared Python helpers
firmware/          ESP-IDF source, assembly, build output, and flashing notes
data/              compact downloaded images (not committed)
```

This is the permanent style of the project:

- one obvious runnable script for each phase;
- one readable config file for values that change between experiments;
- small helpers in `python/common/` when two phases genuinely share code;
- dataset-specific parsing only in `python/datasets/`;
- no extra project layers or generated bookkeeping.

The quantization phase will add `python/quantize.py` when it is implemented.
Experiments are represented by small config files, so a Git commit records the
exact values used.

## Download the data

```bash
python3 -m pip install -r python/requirements.txt
python3 python/download.py
```

Edit [`config/download.toml`](config/download.toml) to change image size,
dataset sizes, or worker count. The bounded corpus contains face/head positives
from three sources plus strict Open Images face/head negatives and official
held-out splits. Each source is converted to compact WebP files no larger than
400 by 200, and its raw archive is removed before the next source begins.

See [`docs/download.md`](docs/download.md) for the exact mixture and commands.

## Inspect preprocessing

```bash
python3 python/preprocess.py
```

This writes one local contact sheet to `previews/preprocess.png`. The left side
of each tile shows transformed boxes and the right side shows the heatmap. It
does not duplicate the dataset. Edit [`config/preprocess.toml`](config/preprocess.toml)
to change input size, RGB/luminance mode, or augmentation strengths.

## Train

```bash
python3 python/train.py
```

The model trains directly from the compact data and generates augmentation as
it loads each batch. Edit [`config/train.toml`](config/train.toml) to change the
model or optimizer. Checkpoints and metrics are written under `runs/` and stay
local. Run `python3 python/train.py --smoke` for a short real-data check.

## Phases

1. [Download](docs/download.md)
2. [Preprocess and augment](docs/preprocess.md)
3. [Train](docs/train.md)
4. [Characterize the hardware](docs/hardware.md)
5. Quantize, fit, and retrain until the largest useful model fits
6. Implement and measure the complete camera-to-output firmware

The later steps intentionally feed back into input size and model size. If the
model is too small, grow it; if activations, latency, or memory traffic do not
fit, shrink or restructure it and train again.
