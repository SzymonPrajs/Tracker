# Tracker

A small hobby project: find a head in a camera stream and run that model on a
Waveshare **ESP32-P4-Module-DEV-KIT** with an **OV5647** camera.

If you just cloned this, do these in order.

## What you need

- a Mac
- Python 3.10+
- the P4 board, the OV5647, and a data-capable USB-C cable
- later: ESP-IDF v5.5.2 (the hardware page installs it)

## 1. Python

```bash
python3 -m pip install -r python/requirements.txt
```

## 2. Download the still-image corpus

```bash
python3 python/download.py
```

This fills `data/` with compact WebP images and one `labels.jsonl` per source.
Edit `config/download.toml` if you want a smaller set. Details:
[docs/download.md](docs/download.md).

## 3. Look at preprocessing

```bash
python3 python/preprocess.py
```

Writes `previews/preprocess.png`. Same code path training uses. Config:
`config/preprocess.toml`. Details: [docs/preprocess.md](docs/preprocess.md).

## 4. Train

The network is already the streaming one: luminance + motion + prior + a
small recurrent state. The first training loop is **spatial control** —
still images, motion/prior/state are zeros, only heatmap and offset are
trained. That is deliberate. Video pairs come next without changing the
graph.

```bash
python3 python/train.py --smoke          # a minute, real data
python3 python/train.py                  # the real run
```

Checkpoints land in `runs/temporal_spatial/`. Config: `config/train.toml`
and `config/temporal.toml`. Details: [docs/train.md](docs/train.md).

Inspect the graph without data:

```bash
python3 python/show_model.py
```

After a checkpoint exists, export ONNX (INT8 for the chip is the next
addition to this same script):

```bash
python3 python/export.py --checkpoint runs/temporal_spatial/best.pt
```

## 5. Plug in the board

[docs/hardware.md](docs/hardware.md) is the Mac install, the CSI cable, and
the flash command. Short version, after ESP-IDF is installed:

```bash
. $HOME/esp/esp-idf/export.sh
cd firmware
idf.py set-target esp32p4
idf.py -p /dev/cu.usbmodemXXXX flash monitor | tee ../hardware-check.log
```

You want `BOARD: OK` and `CAMERA: OK`, plus `PID=0x5647` in the log. Keep
`hardware-check.log`.

## Layout

```text
config/         numbers you change between runs
docs/           how to run each script, plus the Mac board guide
research/       the motion-first / temporal design notes
python/
  download.py   get and compact the datasets
  datasets/     one parser per external source
  preprocess.py preview
  train.py      temporal model, spatial control first
  show_model.py shapes, MACs, state bytes
  export.py     checkpoint → ONNX
  common/       code two scripts actually share
firmware/       ESP-IDF app. Today: board + camera check.
data/           local images, not committed
```

## Research

The model and the training order live in [research/README.md](research/README.md).
Read that if you want to know why motion surfaces and a two-pole state exist.
Do not treat those notes as a second codebase.

## For agents

See [AGENTS.md](AGENTS.md). Keep this a handful of scripts a person can
open. Do not add process documents or extra packages.
