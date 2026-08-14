# Train

```bash
python3 python/train.py --smoke
python3 python/train.py
python3 python/train.py --resume runs/temporal_spatial/last.pt
```

`config/train.toml` is the run (epochs, batch, sampling, optimizer).
`config/temporal.toml` is the network. `config/preprocess.toml` is the
image size and augmentation. Change numbers there, not in the script.

## What this run is

The model in `python/common/model.py` is the streaming detector:

```text
Y(t)  +  P(t)/N(t)  +  previous-owner prior  +  fast/slow state
    → heatmap, offset, displacement, next state
```

This first loop only teaches **where a head is** from still images:

- input is one luminance plane
- motion, prior, and recurrent state are zeros
- loss is heatmap + offset
- displacement is present in the graph and left unused

That is the spatial control in the research notes. The same `train.py`
should grow into pairs and short clips later. Do not add
`train_temporal.py`.

## What good looks like

`--smoke` should finish one epoch and write `runs/smoke/last.pt`.

A real run writes `runs/temporal_spatial/metrics.jsonl`, `last.pt`, and
`best.pt`. `best.pt` is picked by validation center AP at 8 pixels.

Useful printed numbers:

- `center_ap_8` — ranking quality
- `center_precision_8` / `center_recall_8` — at the threshold in the config
- `negative_false_positives_per_image` — Open Images verified negatives
- `recall_8_<source>` — per dataset

Old checkpoints under `runs/focused_400x200/` are from the previous
still-image-only network. They will not load.

## After training

```bash
python3 python/export.py --checkpoint runs/temporal_spatial/best.pt
```

That writes an ONNX file with named inputs `luminance`, `motion`,
`prior`, `fast_state`, `slow_state`. INT8 / ESP-DL belongs in
`export.py` once the camera bytes are known. It is a deploy step, not a
separate project phase.
