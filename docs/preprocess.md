# Preprocess

```bash
python3 python/preprocess.py
python3 python/preprocess.py --clean
```

Reads `data/*/labels.jsonl` and writes `previews/preprocess.png`. It does
not copy the dataset. Training calls the same functions in
`python/common/preprocessing.py` on each batch.

Default input is **320×160 luminance**, which matches the temporal model.
Change `config/preprocess.toml` to try 200×100 or 400×200. The compact
cache is 400×200 WebP, so those three sizes work without a re-download.
Going above 400×200 needs a new download.

Each example:

- one luminance (or RGB, if you change `color`) tensor
- a `head_center` heatmap
- a sub-cell offset at stride 4
- ignored regions for tiny or marked-difficult boxes

Heads shorter than `minimum_target_pixels` are masked, not treated as
background. Training skips frames with more than
`maximum_train_targets` usable heads.

Augmentation: a clean branch, darkness, noise, blur, and a full-canvas
fisheye-like warp that keeps the rectangle boundary fixed. The same warp
will later be applied to every frame of a clip so it does not invent
motion.
