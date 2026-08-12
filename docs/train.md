# Train

Run training from the repository root:

```bash
python3 python/train.py
```

`config/train.toml` controls model widths and depths, source mixture, optimizer,
batch size, device, and output folder. Input size, RGB versus luminance,
minimum target size, and augmentation remain in `config/preprocess.toml`, so
the preview and trainer use identical preprocessing.

The focused model uses ordinary and depthwise convolutions, batch normalization,
ReLU, nearest-neighbor resize, and addition. It keeps two residual blocks at
stride 4, moves six blocks to the cheaper stride-8 map, then merges the result
back into stride 4. Its three outputs are one center logit and two sub-cell
offsets. The default has 38,403 trainable parameters and the following computed
convolution cost:

| Input | Model compute |
|---|---:|
| 200 by 100 | 16.915 MMAC |
| 320 by 160 | 42.266 MMAC |
| 400 by 200 | 66.040 MMAC |

This is substantially cheaper than keeping all twelve old blocks at stride 4.
The 400 by 200 choice is an experiment, not a claim that it fits the complete
camera pipeline: export and board profiling still have to measure allocator
workspace, camera buffers, conversion, PSRAM traffic, and latency together.

Official WIDER validation, CrowdHuman validation, and SCUT test records are
used when present. A stable content-group split is used only for a source whose
held-out data has not yet been appended. Exact duplicate content is kept on one
side, with held-out membership taking priority. Training samples approximately
50% WIDER, 25% CrowdHuman, 20% SCUT-HEAD, and 5% verified Open Images
negatives. Validation is clean; training creates fresh configured augmentation
each epoch.

Each epoch writes readable JSON metrics plus `last.pt` and the best validation
checkpoint. Predictions are decoded with their offsets and matched one-to-one
to transformed raw centers. Metrics include center AP within 4 and 8 input
pixels, precision/recall at the configured operating threshold, source/size
recall, and false positives per verified-negative image. `best.pt` is selected
by center AP at 8 pixels, not validation loss. Resume explicitly with:

```bash
python3 python/train.py --resume runs/focused_400x200/last.pt
```

Before a full run, exercise the whole real-data path quickly with:

```bash
python3 python/train.py --smoke
```

The first run uses a 12-pixel minimum. After it converges, the next controlled
run lowers the minimum to 8 pixels. Resolution comparisons use the same compact
source images at 200 by 100, 320 by 160, and 400 by 200. Finalists then compare
RGB, luminance, and a RAW-derived one-channel input. Quantization-aware training
comes only after an ordinary INT8 export shows whether it is needed.

The optimization loop is deliberately simple:

1. train a model;
2. export it and measure activations, memory traffic, and board latency;
3. grow it when useful headroom remains;
4. shrink or restructure it when it does not fit;
5. repeat until the best useful model fills the measured hardware envelope.
