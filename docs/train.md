# Train

Run training from the repository root:

```bash
python3 python/train.py
```

`config/train.toml` controls model width and depth, optimizer values, batch
size, validation fraction, device, and output folder. Input size, RGB versus
luminance, and augmentation remain in `config/preprocess.toml`, so the same
preprocessing code is used by the preview and the trainer.

The current model uses ordinary and depthwise convolutions, batch normalization,
and ReLU6. It keeps a stride-four feature map and emits 25 channels: five
heatmaps, ten box-size values, and ten center-offset values. Width and depth can
be changed without editing Python. The baseline's twelve stride-one body blocks
give each output cell an approximately 103 by 103 pixel receptive field, which
covers the full height of the initial 200 by 100 input without an unsupported
resize or uncertain dilated-convolution path.

The split is deterministic and keeps examples from every source in both train
and validation. Training can sample the four sources evenly even though their
download sizes differ. Validation uses clean images; training creates fresh
configured augmentation each epoch. Unknown annotation kinds and ignored
regions do not contribute false-negative heatmap loss.

Each epoch writes readable JSON metrics plus `last.pt` and the best validation
checkpoint. A checkpoint contains the model, optimizer, learning-rate schedule,
preprocessing values, and current epoch. Resume explicitly with:

```bash
python3 python/train.py --resume runs/baseline/last.pt
```

Before a full run, exercise the whole real-data path quickly with:

```bash
python3 python/train.py --smoke
```

Separate runs will compare RGB, luminance, and a RAW-derived one-channel input.
The model will be trained from scratch. Quantization-aware training can be
selected in a later experiment config; it will not require a second training
framework.

The optimization loop is deliberately simple:

1. train a model;
2. export it and measure its activations, memory traffic, and board latency;
3. grow it when useful headroom remains;
4. shrink or restructure it when it does not fit;
5. repeat until the best-performing useful model fills the real hardware
   envelope.
