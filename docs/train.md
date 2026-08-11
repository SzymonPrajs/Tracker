# Train

Training will be one script, `python/train.py`, controlled by readable config
files under `config/`. Each experiment config will state the input format and
size, model widths/depths, optimizer, augmentation strength, and output path.

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
