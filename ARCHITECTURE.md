# Tracker

The repository has one path:

```text
generated/real images -> PyTorch HC-DS31 -> ONNX -> ESP-DL INT8
camera -> PPA 160x288 input -> model -> C centroid decoder
```

- `training/` trains and converts the fixed model.
- `src/tracker.c` contains preprocessing and centroid decoding.
- `firmware/` cross-compiles the C code and two standalone RISC-V assembly helpers.

The model has a `160x288` input, a `40x72x16` HWC16 output, 31.26 MMAC per
frame and 66,224 convolution weights. Channels are confidence, X/Y offset,
width, height, then eleven zero padding channels.

Nothing else is implemented until the baseline runs on the physical board.
