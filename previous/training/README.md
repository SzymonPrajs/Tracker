# Training

Reset status, 2026-08-10: this training implementation is archived and its
real-image datasets and calibration tensors have been deleted. The scripts,
checkpoints, metrics, and synthetic smoke fixtures remain. Run commands from
the `previous/` directory only after intentionally rebuilding the data.

The baseline uses one path: PyTorch on the Mac, ONNX, then ESP-PPQ INT8 for ESP32-P4.

```sh
./training/setup.sh
make train
make model
```

`make train` reads `data/packed/288x160`: fixed-shape uint8 RGB plus saved 72x40
heatmap, offset, size, and mask tensors. No JPEG decoding, resizing, heatmap
pooling, or target construction occurs inside an epoch. It applies the
room-camera mixture in `datasets/room_mix.json`; the sampler favours
one-to-four-head scenes and limits dense-crowd influence. WIDER remains a
separate face-only cache and never becomes head-size supervision.

The loss is the standard CenterNet modified focal loss for overlapping Gaussian
centres, robust Smooth-L1 for sparse sub-cell offset and normalized head size,
and a small penalty keeping the HWC16 padding channels zero. AdamW plus a
One-Cycle schedule, best/last checkpoints, validation, early stopping, resume,
gradient accumulation/clipping, CUDA AMP, optional `torch.compile`, and
device-aware memory layout are all command-line parameters. On this Mac, batch
128 and NCHW MPS reached about 424 images/s with augmentation disabled and 302
images/s with the default device-side flip and colour augmentation in 20-step
measured runs. Channels-last is reserved for CUDA because PyTorch 2.9 MPS cannot
backpropagate through this FPN layout.

Training writes `model.pt`, restartable `last.pt`, `history.jsonl`, the resolved
configuration, and a real-image `calibration.npy`. See all controls with:

```sh
PYTHONPATH=training .tools/tracker/bin/python training/train.py --help
```

The old 24-scene generator remains only as an explicit smoke test:

```sh
PYTHONPATH=training .tools/tracker/bin/python training/train.py --synthetic --epochs 1
```

`make model` writes the fixed `160x288` deployment ONNX and an ESP32-P4-specific
INT8 `model.espdl`. Quantization uses 256 deterministic training scenes balanced
across dataset source, head count, and largest-head size rather than adjacent
records. It embeds a real input/output test vector for ESP-DL's on-board model
test, exports the ESP-PPQ graph/configuration, generates the firmware exponents,
and compares floating-point ONNX with simulated INT8 detection AP on 256 held-out
validation scenes. The full provenance, hashes, scales, sample IDs, numerical
error, saturation, and metric delta are saved in `model.quantization.json`.

The camera path is exact for the required input scale: `(uint8 - 128) / 128` in
training becomes `uint8 ^ 0x80` on the device at exponent `-7`. See
`data/README.md` for reproducible downloads and conversion.

The default training augmentation now models exposure in stops, severe low
light, white balance, gamma, saturation, uneven illumination, local shadows,
vignetting, shot/read noise, mild blur, 130/180-degree-class fisheye distortion,
and 2x-6x telephoto crops. Spatial transforms warp the saved probability map and
recompute head centres, offsets, and boxes. See
`docs/15-camera-augmentation.md` and render a review sheet with
`training/preview_augmentations.py` before changing its ranges.

After the optimizer has selected `training/artifacts/optimized_model.pt`, run
`make model-optimized` to export, calibrate, evaluate, and install the resulting
ESP-DL binary at `firmware/models/hcds31-int8.espdl`. `make firmware` embeds that
binary and builds an on-device startup check which verifies tensor shapes and
exponents, runs the embedded ESP-DL golden test vector, and prints a layer
profile. The firmware profile targets a 16 MB flash part with a 4 MB factory
application partition.
