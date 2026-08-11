# Preprocess and augment

Run the implemented preprocessing stage from the repository root:

```bash
python3 python/preprocess.py
```

It reads the plain `images/` and `labels.jsonl` downloads and writes a small
visual check to `previews/preprocess.png`. It does not make an augmented copy
of the dataset. The future training script will import the same
`common/preprocessing.py` functions and create different augmentations while it
loads each batch.

The augmentation mixture includes clean images, darkness, sensor-like noise,
blur, and several strengths of full-canvas fisheye-like warping. The warp is a
one-to-one remap with a fixed rectangular boundary: it changes the interior
geometry without zooming, cropping, or hiding the original canvas. Boxes pass
through the same map before targets are made.

Each example contains:

- an RGB or luminance tensor at the configured input size;
- separate heatmaps for head, face, COCO person, visible person, and full
  person;
- center offsets and box sizes at the configured output stride;
- validity masks so labels absent from a particular dataset are unknown rather
  than false negatives;
- masked-out regions for annotations marked difficult or ignored.

All changing values are in `config/preprocess.toml`. The default input is 200
by 100 with targets at stride 4. Use `--clean` to inspect resizing and target
generation with random augmentation disabled:

```bash
python3 python/preprocess.py --clean
```

The stored 400 by 200 maximum is intentionally larger than the likely initial
model input. If experiments require a larger input, change the download config
and regenerate the compact data rather than stretching small images.
