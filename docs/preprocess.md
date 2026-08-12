# Preprocess and augment

Run the implemented preprocessing stage from the repository root:

```bash
python3 python/preprocess.py
```

It reads the plain `images/` and `labels.jsonl` downloads and writes a small
visual check to `previews/preprocess.png`. It does not make an augmented copy
of the dataset. Training imports the same `common/preprocessing.py` functions
and creates different augmentations while it loads each batch.

The augmentation mixture includes clean images, darkness, sensor-like noise,
blur, and several strengths of full-canvas fisheye-like warping. The warp is a
one-to-one remap with a fixed rectangular boundary: it changes the interior
geometry without zooming, cropping, or hiding the original canvas. Boxes pass
through the same map before targets are made.

Each example contains:

- an RGB or luminance tensor at the configured input size;
- one `head_center` heatmap shared by WIDER face centers and SCUT/CrowdHuman
  head centers;
- a sub-cell center offset at the configured output stride;
- transformed raw centers for the decoded evaluator;
- masked-out regions for annotations marked difficult or ignored.

There is deliberately no box-size regression: a WIDER face box and a full-head
box have different extents even when their centers describe the same tracking
target. There are also no body channels.

All changing values are in `config/preprocess.toml`. The first clean run uses
400 by 200 with targets at stride 4. An annotated head shorter than 12 input
pixels becomes an ignored region, not background. Training frames with more
than 20 usable targets are skipped so dense crowds do not dominate this small
room/outdoor tracker. Use `--clean` to inspect resizing and target generation
with random augmentation disabled:

```bash
python3 python/preprocess.py --clean
```

The existing compact cache can compare 200 by 100, 320 by 160, and 400 by 200
without another download. It cannot support a meaningful experiment above 400
by 200; that would require regenerating compact images from the originals
rather than stretching WebP files.
