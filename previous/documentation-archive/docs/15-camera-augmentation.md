# Camera and lens augmentation

Status after the 2026-08-10 workspace reset: this chapter records the prior
training experiment. Its code, checkpoints, and logs are preserved under
`../previous/training/`, while the downloaded and derived datasets have been
deleted. Reproduction therefore requires rebuilding the data before entering
the archived snapshot and running these commands.

The first 30-epoch search converged: decoded validation AP peaked at epoch 27,
validation loss changed by less than 0.5% through epoch 30, and the final epoch
did not improve either criterion. Its best checkpoint scored 0.7406 room mAP on
validation but only 0.4485 on the held-out test split. More unmodified epochs
would therefore reinforce the same domain rather than address the observed
generalization gap.

The augmented run starts from the selected epoch-27 weights, but deliberately
resets AdamW and the cosine schedule. This retains the already learned head
features while treating the broader camera distribution as a new optimization
phase.

## Online mixture

Augmentation runs on the accelerator after packed uint8 batches are transferred.
It does not duplicate the 288x160 cache. Every epoch draws new parameters.

| Effect | Default probability | Default range |
|---|---:|---|
| Completely clean sample | 15% | bypasses every effect |
| Horizontal mirror | 50% | exact label transform |
| Normal exposure | within 80% exposure gate | -1.25 to +0.75 EV |
| Severe low light | 30% | -4.0 to -1.5 EV |
| White balance / tint | 60% | up to 18% channel gain |
| Gamma / tone response | 40% | log magnitude 0.25 |
| Saturation | 45% | +/-35% |
| Directional illumination | 35% | up to 45% gradient |
| Local shadow | 25% | up to 65% attenuation |
| Optical vignette | 30% | up to 45%; forced for fisheye samples |
| Shot and read noise | 35% | signal-dependent plus fixed floor |
| Mild defocus / resampling blur | 12% | 3x3 average |
| 130-degree fisheye class | 18% | radial strength 0.10 to 0.25 |
| 180-degree fisheye class | 14% | radial strength 0.25 to 0.45 |
| 10-degree telephoto class | 18% | 2x to 6x crop, log-uniform |
| No lens warp | 50% | preserves the ordinary-room domain |

Exposure, white balance, illumination, vignetting, and sensor noise are applied
in an approximately linear-light RGB space. Gamma, saturation, contrast, blur,
and 8-bit clipping follow. This ordering is intentional: adding a fixed offset
to gamma-encoded pixels is not a useful simulation of low photon counts.

The lens classes are stochastic robustness approximations, not claims that an
uncalibrated source photograph has been optically converted to an exact 130- or
180-degree camera. A source image contains no scene content outside its original
field of view. The radial transform instead reproduces the dominant off-axis
compression and edge deformation. Telephoto augmentation crops 2x to 6x and
usually follows a randomly selected head, with 25% random crops retained for
empty and partial scenes.

## Label geometry

An image-only warp would silently corrupt supervision. The implementation uses
one inverse sampling map for the RGB image and probability heatmap, then:

1. recovers every source head centre from the sparse mask and sub-cell offset;
2. transforms the centre and a 3x3 sampling of each box through the forward lens
   model;
3. clips the transformed box to the frame and drops centres outside the crop;
4. recomputes the 72x40 cell, centred offset, normalized width and height;
5. restores a unit heatmap peak at every retained centre.

The 3x3 box sampling matters because under radial distortion an edge midpoint,
not a corner, can become the axis-aligned extremum. The transform is monotonic
through the configured maximum strength and its forward/inverse round trip is a
unit-tested contract.

OpenCV's fisheye model likewise expresses distortion through a radial angle and
polynomial terms: <https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html>.
The implementation uses PyTorch's normalized sampling grid so image and target
warps share one coordinate system:
<https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html>.

## Reproduction and review

After intentionally rebuilding the dataset, run the historical workflow from
the archived snapshot:

```sh
cd previous
```

Render a labelled sheet before changing augmentation ranges:

```sh
PYTHONPATH=training .tools/tracker/bin/python training/preview_augmentations.py \
  --index 12060 --output training/artifacts/augmentation_preview.png
```

Start the selected augmented continuation:

```sh
PYTHONPATH=training .tools/tracker/bin/python training/train.py \
  --initialize training/artifacts/optimized_model.pt \
  --output training/runs/augmented-v1 \
  --epochs 24 --schedule-epochs 24 --scheduler cosine --lr 0.0015 \
  --patience 0 --save-every-epoch --log-interval 100
```

The clean validation and test splits remain unaugmented. They measure whether
robustness was gained without losing the ordinary-room task; lens- and
lighting-stratified evaluation should additionally be used to select the final
checkpoint rather than relying on training loss alone.
