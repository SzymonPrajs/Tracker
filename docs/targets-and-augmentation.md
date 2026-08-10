# Targets and augmentation

## Target contract

Keep canonical annotations vector-based. Geometry is transformed first; model
targets are regenerated afterwards.

The first baseline is `head_full` or `head_visible` heatmap + offset + size,
chosen according to the admitted corpus and never merged implicitly. Optional
`head_point`, `face_visible`, `person_full`, `person_visible`, mask, and pose
tasks use independent sigmoid channels or separate heads. They overlap
hierarchically and must not be a softmax-exclusive class set. Every semantic
task owns its own offset/size regression channels and validity mask; no shared
regression cell may receive incompatible head, face, and person targets.

For input width `W`, height `H`, and output stride `S`, an object's center is

```text
p = (cx / S, cy / S)
q = floor(p)
offset = p - q
size = (box_width / W, box_height / H)
```

The class heatmap is the maximum of object Gaussians:

```text
H_c(x,y) = max_i exp(-((x-qx_i)^2 + (y-qy_i)^2) / (2 sigma_i^2))
```

Start with a box-relative sigma such as
`max(0.5, min(box_width, box_height) / (6*S))`, then treat radius as a searched
parameter. Offset and size losses exist only at valid positive centers.
The integer centre `q` has heatmap value exactly one, which is required by a
CenterNet-style focal loss that defines positives with `target == 1`.

Required additional masks:

- a positive-center regression mask;
- a validity/ignore mask for crowd regions, uncertain labels, clipping, and
  unlabelled semantics;
- native instance masks only where polygons/RLE really exist;
- a named `box_proxy_mask` if a rectangular support mask is useful.

A max-combined detection heatmap is not a count-density map. If counting is
later required, give it a separate target whose Gaussian mass integrates to one.

Measure same-cell collisions within every semantic task and across any
regression channels that share storage at every candidate stride. Do not
silently drop a second object in the same cell; reduce stride, add scale levels,
separate heads, or define an explicit collision policy. Negative or unknown
semantic channels have zero heatmap and regression loss only where their
coverage mask makes that absence valid.

## Augmentation order

1. Select the source and source/scale-balanced sample.
2. Choose the exact identity path or apply lens/full-canvas geometry to the
   image and vector labels. Identity bypasses remapping and interpolation.
3. Apply exposure and sensor-noise transforms at the declared source/sensor
   resolution.
4. Apply the final no-crop, hardware-matching resize or letterbox to the image
   and vector labels.
5. Clip and validate the final transformed geometry and its representability.
6. Apply exact colour/range conversion and input normalization.
7. Generate heatmaps, offsets, sizes, semantic validity masks, and optional
   auxiliary targets at final output resolution.

There is no random resized crop, head-centered zoom, or telephoto substitute in
the default augmentation. Optional deployment crop logic is a later firmware
feature, not training data diversity.

The resolved input profile states whether its full-frame mapping stretches to
the model aspect ratio or preserves aspect ratio with deterministic letterbox
padding. It must reproduce an exact PPA-supported geometry established during
hardware characterization.

## Full-canvas radial warp

For pixel `p`, optical center `c`, radial distance `r = ||p-c||`, and direction
`d = (p-c)/r` for `r>0`, let `R(d)` be the distance from `c` to the rectangular image
boundary along the same ray. Define normalized radius `rho = r/R(d)`.

Use the boundary-preserving family

```text
g(lambda,kappa,rho)
  = (1-lambda)*rho
    + lambda*atan(kappa*rho)/atan(kappa)

p' = c + R(d)*g(lambda,kappa,rho)*d
```

with `0 <= lambda <= 1` and `kappa > 0`.

- `lambda = 0` is exact flat identity.
- Intermediate values provide mild and medium warps.
- Severity is jointly controlled by `lambda` and `kappa`; `lambda = 1` tends
  towards identity as `kappa` tends to zero and is not inherently strong.
- The center and every boundary ray remain fixed because `g(0)=0` and `g(1)=1`.
- The derivative is positive, preventing radial foldover.
- The transform redistributes the complete scene instead of cropping or
  uniformly zooming it.

For `r=0`, define `p'=c` without evaluating `d`. The map is a continuous
full-canvas bijection and smooth inside each rectangular angular sector, but
`R(d)` changes branch at rays through frame corners, so it is only piecewise
differentiable there. Do not claim a globally defined positive 2-D Jacobian.
Test one-sided finite differences and seam energy on those rays. If visible
seams remain, use a separately validated smooth boundary-tapered family, for
normalized `x,y` in `[-1,1]`:

```text
B(x,y) = (1-x^2)(1-y^2)
r2 = x^2 + y^2
(x',y') = (x,y) * (1 + a*B(x,y)*r2)
```

Admit only sampled `a` ranges whose dense containment, inverse, boundary, and
numeric-Jacobian tests pass.

Use inverse mapping for image resampling and forward mapping for annotations.
Warp polygon and keypoint coordinates directly. Sample bounding-box edges
dense enough to enclose curved transformed edges; four corners are insufficient.
Use image interpolation appropriate to the source, nearest-neighbor for discrete
masks, and regenerate heatmaps from transformed geometry.

For a measured lens, also support a calibrated OpenCV fisheye/Kannala–Brandt
model, but keep it a distinct profile. Synthetic warping cannot create scene
content outside the source field of view or prove native-lens robustness.

## Head size and distance

Apparent head size is a useful scale signal but not metric depth without camera
intrinsics and physical-size assumptions. Pose, age, crop, truncation, source
focal length, and existing distortion all confound it.

A normalized far-object weight may be used for sampling and stress selection:

```text
w = clip(log(h_ref / h) / log(h_ref / h_min), 0, 1)
```

Use it to oversample small heads, tune anti-aliasing/blur, set loss weights, or
condition conservative noise severity. Lens distortion remains one global
radius-based scene transform. If warp strength is conditioned on head scale,
name it a nonphysical curriculum experiment and require an ablation.

For pre-augmentation sampling, `h`, `h_ref`, and `h_min` use clean canonical
normalized height `h_source/H_source` and exclude truncated/weak boxes. For
evaluation strata, scale uses final model-input pixels after geometry and
resize. Thresholds are not reused between these coordinate systems.

## Low-light and noise

Use this sensor-order approximation:

```text
x_lin = inverse_tone_curve(x)
mu = exposure * x_lin
x_sensor = Poisson(photon_scale * mu) / photon_scale
x_sensor += Normal(0, read_sigma^2)
x_out = tone_curve(clip(x_sensor, 0, 1))
```

Profiles may add bounded white balance, vignetting, local illumination,
fixed-pattern/row noise, mild blur, and a measured output tone curve. Here
`photon_scale` is photons per normalized linear unit and `read_sigma` uses the
same normalized sensor-domain units; record black-level and white-balance order.
Add JPEG artifacts
only if the actual inference path uses JPEG. Fit ranges later from OV5647 frames
rather than assuming generic camera noise is representative, and keep all
pre-characterization ranges explicitly provisional.

## Input-representation simulation

Photometric augmentation first constructs one common latent sensor exposure.
The measured input profile then produces one of three static branches:

- OV5647-like RAW10 Bayer: apply the measured colour-filter pattern, black and
  white levels, gain/noise, packing-independent numeric values, and the exact
  Bayer-aware reduction plus INT8 mapping used on the board;
- ISP luminance: reproduce measured ISP operations, Y range, PPA scaling, and
  GRAY8/INT8 rounding;
- ISP RGB: reproduce measured ISP colour/tone operations, PPA scaling/format,
  channel order, range, and INT8 rounding.

RAW10 is not treated as ten-bit luminance. Synthetic Bayer conversion from
ordinary RGB data is an approximation whose parameters are fitted on paired or
controlled OV5647 captures and whose value is separately ablated. Geometry is
applied before the camera representation is sampled so Bayer phase and final
resize behavior remain explicit.

The starting mixture is a configuration, not a permanent constant. It must
contain identity, flat, mild, medium, and strong branches and must report their
observed frequencies. Preserve a meaningful clean branch so robustness training
does not destroy ordinary-room accuracy.

## Property and adversarial tests

- Identity produces pixel and label identity within declared interpolation
  tolerance.
- The center and sampled frame boundary remain fixed.
- Radial derivative stays positive; sector-interior numeric Jacobians stay
  positive; one-sided corner-ray derivatives and visible seam energy pass.
- Every output pixel inverse-maps to a valid source coordinate.
- No pre-warp annotation disappears; any later loss is documented clipping.
- Forward/inverse point round-trip error stays below a subpixel limit.
- Polygons, keypoints, enclosing boxes, validity masks, and rendered objects
  remain aligned.
- Heatmaps are regenerated, not warped.
- Class-certified negative samples remain finite-loss with zero positive
  regression loss only for valid covered semantics.
- Fixed sample and worker seeds replay exactly.
- Measured flat/mild/medium/strong frequencies match the resolved config.
- Reviews are stratified by head size, source, crowding, darkness, border
  position, occlusion, and warp strength.
- Native fisheye, synthetic fisheye, and flat metrics are reported separately.
- Final-input head-size histograms expose objects below one output cell or the
  declared useful-pixel limit; those instances are ignored or handled by an
  explicit policy rather than silently becoming positives.
