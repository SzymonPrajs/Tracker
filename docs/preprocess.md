# Preprocess and augment

This phase will be implemented as `python/preprocess.py` with a small editable
config file. It will read the plain `images/` and `labels.jsonl` outputs from
the downloader and create training batches directly; it will not introduce a
second artifact-management system.

The augmentation mixture will include clean images, darkness, sensor-like
noise, blur, and several strengths of full-canvas radial/fisheye warping. A
warp changes the geometry without using a zoom or random resized crop. Every
box is transformed with the same geometry before heatmaps and masks are made.

The stored 400 by 200 maximum is intentionally larger than the likely initial
model input. If experiments require a larger input, change the download config
and regenerate the compact data rather than stretching small images.
