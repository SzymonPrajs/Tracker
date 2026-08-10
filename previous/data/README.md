# Head-detection data

Reset status, 2026-08-10: all downloaded, extracted, processed, packed, and
state payloads have been deleted. This README and the acquisition/conversion
scripts are preserved so a future rebuild remains possible.

This directory is managed by `training/datasets/pipeline.py`.

- `raw/`: temporary resumable downloads; empty after a successful full rebuild.
- `extracted/`: temporary safely unpacked sources; empty after a successful rebuild.
- `processed/576x320/`: letterboxed RGB JPEGs, canonical head/face boxes, and
  lossless 144x80 16-bit centre-probability maps.
- `processed/288x160/`: deployment-size JPEGs plus directly regenerated 72x40
  maps for inspection and target-architecture experiments.
- `packed/288x160/`: fixed-shape RGB, heatmap, offset, size, and mask memory maps
  used by the fast training loop; WIDER face-only rows are intentionally absent.
- `state/`: resumable download and extraction state.

Only this README is committed. The data itself is ignored because several
sources prohibit redistribution and because the generated cache is large.
Every source URL, locally verified checksum, terms, and conversion choice lives
in `training/datasets/sources.json` so the entire acquisition is repeatable.

Rebuild everything, one download at a time, and remove archives/extracted
source images only after both caches and every packed split validate:

```sh
make data-all
```

Pass `--keep-raw` directly to `training/datasets/rebuild.py` only for a
diagnostic rebuild. Existing valid 576x320 sources are reused; use
`--force-acquire` to reacquire them.

Run one source at a time:

```sh
make data-status
.tools/tracker/bin/python training/datasets/pipeline.py download rpee_heads
.tools/tracker/bin/python training/datasets/pipeline.py extract rpee_heads
.tools/tracker/bin/python training/datasets/pipeline.py convert rpee_heads
.tools/tracker/bin/python training/datasets/pipeline.py validate rpee_heads
```

Or run the entire resumable sequence with `pipeline.py prepare SOURCE`. Use
`download SOURCE --file NAME` when a multi-file source must be fetched one
archive at a time. `pipeline.py status` reports partial byte counts and expected
percentages while downloads run.

Open Images has an additional reproducible selection and image-ID stage so the
complete nine-million-image corpus is never downloaded:

```sh
.tools/tracker/bin/python training/datasets/pipeline.py download open_images_human_head --file train_boxes
.tools/tracker/bin/python training/datasets/pipeline.py download open_images_human_head --file validation_boxes
.tools/tracker/bin/python training/datasets/pipeline.py select open_images_human_head
.tools/tracker/bin/python training/datasets/pipeline.py download-images open_images_human_head --workers 16
.tools/tracker/bin/python training/datasets/pipeline.py convert open_images_human_head
.tools/tracker/bin/python training/datasets/pipeline.py validate open_images_human_head
```

Each canonical JSONL row permanently records the original box, source image
dimensions, transformed cache box, provenance path, source split, and configured
canonical split before transient inputs are removed. Video sources are grouped
by recording; sources without reliable recording IDs preserve publisher splits.
Face-only corpora retain `faces` and a separate `face_heatmap`; they are excluded
from full-head size supervision. Heatmaps are derived caches and can always be
regenerated from the retained canonical boxes.
