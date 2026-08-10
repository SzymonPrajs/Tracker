# Head-detection dataset investigation

Investigated 2026-08-09 for the initial 288x160 room tracker. The required
label is a **head box**, not a face box. Face-only corpora are therefore not
silently mixed into the target.

Reset status, 2026-08-10: the validated counts below are a historical audit.
All downloaded and derived dataset payloads were subsequently deleted to reclaim
space. Acquisition, conversion, validation, and sampling scripts remain in
`../previous/training/datasets/`, but there is no current local canonical or
packed dataset until those scripts are deliberately rerun.

## Acquisition decision

| Source | Real scenes and labels | Scale | Terms | Decision |
|---|---|---:|---|---|
| [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) | Part A is university classrooms. Pascal-VOC boxes cover the entire visible or occluded head without background. Part B adds diverse Internet scenes. | 4,405 images; 111,251 heads claimed by the page | Academic research only | Download first. Best immediately accessible room data. |
| [R2PPE](https://doi.org/10.5281/zenodo.13851664) | Indoor resuscitation-room simulations, COCO labels, multiple people and PPE. The `Head` category has 29,606 audited boxes. | 10,034 images; 34.5 GB image archive | CC BY-NC 4.0 | Download. Closest domain, despite the large archive. Re-split all 26 recording prefixes because every publisher sequence appears in both supplied train and test JSON. |
| [RPEE-Heads](https://ped.fz-juelich.de/da/2024rpee_heads) | Outdoor field-study crowds, YOLO head boxes, many very small and occluded heads. | 1,886 4K frames; 109,913 boxes; 1.1 GB | CC BY-SA 4.0 | Download. Not room-domain, but unusually useful for tiny/dense heads. Re-split by video. |
| [CrowdHuman](https://www.crowdhuman.org/) | Diverse crowded scenes; each person has a head (`hbox`), visible-body, and full-body box. | 15,000 train + 4,370 validation images; 470K person instances | Non-commercial research/education; no image redistribution | Selected, but official Google Drive currently refuses unattended access. Retry rather than use an unverified mirror. |
| [JRDB/JRDB-Pose](https://jrdb.erc.monash.edu/) | Mobile-robot video in indoor and outdoor human environments, tracked head boxes and pose. The paper reports 600K 2D head boxes. | 60K annotated frames overall | CC BY-NC-SA 3.0 | Excellent later addition, but the official site requires an account/login. |
| [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | Real web images have separate `Human head` and `Human face` classes. | Targeted local selection: 32,144 images, 66,194 true heads, 49,751 paired faces | Annotations CC BY 4.0; images listed as CC BY 2.0, with a warning to verify each image | Acquired by exact image ID after selecting one-to-six-head scenes with a head at least 15% of image height. Primary close/mid-range source. |
| [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) | Close through distant real-scene face boxes, with blur, pose, illumination, invalid, and occlusion attributes. | 12,880 train + 3,226 validation images before local close-range filtering | CC BY-NC-ND on the official page | Acquire train/validation as face-only auxiliary data. Never use its rectangles as full-head size truth. |
| [VGG Hollywood Heads](https://www.robots.ox.ac.uk/~vgg/software/headmview/) | Movie frames with head and upper-body boxes and broad yaw. | 1,122 frames; 273 MB | No explicit dataset grant found on the release page | Acquire. Small but strongly relevant close/mid-shot movie material. |
| [HollywoodHeads](https://www.di.ens.fr/willow/research/headdetection/) | Natural movie frames with complete visible-head rectangles, broad pose, lighting, occlusion, and tracks. | 224,740 frames; 369,846 heads; 5.4 GB | MIT license included in archive | Acquired. Preserve movie-disjoint splits and difficult flags; canonicalize a scale-filtered, temporally subsampled room-oriented subset. |
| [DAD-3DHeads](https://www.pinatafarm.com/research/dad-3dheads/dataset) | Close in-the-wild head boxes, dense 3D meshes, pose, occlusion, expression, illumination, and quality. | 44,898 images; 39.8 GB | CC BY-NC-SA 4.0 | Blocked on the publisher's Hugging Face terms/login flow; acquire after the user accepts access. |
| [VGGHeads](https://www.robots.ox.ac.uk/~vgg/research/vgg-heads/) | Synthetic multi-head scenes with boxes, landmarks, and meshes. | More than 1M images; approximately 187 GB | Dataset terms not identified; code MIT | Released, but the archive alone exceeds current free space before extraction. Defer behind real room-scale sources. |

## Explicit exclusions

- Face-detection corpora box facial appearance, not the complete head. They are
  retained in separate `faces` fields and face heatmaps. Expanding them by a
  fixed heuristic would encode systematic label error, especially for profiles
  and backs of heads, so they never supervise full-head size.
- Point-supervised crowd-counting sets are not used for size or edge targets.
  Their points could support a separately masked centre-only loss later, but a
  point must not be presented as a measured head box.
- Brainwash is not acquired from third-party copies because its official
  distribution was withdrawn.
- R2PPE videos are not downloaded: the released image archive already contains
  the annotated frames, so the 6 GB video archive would duplicate storage.

## Local acquisition result

The scripted run completed six full-head sources plus one separate face-only
auxiliary source:

| Source | Cached images | Usable heads | Rejected source boxes | Split (train/val/test) |
|---|---:|---:|---:|---:|
| SCUT-HEAD | 4,405 | 111,248 | 6 zero-area boxes | 2,543 / 862 / 1,000 (publisher split) |
| RPEE-Heads | 1,886 | 109,895 | 18 boxes wholly outside their images | 1,502 / 192 / 192 (66 whole recordings) |
| R2PPE | 10,034 | 29,606 | 0 | 7,938 / 1,013 / 1,083 (26 whole recordings) |
| Open Images Human head | 32,144 | 66,194 | 0 | 29,792 / 2,352 / 0 (publisher split) |
| VGG Hollywood Heads | 1,122 | 2,067 | 0 | 884 / 119 / 119 (33 whole movies) |
| HollywoodHeads | 27,915 selected from 224,740 | 46,313 | 0; 402 difficult retained as ignored | 20,634 / 6,155 / 1,126 (publisher movie-disjoint split) |
| **Full-head total** | **77,506** | **365,323** | **24 rejected; 402 ignored** | **63,293 / 10,693 / 3,520** |

WIDER FACE contributed another 6,964 selected images and 12,905 usable face
boxes (5,552 train / 1,412 validation). Open Images also preserves 49,631
usable paired face boxes beside its true head boxes. At validation time, the
local canonical store contained 84,470 images and 62,536 usable face
annotations, but face-only rows were excluded from the initial full-head
training loader.

Every image, JSONL row, transformed box, sequence assignment, and 16-bit
heatmap passed the pipeline validator. The reproducible raw downloads and
extracted originals were deleted after successful derivation. The retained
576x320 future cache, 288x160 deployment cache, and fixed-shape packs were the
working data at validation time; they too were deleted in the 2026-08-10 reset.
CrowdHuman remains
selected but not acquired because its official Google Drive endpoint rejected
unattended access; its dense-crowd geometry is also lower priority for this
room-camera model.

The two largest new head sources match the intended deployment distribution:
Open Images has median head height 24.6% and 91.2% one-to-four-head images;
the HollywoodHeads subset has median head height 31.9% and 99.2%
one-to-four-head images. The smaller VGG movie set has median head height 27.0%.

## Canonical representation

The transient raw archives and original images are removed after validation. A
canonical JSONL row retains their provenance, original dimensions and split,
source `[x,y,w,h]` boxes,
letterboxed cache boxes, ignore/occlusion attributes, recording ID, and the
training split. SCUT and both Hollywood sources preserve publisher splits;
RPEE-Heads and R2PPE are re-split by whole recording with an image-balanced
80/10/10 assignment because their supplied layouts do not provide safe
sequence-independent splits.

The initial room-camera sampler is preserved in
`../previous/training/datasets/room_mix.json`.
Its per-epoch probability mass is 42% Open Images, 18% HollywoodHeads, 18%
SCUT-HEAD, 15% R2PPE, 4% VGG movie heads, and 3% RPEE-Heads. Within each source,
one-to-four-head frames receive full weight, five-to-eight receive 0.5, dense
frames receive 0.15, and true empty frames receive 0.35 for negative training.

The future cache is 576x320 RGB JPEG (quality 90), exactly twice the current
288x160 model input. It preserves room-scale appearance for a later model up to
roughly twice the current dimensions without retaining multi-megapixel source
images. Images are letterboxed rather than stretched.

Each cached image also has a lossless 144x80 uint16 probability PNG at output
stride 4. For a cache-space head box with centre `(cx,cy)` and half extents
`(hx,hy)`, its continuous target is

```text
p(x,y) = exp(log(0.05) * (((x-cx)/hx)^2 + ((y-cy)/hy)^2))
```

Thus the probability is 1 at the quantized centre, 0.05 at the midpoint of
each box edge, and 0.0025 at a corner. Multiple heads combine by probabilistic
union, `1 - product(1 - p_i)`. Ignored boxes do not enter the positive map.
The formula, edge probability, stride, overlap rule, and integer scale are
stored in every record, so augmentations can regenerate the target from the
original boxes.

The secondary 288x160 cache regenerates its 72x40 target directly from scaled
canonical boxes. The training pack additionally stores fixed-shape RGB,
heatmap, offset, size, and mask arrays, eliminating decode and target-generation
work during training. A global 576x320 cache should not be mistaken for
preserving every tiny head that existed in a multi-megapixel source.
