# Data acquisition and canonical corpus

## Objective

Build a broad but auditable corpus for heads in real scenes. The data should
cover full and visible heads, faces, full and visible bodies, poses, indoor and
outdoor scenes, ordinary rooms, crowds, occlusion, darkness, distance, flat
optics, and native fisheye optics. Wider coverage is useful only when annotation
meaning and split independence remain explicit.

## Research authorization and corpus policy

The user has authorized every listed source for this personal, non-commercial
research project. There is no licence-admission gate, but authorization does not
justify unbounded local storage. Acquire complete moderate sources and
deterministic useful subsamples of very large sources. A source that requires a
multi-hundred-gigabyte monolithic download merely to retain a small fraction is
skipped until partial retrieval is possible.

Every source still has a task role:

- **direct**: native head boxes or points can supervise a compatible head task;
- **auxiliary**: face, body, mask, or pose semantics remain separate tasks;
- **negative/stress**: scenes strengthen negatives, low light, density,
  occlusion, or native-fisheye evaluation.

## Ordered portfolio

| Source | Native value | Research role | Technical/semantic constraint |
|---|---|---|---|
| CrowdHuman | head, visible-body, full-body boxes and ignore/occlusion metadata | direct + auxiliary | head semantics and ignore regions must remain exact |
| RPEE-Heads | visible heads at distance and high density across 66 videos | direct | split by recording; outdoor bias |
| JRDB-Pose | indoor/outdoor robot sequences, head boxes, body poses, occlusion | direct + auxiliary | split by sequence/location; 360° camera bias |
| SCUT-HEAD | full/amodal head boxes, including classroom scenes | direct | full/amodal semantics; classroom similarity |
| Open Images V7 | class-scoped head/face/person boxes and human-verified negative labels | direct + auxiliary + negative | exact MID, group flags, and class-coverage parsing |
| COCO 2017 | person boxes, instance masks, keypoints, diverse contexts | auxiliary + scene diversity | no direct full-head truth |
| JHU-CROWD++ | tiny head points/approximate boxes, adverse weather and low light | direct point + stress | approximate geometry and extreme density |
| WIDER FACE | face boxes across scale, pose, occlusion, and events | auxiliary + stress | face is not full head |
| DARK FACE | annotated faces in real nighttime scenes | auxiliary + low-light stress | face-only supervision |
| NWPU-Crowd | points/boxes at extreme density | direct point + stress | distribution imbalance and very small instances |
| HollywoodHeads | movie-frame head boxes and pose/lighting diversity | direct + stress | temporal/domain leakage must be grouped |
| OCHuman / CrowdPose | occluded masks, body boxes, and keypoints | auxiliary + occlusion stress | body/pose cannot become exact head boxes |
| FishEye8K | native fisheye person boxes across 18 cameras | auxiliary + native-fisheye stress | person, not head |
| NightOwls | night pedestrian/background frames and ignore regions | auxiliary + negative/stress | bounded deterministic subsample only; thresholded annotation semantics |
| WoodScape | native fisheye masks/boxes | auxiliary + native-fisheye stress | automotive domain and non-head semantics |

Initial download order:

```text
CrowdHuman
→ RPEE-Heads
→ JRDB-Pose
→ SCUT-HEAD
→ Open Images verified negatives
→ COCO
→ JHU-CROWD++
→ WIDER FACE and DARK FACE
→ NWPU-Crowd, HollywoodHeads, OCHuman, CrowdPose
→ FishEye8K, NightOwls, WoodScape, and every remaining listed source
```

Download annotations and metadata before pixels whenever the source permits.
Confirm counts, schema, transfer layout, checksums, and access mechanics, then
freeze the complete-source or subsample manifest before pixel transfer.

## Bounded sampling and storage profile

The corpus configuration declares `max_corpus_bytes` and `min_free_bytes`;
each source configuration declares `max_selected_images`,
`max_selected_packet_bytes`, `max_temporary_bytes`, `max_sequences`, a seed,
and required strata. Select by source group before
pixels when possible, balancing head scale, density, lighting, indoor/outdoor,
occlusion, pose, native distortion, negative type, and sequence/camera. Preserve
rare cases deliberately, but prevent a massive source from dominating merely
because it contains more frames.

The packet storage profile is derived from the maximum current candidate model
input `(W_max, H_max)`. For an image larger than the envelope, preserve aspect
ratio and resize once so it fits within `(2*W_max, 2*H_max)`—twice each target
dimension, four times its pixel count. Do not crop, letterbox, or upscale an
image that is already smaller. Transform vector geometry by the same scale.
Record the source and stored dimensions, filter, codec, encoder version, and
quality/lossless setting. Select the storage codec only after a small
decode-speed, image-quality, and space comparison; do not hide that choice.

If the hardware/model feedback loop later selects an input larger than half a
packet's stored dimensions, that packet is insufficient. Regenerate it from its
source/subsample manifest; never upscale the minimized packet and call it new
source evidence.

## Class-specific negative evidence

Negative status is a label, not a missing row.

Use, in priority order:

1. target-camera frames manually certified as `no_human_verified` against a
   frozen ontology covering people, heads, and faces;
2. Open Images records where an exact class MID is explicitly verified absent;
3. source-specific negatives such as
   `pedestrian_above_50px_verified_negative` from NightOwls;
4. manually audited candidates from other selected sources.

For every image/class pair, store a `CoverageRecord` with exact class/MID,
`positive_exhaustive | verified_absent | partial | unknown`, annotation size
threshold if any, and evidence origin. An Open Images `Person=0` record does not
establish `Human face=0` or `Human head=0`; an unmentioned class is `unknown`.
NightOwls background frames are not `no_human` because its pedestrian
annotation rules include size thresholds and ignore cases.

The negative audit deliberately includes posters/screens, mannequins/statues,
reflections, helmets, balls, foliage, lamps, skin-coloured objects, empty rooms,
pets, vehicles, and moving non-human objects. Each region remains verified
negative, ignore, or unknown according to the evidence actually available.

## Canonical records

```text
ImageRecord
  source, source_version, source_url, source_image_id
  source_sha256, source_width, source_height, orientation
  stored_sha256, stored_width, stored_height, storage_profile
  sequence_id, camera_id, scene_id, duplicate_group
  source_split, internal_split
  research_authorization_id, acquisition_checksum
  coverage_records[] and negative/unknown state

CoverageRecord
  exact_class_or_mid
  status: positive_exhaustive | verified_absent | partial | unknown
  annotation_size_threshold
  evidence_origin

InstanceRecord
  source_instance_id
  semantic:
    head_full | head_visible | face_visible |
    person_full | person_visible | person_mask | pose | head_point
  geometry: bbox | point | polygon/RLE | keypoints
  quality: exact | approximate | derived | weak
  occluded, truncated, ignored, uncertain
  derivation_rule and parent_instance_id
```

Native semantics have explicit derivation limits:

| Native semantic | Allowed target | Forbidden derivation |
|---|---|---|
| exact full/amodal head box | `head_full` centre and size | visible-head claim without a native rule |
| exact visible head box | `head_visible` centre and size | full/amodal head size |
| head point | `head_point` localization | any head box or mask |
| face box | `face_visible` auxiliary task | head positive or enlarged “ground truth” head |
| full/visible body box or mask | matching `person_*` auxiliary task | head position or size |
| pose keypoints | pose task; optional weak head centre only when the declared visible-keypoint rule passes | trusted full/visible head box |

Derived and weak targets are optional, separately ablated, and never enter
strict exact-head validation. Pseudo-label experiments require their own
manifest and gate and remain excluded from strict validation.

## Per-source packet contract

Each completed source is an independently inspectable directory:

```text
data/packets/<source>/<packet-version>/
  README.md
  packet.json
  selection.json
  records.jsonl
  images/
  reports/validation.json
  checksums.sha256
```

`packet.json` records source/version URLs, streamed archive checksums, adapter
and code versions, storage profile, record counts, and output hashes.
`selection.json` records the deterministic complete/subsample decision, seed,
caps, groups, strata, inclusions, and exclusions. `records.jsonl` retains native
semantics and canonical vector labels for selected images. The README explains
the source adapter and one-command validation. No heatmaps, masks, augmentation,
model-sized tensors, raw archives, or extraction trees live in the packet.

Packets are immutable. A changed source selection, storage envelope, adapter,
or encoder creates a new packet version; it never mutates a packet consumed by
a recorded training run.

## Acquisition pipeline

Only one source may occupy raw staging at a time:

1. Fetch annotations/metadata and freeze the complete/subsample selection.
2. Calculate peak temporary and packet space; fail before transfer if the
   per-source/global cap or free-space reserve would be crossed.
3. Create a unique guarded temporary staging directory for exactly this source.
4. Stream selected files where possible. If a seekable archive is unavoidable,
   retain only that one temporary archive. Hash bytes during transfer.
5. Verify archive type/checksum and safe extraction paths; reject traversal,
   links outside staging, unexpected files, and corrupt decodes.
6. Parse native annotations, select records, normalize orientation, resize only
   to the 2× storage envelope, encode, and transform vector labels.
7. Preserve publisher `source_split`; create group-safe `internal_split` only
   within source training data. Never promote official validation/test records.
8. Reconcile counts, rejections, ignore flags, bounds, duplicates, selection
   strata, stored dimensions, and output hashes. Render conformance samples.
9. Build the packet in a temporary output directory and run its independent
   validator. Atomically promote it only after every check passes.
10. Delete only the resolved source staging directory, using its exact guarded
    path. Verify that its archive and extraction tree no longer exist and record
    cleanup success in the run report.
11. Refuse to start the next source unless the prior packet passes and the raw
    staging area is empty. On failure, write the diagnostic and clean staging;
    tiny versioned fixtures—not downloaded raw data—support adapter debugging.

Production acquisition has no keep-raw mode. Raw bytes exist only transiently
inside the active source's staging directory. Scripts, source/subsample
manifests, checksums, adapters, reports, and compact packets are retained.

## Efficiency policy

- Store each selected image once in its source packet; keep labels as vector
  geometry.
- Rasterize heatmaps and masks only at the final run/output resolution.
- Use source-balanced and scale-balanced sampling rather than instance-count
  sampling that lets dense crowds dominate.
- Keep native validation images unaugmented.
- At the actual model resolution, report transformed head-width/height
  histograms and all instances below one output cell or the declared minimum
  useful pixel size; those cases cannot silently remain valid box targets.
- Record decode, augmentation, transfer, and training throughput separately so
  a fast model is not hidden behind a slow data loader.
- Do not duplicate packets into permanent model-resolution datasets. Training
  reads packets and creates targets in memory; any optional performance cache is
  content-addressed, bounded, disposable, and independently regenerable.

## Stage gate

- Every listed source has a machine-readable result: complete packet, bounded
  subsample packet, partial-retrieval-impossible, retryable access failure, or
  source-unavailable evidence.
- Every packet records a deterministic selection and 2× storage profile; no
  stored image was cropped or upscaled.
- Peak staging stayed below its configured cap and free-space reserve.
- Total promoted packet bytes stay below `max_corpus_bytes`.
- All selected files, streamed source checksums, packet files, and counts
  reconcile.
- The raw staging directory is empty, and cleanup success is recorded, before
  the next source begins.
- Native and canonical annotation counts reconcile with documented rejections.
- No exact or reviewed near-duplicate crosses a split.
- Video/movie/camera/location groups do not cross splits.
- Random rendered samples pass per source and scale/occlusion/lighting stratum.
- Certified negatives are manually audited and retain class-specific coverage.
- Source-specific `adapter_conformance_fixtures` remain visible to parser
  development and pass exact numeric/rendered checks.
- An independently labelled `sealed_audit_gold` set is never used for parser,
  augmentation, model, calibration, or threshold tuning.
- Coverage audits include age/child scale, skin tone under low light, headwear,
  hairstyles, mobility aids, seated/lying people, camera height, geography, and
  indoor-room type. Face-heavy pixels receive a privacy/data-minimisation review.
