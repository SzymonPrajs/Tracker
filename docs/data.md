# Data acquisition and canonical corpus

## Objective

Build a broad but auditable corpus for heads in real scenes. The data should
cover full and visible heads, faces, full and visible bodies, poses, indoor and
outdoor scenes, ordinary rooms, crowds, occlusion, darkness, distance, flat
optics, and native fisheye optics. Wider coverage is useful only when annotation
meaning, licensing, and split independence remain explicit.

## Admission policy

Every source receives one of three states:

- **candidate**: high task value, but admitted only after exact licence/access,
  annotation, and lineage checks pass;
- **auxiliary**: useful semantics that do not directly supervise a full head;
- **gated**: unresolved image rights, login/terms, size, or domain risk.

Maintain separate `research-only` and `deployable-lineage` manifests if later
commercial use is conceivable. A research-only source must never silently
contaminate the deployable lineage.

## Ordered portfolio

| Source | Native value | Admission | Principal risk |
|---|---|---|---|
| CrowdHuman | head, visible-body, full-body boxes and ignore/occlusion metadata | primary candidate, research-only | non-commercial/research terms; no redistribution |
| RPEE-Heads | visible heads at distance and high density across 66 videos | primary candidate | split by recording; preserve DOI/licence evidence; outdoor bias |
| JRDB-Pose | indoor/outdoor robot sequences, head boxes, body poses, occlusion | primary candidate, research-only | split by sequence/location; 360° camera bias |
| SCUT-HEAD | full/amodal head boxes, including classroom scenes | primary candidate, research-only | classroom similarity and academic-only terms |
| Open Images V7 | class-scoped head/face/person boxes and human-verified negative labels | conditional subset | exact MID/coverage parsing and per-image licence allowlist |
| COCO 2017 | person boxes, instance masks, keypoints, diverse contexts | auxiliary | no direct full-head truth; source-image licenses vary |
| JHU-CROWD++ | tiny head points/approximate boxes, adverse weather and low light | gated auxiliary | unresolved pixel terms, approximate geometry, extreme density |
| WIDER FACE | face boxes across scale, pose, occlusion, and events | gated auxiliary | unresolved pixel terms; face is not full head |
| DARK FACE | annotated faces in real nighttime scenes | gated auxiliary | unresolved pixel terms; face-only supervision |
| NWPU-Crowd | points/boxes at extreme density | gated | access/license and distribution imbalance |
| HollywoodHeads | movie-frame head boxes and pose/lighting diversity | gated | unclear pixel rights and severe temporal/domain leakage |
| OCHuman / CrowdPose | occluded masks, body boxes, and keypoints | gated auxiliary | repository license may not grant image rights |
| FishEye8K | native fisheye person boxes across 18 cameras | gated validation | person, not head; dataset license not explicit |
| NightOwls | night pedestrian/background frames and ignore regions | gated subset | non-commercial and roughly 285 GB |
| WoodScape | native fisheye masks/boxes | excluded by default | proprietary data |

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
→ gated sources only after review
```

Download annotations and terms before large image archives. Confirm source
counts, schema, archive size, checksums, and access conditions first.

## Class-specific negative evidence

Negative status is a label, not a missing row.

Use, in priority order:

1. target-camera frames manually certified as `no_human_verified` against a
   frozen ontology covering people, heads, and faces;
2. Open Images records where an exact class MID is explicitly verified absent;
3. source-specific negatives such as
   `pedestrian_above_50px_verified_negative` from NightOwls;
4. manually audited candidates from other admitted sources.

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
  sha256, width, height, orientation
  sequence_id, camera_id, scene_id, duplicate_group
  source_split, internal_split
  license_id, license_snapshot, acquisition_checksum
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

## Acquisition pipeline

1. Add the source manifest, expected files, terms snapshot, and checksums.
2. Download resumably into an ignored immutable raw area.
3. Verify archive type, size, checksum, and safe extraction paths.
4. Decode-test every image and normalize orientation without losing original
   dimensions or identity.
5. Parse native annotations without target generation.
6. Reconcile native and parsed counts, rejected rows, ignore flags, and bounds.
7. Preserve the publisher's split as immutable `source_split`; never promote
   official validation/test records into training.
8. Deduplicate exact bytes, then review perceptual-hash and embedding neighbours.
9. Within source training data, create a group-safe `internal_split` by video,
   movie, camera, location, URL, and duplicate cluster. Preserve the official
   split separately for benchmark comparison and identify which each report uses.
10. Freeze canonical manifests before augmentation.
11. Materialize immutable efficient shards for a declared run resolution, or
    stream from archives when that is demonstrably faster and smaller.

Raw downloads remain untracked. Scripts, manifests, checksums, license
snapshots, parser reports, and derived-data recipes are versioned. Large raw
archives may be deleted only after a complete reproducibility and validation
report exists.

## Efficiency policy

- Store source pixels once; keep canonical labels as vector geometry.
- Rasterize heatmaps and masks only at the final run/output resolution.
- Use source-balanced and scale-balanced sampling rather than instance-count
  sampling that lets dense crowds dominate.
- Keep native validation images unaugmented.
- At the actual model resolution, report transformed head-width/height
  histograms and all instances below one output cell or the declared minimum
  useful pixel size; those cases cannot silently remain valid box targets.
- Record decode, augmentation, transfer, and training throughput separately so
  a fast model is not hidden behind a slow data loader.

## Stage gate

- Every admitted source has explicit access and license evidence.
- All expected files and checksums reconcile.
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
