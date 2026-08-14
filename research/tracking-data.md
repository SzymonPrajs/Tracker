# Direction: temporal head-tracking data

## Conclusion

There is no single perfect dataset for this project. The smallest defensible
training set combines four different kinds of evidence:

1. **CAVIAR's 17 head-enriched sequences** supply small, directly downloadable
   indoor walking, stopping, meeting, entry, exit, and occlusion clips with
   persistent IDs and manually placed head circles for some tracked people. Head
   coverage must be audited before any frame is used as exhaustive supervision.
2. **JRDB-Pose** supplies real head boxes, persistent identities, occlusion, and
   fixed and moving cameras. It is the best public structural match.
3. **ChokePoint** supplies the exact behaviour we care about first: one person
   walking into or out of a fixed-camera portal. Its labels are faces rather than
   full heads, so they need conversion and correction.
4. **Our own OV5647 recordings** supply the actual lens, ISP, exposure behaviour,
   room, viewing height, and target use case. They are not optional merely because
   public video is available.

Add a deliberately small sequence-balanced slice of **HT21/CroHD** only as a hard
multi-person and association stressor. HollywoodHeads is useful later for natural
head-track appearance diversity, but it is not the fixed-room domain. Do not
assemble another indiscriminate hundreds-of-gigabytes corpus.

The practical first target is about **60--90 minutes of our own varied video**,
the useful CAVIAR clips, a small ChokePoint subset, a sequence-balanced JRDB-Pose
subset, and a capped HT21 subset. Existing still-image head data remains useful
for the appearance path.
Temporal data teaches displacement, persistence, entries, exits, occlusion, and
identity; it does not need to relearn every possible face from video.

## What must be labelled

The temporal model currently predicts:

```text
head heatmap + x/y sub-cell offset + x/y displacement to the previous frame
```

The recurrent maps and previous-owner prior are inputs/state, not annotations.
The irreducible human-reviewed annotation is therefore:

```text
sequence, frame, head rectangle, track_id, visibility, ignore/outside
```

Annotate **every head that could be a target**, not only the first entrant. The
first-subject rule is a tracker policy which can be derived deterministically from
track IDs. Labelling only the chosen subject would incorrectly teach every other
head to be background.

Use one consistent head definition: the apparent full head, including hair and
ordinary headwear, but excluding neck and shoulders. For partial occlusion,
estimate the full box only when its centre remains credible; record visibility and
mark uncertain targets `ignore`. A fully invisible head may keep its track identity
through an occlusion interval, but it must not create a positive heatmap target.
Faces alone are insufficient because the deployed camera must also see profile and
back-of-head views.

Do **not** hand-label centres, heatmaps, motion surfaces, displacements, or owner
IDs. Derive them after resizing and augmentation. This lets image size, output
stride, Gaussian radius, temporal spacing, and ownership rules change without
re-annotating video.

## Public data, in integration order

| Source | Useful evidence | Scale and access | Use here |
|---|---|---|---|
| [CAVIAR](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/) | Indoor walking, stopping, browsing, meetings, crossings and occlusion with persistent IDs; 17 sequences add manual head centre/radius for only some tracked targets | 78 hand-labelled sequences and about 90,000 frames, 384x288 at 25 fps; individual MPG/XML pairs are direct small downloads; CC BY-SA | **First integration because it is small and direct, not because its heads are exhaustive.** Use only fully covered frames or correct/mask every missing head. |
| [JRDB-Pose](https://jrdb.erc.monash.edu.au/dataset/pose) | Head boxes with track IDs, pose and occlusion; indoor/outdoor; stationary and moving human-height platform | JRDB has 54 sequences and 60,000 frames; JRDB-Pose reports 600,000 head boxes; account download; CC BY-NC-SA 3.0 | **Primary public temporal source.** Sample by sequence and scene, not by raw box count. |
| [ChokePoint](https://zenodo.org/records/815657) | People naturally enter and leave fixed-camera portals; the first 100 frames of each sequence are empty background; two crowded sequences | 48 sequences, 64,204 labelled face images, 800x600 at 30 fps; 11.7 GB total, but individual sequence archives are downloadable | **Primary behaviour source.** Start with a few separately downloadable sequences; convert face tracks to head boxes and correct them. |
| [HT21 / CroHD](https://motchallenge.net/data/Head_Tracking_21/) | Exact head trajectories under severe density, occlusion and distractors | 9 full-HD 25 fps sequences, 11,463 frames, 2,276,838 head boxes and 5,230 tracks; elevated cameras; 4.1 GB | **Small stress subset only.** Its millions of correlated crowd boxes must not dominate the room task. |
| [HollywoodHeads](https://www.di.ens.fr/willow/research/headdetection/) | Natural head appearance and short head tracks across films | 224,740 frames, 369,846 head boxes and 3,872 tracks; 5.4 GB | Later selected continuous clips only. Existing still data already covers much of its appearance value, and film cuts/domain are not the first room experiment. |
| [HDA](https://vislab.isr.tecnico.ulisboa.pt/hda-dataset/) | Synchronized indoor office cameras, identities, crowd and occlusion | 18 cameras, 75,207 labelled frames and 85 people, sampled at 1--5 fps | Secondary domain evidence; sampling is too sparse for the main motion recurrence. |
| [PoseTrack21](https://github.com/anDoer/PoseTrack21) | Diverse real videos, person tracks, keypoints and occlusion | Download requires its licence form/token; head boxes must be inferred from keypoints | Later moving-camera/generalization source, not necessary for the first fixed-camera result. |
| [EasyCom](https://github.com/facebookresearch/EasyComDataset) | Egocentric restaurant conversations, close heads, occlusion and moving viewpoint | About 6.6 hours, 1920x1080 at 20 fps, with generated face/head boxes, participant IDs and head pose | Best later egocentric supplement; too close/conversational and too large for fixed-camera stage one. |
| [PersonPath22](https://amazon-science.github.io/tracking-dataset/personpath22.html) | 236 mostly static-camera videos with exhaustive person tracks, stationary and severe-occlusion tags | Visible/amodal person boxes, not heads; CC BY-NC 4.0 | Useful only if a teacher-generated head layer is worth reviewing. |
| [JTA](https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=25) | Synthetic perfect pose, identity and occlusion | 512 full-HD 30-second videos and roughly 10 million poses; GTA-derived conditions | Optional ablation for rare occlusion, never the validation set or primary visual domain. |

[Oxford Town Centre](https://www.robots.ox.ac.uk/~lav/Papers/benfold_reid_cvpr2011/benfold_reid_cvpr2011.html)
is another strong direct-head lead: its project reports 71,500 manual head
locations in 1080p/25 fps video. The original data download currently returns
404, so it is recorded as unavailable rather than placed in the executable plan.

Other avenues were deliberately rejected for stage one:

- [LaSOT](https://hengfan2010.github.io/projects/LaSOT/),
  [GOT-10k](https://got-10k.aitestunion.com/), and TrackingNet label one selected
  generic object, so other people can become false negatives; LaSOT alone is
  roughly 227 GB.
- [TAO](https://github.com/TAO-Dataset/tao), AVA, and Kinetics have
  sparse/federated boxes or no exhaustive spatial tracks; unlabelled people cannot
  safely be treated as background.
- MOT17/20, DanceTrack, [BDD100K](https://github.com/bdd100k/bdd100k), JAAD, and
  PersonPath22 are body-track sources that need another pseudo-head layer;
  CroHD/JRDB already cover the immediate value.
- MOTSynth and JTA are synthetic and huge; a tiny JTA ablation is enough if real
  occlusion remains underrepresented.
- SCUT-HEAD and the existing still corpus remain valuable appearance data but are
  deliberately de-redundified images, not temporal evidence.
- [Cchead](https://github.com/kailaisun/Cchead) is a useful CroHD alternative with
  more than two million head boxes, but its dense classroom/overhead domain
  repeats the same crowd imbalance.

More raw frames would add storage and annotation work without filling a current
evidence gap.

### Exact first public download

Begin with data that can answer a question, not every available archive:

- JRDB-Pose: choose indoor fixed-camera, indoor moving-camera, outdoor, and
  occlusion-heavy sequences as separate groups; keep official validation/test
  boundaries where supplied. Its annotations are made at 7.5 fps with intervening
  frames linearly interpolated, and people below its body-area threshold can be
  unlabelled. Preserve that provenance, prefer native labelled instants initially,
  and never treat an unverified small person as negative background.
- CAVIAR: begin with `Meet_WalkTogether2` and `Meet_WalkSplit`, which the official
  index marks as having all tracked targets enriched, then expand through the 17
  listed clips. Convert each head circle `(xc, yc, radius)` to a centre plus square
  head box; retain its occlusion and persistent object ID. Still count head
  coverage against all person tracks frame by frame. Keep a frame only if every
  relevant person has a head, otherwise correct the missing heads or mask their
  regions. The source warns that some Lisbon videos contain an occasional
  duplicate frame and missing successor, so preserve timestamps/frame mapping
  rather than assuming every adjacent frame is new.
- ChokePoint: download its 456 kB ground-truth archive and two to four sequence
  archives covering entering, leaving, two portals, and at least two cameras.
- HT21: download annotation-only files first, inspect scale/density, then retain a
  bounded set of short contiguous segments or moving crops from each training
  sequence. Never sparsely pluck independent frames for a motion experiment.

Each source should be downloaded, converted to the compact local form, checked,
and have its raw archive removed before the next source is fetched. Keep a source
record containing the source URL, split, licence, archive checksum, conversion
version, and retained sequence list.

The first mature packet should be about **110,000--120,000 compact frames**, not
millions:

| Source | Retain initially |
|---|---:|
| Own OV5647, including its negatives | about 54,000 frames / 60 minutes |
| CAVIAR audited-head and selected behaviour clips | 15,000--20,000 frames |
| JRDB-Pose, sequence-balanced | about 20,000 frames |
| ChokePoint, one camera per selected event initially | 15,000--20,000 frames |
| HT21 association stress | at most 5,000 frames |

This is a ceiling for the first serious experiment, not a prerequisite. Training
should start after the pilot and first CAVIAR conversion so loader and label errors
are discovered before the corpus is filled.

## Record our own data

Record the original OV5647 stream at 1080p30 when practical. Preserve a small
immutable original-quality holdout, then make a compact constant-frame-rate copy
with maximum width **800 pixels at 15 fps**, preserving the source aspect ratio.
For a 1920x1080 mode that is normally 800x450: it keeps at least twice the largest
current 400x200 input while leaving the later crop-versus-letterbox decision open.
Prefer the ISP's YUV420 and store our compact master as lossless intra-frame FFV1
in Matroska when the capture path permits it. [FFV1 is a specified lossless
intra-frame codec](https://www.rfc-editor.org/rfc/rfc9043.html): it preserves exact
frames, timestamps, and ordinary video handling without inter-frame codec motion
artefacts. At 800x450 and 15 fps, raw 8-bit Y alone is about 19.4 GB/hour and full
8-bit YUV420 is about 29.2 GB/hour before lossless compression. Bound the retained
original-quality holdout to 10--15 minutes; verify and delete other temporary 1080p
recordings after making the compact master.

Do not upscale small public videos such as CAVIAR, and do not transcode an already
compact source merely to change its extension. Preserve or remux it when possible;
only large public video needs a single controlled downscale.

When creating a 15 fps compact copy, deterministically select original frames by
presentation timestamp. Never let an implicit `fps` conversion manufacture
duplicates. Record the original frame index and integer timestamp for every
retained frame; P/N and displacement use the previous retained frame and its real
elapsed interval.

### Staged capture

1. **Pilot: 10--20 minutes.** Record 12--20 clips of 30--60 seconds. One person
   enters, crosses, stops, turns, approaches, recedes, and exits. Complete the
   entire annotation and training path before recording more.
2. **Single-person core: build to 40--60 minutes total.** Use at least three sessions on
   different days, lighting, clothing, camera distances, speeds, and routes.
   Include every entrance edge, profile/back views, sitting or bending, short
   2--10 second stops, and long 20--60 second stationary periods.
3. **Ownership and distractors: extend the whole local set to 60--90 minutes.** A
   second person enters after lock, both enter together, people cross, the owner
   is occluded, and one person replaces another. Include doors, curtains, fans,
   screens, shadows, lights, pets, and an empty room as non-head motion. Clear
   faces on monitors, photographs, masks, dolls, and clothing are deliberate hard
   negatives: give them no positive and no ignore mask. Only genuinely ambiguous
   regions are ignored.
4. **Moving-camera set, later.** Walk, pan, turn, stop, and follow a person only
   after the fixed-camera baseline is measured. Keep this as a separately named
   domain because global image motion changes the problem.

Most clips should be 25--45 seconds with a short empty-room lead-in and tail so
background initialization and release are observable. Keep roughly 10% as
uninterrupted 2--5 minute clips: short clips cannot reveal slow recurrent-state
drift or a background model gradually absorbing a stationary person.

Five to ten people giving even short sessions is more valuable than hours of
adjacent frames from one person. If only one person is initially available, that
is enough to build the system, but the final holdout must say that it measures a
person-specific prototype rather than broad generalization.

Use participant codes rather than names, obtain consent from anyone deliberately
recorded, disable audio, and keep footage under the ignored local `data/` tree—not
in Git. Store capture metadata and checksums with the compact packet.

Assign each complete recording session to one split before annotation or
augmentation. Never put adjacent parts of one recording in train and validation.
Where the corpus permits it, keep separate tests for a new day, a new room, and a
new person rather than accidentally requiring every split to differ in every way.
Reserve 15--20 minutes from a different session as a densely reviewed target-camera
test set; public video must never substitute for this result. Synchronized camera
views and all derivatives of one event always stay in the same split.

## Local pre-labelling on a Mac

The simplest maintainable workflow is:

```text
native Mac teacher model -> provisional head tracks -> local CVAT -> corrected tracks
```

1. Run a larger pose/person model natively with PyTorch MPS. Ultralytics supports
   [pose estimation](https://docs.ultralytics.com/tasks/pose/) and persistent IDs
   with [BoT-SORT or ByteTrack](https://docs.ultralytics.com/modes/track/). Derive a
   provisional head box from nose/eye/ear/shoulder evidence and inherit the person
   track ID. Core ML export is a later speed benchmark, not part of this untested
   first tracking path.
2. For better full-head labels, fine-tune a much larger desktop teacher on the
   existing CrowdHuman and SCUT **full-head** boxes plus corrected local tracks.
   Do not mix WIDER face boxes into that class without a separately validated
   face-to-head conversion. Using a large teacher is compatible with this project:
   the novelty is making the result run efficiently on the tiny device.
3. Simplify each dense proposed trajectory into sparse keyframes at a one-to-two
   final-input-pixel tolerance while retaining entries, exits, sharp turns,
   confidence collapses, occlusions, and ID changes. Import **CVAT for video XML**
   into a local [CVAT installation](https://docs.cvat.ai/docs/administration/basics/installation/).
   CVAT's [track mode](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/)
   interpolates rectangles between sparse keyframes and retains occluded/outside
   state. Correct the model's tracks rather than accepting them as truth.
4. Annotate a keyframe roughly every second during smooth motion. Add keyframes
   every 3--5 frames around fast motion, turns, crossings, scale changes,
   occlusion, entry, and exit. Inspect every interpolation span. Review every
   frame of validation and test densely.
5. Export CVAT video tracks, normalize them once into the project's simple JSONL,
   and keep the original CVAT export beside the normalized annotation as an audit
   source.

The current large-model starting point is below. It deliberately runs inside the
ignored `data/` tree so its environment, downloaded weights, and outputs cannot
pollute the repository:

```bash
mkdir -p data/teacher
python3 -m venv data/teacher/.venv
source data/teacher/.venv/bin/activate
python -m pip install "ultralytics==8.4.120"
cd data/teacher
python - <<'PY'
import platform
import torch
import torchvision
import ultralytics
from ultralytics import YOLO

print(platform.python_version(), ultralytics.__version__)
print(torch.__version__, torchvision.__version__)
assert torch.backends.mps.is_available()

model = YOLO("yolo26l-pose.pt")
seen_track = False
for frame, result in enumerate(model.track(
    source="../video/own/pilot.mkv",
    stream=True,
    device="mps",
    tracker="botsort.yaml",
    imgsz=960,
    conf=0.15,
)):
    if result.boxes is None or not result.boxes.is_track:
        continue
    assert result.keypoints is not None
    print(frame, result.boxes.id, result.boxes.xyxy, result.keypoints.data)
    seen_track = True
    break
assert seen_track, "no provisional track was produced"
PY
python -m pip freeze > environment.txt
```

This only proves that pose detections and provisional track IDs can be produced;
it does not yet export this project's head schema. A short conversion script must
associate the returned pose/person box with its tracker ID, seed a conservative
head rectangle, mark the proposal origin/confidence, simplify the trajectory, and
write CVAT video XML. It must flag entirely missed heads and tracker gaps for human
review rather than merely correcting existing boxes. Compare `m`, `l`, and `x`,
BoT-SORT and ByteTrack, and 640 versus 960 input on the same pilot by missed heads
and correction count—not just speed. `l` at 960 is a hypothesis, not an established
optimum, and this exact inference has not yet been benchmarked on this Mac.

Install CVAT outside this repository so the project itself stays small:

```bash
git clone --branch v2.73.0 --depth 1 \
  https://github.com/cvat-ai/cvat.git /Users/szymon/Tools/cvat
cd /Users/szymon/Tools/cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
open -a "Google Chrome" http://localhost:8080
```

Keep CVAT's native XML export as the human-editable annotation archive because it
preserves keyframes, track structure, `occluded`, and `outside`. The small JSONL
below is the normalized training input, regenerated from that archive. Create the
task with zero-based frame indexing and test one export/import/export round trip
before bulk annotation. Uploading the small pilot is acceptable; for the full set,
configure CVAT's [shared storage](https://docs.cvat.ai/docs/administration/community/basics/installation/#share-path)
read-only so video is not duplicated into Docker-managed storage.

Run the teacher outside CVAT's Docker containers. Docker Desktop on macOS does not
expose Apple's MPS GPU path to a Linux container, whereas the native process can
use MPS/Core ML. CVAT remains the browser-based correction interface.

[SAM 2](https://github.com/facebookresearch/sam2) can propagate a manually prompted
head mask through difficult short clips, and [CoTracker](https://github.com/facebookresearch/co-tracker)
can propagate manually seeded points. Neither discovers semantic heads by itself;
use them as repair tools for difficult tracks, not as the main labeller. A generic
face detector is also insufficient because it systematically misses profiles,
back-of-head views, headwear, and occlusion.

## Compact local packet

Keep the temporal pipeline separate from the current still-image packet, but no
more complicated than this:

```text
data/video/<source>/source.json
data/video/<source>/<sequence>.<video-extension>
data/video/<source>/<sequence>.cvat.xml
data/video/<source>/<sequence>.jsonl
```

The CVAT XML is present only for locally corrected sources. It is the editable
archive; JSONL is the small normalized training representation.

One annotation row represents one frame:

```json
{"frame":42,"source_frame":84,"pts_us":2800000,"reset":false,"heads":[{"track_id":3,"box":[121.2,44.1,31.0,38.5],"visibility":"partial","outside":false,"ignore":false,"origin":"manual_head"}],"ignore_regions":[]}
```

`box` is `x, y, width, height` in compact-video pixels. `source.json` records the
original FPS, compact FPS, dimensions, session/split groups, original frame-to-
compact-frame mapping, URLs/checksums, and conversion command. Scene cuts and
missing/dropped frames must be explicit so recurrent state can reset and
displacement can use the true elapsed interval.

Frames are zero-based. Every active track has a row on every retained frame:

- `full` or `partial`: `box` is present and `outside=false`;
- `hidden`: `box=null`, `outside=false`, and no positive target is generated;
- exited: one transition with `box=null`, `visibility="outside"`, and
  `outside=true`, after which the track is omitted until it really returns.

`visibility`, `ignore`, and `origin` are explicit CVAT label attributes; they are
not inferred from a made-up fractional visibility score. A detector gap remains
an uncertain review interval—it must never be converted automatically into an
exit. If the location of an unlabelled or fully hidden head could be mistaken for
background, put the credible region in `ignore_regions`.

From each matched track, build the five supervised maps at stride four:

```text
heatmap             = Gaussian at every valid current head centre
offset              = fractional current centre within its stride-4 cell
displacement        = previous_centre / 4 - current_centre / 4
displacement_mask   = 0 for entries, ignored/invisible heads, and resets
```

The previous-owner prior must come only from an earlier frame or earlier
prediction; using the current label would leak the answer. Derive the expected
owner causally from frames no later than the current one, with the configured
confirmation/lost intervals and a fixed simultaneous-entry tie rule. Looking at a
completed future track to decide who was first is label leakage.

The converter should stream one source at a time:

```text
download archive -> verify -> decode -> resize/frame-rate conversion ->
convert tracks -> sample visual overlay -> delete raw archive
```

Do not store generated P/N surfaces, heatmaps, priors, or resized model inputs.
They are small deterministic products of video, annotations, and configuration
and must be regenerated after augmentation.

## Training mixture and clip sampling

Start by pretraining the appearance path on the existing still-head data. Then
fine-tune temporal clips with an initial temporal-batch sampling target such as:

```text
45% own OV5647 clips
25% JRDB-Pose
15% CAVIAR audited-head clips
10% ChokePoint
 5% HT21 stress clips
```

These are sampling weights, not required storage proportions. Sample sequences
first and start positions second. Never drop an annotated head within a retained
frame: that makes it false-negative background. Instead cap how often dense
sequences/segments are sampled. If two centres land in the same stride-4 cell,
mask that cell's ambiguous offset/displacement regression or exclude that frame;
one cell cannot encode two vectors. Begin with 8-frame clips for debugging, then
test 16--64 frames once memory and gradients are stable. Include empty and
non-head-motion clips in every epoch.

Transfer does not require a different network. Once the current still-image
spatial control has a sound checkpoint, load its luminance stem, backbone,
heatmap, and offset weights into the same streaming graph. The motion stem,
displacement channels, prior scale, and temporal mixing then learn from video.
Keep some still-image batches during fine-tuning so temporal shortcuts do not
erase head appearance.

The first causal ladder should be:

1. still-image spatial model;
2. real frame pairs with displacement, but zero prior and reset recurrent state;
3. short real clips with state;
4. prior heatmap and deliberate prior corruption;
5. longer clips containing stops and occlusion;
6. ownership state machine, evaluated but not encoded as a detection label;
7. moving-camera clips only after fixed-camera performance is sound.

## Temporal augmentation rules

Apply the same geometry to every frame and every box in a clip. A physical
fisheye/lens warp is a full-canvas image-coordinate transform; it is not a zoom and
must not depend on one head's estimated distance. Head size may control blur,
noise, and sampling difficulty, but not the lens equation. Freeze one exact
crop-or-letterbox transform per experiment; validation/test must use the transform
that deployment will use.

Vary these deliberately:

- flat, intermediate, and fisheye-like full-canvas warps;
- darkness, gain noise, backlight, motion blur, and rolling-shutter-like shear,
  correlated over time rather than independently randomized per frame;
- gradual and abrupt exposure changes;
- dropped and duplicated frames, with elapsed-frame count passed to the model;
- synthetic entry/exit and occluders: update boxes/visibility consistently, stop a
  fully covered head from creating a positive, but preserve its track continuity;
- scene cuts with an explicit recurrent-state reset.

Use codec degradation only for already-compressed public-source normalization or
a small robustness ablation. The deployed ISP Y plane is not video-compressed, so
compression must not dominate OV5647 training.

The complete derivation order is fixed:

```text
select retained frames and elapsed interval
-> apply shared/smooth geometry and transform boxes
-> apply temporally correlated exposure, noise, and blur
-> convert to luminance
-> set first-frame P/N to zero and update P/N sequentially
-> build heatmaps, offsets, displacement, masks, and prior
```

The recurrent neural poles already accept elapsed frames, but the current P/N
motion update applies only one decay step. Extend it to `decay ** elapsed_frames`
before enabling variable-frame-spacing augmentation.

Synthetic pairs made from still heads are useful for debugging displacement and
rare transitions, but they cannot validate recurrence, identity retention, real
motion blur, or sensor exposure behaviour.

## Validation and the stopping rule

Frame AP alone cannot answer whether the system behaves correctly. Report:

- head recall and centre error by apparent head size;
- acquisition delay after entry;
- first-subject retention and identity switches;
- survival during stationary pauses;
- recovery time after short and long occlusion;
- false acquisition per minute on empty/non-head-motion video;
- complete camera-to-centre latency, peak memory, and achieved frame rate on the
  ESP32-P4.

Keep a short failure reel with one example per error category. Record another
session only when validation review exposes a missing condition. Before comparing
increments, freeze the centre-match rule, confidence-selection method, head-size
bins, confirmation/lost timeouts, occlusion durations, transform, training
schedule, and seeds. A sensible centre match is distance no greater than half the
annotated head short side; use input-size bins `8--12`, `12--24`, `24--48`, and
`>48` pixels.

As a first coverage target, collect at least 100 clean acquisitions, 60 stops of
10 seconds or more, 50 occlusions, 50 later-person challenges, 50 crossings, and
100 empty/non-head-motion intervals, with at least 20 independent validation
events in every evaluated bucket. Stop after two successive additions of 25% more
independent sessions each improve the preselected validation metrics by less than
one absolute percentage point under the frozen procedure and no major failure is
concentrated in an underrepresented scenario. Evaluate the locked target-camera
test only once after dataset/model choices are frozen. This is a coverage-driven
hobby dataset, not a competition to maximize frame count.

## Immediate next action

Do not begin with a large public download. Record the 10--20 minute OV5647 pilot,
set up CVAT locally, pre-label it with a desktop teacher, correct the tracks, and
make one 8-frame loader batch produce the current five targets. In parallel,
convert the CAVIAR `Meet_WalkTogether2` and `Meet_WalkSplit` video/XML pairs,
download the ChokePoint ground truth plus two sequence archives, and request/access
JRDB-Pose. That is enough to expose the real annotation and loader problems before
committing storage or days of correction work.
