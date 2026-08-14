# Direction: temporal neural architectures

## Decision

There are two worthwhile meanings of “a network not run on an ESP32 before”:

1. **A known temporal network with a new deployment.** A very small
   CenterTrack-style head detector is the safest candidate. It is online, solves
   detection and association together, and needs only ordinary convolutions plus
   a previous-result heatmap and two displacement outputs [1].
2. **A new deployment-specific synthesis.** The stronger research direction is a
   **dual-timescale motion-state detector**: current appearance, decaying positive
   and negative change surfaces, a previous-centre heatmap, and a tiny diagonal
   recurrent state inside the existing stride-8 feature map.

Fast/slow recurrence itself is established prior art: Multiple Spatio-Temporal
Scales Neural Networks used leaky convolutional visual units with different time
constants in 2015 [16]. The potentially original part is the exact constrained
combination of eight-channel fast/slow state, pseudo-event surfaces, stride-4
previous-point injection, and head-centre displacement, designed for this memory
budget. It should be treated as an unvalidated hypothesis until controlled
experiments beat the simpler baselines.

A public English/Chinese web, GitHub, academic-index, and patent audit on 14 August
2026 found no ESP32 implementation of CenterTrack,
DeltaCNN/CBinfer/Skip-Convolutions, online TSM, MoViNet, VideoMamba, or a ConvLSTM
video detector. Patents do cover TSM in general video systems, and generic ESP32
tracking demos exist; neither establishes one of these neural temporal models on
the chip. This is evidence of a plausible novelty opportunity, not proof that
nobody has ever run one privately, left it unindexed, or used another name.

## What the wider model families teach us

| Family | Useful idea | Why not copy the published model directly? | Decision |
|---|---|---|---|
| CenterTrack | Detect the current centre and regress its displacement to the previous centre [1] | Its published backbone is much larger than this project needs | **Build a tiny version first** |
| Online TSM | Replace some current feature channels with cached channels from the previous frame; no added parameters or MACs [2] | Feature copies and layout changes are still real costs | Strong ablation |
| Bottleneck ConvLSTM | Learn a spatial hidden state for mobile video detection [3] | Hidden state, cell state, gates, and temporary activations are expensive | Do not start here |
| Diagonal and multi-timescale state | Independent recurrent components can retain different timescales [4][16] | VideoMamba applies far larger selective state-space machinery to video recognition [5] | Use the principle, not VideoMamba |
| Streaming 3-D CNN | Causal stream buffers make memory independent of video duration [6] | Even MoViNet-A0-Stream is 3.1 M parameters, 2.73 GFLOP per 50-frame clip, and reported 71 MB peak memory [6] | Inspiration only |
| Change-sparse CNN | Cache activations and update only changed spatial locations [7][8][9] | Irregular work, indexing, and cache traffic can cost more than the skipped arithmetic on a tiny dense model | Later custom-kernel research |
| Moving-camera sparse CNN | Align cached activations in a spherical buffer before sparse updates [10] | Much more state and scheduling complexity; movement can make early layers nearly dense | Much later |
| Event surface / SNN | Preserve the sign and recency of brightness changes [11][12] | The OV5647 is a frame camera and the P4 is not a neuromorphic accelerator | Use pseudo-event inputs, not a full SNN |
| Siamese/correlation tracker | Compare a stored target template with a search region [13][14] | Dynamic correlation/FFT kernels are awkward, a detector is still needed for acquisition, and template drift remains | Later only if association fails |

Three distinct efficiency strategies should not be confused:

- **Temporal representation** gives the network motion evidence.
- **Temporal memory** lets the network remember evidence after motion stops.
- **Conditional computation** avoids running all work on every pixel or frame.

The first two can improve accuracy. The third can improve speed, but only after
measurement shows that its bookkeeping costs less than the dense work it skips.

## Proposed network: dual-timescale motion-state detector

### 1. Frame-derived event surfaces

Let `Y(t)` be the current 8-bit luminance plane and let:

```text
d(t)  = clip(Y(t) - Y(t-1), -T, T)
P(t)  = max(lambda * P(t-1), max( d(t), 0))
N(t)  = max(lambda * N(t-1), max(-d(t), 0))
input = [Y(t), P(t), N(t)]
```

`P` and `N` retain recent brightening and darkening after the instantaneous
difference has vanished. The three-channel input costs no more model-input storage
than RGB. It is a **pseudo-event representation from frames**, not event-camera
data and not a spiking network. Event-vision research supports time surfaces as a
useful way to preserve the timing of changes, while also showing that learned
surfaces can outperform a single hand-designed representation [11].

### 2. Previous-result injection

Keep the previous centre heatmap at stride 4. Instead of upsampling it into another
full input plane, add it to the stride-4 features with one learned scalar per
channel:

```text
X4(t, c, y, x) += w(c) * previous_heatmap(y, x)
```

At 400 by 200, this heatmap is 100 by 50, or 5,000 INT8 bytes. This preserves the
most useful part of CenterTrack's interface without another full-frame buffer.

### 3. Fast and slow feature memory

After downsampling, the existing model has a 64 by 50 by 25 stride-8 map at 400 by
200 input. Select `m = 8` temporal channels `U(t)` and maintain two states:

```text
F(t) = a_fast * F(t-1) + (1 - a_fast) * U(t)
S(t) = a_slow * S(t-1) + (1 - a_slow) * U(t)

M(t) = U(t)
     + q_fast * (U(t) - F(t-1))
     + q_slow * (F(t-1) - S(t-1))
```

Replace those eight channels with `M(t)` before the six stride-8 convolutional
blocks. The remaining backbone then mixes the temporal signal spatially and across
channels. Learn `a_fast`, `a_slow`, `q_fast`, and `q_slow` per channel. Constrain
the two `a` values to `[0, 1]`, initialize the fast state near `0.5`, the slow state
near `0.9375`, and initialize both `q` values to zero. The zero initialization
makes this temporal adapter an exact pass-through rather than immediately
perturbing the stride-8 features of a useful checkpoint. Initialize the
previous-heatmap injection weights to zero for the same reason.

This is a tiny diagonal state-space-inspired recurrence [4] and a close relative
of earlier leaky multi-timescale visual units [16], not an invention of
multi-timescale memory. For firmware, quantize the coefficients to Q7 and
implement this recurrence in one small saturating C loop outside the exported
convolution graph. Initial powers of two make the update expressible with shifts;
learned Q7 coefficients are kept only if the measured accuracy gain justifies
multiplication.

### 4. Outputs

Use five stride-4 output planes:

```text
head centre confidence                       1
sub-cell location offset                     2
displacement to the previous head centre     2
```

The deterministic ownership state machine still decides that the first confirmed
subject remains primary. Neural displacement improves association; it does not
replace the ownership policy.

## Exact state pressure at 400 by 200

The current two-scale model performs about 66.04 MMAC at this resolution. The
important temporal comparison is memory traffic, not merely parameter count:

| Temporal mechanism | Persistent INT8 state before alignment/workspace |
|---|---:|
| Previous stride-4 centre heatmap | 5,000 B |
| Online TSM caching 1/8 of 64 stride-8 channels | 10,000 B |
| Proposed fast + slow state, `m = 8` | 20,000 B |
| Proposed fast + slow state, `m = 16` | 40,000 B |
| ConvLSTM hidden + cell at 64 stride-8 channels | 160,000 B |
| One cached output for each of six stride-8 body blocks | 480,000 B |

The last row is only a lower bound for a DeltaCNN-like conversion: it excludes
the stride-4 layers, truncation buffers, sparse masks, halos, camera buffers, model
workspace, and output maps. DeltaCNN itself identifies memory as a limitation on
low-end devices [9]. Caching every layer is therefore the wrong first P4 design.

With `m = 8`, the proposed recurrence updates 20,000 state values per frame. That
arithmetic is tiny beside 66 million convolution MACs; whether it is actually
cheap depends on where those 20 KB live and how often they are read and written.
It still requires a physical board profile.

## A second original direction: tile-delta inference

CBinfer, Skip-Convolutions, DeltaCNN, and MotionDeltaCNN form a serious research
lineage rather than a simple frame-difference trick [7][8][9][10]. They propagate
changes through the network's own activations so unchanged regions do not repeat
all convolution work. DeltaCNN uses tiled processing because structured blocks are
more hardware-friendly than arbitrary per-pixel sparsity [9].

A plausible microcontroller adaptation would:

- run the stem and stride-4 path densely;
- permit skipping only in the six stride-8 blocks;
- use fixed rectangular tiles and a compact bit mask, never lists of sparse pixel
  coordinates;
- expand each active tile by the convolution halo;
- run ordinary dense INT8 kernels inside active tiles;
- force a complete refresh periodically and after camera motion or exposure jumps.

This could be more novel than the recurrent model, but it is also much riskier.
Sparse GPU speedups do not predict MCU speedups: a 66 MMAC dense network may finish
before irregular tile scheduling and PSRAM traffic pay for themselves. Implement
it only after the dense temporal model gives a correct reference and a layer-level
board benchmark establishes the break-even changed-area fraction.

## Models deliberately not chosen

- **MoViNet and VideoMamba:** both answer broad video-recognition questions. This
  project needs one dense point per frame. Their streaming/SSM ideas transfer; the
  architectures do not.
- **A full spiking network:** event-driven energy claims depend on event sensors
  and neuromorphic execution. With frame-derived pseudo-events on a conventional
  processor, most of that advantage disappears [12].
- **A Siamese tracker first:** it can follow a supplied target but cannot decide
  which new moving structure is a head. Published “lightweight” trackers still
  operate at phone-class budgets; LightTrack reports 790 M FLOP [14].
- **A full ConvLSTM:** four gated convolutions plus hidden and cell maps spend too
  much of the memory budget before showing whether simple temporal state helps.
- **A transformer:** the problem has local motion, a single owned target, and a
  strict streaming state. Attention is not the missing capability.

## Controlled experiment ladder

Do not combine all changes into one opaque training run:

1. **Spatial control:** current luminance only.
2. **Existing motion control:** current luminance plus signed one-frame change.
3. **Tiny CenterTrack:** add the previous heatmap and displacement loss.
4. **Time-surface input:** replace one-frame change with positive and negative
   decaying surfaces.
5. **Dual-timescale state:** add `m = 8`; compare no state, fast only, and fast plus
   slow before trying `m = 16`.
6. **Online TSM:** cache eight channels instead of the learned recurrence at the
   same insertion point and state budget.
7. **Conditional execution:** only after selecting the best dense temporal model,
   measure fixed stride-8 tile skipping on the board.

Train contiguous clips, initially 8 to 16 frames, and reset recurrent state only
at real sequence boundaries. Use truncated backpropagation through time. Apply the
same spatial augmentation to every frame in a clip; independent warps create fake
motion. Include stationary people, stops after entry, occlusions, two-person
crossings, exposure changes, and moving-camera clips.

Every model is selected on head detection, acquisition time, first-subject
retention, identity switches, stationary-pause survival, false acquisition from
non-head motion, and recovery after occlusion. Also report actual camera-to-point
latency, persistent state bytes, peak internal and external memory, and measured
bytes moved per frame. FLOP or MAC counts alone cannot select this design.

## Novelty audit boundary

The audit used exact English and Chinese web searches, Google Patents, academic
indexes, and these public GitHub searches:

```text
gh search repos 'DeltaCNN ESP32'
gh search repos 'CenterTrack ESP32'
gh search repos 'Temporal Shift Module ESP32'
gh search repos 'MoViNet ESP32'

gh search code 'DeltaCNN ESP32'
gh search code 'CenterTrack ESP32'
gh search code '"Temporal Shift Module" ESP32'
gh search code 'MoViNet ESP32'
gh search code 'VideoMamba ESP32'
gh search code 'ConvLSTM ESP32'
```

All returned no repository or code matches on 14 August 2026. Public ESP32 object
detection and classical OpenCV tracking do exist [15], so “tracking on ESP32” is
not novel. The defensible claims, if this work succeeds, would be:

- “no prior public ESP32 implementation was found in the documented search”; and
- “this repository implements a tiny CenterTrack-style or dual-timescale
  recurrent head detector on ESP32-P4.”

Do not say “the first ever” without a much broader paper, patent, package-registry,
conference-demo, and multilingual search immediately before publication.

## Sources

1. Zhou, Koltun, and Krähenbühl, [Tracking Objects as Points](https://arxiv.org/abs/2004.01177), 2020.
2. Lin, Gan, and Han, [TSM: Temporal Shift Module for Efficient Video Understanding](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html), 2019.
3. Liu and Zhu, [Mobile Video Object Detection With Temporally-Aware Feature Maps](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Mobile_Video_Object_CVPR_2018_paper.html), 2018.
4. Gu et al., [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893), 2022.
5. Li et al., [VideoMamba: State Space Model for Efficient Video Understanding](https://arxiv.org/abs/2403.06977), 2024.
6. Kondratyuk et al., [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511), 2021.
7. Cavigelli et al., [CBinfer: Change-Based Inference for Convolutional Neural Networks on Video Data](https://arxiv.org/abs/1704.04313), 2017.
8. Habibian et al., [Skip-Convolutions for Efficient Video Processing](https://openaccess.thecvf.com/content/CVPR2021/html/Habibian_Skip-Convolutions_for_Efficient_Video_Processing_CVPR_2021_paper.html), 2021.
9. Parger et al., [DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html), 2022.
10. Parger et al., [MotionDeltaCNN](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html), 2023.
11. Cannici et al., [A Differentiable Recurrent Surface for Asynchronous Event-Based Data](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3398_ECCV_2020_paper.php), 2020.
12. Ji et al., [SCTN: Event-Based Object Tracking with Deep Convolutional Spiking Neural Networks](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1123698/full), 2023.
13. Wang et al., [DCFNet: Discriminant Correlation Filters Network for Visual Tracking](https://arxiv.org/abs/1704.04057), 2017.
14. Yan et al., [LightTrack: Finding Lightweight Neural Networks for Object Tracking](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_LightTrack_Finding_Lightweight_Neural_Networks_for_Object_Tracking_via_One-Shot_CVPR_2021_paper.html), 2021.
15. Espressif, [OpenCV object-tracking example for ESP32-S3-EYE](https://components.espressif.com/components/espressif/opencv/versions/4.7.0~2/examples/object_tracking?language=en).
16. Jung et al., [Self-Organization of Spatio-Temporal Hierarchy via Learning of Dynamic Visual Image Patterns](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492609/), 2015.
