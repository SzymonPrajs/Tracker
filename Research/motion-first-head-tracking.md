# Direction: motion-first head tracking

## Conclusion

Do not replace head detection with frame differencing. Use motion as an
**attention signal and compute gate**, use a small neural network to decide
whether the changed structure is a head, and keep the chosen subject in a
persistent tracker.

The best first system for this project is:

```text
OV5647 -> ISP/PPA -> small luminance frame
                         |
             current luminance + signed frame difference
                         |
             tiny head-centre neural network
                         |
             first-subject ownership tracker
                         |
                    one target point
```

This is biologically inspired without taking the analogy literally. The classic
frog-retina work found parallel outputs for contrast, curved dark boundaries,
moving edges, and local dimming; it did not establish that a frog has no visual
response at all to stationary structure. The useful engineering lesson is that
change can be extracted before general recognition, so later computation receives
a much smaller and more relevant signal [1].

## Why motion cannot be the whole detector

Pure differencing has three fatal cases:

- a tracked person stops moving and disappears from the difference image;
- a moving camera makes almost the entire image appear to move;
- a shadow, door, monitor, tree, or exposure change moves without being a head.

There is a fourth case when the camera follows a walking person: the selected head
can become nearly stationary in image coordinates while the whole background
moves. A semantic head signal and remembered track are therefore necessary even
in a motion-first system.

The project should separate three responsibilities:

1. **Motion proposes attention.** Cheap temporal image processing finds changed
   regions or decides that a neural pass is needed.
2. **The neural network confirms heads.** It sees appearance and change together,
   rather than trying to classify an otherwise context-free difference blob.
3. **The tracker owns the first subject.** A deterministic state machine, not the
   neural network, prevents a later person from stealing the track.

## First implementation

### Neural input

Start with two 8-bit planes at the model resolution:

```text
channel 0 = current ISP luminance Y(t)
channel 1 = clipped signed difference Y(t) - Y(t-1)
```

Signed change preserves whether a boundary became lighter or darker. The current
luminance plane preserves the head evidence that absolute differencing destroys.
The first experiment should keep the existing model body and its head-centre and
sub-cell-offset outputs; only the first convolution changes from RGB to two input
channels. This gives a causal test of temporal input before changing the network
and the tracker at the same time.

To reuse the trained spatial model, initialize the new luminance channel by summing
its three RGB stem weights. This preserves the stem response that the RGB model
would have produced if the same luminance were replicated into R, G, and B. Set the
delta-channel weights to zero. The converted model then begins with an approximate
old still-image behaviour and learns how much temporal evidence to add, instead of
discarding that training.

Pretrain or initialize from the existing still-head corpus. During fine-tuning,
make frame pairs from video. Apply the same lens warp and base geometry to both
frames, then apply a small relative transform for real object or camera motion.
Independently warping the two frames would teach synthetic motion artefacts.

### First-subject ownership

“Track the first person” is a policy, not an image class. Use this small state
machine:

```text
SEARCHING -> CANDIDATE -> LOCKED -> COASTING -> LOST -> SEARCHING
```

- `SEARCHING`: motion opens the detector; no target is owned.
- `CANDIDATE`: require consistent head detections over a few frames. The first
  confirmed candidate wins; simultaneous ties use a documented deterministic
  rule such as confidence and then distance from the image centre.
- `LOCKED`: predict the owned centre with a constant-velocity alpha-beta or
  Kalman filter. Associate only a plausible detection near that prediction. New
  heads may be observed but cannot replace it.
- `COASTING`: if the head stops, blurs, or is briefly occluded, continue the
  prediction and run periodic semantic checks. Do not release ownership merely
  because the motion mask vanished.
- `LOST`: release the target only after a configured absence interval. A later
  person can become primary only after this transition.

For one owned subject, nearest gated association is enough. SORT shows that a
Kalman predictor plus simple assignment can be a strong real-time baseline [2];
the Hungarian algorithm is unnecessary until this project deliberately maintains
several competing tracks.

A small luminance template or correlation filter can refine the centre between
neural passes, but it must never be allowed to update from a low-confidence patch:
that is how template trackers drift onto a wall or a second person [3].

### Background for a fixed camera

Keep both short- and long-term evidence:

```text
frame change:       D(t) = abs(Y(t) - Y(t-1))
background change:  R(t) = abs(Y(t) - B(t))
background update:  B(t) <- (1-a) B(t) + a Y(t)
```

Update `B` only in pixels currently believed to be background. Protect the locked
head region and a guard band around it so a stationary subject is not absorbed
into the room. A widespread residual usually means exposure or lighting changed;
adapt or reset the background instead of declaring the whole room foreground.
This fixed-camera mode should be built first. Conventional background subtraction
is explicitly premised on a static camera [4].

### A moving camera is a separate mode

Raw frame difference is invalid once the camera moves. Estimate global background
motion from low-resolution corners or blocks, map the previous image toward the
current image, and difference only after that compensation. An optional gyro can
provide a useful prior, but visual correction is still needed for translation and
parallax.

This has limits. One affine transform or homography cannot align a close wall and
a distant street simultaneously, and people must be excluded from the background
motion estimate. Research on non-stationary cameras reaches the same design:
compensate camera motion, protect the background model from foreground pollution,
and tolerate imperfect alignment [5].

On the ESP32-P4, arbitrary perspective warping is not an advertised ESP-DL
operator, and the PPA advertises scaling, right-angle rotation, mirroring, blending,
and colour conversion rather than a general homography [6][7]. Therefore the first
mobile-camera experiment should compare:

1. no background model, using the temporal neural detector and owned-track state;
2. coarse translation or small affine compensation at low resolution;
3. more general warping only if the measured improvement justifies its memory
   traffic and custom code.

## Neural alternatives

| Design | What it adds | Decision |
|---|---|---|
| Current `Y + signed delta` detector, classical association | Motion-aware recognition with one previous frame | **Build first** |
| CenterTrack-lite | Previous result heatmap and a two-value displacement from each current head to its previous centre | **Build second if identity switches remain** |
| Online Temporal Shift Module | Reuses a fraction of intermediate channels from the previous frame without 3-D convolution | Research candidate after export and buffer profiling |
| Siamese/correlation neural tracker | Learns a target template after acquisition | Useful only if simple template tracking is inadequate; drift and distractors remain |
| Optical-flow network, 3-D CNN, or ConvLSTM | Dense learned temporal state | Reject initially: too much compute, activation storage, and deployment complexity |

CenterTrack is the closest published model to the desired behaviour: it consumes
the current frame, previous frame, and previous detection heatmap and predicts both
new centres and their displacement to prior centres [8]. The project does not need
its large backbone. Its **interface and loss** can be copied onto this tiny
head-centre model.

An online Temporal Shift Module moves some feature channels forward in time and
keeps ordinary 2-D convolutions, adding no arithmetic parameters [9]. However,
“zero computation” does not mean zero cost here: the P4 must retain, address, and
possibly copy feature planes. It is a later board-measured experiment, not the
starting architecture.

## ESP32-P4 data path and memory

The OV5647 driver currently exposes RAW8 modes at 800-wide resolutions and RAW10
at 1920 by 1080 at 30 FPS or 1280 by 960 at 45 FPS [10]. The P4 ISP accepts RAW
camera data and can output RGB or YUV, including YUV420; the PPA can scale and has
a GRAY8 mode [6][11]. The most promising path is therefore:

```text
OV5647 RAW -> ISP YUV420 -> use/scale the Y plane -> temporal input -> INT8 model
```

This is a logical path, not yet a zero-copy claim. The ISP documentation says its
processed output is written to system memory through DMA, and PPA scaling may read
that image again. The board benchmark must count actual buffers, cache maintenance,
and bytes moved between camera capture, PPA, and ESP-DL.

Do not begin by training directly on packed RAW10. RAW10 is a Bayer mosaic rather
than luminance: each site measures only one colour, while exposure, gain, and
black-level changes also appear in its temporal residual. RAW-derived input remains
a valid later experiment, but ISP luminance is the clean baseline and avoids an RGB
frame solely to recreate luminance.

The temporal buffers themselves are small but not free:

| Resolution | One Y plane | `Y + delta` model input | Input + previous Y + background |
|---|---:|---:|---:|
| 200 by 100 | 19.5 KiB | 39.1 KiB | 78.1 KiB |
| 320 by 160 | 50.0 KiB | 100.0 KiB | 200.0 KiB |
| 400 by 200 | 78.1 KiB | 156.3 KiB | 312.5 KiB |

These are only raw pixel buffers. They exclude camera queues, alignment, neural
activations, ESP-DL workspace, output maps, and firmware. The chip has 768 KiB of
HP L2 memory plus in-package PSRAM [12]. Espressif also warns that PPA performance
depends heavily on PSRAM bandwidth and degrades when other peripherals compete for
it [7]. The final buffer schedule must therefore be measured on the board and
should:

- avoid ever materializing full-resolution RGB;
- let DMA/ISP/PPA produce the smallest useful luminance image;
- fuse subtract, clip, and NHWC interleave into one measured C loop so no separate
  delta image is written; use PIE/assembly only if profiling proves this loop hot;
- reuse the current Y buffer as the next previous frame;
- keep only hot activations in internal memory and benchmark every PSRAM pass.

ESP-DL supports INT8 convolution, depthwise convolution, addition, subtraction,
and resize on ESP32-P4, so the proposed tiny detector remains in its supported
operator set [13]. This is compatibility evidence, not a latency or memory result.

## Data and experiments

The existing still images remain useful for spatial head recognition. Add temporal
evidence narrowly:

- record short OV5647 room clips containing one entrant, two entrants, stops,
  exits, partial occlusion, empty-room motion, lights, monitors, doors, and shadows;
- use PoseTrack for head boxes, persistent person IDs, occlusion, fast people, and
  camera motion [14];
- use a sequence-balanced subset of CroHD for true head tracks and identity stress,
  not all 2.27 million densely redundant boxes [15];
- split whole videos, never adjacent frames, between train and validation.

Run one controlled sequence:

1. existing spatial model on current luminance only;
2. the same model on current luminance plus signed difference;
3. add the ownership state machine;
4. add a learned previous-centre displacement only if step 3 still switches;
5. test 200 by 100, 320 by 160, and 400 by 200 on the board;
6. introduce camera compensation only after the fixed-camera system is sound.

Frame-wise AP is no longer enough. Report acquisition time, centre error, false
acquisitions on non-head motion, first-subject retention, identity switches,
stationary-pause survival, occlusion recovery time, and complete camera-to-centre
latency and peak memory. HOTA is a useful standard reference because it separates
detection, association, and localization [16], but the first-subject retention and
release rules are project-specific and must be measured explicitly.

## Sources

1. Lettvin et al., [What the Frog's Eye Tells the Frog's Brain](https://ieeexplore.ieee.org/document/4065609), 1959.
2. Bewley et al., [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763), 2016.
3. Henriques et al., [High-Speed Tracking with Kernelized Correlation Filters](https://arxiv.org/abs/1404.7584), 2015.
4. OpenCV, [Background subtraction for static cameras](https://docs.opencv.org/4.10.0/d1/dc5/tutorial_background_subtraction.html).
5. Yi et al., [Detection of Moving Objects with Non-stationary Cameras](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W03/html/Yi_Detection_of_Moving_2013_CVPR_paper.html), 2013.
6. Espressif, [ESP32-P4 image signal processor](https://docs.espressif.com/projects/esp-idf/en/latest/esp32p4/api-reference/peripherals/isp.html).
7. Espressif, [ESP32-P4 pixel-processing accelerator](https://docs.espressif.com/projects/esp-idf/en/latest/esp32p4/api-reference/peripherals/ppa.html).
8. Zhou et al., [Tracking Objects as Points / CenterTrack](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490460.pdf), 2020.
9. Lin et al., [Temporal Shift Module](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf), 2019.
10. Espressif, [OV5647 modes in esp-video-components](https://github.com/espressif/esp-video-components/blob/master/esp_cam_sensor/sensors/ov5647/Kconfig.ov5647).
11. Espressif, [ESP-Video camera framework](https://docs.espressif.com/projects/esp-video-components/en/latest/esp32p4/Get_Started/index.html).
12. Espressif, [ESP32-P4 series datasheet](https://documentation.espressif.com/esp32-p4_datasheet_en.html).
13. Espressif, [ESP-DL operator support](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md).
14. Iqbal et al., [PoseTrack](https://antonmil.github.io/files/cvpr2017/cvpr2017-iqbal.pdf), 2017.
15. Sundararaman et al., [Tracking Pedestrian Heads in Dense Crowd / CroHD](https://arxiv.org/abs/2103.13516), 2021.
16. Luiten et al., [HOTA](https://arxiv.org/abs/2009.07736), 2021.
