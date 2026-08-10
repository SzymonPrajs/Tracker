# Camera pipeline, bandwidth, buffers, and efficient implementation

Last verified: 2026-08-08.

## What computes each stage

```text
photons
  -> sensor pixel array, rolling shutter, analog gain/exposure
  -> RAW8/RAW10 MIPI packets
  -> ESP32-P4 MIPI CSI receiver
  -> ISP: black-level/defect/lens correction, demosaic, color matrix,
          gamma, statistics, crop and output-format conversion
  -> DMA-backed frame buffers in PSRAM
  -> PPA: scale/crop/rotate/mirror/color conversion
  -> one or more consumers:
       small preview -> SPI/DSI display
       YUV420 -> hardware H.264 -> storage/network
       RGB/YUV -> hardware JPEG -> photo/MJPEG
       reduced tensor/image -> HP CPU + PIE/ESP-DL -> inference
       application kernel -> HP CPU C/assembly
```

There are two planes:

- **Control plane**: the HP CPU drives SCCB/I2C to identify the sensor and write exposure, gain, timing, crop, flip, and mode registers. `esp_ipa` reads ISP statistics and performs software AE/AWB control. FreeRTOS tasks manage V4L2 queues, muxing, filesystems, UI and networking.
- **Data plane**: MIPI CSI, ISP, DMA, PPA, JPEG, H.264, 2D-DMA and display engines move or transform bulk pixels. The CPU should schedule these blocks and process reduced results, not manually copy every pixel.

The ESP32-C6 on these boards is a radio companion. Camera pixels and ISP/PPA/codec work remain on the P4. Sending a stream over Wi-Fi adds an SDIO transfer to the C6; it does not offload vision processing to it.

## Software stack

Espressif's [ESP-Video-Components](https://docs.espressif.com/projects/esp-video-components/en/latest/esp32p4/Get_Started/index.html) divides the system into:

- `esp_cam_sensor`: sensor identification and register-mode tables;
- `esp_sccb_intf`: sensor control bus;
- `esp_ipa`: image-processing algorithms such as AE/AWB that close the loop around ISP statistics;
- `esp_video`: Linux-V4L2-compatible capture and codec devices;
- ESP-IDF camera, ISP, PPA, JPEG and H.264 drivers underneath.

The current `esp_video` device convention is:

| Device | Function | Important format constraint |
|---|---|---|
| `/dev/video0` | MIPI CSI capture | Sensor RAW input; ISP-selected application output |
| `/dev/video10` | Hardware JPEG encode | Several RGB/YUV inputs, revision-dependent |
| `/dev/video11` | Hardware H.264 encode | YUV420 input only |
| `/dev/video12` | Hardware JPEG decode | JPEG input, decoded image output |
| `/dev/video20` | ISP metadata/statistics | Control-plane data for IPA/tuning |

The lower-level [ESP-IDF camera controller](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/camera_driver.html) exposes direct transaction callbacks and buffer allocation. The callbacks run in interrupt context and must not block. V4L2 is generally the better application boundary; use the direct driver only when its ownership model or latency is genuinely needed.

## Hard limits that shape a design

- MIPI CSI: two lanes, up to 1.5 Gbit/s per lane; 3 Gbit/s aggregate signaling ceiling.
- ISP: RAW8/10/12 input, maximum specified image size 1920 x 1080.
- H.264: baseline YUV420 encoder, up to 1920 x 1080 at 30 fps aggregate.
- JPEG: separate fixed-function codec; see the architecture chapter for still/dynamic limits.
- PPA: scale, rotate, mirror, blend, fill and format conversion; use it to make consumer-specific frames.
- PSRAM: 32 MiB physical on the target part, less mappings, XIP, heap reservations, stacks, models and application allocations.

The lane ceiling does not imply that 375 MB/s of frames can be stored, processed, and reread indefinitely. CSI signaling, ISP output DMA, PSRAM, cache, consumers, display, flash XIP and the CPU all have different arbitration and overhead.

## Bandwidth formulas

For a tightly packed active image:

```text
bytes_per_frame = width * height * bits_per_pixel / 8
active_MB_per_s = bytes_per_frame * fps / 1,000,000
active_Mbit_per_s = active_MB_per_s * 8
frames_per_minute = fps * 60
pixels_per_minute = width * height * fps * 60
```

Use decimal MB/s for link-rate comparisons. Use MiB for memory capacity. Actual V4L2 buffers may include line stride, height rounding, alignment or metadata; after `VIDIOC_G_FMT`, use `fmt.fmt.pix.bytesperline` and `sizeimage`, not the tight-pack calculation.

RAW10 is normally packed as four 10-bit pixels in five bytes. Do not model it as a 16-bit array unless the chosen receiver/output path expands it; expansion raises memory traffic from 1.25 to 2 bytes/pixel.

## Active-pixel bandwidth

### 1920 x 1080

| Format | Bytes/frame | 25 fps | 30 fps | Two buffers |
|---|---:|---:|---:|---:|
| RAW8 | 2,073,600 | 51.84 MB/s | 62.21 MB/s | 3.96 MiB |
| Packed RAW10 | 2,592,000 | 64.80 MB/s | 77.76 MB/s | 4.94 MiB |
| YUV420 | 3,110,400 | 77.76 MB/s | 93.31 MB/s | 5.93 MiB |
| RGB565 or YUV422 | 4,147,200 | 103.68 MB/s | 124.42 MB/s | 7.91 MiB |
| RGB888 | 6,220,800 | 155.52 MB/s | 186.62 MB/s | 11.87 MiB |

### 1280 x 720

| Format | Bytes/frame | 25 fps | 30 fps | 60 fps | Two buffers |
|---|---:|---:|---:|---:|---:|
| RAW8 | 921,600 | 23.04 MB/s | 27.65 MB/s | 55.30 MB/s | 1.76 MiB |
| Packed RAW10 | 1,152,000 | 28.80 MB/s | 34.56 MB/s | 69.12 MB/s | 2.20 MiB |
| YUV420 | 1,382,400 | 34.56 MB/s | 41.47 MB/s | 82.94 MB/s | 2.64 MiB |
| RGB565 or YUV422 | 1,843,200 | 46.08 MB/s | 55.30 MB/s | 110.59 MB/s | 3.52 MiB |
| RGB888 | 2,764,800 | 69.12 MB/s | 82.94 MB/s | 165.89 MB/s | 5.27 MiB |

### Other board-relevant modes

| Mode | RAW input | RAW active rate | RGB565 rate | YUV420 rate |
|---|---:|---:|---:|---:|
| 1600 x 1200 at 30, RAW8 | 1,920,000 B/frame | 57.60 MB/s | 115.20 MB/s | 86.40 MB/s |
| 1600 x 1200 at 30, RAW10 | 2,400,000 B/frame | 72.00 MB/s | 115.20 MB/s | 86.40 MB/s |
| 1600 x 900 at 30, RAW10 | 1,800,000 B/frame | 54.00 MB/s | 86.40 MB/s | 64.80 MB/s |
| 1280 x 960 at 45, RAW10 | 1,536,000 B/frame | 69.12 MB/s | 110.59 MB/s | 82.94 MB/s |
| 800 x 1280 at 50, RAW8 | 1,024,000 B/frame | 51.20 MB/s | 102.40 MB/s | 76.80 MB/s |

The ISP input rate and the stored output rate are different. A RAW10 1080p30 sensor sends 77.76 MB/s of active samples; choosing RGB565 makes the stored output 124.42 MB/s. If a consumer then reads every output byte once, the simple write-plus-read lower bound is 248.83 MB/s. A second consumer, a full-frame copy, or a PPA output adds more traffic.

Do not automatically add RAW input bytes to PSRAM traffic: a streaming CSI-to-ISP path need not store the RAW frame. Add only transfers that actually reach memory in the configured pipeline.

## Lane-rate headroom

| Sensor driver mode | Configured signaling | Active RAW payload | Active/signaling ratio |
|---|---:|---:|---:|
| OV2710 RAW10 1080p25 | 1 x 800 Mbit/s | 518.40 Mbit/s | 64.8% |
| OV5647 RAW10 1080p30 | 2 x 408.33 = 816.67 Mbit/s | 622.08 Mbit/s | 76.2% |
| SC2336 RAW10 1080p30 | 2 x 405 = 810 Mbit/s | 622.08 Mbit/s | 76.8% |
| SC2356/SC202CS RAW10 1600 x 1200p30 | 1 x 720 Mbit/s | 576.00 Mbit/s | 80.0% |
| SC2356/SC202CS RAW8 1600 x 1200p30 | 1 x 576 Mbit/s | 460.80 Mbit/s | 80.0% |

The difference pays for horizontal/vertical blanking, packet headers, synchronization and sensor timing. It is not spare application bandwidth. A mode's exact lane rate is part of its register table and must match the CSI receiver configuration.

## Display traffic

| Board display | Tight RGB565 frame | 30 full frames/s | Interface implication |
|---|---:|---:|---|
| P4X-EYE, 240 x 240 | 115,200 B | 3.46 MB/s, 27.65 Mbit/s active | SPI; crop/scale in PPA first |
| Waveshare 3.5, 320 x 480 | 307,200 B | 9.22 MB/s, 73.73 Mbit/s active | SPI; full-rate refresh may become a bus limit |
| Function EV, 1024 x 600 | 1,228,800 B | 36.86 MB/s | MIPI DSI; include blanking/protocol overhead |
| Tab5, 1280 x 720 | 1,843,200 B | 55.30 MB/s | MIPI DSI; display reads materially load PSRAM |

Dirty rectangles can reduce UI transfers, but a live full-screen camera preview normally changes almost every pixel. Double-buffered displays also reserve two screen-sized allocations.

## Buffer ownership and queue depth

Use a strict state machine:

```text
FREE -> queued to CSI -> FILLED -> owned by one consumer/refcounted fan-out
     -> returned to CSI only after every zero-copy consumer is finished
```

- Two capture buffers minimize memory and allow capture of frame N+1 while frame N is consumed.
- Three buffers absorb one scheduling hiccup but add an entire frame of memory and can add latency.
- Deeper queues hide overload until latency becomes unacceptable. For real-time vision, dropping the oldest unprocessed frame is usually better than building a seconds-long backlog.
- Never let the producer overwrite a buffer still read by PPA, encoder, display, CPU or AI. Use completion callbacks/fences, not timing guesses.
- Allocate DMA-capable, cache-line-aligned PSRAM through the supported capability allocator or V4L2 MMAP path. Apply the cache-maintenance rules in the memory chapter at every CPU/DMA ownership change.

The [ESP-Vision camera pipeline](https://docs.espressif.com/projects/esp-vision/en/latest/esp32p4/concepts/camera-pipeline.html) also uses double buffering and PPA for P4 scale/color conversion. This is the correct default pattern.

## 32 MiB budgeting examples

### Full 1080p RGB565 capture

- two capture frames: 7.91 MiB;
- one full-size work frame: 3.96 MiB;
- two 240 x 240 previews: 0.22 MiB;
- one 720p work frame: 1.76 MiB;
- five 240 x 240 AI frames: 0.55 MiB;
- nominal JPEG output: 0.80 MiB.

That is already about 15.2 MiB with the small auxiliary allocations in the P4-EYE demo, before model weights/activations, code/rodata XIP, task stacks, audio, filesystem caches, network buffers, heap fragmentation and safety margin.

### Encoder-oriented 1080p YUV420

Two YUV420 input frames consume 5.93 MiB instead of 7.91 MiB for RGB565. H.264 output must use a separate variable-size buffer/ring. Size it for peak I-frame output and backpressure, not average bitrate alone. A 10-Mbit/s average stream is only 1.25 MB/s on average, but individual access units are bursty.

At startup, log:

- total and largest free internal block;
- total and largest free PSRAM block;
- every V4L2 `sizeimage`, requested buffer count and returned buffer count;
- PPA/JPEG/H.264 output allocation sizes;
- model parameter and activation allocations.

Largest-free-block is often more useful than total free bytes because frame allocations are large and contiguous.

## JPEG and H.264 behavior

### JPEG/MJPEG

JPEG size depends on scene detail, noise, subsampling and quality. The P4-EYE demo uses quality 90 for photos and 80 for MJPEG video. Its `5:1` allocation is a chosen assumption. Test a noisy, high-detail scene at the highest gain; it is more likely to produce a large JPEG than a clean static scene.

In the pinned [`example_encoder.c`](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_video/examples/common_components/example_video_common/example_encoder.c), hardware-JPEG input support for YUV420 and YUV444 is compiled only when the minimum ESP32-P4 revision is v3.0 or later. On older non-`X` boards, negotiate an actually supported input such as RGB565/YUV422 or check the exact component and silicon combination; do not assume a P4X format matrix applies.

MJPEG is operationally simple and every frame is independently decodable, but its storage/network rate is normally much larger than inter-frame H.264. It can still be useful for random access, resilient capture, and simple tooling.

### H.264

The pinned [`esp_video_h264_device.c`](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_video/src/device/esp_video_h264_device.c) accepts YUV420 only and exposes:

| Control | Default | Range/step |
|---|---:|---:|
| Bitrate | 10,000,000 bit/s | 25,000 to 25,000,000 in 25,000 steps |
| I-frame period/GOP | 30 | 1 to 120 |
| Minimum QP | 25 | 0 to 51 |
| Maximum QP | 26 | 0 to 51 |

The average compressed rates are 1.25 MB/s at the 10-Mbit/s default and 3.125 MB/s at the 25-Mbit/s control maximum. They do not include container, audio, filesystem, transport or retransmission overhead. Short-term output is not constant.

Do not confuse the H.264 GOP control with camera frame rate. Configure the sensor frame interval on the capture device, then make the encoder's timing/rate-control configuration agree with the measured input.

Source-audit caveat: at the pinned commit, the V4L2 H.264 wrapper initializes the low-level encoder's `fps` field from the same value used for `gop`; it does not expose an independent encoder-fps control in that constructor. This makes it especially important to inspect the component version, keep camera cadence and encoder configuration consistent, and validate timestamps and output over a full minute when changing the GOP away from its default of 30.

## External-output ceilings

- USB 2.0 High-Speed signals at 480 Mbit/s, but protocol and implementation reduce payload. Uncompressed 1080p30 RGB565 alone is 995.3 Mbit/s and cannot fit; compressed video can.
- 100-Mbit Ethernet has a 12.5 MB/s raw line ceiling and less application payload. A 10-Mbit/s H.264 stream is plausible; raw 1080p is not.
- Wi-Fi through ESP32-C6 adds SDIO, radio, protocol, contention and retransmission. No fixed camera-stream throughput should be assumed; benchmark the complete P4-to-C6-to-access-point path.
- SD-card write rate and worst-case pause behavior vary dramatically. Use 4-bit SD mode where the board supports it, preallocate files where possible, buffer bounded bursts, and test the exact card after sustained writes.

## Efficient C design

1. Use `esp_video`/V4L2 and MMAP buffers for the normal application path. `VIDIOC_DQBUF` transfers ownership; `VIDIOC_QBUF` returns it.
2. Query capabilities and enumerate modes. Set one exact driver-supported sensor mode; read the negotiated format back.
3. Request the final ISP format needed by the dominant consumer. H.264 wants YUV420; a display often wants RGB565. Avoid RGB888 unless its precision is valuable enough to justify 50% more traffic than RGB565.
4. Fan out with hardware transforms. Create a small PPA output for display/AI instead of copying or rescanning the 1080p frame on the CPU.
5. Pass buffer descriptors—pointer, stride, width, height, format, sequence and timestamp—through queues. Do not pass naked pointers whose ownership is ambiguous.
6. Bound every queue. Count drops and late frames. A system that silently accumulates latency is not real time.
7. Keep callbacks/ISRs minimal: stamp, enqueue, notify, return. Run ISP control, inference, encoding and I/O in tasks.
8. Pin only where measurement shows value. Separate capture/control from heavy inference or storage, but remember both HP cores share cache and PSRAM.
9. Avoid per-frame allocation. Allocate all frame, codec and model buffers at initialization and fail early with a complete memory report.
10. Measure each stage with cycle/time stamps and the hardware performance counters described in the assembly chapter.

## Where direct assembly helps—and where it does not

Hand assembly should not replace CSI, ISP, PPA, JPEG, H.264 or DMA. It can help a residual CPU kernel after the frame has been reduced to a cache-friendly tile or tensor:

- threshold/mask generation on a small luma plane;
- fixed-size convolution or morphology not supplied by ESP-DL;
- packed-channel rearrangement after proving PPA cannot express it;
- checksum, feature accumulation, or drawing kernels with predictable alignment.

Keep the kernel in a `.S` file, obey the RISC-V ABI, and use PIE only with the assembler/toolchain snapshot validated in the ISA chapters. Tile data into internal L2/SPM or a cache-resident working set. A theoretically fast vector loop that streams full RGB888 frames from PSRAM may lose to a PPA operation plus a small scalar cleanup.

RAW10 unpack is a particularly poor first assembly target: if the ISP can directly deliver the required RGB/YUV form, unpacking RAW10 on the CPU adds reads, writes and bit manipulation that the data path already knows how to perform.

## One-minute acceptance test

For each board/mode, record these counters over at least 60 seconds:

| Counter | Required interpretation |
|---|---|
| Sensor/capture frames | Successful DQBUF or transaction completions |
| Sequence gaps | Frames lost before application processing |
| Repeated frames | Container/display duplicates used to meet a nominal cadence |
| ISP/PPA completions and failures | Hardware-stage reliability and time |
| AI frames and inference latency distribution | Do not report only the mean |
| Encoder input/output frames and `bytesused` | Detect drops and output bursts |
| Displayed frames | Separate from captured frames |
| SD/network bytes and stalls | Include maximum blocking interval |
| Queue high-water marks | Reveal overload and latency accumulation |
| PSRAM/internal minimum free and largest block | Reveal fragmentation/leaks |

At a configured 30 fps, exactly 1,800 input frame opportunities occur in one minute. Report every divergence rather than rounding an observed 27.4 fps to “30 fps.” Repeat with a high-detail/noisy scene, radio traffic, display enabled, flash logging, and the intended AI model active.
