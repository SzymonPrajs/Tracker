# Practical purchasable board and camera: ESP32-P4-NANO-KIT-A

Last verified: 2026-08-08.

## Recommendation

Buy the **Waveshare ESP32-P4-NANO-KIT-A**, manufacturer SKU **29027**. This is the ordinary development-board-plus-removable-camera arrangement required here:

- one `ESP32-P4-NANO` development board;
- one separate Waveshare `RPi Camera (B)` using the OV5647 sensor;
- one matched 15-pin FFC camera cable;
- one 8-ohm, 2-watt speaker.

The camera is not soldered to the board and the product is not a closed camera appliance. The board has a normal two-lane MIPI CSI socket. The matched kit is preferable to buying an arbitrary “Raspberry Pi camera” because Raspberry Pi-branded modules now use several sensors, connector widths, cable pitches and cable orientations, while ESP32-P4 software support is sensor-specific.

Purchase references:

- [Waveshare product page](https://www.waveshare.com/esp32-p4-nano.htm): select `ESP32-P4-NANO-KIT-A`; verify SKU `29027`.
- [Verified Amazon.com camera-bundle listing](https://www.amazon.com/dp/B0DKT7ZP48): ASIN `B0DKT7ZP48`, title includes “Bundle with RPi Camera (4 Items).”
- [Amazon UK exact search](https://www.amazon.co.uk/s?k=B0DKT7ZP48): regional availability and price can change. If the ASIN is unavailable, search for `ESP32-P4-NANO-KIT-A 29027` and check the package list against the four items above.

Do **not** confuse it with Amazon ASIN `B0DKT7FXK4`, whose verified title says “Bundle with Speaker (2 Items).” That option does not include the camera bundle. Also do not choose the Waveshare basic SKU `29026` unless buying the exact camera and cable separately.

## What is on the board

| Item | Specification | Design consequence |
|---|---|---|
| Main processor | `ESP32-P4NRW32`; two 32-bit RISC-V HP cores up to 360 MHz and one LP core up to 40 MHz | This widely sold board is the 360-MHz non-`X` product, not automatically the current 400-MHz `ESP32-P4NRW32X` target used elsewhere in this reference. |
| In-package PSRAM | 32 MB | Large camera buffers fit, but this is external PSRAM behind the cache, not 32 MB of internal SRAM. |
| On-board flash | 16 MB NOR | The `32 MB` in the product name/specification refers to PSRAM; this kit does not provide 32 MB flash. |
| Internal memory called out by vendor | 128 KB HP ROM, 16 KB LP ROM, 768 KB HP L2 memory, 32 KB LP SRAM, 8 KB TCM | Keep hot code/data in internal memory and bulk frames in PSRAM. Do not add these values together and call the result freely allocatable application RAM. |
| Radio companion | `ESP32-C6-MINI-1`, connected over SDIO | C6 provides 2.4-GHz Wi-Fi 6 and Bluetooth 5/BLE. It does not receive CSI pixels or run the P4 ISP/H.264 workload. |
| Camera | 15-pin, two-lane MIPI CSI socket; schematic labels it `15PIN--PI4B` | Use the included opposite-contact 15-pin cable and OV5647 module. |
| Camera control | SCCB/I2C: SCL `GPIO8`, SDA `GPIO7`; CSI data/clock use dedicated MIPI pins | These are the relevant board values when a generic example asks for sensor-control pins. The schematic does not route ordinary GPIOs as camera data pins. |
| Storage | microSD/TF slot using SDIO 3.0 | Interface capability is not a guaranteed sustained card write rate. Test the exact card and worst-case write pauses. |
| Wired network | 100-Mbit Ethernet | Suitable for compressed video; not for uncompressed 1080p frames. |
| USB | USB Type-A 2.0 High-Speed OTG; USB-C power/program/debug through USB-to-UART | USB HS can carry compressed video, but raw 1080p RGB565 at 30 fps exceeds the 480-Mbit/s signaling rate before overhead. |
| Other | two-lane MIPI DSI, microphone, audio codec/amplifier and speaker socket, RTC-battery header, optional PoE header, 28 programmable GPIOs on two 2x13 headers | The board remains a general development platform rather than a camera-only board. |
| Physical size | 50.00 x 50.00 mm | Compact enough for a camera prototype while retaining RJ45, USB-A, microSD and headers. |

The board vendor advertises 1080p30 H.264/JPEG encoding. The defensible implementation boundary is: OV5647 RAW10 1080p30 is present in Espressif's current sensor driver, and ESP32-P4 has a hardware H.264 encoder whose aggregate limit is 1080p30. Sustained application behavior still has to be measured with the selected storage/network/display/AI workload.

### Silicon-revision warning

The product page names `ESP32-P4NRW32` at 360 MHz, whereas the main chapters in this folder target the newer `ESP32-P4NRW32X` v3.x family at up to 400 MHz. Waveshare's current BSP explicitly supports both rev v1.3 and rev v3.x for some functions, which means the product family can span revisions. Amazon inventory can also be older than direct-vendor inventory.

On first boot, log the full eFuse chip revision. Build with a compatible `CONFIG_ESP_REV_MIN_FULL`, apply the matching errata, and benchmark at the clock actually reported by the unit. Do not use the 400-MHz operation ceilings from the P4X chapters for a 360-MHz board.

## Camera module

The included `RPi Camera (B)` has:

- OmniVision OV5647 rolling-shutter sensor;
- 5-megapixel, 1/4-inch optical format;
- 6-mm, f/2.0 lens;
- manual focus;
- no infrared night-vision support.

Waveshare's kit page states a 60.6-degree field of view, while the separate current `RPi Camera (B)` product page states 43 degrees. Treat the optical FOV as a vendor-documentation conflict and measure the delivered module if calibration or scene coverage matters. The camera's 5-MP native array does not mean the current ESP32-P4 driver/ISP path offers a supported 5-MP capture mode.

### Modes in the current Espressif OV5647 driver

| Sensor output | Lanes | Configured lane rate | Intended use |
|---|---:|---:|---|
| RAW10 1920 x 1080 at 30 fps | 2 | approximately 408.33 Mbit/s per lane | The correct mode for this project's 1080p requirement |
| RAW10 1280 x 960, binned, at 45 fps | 2 | driver-defined | Higher-rate, smaller image |
| RAW8 800 x 1280 at 50 fps | 2 | driver-defined | Portrait display/demo path |
| RAW8 800 x 800 at 50 fps | 2 | driver-defined | Square processing |
| RAW8 800 x 640 at 50 fps | 2 | driver-defined | Reduced processing |

There is no 1080p25 or 1080p20 register table in the current OV5647 driver. The reliable starting mode is therefore **1080p30**.

For a nominally lower application rate:

- easiest: capture, ISP-process and encode at 30 fps and let the receiver/display consume the cadence it needs;
- lower algorithm load: always retain the newest frame and schedule the CPU algorithm at 25 or 20 Hz, returning skipped capture buffers immediately;
- lower encoded-frame count: feed selected frames to the encoder and generate timestamps from the selected cadence, then verify playback duration;
- lowest sensor/link/ISP load: create and validate a new OV5647 timing/register mode. This is driver work and should not be assumed to result from changing an application `fps` field.

Selecting five of every six 30-fps frames produces 25 frames each second; selecting two of every three produces 20. Simple dropping creates uneven inter-frame spacing, so timestamp or rate-control logic must be explicit when uniform presentation timing matters.

## 1080p resource envelope

### Frames and CPU time

| Application cadence | Frames/minute | Pixels/minute | 360-MHz cycles/frame, one HP core | Cycles/pixel, one HP core |
|---:|---:|---:|---:|---:|
| 20 fps | 1,200 | 2.48832 billion | 18.0 million | 8.68 |
| 25 fps | 1,500 | 3.11040 billion | 14.4 million | 6.94 |
| 30 fps | 1,800 | 3.73248 billion | 12.0 million | 5.79 |

One 360-MHz core has 21.6 billion clock cycles per minute; both cores have a 43.2-billion-cycle arithmetic ceiling. These are cycle budgets, not useful-operation guarantees. Interrupts, RTOS work, stalls, cache misses, shared PSRAM, DMA arbitration and serial I/O consume part of the budget. Fixed-function ISP/PPA/JPEG/H.264 operations should not be charged as if the CPU executed one scalar instruction per pixel.

The low cycles-per-pixel row explains the correct architecture: CSI, ISP, PPA and the codecs handle full frames; C or assembly handles a cropped/scaled luma plane, a tensor, metadata, or a small tiled residual kernel.

### Active data and buffer sizes

These are tight-packed active-pixel values. The negotiated V4L2 `bytesperline` and `sizeimage` remain authoritative.

The 20/25-fps columns describe traffic if a stage truly runs at that cadence. With the current supported sensor mode, the OV5647, CSI receiver and normal capture/ISP path still run at 30 fps and therefore retain the 30-fps RAW/capture traffic even when the application discards frames. Only work downstream of the discard point falls to 25 or 20 fps.

| Format | Bytes/frame | 20 fps | 25 fps | 30 fps | Two buffers |
|---|---:|---:|---:|---:|---:|
| Packed RAW10 sensor payload | 2,592,000 | 51.84 MB/s | 64.80 MB/s | 77.76 MB/s | 4.94 MiB |
| YUV420 encoder input | 3,110,400 | 62.21 MB/s | 77.76 MB/s | 93.31 MB/s | 5.93 MiB |
| RGB565/YUV422 | 4,147,200 | 82.94 MB/s | 103.68 MB/s | 124.42 MB/s | 7.91 MiB |
| RGB888 | 6,220,800 | 124.42 MB/s | 155.52 MB/s | 186.62 MB/s | 11.87 MiB |

At 1080p30 the two CSI lanes signal at about 816.67 Mbit/s combined. Active RAW10 pixels account for 622.08 Mbit/s; the remainder covers sensor blanking and MIPI protocol/timing rather than application data.

A practical encoder-oriented allocation starts with two YUV420 frames, consuming about 5.93 MiB. Add a separately sized H.264 bitstream ring, small PPA outputs for preview/vision, models and activations, task stacks, network/filesystem buffers and a safety margin. Never allocate the remaining PSRAM by subtracting only 5.93 MiB from 32 MiB; mappings, firmware configuration, fragmentation and other consumers reduce what is usable.

### Compressed output

The current `esp_video` H.264 wrapper defaults to 10 Mbit/s and permits up to 25 Mbit/s. At a steady average:

| H.264 bitrate | Payload per second | Payload per minute | Payload per hour |
|---:|---:|---:|---:|
| 10 Mbit/s | 1.25 MB | 75 MB | 4.5 GB |
| 25 Mbit/s | 3.125 MB | 187.5 MB | 11.25 GB |

These omit filesystem/container/network overhead and do not describe peak I-frame size. A 100-Mbit Ethernet link has enough line-rate headroom for either compressed setting, but application throughput is lower than 100 Mbit/s. Wi-Fi uses the C6 companion and must be measured end to end. Raw 1080p is not a sensible external transport for this board.

For 1080p at only 20--30 fps, begin below 10 Mbit/s if scene quality is acceptable, then raise bitrate using recorded high-detail and low-light scenes. Noise increases compression cost.

## Compute path to implement

```text
OV5647 rolling-shutter sensor
  -> two-lane MIPI CSI, RAW10 1920x1080@30
  -> P4 CSI receiver
  -> P4 ISP + esp_ipa exposure/white-balance control
  -> DMA-backed PSRAM buffers
  -> branch A: YUV420 -> hardware H.264 -> Ethernet/Wi-Fi/microSD
  -> branch B: PPA scale/crop -> small preview or algorithm input
  -> branch C: HP cores + PIE/ESP-DL -> custom vision algorithm
```

The P4 owns all camera and image compute in this diagram. The C6 only provides Wi-Fi/BLE connectivity. For a custom algorithm, avoid scanning a full RGB888 1080p frame on both cores. Ask the ISP for the most useful output, use PPA to create the smallest useful input, and keep the CPU kernel's hot tile in internal memory/cache.

## Bring-up route

Use ESP-IDF rather than making Arduino the reference environment. Waveshare currently warns that ESP32-P4 Arduino support is limited, while the ESP-IDF stack exposes CSI, ISP, PPA and hardware codecs.

1. Connect the included OV5647 with the included 15-pin cable while the board is unpowered. Fully release and relock both FFC latches; preserve the kit's documented contact orientation.
2. Install the ESP-IDF version required by the checked-out example. Waveshare's current platform repository says most examples target ESP-IDF 5.4 or later.
3. Run Waveshare's `00_board_check`, then record the chip revision, clock, flash and PSRAM results.
4. Use the current Waveshare `ESP32-P4-Platform` repository's `16_video_lcd_display` or `17_simple_video_server` as a bring-up reference. The compatibility matrix marks both for ESP32-P4-NANO.
5. In `menuconfig`, use MIPI CSI, select OV5647, select `RAW10 1920x1080 30fps`, and verify SCCB/I2C SCL `GPIO8`, SDA `GPIO7`. The vendor examples default to a smaller `800 x 1280` mode, so 1080p must be selected deliberately.
6. Start with two capture buffers. Log sensor PID, selected mode, lane rate, negotiated format, stride, `sizeimage`, buffer count, sequence and timestamp.
7. Validate capture-only for one minute: at 30 fps expect 1,800 capture opportunities. Count successful buffers, sequence gaps, duplicates and late frames.
8. Add ISP/PPA, then H.264/JPEG, then Ethernet or microSD, and only then the custom CPU/PIE algorithm. Re-run the one-minute counters after each stage.

Waveshare's simple server is useful for an initial MJPEG/browser check. For the final bandwidth-efficient recorder or stream, use Espressif's `esp_video` V4L2 H.264 path and YUV420 buffers. Keep container timestamps tied to measured selected frames; an open 2026 report describes accelerated playback when an OV5647 1080p30 recording's declared cadence and captured timing did not agree.

## Acceptance decision

This kit meets the stated requirement if the delivered unit passes all of these:

- OV5647 PID detected and 1080p30 RAW10 mode selected;
- 1,800 capture opportunities in a 60-second capture-only test with acceptable drops;
- sufficient largest contiguous PSRAM block for two negotiated YUV420/RGB buffers plus the selected codec/model allocations;
- stable H.264 or JPEG output to the intended Ethernet, Wi-Fi or microSD destination;
- exact delivered silicon revision recorded and matched to build configuration/errata;
- application produces 30 fps directly, or deliberately selects and timestamps 25/20 fps without silently accumulating latency.

For deeper design, use [camera development boards](09-camera-development-boards.md) for alternatives and [camera pipeline and bandwidth](10-camera-pipeline-bandwidth.md) for ownership, queueing, formats and performance-test methodology.
