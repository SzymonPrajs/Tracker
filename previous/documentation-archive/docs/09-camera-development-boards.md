# Camera-equipped ESP32-P4 development boards

Last verified: 2026-08-08.

This chapter covers boards that ship with, enclose, or have an official kit option for a camera. It separates three quantities that vendors often blur together:

- **Sensor capability**: what the image sensor silicon or module data sheet advertises.
- **Driver mode**: a resolution, format, lane count, lane rate, and frame rate for which the current Espressif driver contains a register table.
- **Demonstrated application behavior**: what a particular example actually configures and records. A constant named `30 FPS` is not a measured guarantee.

All five boards below are relevant to a 32 MB PSRAM target. Only the two `P4X` Espressif boards explicitly identify the newer ESP32-P4 revision family; confirm the exact eFuse revision of every purchased unit.

For an ordinary purchasable board with a removable camera rather than a camera-specific appliance, the first choice is now the [Waveshare ESP32-P4-NANO-KIT-A](11-buyable-board-camera-kit.md). It bundles the general-purpose board, OV5647 module and matched 15-pin cable and supports the current driver's RAW10 1920 x 1080 at 30 fps mode.

## Quick selection

| Board | Camera supplied | Display and output path | P4 memory | Camera link | Most defensible current capture mode | Best use |
|---|---|---|---|---|---|---|
| [Waveshare ESP32-P4-NANO-KIT-A](11-buyable-board-camera-kit.md) | Removable OV5647, 5 MP, supplied with matched cable | No display required; MIPI DSI optional; microSD, USB HS, 100-Mbit Ethernet | ESP32-P4NRW32, 32 MB PSRAM; 16 MB flash | 2-lane MIPI CSI, Pi-style 15-pin | RAW10 1920 x 1080 at 30 fps in the current Espressif driver | Recommended generic board + replaceable camera |
| [ESP32-P4X-EYE](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32p4/esp32-p4x-eye/user_guide.html) | OV2710, 2 MP, attached in enclosure | 240 x 240 SPI LCD, microSD, USB 2.0 HS device | 16 MB flash; official demo detects 32 MB PSRAM | 1-lane MIPI CSI | RAW10 1920 x 1080 or 1280 x 720 at 25 fps in the current driver | Small camera product, AI, recorder, battery prototype |
| [ESP32-P4X-Function-EV-Board](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32p4/esp32-p4x-function-ev-board/user_guide.html) | Optional 2 MP module, adapter and cable | Optional 1024 x 600 MIPI DSI LCD; 100-Mbit Ethernet, SD, USB HS | 16 MB flash; official examples detect 32 MB PSRAM | 2-lane MIPI CSI | Module specifies RAW10 1920 x 1080 at 30 fps; exact sensor identity is not published | Instrumented multimedia pipeline and large-display work |
| [Waveshare ESP32-P4-WIFI6-Touch-LCD-3.5](https://www.waveshare.com/esp32-p4-wifi6-touch-lcd-3.5.htm) | OV5647, 5 MP, supplied | 320 x 480 SPI LCD, microSD, USB HS | ESP32-P4NRW32, 32 MB PSRAM; 16 MB flash | 2-lane MIPI CSI, 15-pin/0.5 mm | RAW10 1920 x 1080 at 30 fps in the current Espressif driver | Compact 5-MP-sensor experiments and small HMI |
| [M5Stack Tab5](https://docs.m5stack.com/en/core/Tab5) | SC2356, 2 MP, built in | 1280 x 720 MIPI DSI; microSD, USB, audio, optional battery | ESP32-P4NRW32 at 360 MHz; 32 MB octal PSRAM; 16 MB flash | MIPI CSI, board routes both data lanes | RAW8/RAW10 1600 x 1200 at 30 fps is registered; use 1600 x 900 or smaller when the ISP/H.264 path is required | Finished portable terminal, large preview and UI |

“Most defensible” means a mode present in source or in the supplied module specification. It is not a sustained end-to-end benchmark with AI, display, storage, and radio active.

## 1. ESP32-P4X-EYE

### Board and optical module

The P4X-EYE is the most direct reference for a compact ESP32-P4 camera product. The board adds an ESP32-C6-MINI-1U for radio, a 16 MB SPI flash, 4-bit microSD, USB 2.0 High-Speed device port, digital microphone, fill light, battery connector, and a 240 x 240 ST7789 SPI display. The P4—not the C6—receives and processes camera pixels. The C6 carries Wi-Fi/Bluetooth traffic.

The attached [HDF2710-47-MIPI-V2.0 module drawing](https://dl.espressif.com/AE/esp-dev-kits/HDF2710-47-MIPI-V2.0.pdf) identifies:

- OV2710, 1920 x 1080 array, 1/2.7-inch optical format;
- one MIPI data pair plus a MIPI clock pair on a 24-pin flex;
- 4.1 mm effective focal length, f/2.0, approximately 107-degree field of view;
- manual lens adjustment, rolling shutter, and SCCB/I2C control;
- approximately 18 x 13 mm module body and a 47 mm flex.

The [OV2710 product brief](https://dl.espressif.com/AE/esp-dev-kits/ov2710pbv1.1web.pdf) advertises 1080p30, cropped 720p60, VGA120 and QVGA240, RAW8/RAW10, and a single MIPI lane up to 800 Mbit/s. Those are sensor maxima, not the modes shipped by Espressif's current component.

### Modes actually present in Espressif's driver

The pinned [`ov2710.c`](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_cam_sensor/sensors/ov2710/ov2710.c) contains only:

| Mode | Lanes | Configured lane rate | Active-pixel payload | Frames/minute |
|---|---:|---:|---:|---:|
| RAW10 1920 x 1080 at 25 fps | 1 | 800 Mbit/s | 64.80 MB/s | 1,500 |
| RAW10 1280 x 720 at 25 fps | 1 | 800 Mbit/s | 28.80 MB/s | 1,500 |

The 800-Mbit/s lane setting includes blanking and link overhead; the active-pixel figure is the lower bound `width x height x 10 x fps / 8`. Do not change `fps` in an application structure and assume the sensor register timing changed with it.

### What the factory demo really does

Espressif's [factory demo](https://github.com/espressif/esp-dev-kits/tree/ddb440c409da7adb0a4ff7902b0133fb3fb6cfa3/examples/esp32-p4-eye/examples/factory_demo) is unusually useful because it is a full application rather than a one-buffer camera sample. It uses:

- `/dev/video0` through the V4L2-compatible `esp_video` layer;
- two MMAP capture buffers and RGB565 as the application camera format;
- ISP/IPA for RAW processing and automatic image control;
- PPA for scale, crop, rotation and preview preparation;
- hardware JPEG encoding for photographs and video frames;
- ESP-DL for face, pedestrian, and YOLO11-nano inference;
- microSD and an MP4 muxer for recording.

The recorder is **MJPEG in MP4**, not H.264. Its source sets JPEG quality 80, labels the stream as 30 fps, starts its measured-rate filter at 15 fps, and clamps that estimate to 5–30 fps. The current OV2710 register mode is 25 fps, so the `30` is container/timestamp intent, not proof that the camera supplies 30 unique frames each second. The checked-in startup log also comes from the earlier `ESP32-P4-EYE` and reports chip revision v1.0; it must not be used as evidence for the silicon revision in a new P4X-EYE.

### Explicit buffer footprint in the demo

At 1920 x 1080 RGB565, the source allocates or requests approximately:

| Allocation | Count x bytes | Total |
|---|---:|---:|
| V4L2 capture buffers | 2 x 4,147,200 | 8,294,400 B |
| 240 x 240 RGB565 LCD canvases | 2 x 115,200 | 230,400 B |
| 1080p RGB565 scale/work buffer | 1 x 4,147,200 | 4,147,200 B |
| Shared 1280 x 720 RGB565 photo/record buffer | 1 x 1,843,200 | 1,843,200 B |
| Nominal JPEG output allocation (`1080p` rounded to 1088 lines, assumed 5:1) | 1 x 835,584 | 835,584 B |
| AI display-frame ring | 5 x 115,200 | 576,000 B |
| Small AI feed/result pipeline allocations | 5 x 7,200 plus 5 x 80 | 36,400 B |
| **Subtotal, excluding models, stacks, heap metadata, filesystem, audio and UI** |  | **15,963,184 B (15.22 MiB)** |

The calculation uses the requested V4L2 format and tight packing; the driver's reported `sizeimage` remains authoritative. The demo's 5:1 JPEG output allocation is an engineering assumption, not a guaranteed maximum compressed-frame size.

The demo startup log detects 32 MB PSRAM but adds about 30,656 KiB to the heap after mappings and reservations. A complex program never begins with all 32 MiB available for camera frames.

### Revision warning

Espressif states that `P4X-EYE` means chip revision v3.1 or later. Prefer v3.2 hardware for the coherency fixes described in the errata chapter. Read the actual eFuse revision at boot; do not infer it from an enclosure or from the old factory-demo log.

## 2. ESP32-P4X-Function-EV-Board camera kit

This is the better board for observing a complete `camera -> ISP -> MIPI DSI display` path. The optional camera kit contains the sensor module, a 15-pin-to-24-pin adapter, and the correct forward-direction flex cable. The separate optional LCD is 1024 x 600. The board also offers 10/100 Ethernet, USB 2.0 HS, SD, audio, and an ESP32-C6-MINI-1.

The [camera-module specification](https://dl.espressif.com/dl/schematics/camera_datasheet.pdf) identifies the module as `AS-AG638A32M2-50`, but does **not** name the sensor die. It specifies:

- 1/3-inch, 2.65 um pixels, 1920 x 1080 RAW10 at 30 fps;
- two MIPI data lanes, rolling shutter, SCCB/I2C register control;
- 12 or 24 MHz input clock;
- 4.6 mm focal length, f/2.0, 66/60/36-degree diagonal/horizontal/vertical FOV;
- less than 1% stated distortion.

The [adapter schematic](https://dl.espressif.com/dl/schematics/esp32-p4-function-ev-board-camera-subboard-schematics.pdf) confirms two lanes, level shifting for SCCB, 1.8/2.8 V rails, reset, and a local 24 MHz oscillator.

The electrical and mode description is consistent with an SC2336-class module, and Espressif examples support SC2336, but the module document does not establish that identity. Production code should detect the sensor ID rather than hard-code this inference.

If the fitted unit is confirmed as SC2336, the current [`sc2336.c`](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_cam_sensor/sensors/sc2336/sc2336.c) includes these useful modes:

| Format and resolution | Frame rate | Lanes x rate | Active payload |
|---|---:|---:|---:|
| RAW10 1920 x 1080 | 30 fps | 2 x 405 Mbit/s | 77.76 MB/s |
| RAW10 1920 x 1080 | 25 fps | 2 x 330 Mbit/s, or 1 x 660 Mbit/s | 64.80 MB/s |
| RAW8 1920 x 1080 | 30 fps | 2 x 336 Mbit/s | 62.21 MB/s |
| RAW10 1280 x 720 | 30/50/60 fps | 2 x 405 Mbit/s | 34.56/57.60/69.12 MB/s |
| RAW8 1280 x 720 | 30 fps | 2 x 336 Mbit/s | 27.65 MB/s |

The official ESP-IDF [`mipi_isp_dsi` example](https://github.com/espressif/esp-idf/tree/7101770dc6db2667b3c477cc31365dd1acd6db4e/examples/peripherals/camera/mipi_isp_dsi) auto-detects supported sensors and demonstrates the direct CSI-to-ISP-to-DSI path. It is a cleaner throughput baseline than an LVGL-heavy phone demo.

Espressif states that P4X Function boards use revision v3.1 or later. Revision v3.1 must not use Secure Download; prefer v3.2 if available.

## 3. Waveshare ESP32-P4-WIFI6-Touch-LCD-3.5

Waveshare's “Smart Vision” board ships with an OV5647 5-MP module, a 320 x 480 SPI touchscreen, ESP32-C6H8 radio coprocessor, microphone, audio codec/amplifier, microSD, and USB HS. Its documentation explicitly identifies the main part as `ESP32-P4NRW32` with 32 MB stacked PSRAM and 16 MB NOR flash, and the camera connector as two-lane MIPI CSI, 15 pins at 0.5 mm pitch.

The [product page](https://www.waveshare.com/esp32-p4-wifi6-touch-lcd-3.5.htm) advertises:

- 2592 x 1944 still capture;
- 1920 x 1080 at 30 fps;
- 1280 x 720 at 60 fps;
- 640 x 480 at 60 or 90 fps;
- 1/4-inch sensor, 3.07 mm lens, f/2.4 and 72.9-degree FOV.

Current Espressif [`ov5647.c`](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_cam_sensor/sensors/ov5647/ov5647.c) exposes a different, smaller set:

| Driver mode | Lanes x rate | Active payload | Frames/minute |
|---|---:|---:|---:|
| RAW10 1920 x 1080 at 30 fps | 2 x 408.33 Mbit/s | 77.76 MB/s | 1,800 |
| RAW10 1280 x 960 binned at 45 fps | 2 x 441.67 Mbit/s | 69.12 MB/s | 2,700 |
| RAW8 800 x 1280 at 50 fps | 2 x 400 Mbit/s | 51.20 MB/s | 3,000 |
| RAW8 800 x 800 at 50 fps | 2 x 400 Mbit/s | 32.00 MB/s | 3,000 |
| RAW8 800 x 640 at 50 fps | 2 x 400 Mbit/s | 25.60 MB/s | 3,000 |

The 5-MP still claim describes the sensor, not the normal ISP/H.264 envelope. The ESP32-P4 ISP is specified for at most 1920 x 1080. A CSI bypass path can receive larger frames if link and memory limits permit, but the pinned OV5647 driver does not include a 2592 x 1944 mode table. Treat full-resolution capture as new sensor-driver and bypass-path work, not a menu selection.

Because the display is SPI, full camera frames should stay in the capture/processing path. Use PPA to create a 320 x 480 or smaller preview buffer; repeatedly scanning a full 1080p RGB frame merely to update the LCD wastes PSRAM bandwidth.

This board uses the non-`X` part name. Do not assume the 400 MHz/v3.x behavior documented for `ESP32-P4NRW32X`; print chip revision and maximum supported CPU frequency on the exact unit. Waveshare recommends ESP-IDF over the less mature Arduino adaptation for demanding work.

## 4. M5Stack Tab5

Tab5 is a finished portable terminal rather than a bare board. It integrates a 5-inch 1280 x 720 MIPI DSI touch display, SC2356 2-MP camera, ESP32-C6-MINI-1U, dual microphones and codecs, microSD, USB host/device, RS-485, sensors, and an optional NP-F550 battery. M5Stack specifies `ESP32-P4NRW32`, two HP cores at 360 MHz, 32 MB octal PSRAM, and 16 MB flash.

M5Stack calls the camera `SC2356`. Espressif's component calls its driver `SC202CS`; the register-table comments identify `FT_SC2356`. Treat these as the board/vendor name and the software-driver name for the same supported module family, but retain sensor-ID detection.

The current [`sc202cs.c`](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_cam_sensor/sensors/sc202cs/sc202cs.c) registers:

| Driver mode | Lanes x rate | Active payload | Frame size |
|---|---:|---:|---:|
| RAW10 1600 x 1200 at 30 fps | 1 x 720 Mbit/s | 72.00 MB/s | 2,400,000 B |
| RAW10 1600 x 900 at 30 fps | 1 x 720 Mbit/s | 54.00 MB/s | 1,800,000 B |
| RAW8 1600 x 1200 at 30 fps | 1 x 576 Mbit/s | 57.60 MB/s | 1,920,000 B |
| RAW8 1280 x 720 at 30 fps | 1 x 576 Mbit/s | 27.65 MB/s | 921,600 B |

Although the connector routes two data lanes, the current tables use one. Also, 1600 x 1200 exceeds the ISP's specified 1080-line maximum even though it is only 1.92 MP. Use 1600 x 900, crop before ISP, or a bypass/raw path when full 4:3 capture is necessary. H.264 remains limited to YUV420 and a 1080p30 aggregate envelope.

The large 720p display itself is a material bandwidth consumer: one RGB565 screen is 1,843,200 bytes, and a full refresh at 30 fps is 55.30 MB/s before DSI blanking/packet overhead. Camera, UI and display buffers therefore compete for the same PSRAM/cache system even though CSI and DSI use separate serial links.

## Frame-rate interpretation

| Rate | Frame interval | Frames/minute | Typical meaning in this board set |
|---:|---:|---:|---|
| 25 fps | 40.00 ms | 1,500 | Current OV2710 modes; SC2336 option |
| 30 fps | 33.33 ms | 1,800 | Normal 1080p or 2-MP target |
| 45 fps | 22.22 ms | 2,700 | OV5647 binned 1280 x 960 driver mode |
| 50 fps | 20.00 ms | 3,000 | Several cropped RAW8/RAW10 modes |
| 60 fps | 16.67 ms | 3,600 | SC2336 720p driver mode; advertised OV5647/OV2710 crop modes |
| 90 fps | 11.11 ms | 5,400 | Advertised OV5647 VGA mode, absent from pinned driver |
| 120 fps | 8.33 ms | 7,200 | OV2710 sensor VGA maximum, absent from pinned driver |
| 240 fps | 4.17 ms | 14,400 | OV2710 sensor QVGA maximum, absent from pinned driver |

For an algorithm costing `K` operations per pixel, the active-pixel work per minute is `width x height x fps x 60 x K`. Examples before any crop or subsampling:

- 1080p25: 3.1104 billion pixels/minute.
- 1080p30: 3.73248 billion pixels/minute.
- 720p60: 3.31776 billion pixels/minute.
- 1600 x 1200 at 30 fps: 3.456 billion pixels/minute.

These are pixel counts, not CPU instruction counts. ISP stages and H.264/JPEG macroblock processing are fixed-function work and should not be converted into fictitious RISC-V “operations.”

## Purchase and bring-up checklist

1. Record the full module marking, board revision, ESP32-P4 eFuse revision, flash ID and PSRAM ID.
2. Confirm that the camera is actually included; the Function EV camera is optional and several larger Waveshare boards offer it only as an option.
3. Confirm connector pitch, contact orientation, lane count, rail voltages, reset/power-down polarity and XCLK source before swapping modules.
4. Let the driver detect the sensor PID. Log the selected register mode, pixel format, lane count, configured lane rate and V4L2 `sizeimage`.
5. First measure capture-only frame rate. Add ISP/PPA, then display, then encoder, then storage/network, and finally AI. Record drops at each stage.
6. Run at least one minute: expected frames are `fps x 60`; report captured, processed, displayed, encoded, saved, dropped and repeated frames separately.
7. Re-run with cold/warm cache and with both flash and PSRAM XIP enabled, because executable traffic can contend with frame traffic.
