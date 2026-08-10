# Amazon UK shopping list: ESP32-P4-Module-DEV-KIT camera build

Last verified: 2026-08-08. Prices, stock and Prime delivery are snapshots and can change by account and postcode. Product identity is recorded by ASIN so that a changed title or search result is easier to detect.

## Board decision

The supplied Amazon item, [Waveshare ESP32-P4-Module-DEV-KIT](https://www.amazon.co.uk/dp/B0FPG45999), ASIN `B0FPG45999`, is a suitable general-purpose ESP32-P4 camera development board. The observed price was **£24.95** with Prime delivery.

It is important to distinguish its two bulk memories:

- **32 MB in-package PSRAM** for frames, models, heaps and other runtime data;
- **16 MB on-board NOR flash** for firmware and assets.

Thus it is the desired “32 MB” board if that means PSRAM. It is not the 32 MB flash variant of Waveshare's separate module.

The board provides two-lane MIPI CSI, the P4 ISP/PPA/JPEG/H.264 hardware, microSD, 100-Mbit Ethernet, USB 2.0 High-Speed OTG, two USB-C programming/power paths, a 40-pin GPIO header and an ESP32-C6 radio companion. The camera pixels and vision compute run on the P4; the C6 supplies 2.4-GHz Wi-Fi 6 and Bluetooth.

### No external antenna is required

The development board already has an **ESP32-C6 SMD antenna**. The small IPEX/u.FL-style provision on the underlying module is an alternative RF route, not an additional antenna that must be fitted. Waveshare documents that the RF signal goes to the board pad by default and can be switched to the IPEX Gen-1 connector by moving a reserved zero-ohm resistor. That is a soldering modification and should be attempted only if enclosure or range measurements justify it.

Initial purchase: **no antenna**.

### Check what is in the board box

Waveshare's official base `ESP32-P4-Module-DEV-KIT` package list includes the board, an 8-ohm 2-watt speaker and two approximately 150-mm SH1.0 four-pin cables. The Amazon description is less explicit. Inspect the delivered box before buying duplicate speaker or SH1.0 leads.

## Core build: buy these

| Purpose | Selected Amazon UK item | Verified snapshot | Why this item / fit note |
|---|---|---:|---|
| Development board | [Waveshare ESP32-P4-Module-DEV-KIT](https://www.amazon.co.uk/dp/B0FPG45999), `B0FPG45999` | £24.95, Prime | 32 MB PSRAM, 16 MB flash, C6 Wi-Fi/BLE, CSI, microSD, Ethernet and 40-pin expansion. |
| Starter camera | [Hailege OV5647 5 MP MIPI CSI camera](https://www.amazon.co.uk/dp/B07Y9WFM13), `B07Y9WFM13` | £6.59, Prime | OV5647 is supported by the current ESP32-P4 sensor stack; listing states 1080p, fixed focus and an included 15-cm FPC cable. This is the sensible inexpensive first camera. |
| Two USB-C data/power cables | [Anker 333 100 W USB-C to USB-C, two-pack, 1 m](https://www.amazon.co.uk/dp/B09LCD851Z), `B09LCD851Z` | £10.99, Prime | Manufacturer text states charging plus USB 2.0 data up to 480 Mbit/s. Use one for UART/programming and one for power. “Not for video output” does not mean “charging only.” |
| Compact battery | [Anker 10,000 mAh 30 W USB-C power bank](https://www.amazon.co.uk/dp/B0CZ9M6X8Q), `B0CZ9M6X8Q` | £17.99, Prime | Safely supplies regulated USB power without designing a Li-ion charger or boost converter. |
| UK mains supply | [Anker Nano 30 W USB-C UK charger](https://www.amazon.co.uk/dp/B0CD75QL17), `B0CD75QL17` | £16.99, Prime | A known-brand regulated USB-C supply; more capacity than this 5-V board needs and reusable elsewhere. Cable is not included, hence the two-pack above. |
| GPIO breakout plus breadboard | [ZDE 40-pin T breakout, ribbon cable and breadboard](https://www.amazon.co.uk/dp/B0DKNDDQX8), `B0DKNDDQX8` | £9.99, Prime | Physically mates with the board's Raspberry-Pi-form-factor 2x20 header and provides a solderless work area. See the pin-label warning below. |
| Jumper leads | [ELEGOO 120-piece M-M/M-F/F-F Dupont set](https://www.amazon.co.uk/dp/B01EV70C78), `B01EV70C78` | £5.94, Prime | Covers breadboard-to-breakout and module-to-module wiring. |
| Video storage | [SanDisk High Endurance 64 GB microSDXC](https://www.amazon.co.uk/dp/B07P3D6Y5B), `B07P3D6Y5B` | £24.88, Prime | U3/V30-class high-endurance media is more appropriate for repeated video writes than a basic card. Format and sustained-write-test the delivered card on the target. |

Observed total for this complete starter set, including the board: **£118.32**.

### Camera upgrade option

For adjustable focus, interchangeable M12 lenses and a small case, substitute the starter camera with [Arducam OV5647 M12 camera with case](https://www.amazon.co.uk/dp/B0867C9SZP), ASIN `B0867C9SZP`, observed at **£17.99 Prime**. Its listing does not clearly promise the required cable, so add [six 15-pin 1.0-mm Raspberry Pi camera FFC cables](https://www.amazon.co.uk/dp/B08SM39RFS), ASIN `B08SM39RFS`, observed at **£7.39 Prime**.

The required camera lead is **15 conductors, 1.0-mm pitch, 15-pin to 15-pin, Pi-4-style/opposite-contact**. Do not buy a Pi Zero or Raspberry Pi 5 **15-to-22-pin** cable for this connection.

#### Lens experimentation: recommended combination

No credible Prime-listed one-box bundle containing both an OV5647 camera and several documented lenses was found. The reliable approach is to buy the camera and matched lens kit as two products.

Pair that camera with [Arducam LK001 10-lens M12 set](https://www.amazon.co.uk/dp/B07L92S9MT), ASIN `B07L92S9MT`, observed at **£74.46 Prime**. This is the strongest general experimentation set found on Amazon UK:

- specifically described for 1/4-inch cameras such as OV5647 and for the Arducam `B0031` M12 camera family;
- ten labelled M12 lenses with nominal angles of view of 10, 20, 40, 60, 80, 100, 120, 140, 160 and 200 degrees;
- telephoto, ordinary/portrait, wide-angle, fisheye and close-focus/macro experimentation in one box;
- four M12 holders covering 18-mm/20-mm screw spacing and 7-mm/13-mm height, plus screws, specifications and a cleaning cloth;
- the manufacturer's lens table specifies visible-light IR filtering, which is preferable to unverified generic CCTV lenses when correct daytime colour matters.

The camera, lens set and spare FFC pack total **£99.84**. The camera's supplied case may have to be removed when changing holder height or using a physically wider lens.

| Nominal lens-angle group | Useful experiments | Optical consequence |
|---|---|---|
| 10--20 degrees | distant targets, alignment, small region inspection | narrow view, precise mounting and vibration control required |
| 40--60 degrees | people, object tracking, ordinary scene capture | most natural-looking and easiest to calibrate |
| 80--100 degrees | room coverage, mobile robot, wider context | moderate edge distortion and fewer pixels per subject |
| 120--140 degrees | close-range navigation, doorway/room coverage | strong perspective; calibration becomes important |
| 160--200 degrees | fisheye, panoramic mapping, optical-flow experiments | severe nonlinear distortion; rectification costs pixels and compute |

For geometry, tracking, pose or measurement work, the more conservative alternative is [Arducam LK002 five-lens low-distortion set](https://www.amazon.co.uk/dp/B07NW8VR71), ASIN `B07NW8VR71`. It contains five lenses tested on 1/4-inch OV5647/OV2640 sensors and covers approximately 45--90 degrees horizontal FOV. It was **£58.94 but not Prime** at inspection, with slower delivery. It is less varied than LK001 but optically more relevant to repeatable computer-vision algorithms.

A cheaper Prime-only trial set can be assembled from a [1.8-mm 180-degree fisheye](https://www.amazon.co.uk/dp/B07PZDL7TV) at £8.39, [4-mm M12 lens](https://www.amazon.co.uk/dp/B0FWK18G23) at £8.99 and [12-mm M12 lens](https://www.amazon.co.uk/dp/B0FWK39G8W) at £9.19. The £26.57 price is attractive, but these generic listings do not document sensor-specific FOV, back-focus, holder height or IR-cut behaviour as well as Arducam's kits. Treat them as optical experiments, not as calibrated substitutes.

#### Other OV5647 camera variants

- [Generic OV5647 day/night module with two IR LEDs](https://www.amazon.co.uk/dp/B095NQT3GJ), £14.99 Prime: useful for a night-vision experiment, but LED current, heat and the IR switching behaviour must be checked on this board.
- [Hailege OV5647 adjustable-focus night-vision camera with case](https://www.amazon.co.uk/dp/B07XD8VB31), £17.49 Prime: a packaged night option, but it is not documented as a multi-holder lens-development system.
- [Arducam automatic day/night OV5647 with interchangeable M12 lens](https://www.amazon.co.uk/dp/B07X1VGQSL), £29.11 at inspection: the technically more interesting day/night variant, but it was not Prime and had slower delivery.

Stay with OV5647 for the first ESP32-P4 work. Amazon's IMX219, IMX708, IMX500 and autofocus Raspberry Pi cameras may share a CSI connector or an M12 mount, but that does not give them a supported ESP32-P4 sensor driver or board configuration.

Every lens change creates a new camera. Record lens identity, holder height, focus position and target resolution, then generate a separate calibration profile for distortion, focal length/principal point and any algorithm thresholds. Do not reuse one camera matrix across the 10-lens set.

### Longer-runtime battery option

Substitute [Anker Zolo 20,000 mAh 30 W power bank](https://www.amazon.co.uk/dp/B0CZ9LH53B), ASIN `B0CZ9LH53B`, observed at **£29.99 Prime**, for approximately twice the stored energy of the 10,000-mAh selection. It also has an integrated USB-C lead, but retain at least one separate data cable for programming.

A 10,000-mAh bank contains about 37 Wh nominal at its internal cell voltage; after conversion and reserve, approximately 25--32 Wh may reach the load. A planning range of 3--6 W for board, camera, radio and storage suggests roughly **4--10 hours**, but this is not a measured board-runtime claim. Measure the actual USB input power under the final workload. The 20,000-mAh choice should be roughly double, subject to the same losses.

### Optional hardware

| Purpose | Item | Verified snapshot | Note |
|---|---|---:|---|
| Spare small-board leads | [SH1.0 1.0-mm prewired 2/3/4/5-pin set](https://www.amazon.co.uk/dp/B07WH5TBQ5), `B07WH5TBQ5` | £8.59, Prime | Optional; wait until the board box is checked. Never infer pin function from insulation colour. |
| PCB mounting | [M2.5 nylon standoff assortment](https://www.amazon.co.uk/dp/B0DCS45XHS), `B0DCS45XHS` | £6.49, Prime | Keeps the board off conductive surfaces; verify hole diameter before forcing any standoff. |

Do not buy a bare 3.7-V LiPo for direct connection. This development board does not document a main-battery charger and power-path controller. The `RTC BAT` header is only for a rechargeable RTC backup cell, not for powering the P4, camera and radio. A custom Li-ion design would need a protected cell, certified charger, load sharing, a regulated 5-V boost stage with adequate continuous and transient current, a fuse, switch and enclosure.

## Breakout-board warning

The development kit is already the carrier/breakout for the soldered ESP32-P4 module; no second module carrier is required. The shopping-list breakout is only for taking the board's 40-pin GPIO connector to a breadboard.

The selected T-cobbler is labelled for Raspberry Pi. It is useful because the **physical 40-pin connector** matches, but its silk-screened Raspberry Pi GPIO names are not the ESP32-P4 signal map.

- Use physical connector pin numbers and Waveshare's ESP32-P4-Module-DEV-KIT pinout.
- Do not copy Raspberry Pi BCM pin numbers from the breakout.
- Treat P4 GPIO as 3.3-V logic unless the board schematic explicitly says otherwise.
- Confirm power and ground with a meter before inserting a peripheral.
- Waveshare says the header supports **some** Raspberry Pi HATs; physical fit does not prove electrical or driver compatibility.

## Camera and storage envelope

The OV5647's supported starting point is two-lane MIPI CSI RAW10 at **1920 x 1080, 30 fps**. If the application needs 25 or 20 fps, keep the sensor/capture path at its supported 30-fps mode and process or encode selected frames with correct timestamps until a separately validated sensor timing table exists.

| Quantity | 1080p30 value |
|---|---:|
| Frames per minute | 1,800 |
| Active packed RAW10 payload | 2,592,000 bytes/frame; 77.76 MB/s |
| YUV420 encoder input | 3,110,400 bytes/frame; 93.31 MB/s |
| Two YUV420 frame buffers | about 5.93 MiB |
| H.264 at 10 Mbit/s | 1.25 MB/s; 75 MB/min; 4.5 GB/hour |

At a steady 10-Mbit/s payload, a nominal 64-GB card holds at most about 14 hours before filesystem/container overhead and reserved capacity. Real record time depends on bitrate control, scene noise, formatting and card behaviour. Keep frames in PSRAM, use DMA/ISP/PPA/H.264 hardware, and expose only reduced/cropped inputs to CPU-heavy algorithms.

## Soldering bench: practical, temperature controlled, not extravagant

No soldering is required to plug in the starter camera, microSD, USB power or 40-pin breakout. The following creates a useful future electronics bench.

| Purpose | Selected Amazon UK item | Verified snapshot | Rationale |
|---|---|---:|---|
| Soldering station | [YIHUA 926LED-IV EVO 110 W station](https://www.amazon.co.uk/dp/B0BJ21NRSY), `B0BJ21NRSY` | £49.99, Prime | Adjustable 90--480 C, display, helping hands, magnifier, five tips and 35 g lead-free solder. A good value choice without paying Weller WE1010 pricing. |
| Fume capture | [Preciva desktop fume extractor with eight filters](https://www.amazon.co.uk/dp/B0BCPLMLS4), `B0BCPLMLS4` | £33.99, Prime | Put the intake close to the joint; it is a capture aid, not a substitute for room ventilation. |
| Multimeter | [AstroAI TRMS 6000-count auto-ranging meter](https://www.amazon.co.uk/dp/B071JL6LLL), `B071JL6LLL` | £23.99, Prime | Continuity, resistance, voltage, current, capacitance, frequency and temperature; useful for verifying header power before connection. |
| ESD work surface | [HPFIX grounded ESD silicone mat and wrist strap](https://www.amazon.co.uk/dp/B098T598G9), `B098T598G9` | £18.99, Prime | Heat-resistant work surface with a stated grounding lead and wrist strap. Connect only to a proper ESD ground point. |
| Flux and rework | [TOWOT no-clean flux plus 10-ft desoldering braid](https://www.amazon.co.uk/dp/B0FL1ZQ5DS), `B0FL1ZQ5DS` | £5.69, Prime | Makes connector/header work and corrections much easier. The station already includes a solder sucker. |
| Spare solder | [TOWOT 0.6-mm lead-free rosin-core solder, 50 g](https://www.amazon.co.uk/dp/B09H2LBYNQ), `B09H2LBYNQ` | £6.97, Prime | Fine enough for PCB headers. Optional initially because the station includes 35 g. |
| Flush cutters | [VCELINK 5-inch precision flush cutter](https://www.amazon.co.uk/dp/B09SL2TCH7), `B09SL2TCH7` | £5.94, Prime | For soft copper leads and component legs; do not use on steel. |
| Wire stripper | [VCELINK 14--24 AWG automatic stripper/cutter](https://www.amazon.co.uk/dp/B0B4J8C8FD), `B0B4J8C8FD` | £9.99, Prime | Covers ordinary hookup wire. Very fine 28-AWG SH leads are better bought pre-crimped than stripped and hand-crimped at first. |

This bench is **£148.58** without the optional spare solder, or **£155.55** with it. Use eye protection, wash hands after handling solder/flux, keep food away from the bench and provide room ventilation even with a desk extractor.

## First connection sequence

1. Inventory the board box; set aside the included speaker and SH1.0 leads if present.
2. With power disconnected, unlock both CSI latches, insert the 15-pin OV5647 cable squarely in the documented orientation, then relock.
3. Insert and format-test the high-endurance microSD card.
4. Power/program first from the computer through a known data-capable USB-C cable. Run Waveshare's board-check example and record chip revision, flash and PSRAM.
5. Bring up OV5647 capture at 1080p30 before enabling Wi-Fi, storage and custom algorithms.
6. Move the same stable build to the USB power bank and measure voltage/current and runtime under the real camera workload.
7. Add the GPIO breakout only after mapping every used physical pin against the Waveshare pinout.

## Primary technical references

- [Waveshare development-board page](https://www.waveshare.com/esp32-p4-module-dev-kit.htm): interfaces, memory, onboard SMD antenna, package contents and camera specification.
- [Waveshare board documentation](https://docs.waveshare.com/ESP32-P4-Module-DEV-KIT): pinout, connector definitions and examples.
- [Waveshare module documentation](https://www.waveshare.com/wiki/ESP32-P4-Module): 360-MHz/40-MHz cores, 32-MB PSRAM and optional IPEX routing through a reserved zero-ohm resistor.
- [Camera pipeline and bandwidth](10-camera-pipeline-bandwidth.md): full implementation and performance-test detail.
