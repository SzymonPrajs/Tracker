# ESP32-P4 setup and hardware characterization

## Fixed hardware and current Mac state

The target is the Waveshare `ESP32-P4-Module-DEV-KIT` with the
`ESP32-P4NRW32`, 32 MB PSRAM, 16 MB flash, and an OV5647 MIPI-CSI camera.
Models are exported as ESP32-P4 ESP-DL `.espdl` artifacts.

Read-only inspection on 2026-08-10 found:

- VS Code 1.130.0 for Apple silicon is installed and its `code` CLI works;
- `espressif.esp-idf-extension` 2.1.1 is installed;
- Git 2.54.0, Python 3.12.2, CMake 4.3.4, Ninja 1.13.2, and Apple Command
  Line Tools are installed;
- `idf.py` is not on `PATH`, `IDF_PATH` is unset, and no EIM-managed ESP-IDF
  installation was found;
- the board was not connected during inspection, so no serial/JTAG device was
  enumerated.

This snapshot is a starting observation, not a permanent dependency. The
scripts below must re-check rather than assume it.

## Reproducible terminal-first setup

Pin one stable ESP-IDF release after checking ESP-DL, esp-video, and board
support compatibility. The current documentation baseline is ESP-IDF v6.0.2.
Use Espressif Installation Manager (EIM), which installs ESP-IDF, the RISC-V
compiler, GDB, OpenOCD, Python environment, and declared tools. Do not rely on a
developer's global shell being preconfigured.

The setup stage will turn the following official procedure into an idempotent
`tools/idf/bootstrap.sh`:

```sh
brew install libgcrypt glib pixman sdl2 libslirp dfu-util cmake python
brew tap espressif/eim
brew install eim
eim install -i v6.0.2 -n true
eim list
eim run "idf.py --version" v6.0.2
```

Before the script changes anything, it prints the planned versions and paths.
It skips matching installations, fails on a conflicting partial setup, and
writes no project-specific paths into the user's global VS Code settings. Save
EIM's generated `eim_config.toml` and the relevant package/tool version report
so another Mac can reproduce the setup.

Create a minimal `firmware/characterize/` ESP-IDF application and prove the
terminal path:

```sh
cd firmware/characterize
eim run "idf.py set-target esp32p4" v6.0.2
eim run "idf.py reconfigure" v6.0.2
eim run "idf.py build" v6.0.2
ls /dev/cu.*
eim run "idf.py -p /dev/cu.SELECTED_PORT flash monitor" v6.0.2
```

The actual serial port is discovered and selected explicitly; it is never
hard-coded. The first application prints the chip model/revision, core/clock
configuration, flash, internal heap, PSRAM size, minimum/largest free blocks,
build ID, and `sdkconfig` hash. A terminal smoke script requires a known boot
marker and saves the monitor log.

## VS Code proof

EIM installations are automatically visible to the Espressif extension through
`$HOME/.espressif/tools/eim_idf.json`. In VS Code:

1. open the repository folder;
2. run **ESP-IDF: Select Current ESP-IDF Version** and choose the pinned setup;
3. run **ESP-IDF: Doctor Command** and save the report in the characterization
   evidence directory after removing machine-private paths;
4. run **ESP-IDF: Set Espressif Device Target** and select `esp32p4`;
5. select the serial port, then build, flash, and monitor the characterization
   app;
6. generate the extension's workspace configuration and prove one OpenOCD/GDB
   breakpoint before relying on interactive debugging.

The terminal build remains authoritative. VS Code must invoke the same EIM
installation and produce the same application hash. Commit only portable
workspace settings and launch tasks; do not commit absolute user paths.

## Benchmark application

The characterization app is deliberately small and separate from production
firmware. It exposes selectable suites and emits one versioned JSON/CSV record
per case. A host runner builds a configuration matrix, flashes it, captures
serial output, validates the schema, and produces comparison tables.

| Suite | Required sweep and output |
|---|---|
| timing overhead | `esp_timer_get_time()` and per-core cycle-counter overhead, warm-up and empty-loop baseline |
| internal memory | aligned sequential read, write, copy, and simple INT8 kernel bandwidth by size and core |
| PSRAM/cache | the same operations from 4 KiB through multi-megabyte working sets; alignment, cache size, one/two cores, CPU versus supported DMA |
| camera only | every OV5647 mode/format; negotiated stride/layout, captured/completed/dropped/corrupt frames and sustained throughput |
| ISP | bypass and enabled pipelines; RAW input, output format/range, frame-completion throughput, black level, lens shading, demosaic, and image checks |
| PPA | full-frame scale and colour conversion for every exact candidate shape; input/output placement, 1/16 scale grid, latency, and bytes moved |
| model only | representative ESP-DL networks and tensors by size; weights/activations in allowed memory placements, cold/warm cache, one/dual-core scheduling |
| combined | camera→ISP/PPA/input mapping→model with two/three capture buffers, one/two small buffers, newest-frame queue, and overlap on/off |

For short kernels use the per-core cycle counter from a task pinned to one core;
for stages use wall-clock timestamps and hardware completion callbacks. Calibrate
measurement overhead, run warm-ups, repeat across boots, and report sample count,
median, p95, maximum, and variability. Logs are compiled out of timed release
regions. Record the firmware commit, tool versions, `sdkconfig`, clock/cache
settings, memory placement, buffer addresses/alignment, and input hashes.

## Byte-movement ledger

Every candidate receives a per-frame ledger rather than only an inference time:

```text
capture DMA writes
+ ISP reads/writes not internal to the streaming path
+ PPA input reads and output writes
+ CPU/DMA format-map reads and writes
+ model weight and activation traffic
+ avoidable copies
= total measured or derived bytes per processed frame
```

Label a number as arithmetic-derived, hardware-counter-measured, or a
contention proxy. Measure end-to-end throughput while stressing competing PSRAM
users. A microbenchmark ceiling is not a sustained camera-loop result.

At 1920×1080, useful lower-bound storage arithmetic is:

| Representation | Bytes/frame | 30 fps write stream |
|---|---:|---:|
| packed RAW10 at exactly 10 bits/pixel | 2,592,000 | 77.76 MB/s |
| RAW10 unpacked into 16-bit words | 4,147,200 | 124.42 MB/s |
| YUV420 | 3,110,400 | 93.31 MB/s |
| RGB888 | 6,220,800 | 186.62 MB/s |

The driver/DMA stride and packing may add padding, so measured allocation and
traffic override this arithmetic.

## RAW10-derived model investigation

RAW10 is ten-bit Bayer mosaic data: each pixel location measures one colour
component according to the OV5647 colour-filter pattern. It is not luminance,
and the packed bytes cannot be sent directly to an INT8 ESP-DL tensor. A direct
RAW candidate must define:

- negotiated packing, stride, Bayer phase, orientation, black/white levels,
  exposure/gain, defective-pixel policy, and lens shading;
- the cheapest correct unpack and downsample path that preserves Bayer phase;
- the exact ten-bit-to-signed-INT8 mapping or learned first-stage conversion;
- public RGB-to-synthetic-RAW generation for pretraining, followed by real
  OV5647 RAW capture for fitting and validation;
- robustness when sensor settings, illumination, and lens change.

The ISP driver is still required by the MIPI camera-controller path, but its
processing pipeline can be bypassed. The current PPA scale/rotate/mirror colour
mode list contains RGB, YUV, and GRAY8 but no RAW Bayer mode, so packed RAW10
cannot be scaled by PPA directly. A direct RAW candidate therefore needs ISP
conversion or measured custom Bayer-aware unpack/reduction before reaching an
INT8 model. Conversely, Y/luminance discards colour but is
already a semantically useful intensity representation. The benchmark and
controlled training comparison decide among RAW-derived Bayer C1, ISP
luminance C1, and ISP RGB C3.

## Characterization gate

- The exact ESP-IDF/tool setup is reproducible from the terminal and selected
  by VS Code; Doctor, build, flash, monitor, and debug proofs pass.
- Fixed board/camera identification and chip revision are recorded.
- Camera formats, strides, ranges, and representative frames are validated.
- Internal/PSRAM/cache, ISP, PPA, model-only, and combined benchmarks have
  machine-readable results and repeatability bounds.
- Every candidate input shape is exactly realizable by its hardware path.
- The RAW-derived, luminance, and RGB contracts are sufficiently precise to
  reproduce in PyTorch.
- A provisional sustained resource envelope and byte-movement budget are frozen
  for float training; later board results may reopen the named upstream decision
  through the controlling feedback loop.

## Primary setup and measurement sources

- [ESP-IDF v6.0.2 macOS installation](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/get-started/macos-setup.html)
- [EIM command-line installation](https://docs.espressif.com/projects/idf-im-ui/en/latest/cli_installation.html)
- [EIM CLI commands and `eim run`](https://docs.espressif.com/projects/idf-im-ui/en/latest/cli_commands.html)
- [ESP-IDF extension installation](https://docs.espressif.com/projects/vscode-esp-idf-extension/en/latest/installation.html)
- [ESP-IDF extension commands](https://docs.espressif.com/projects/vscode-esp-idf-extension/en/latest/commands.html)
- [ESP32-P4 project build/flash/monitor](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/get-started/start-project.html)
- [ESP32-P4 ISP](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/isp.html)
- [ESP32-P4 PPA](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/ppa.html)
- [ESP32-P4 speed measurement](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/performance/speed.html)
- [ESP32-P4 external RAM behavior](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/external-ram.html)
