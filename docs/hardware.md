# Get the board running (Mac)

You bought a **Waveshare ESP32-P4-Module-DEV-KIT** and an **OV5647** camera
(often sold as Raspberry Pi Camera B). This page gets that hardware talking
on a Mac. It is not a lab procedure. After this you should know:

- the chip is an ESP32-P4 with PSRAM
- the USB cable and serial port work
- the camera is on the CSI connector and is producing frames

Keep the serial log. That is the "characterization" for now.

## What you need

- the P4 board
- the OV5647 module and its 15-pin ribbon
- a USB-C cable that carries data, not charge-only
- a Mac with Homebrew and Xcode command-line tools
- ESP-IDF **v5.5.2** (Waveshare's P4 examples want 5.4 or newer; 5.5.2 is the
  current stable that matches Espressif's P4 getting-started guide)

The ESP32-P4 itself has no Wi-Fi. This board talks through a USB-C port and
has an ESP32-C6 for wireless later. You do not need Wi-Fi for the check.

## 1. Install the Mac tools

In Terminal:

```bash
xcode-select --install
brew install cmake ninja dfu-util ccache
```

If `xcode-select` complains that the path is invalid, run it again and finish
the GUI installer.

Apple Silicon: if a later IDF tool install fails with "bad CPU type", install
Rosetta:

```bash
/usr/sbin/softwareupdate --install-rosetta --agree-to-license
```

## 2. Install ESP-IDF

```bash
mkdir -p ~/esp
cd ~/esp
git clone -b v5.5.2 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32p4
```

That downloads the RISC-V compiler and Python tools into `~/.espressif`.
It only has to run once.

Every new terminal that will build firmware needs:

```bash
. $HOME/esp/esp-idf/export.sh
```

The leading `. ` matters. Optional, in `~/.zshrc`:

```bash
alias get_idf='. $HOME/esp/esp-idf/export.sh'
```

Then `get_idf` in a new terminal. Do not put `export.sh` directly in
`.zshrc` — it would activate IDF in every window.

VS Code with the official ESP-IDF extension is optional. The commands below
are enough.

## 3. Plug the hardware in

1. Power the board **off**.
2. Open the **MIPI-CSI** latch (camera, usually labelled CSI). Do **not** use
   the DSI display connector.
3. Seat the OV5647 ribbon. The contacts face the connector pins; the latch
   closes down on the cable. If the first flash says the sensor is missing,
   flip the ribbon and try once.
4. Close the latch.
5. Plug USB-C into the board's UART / flashing Type-C port (the one next to
   BOOT / RESET, not the Type-A OTG port).
6. On the Mac:

```bash
ls /dev/cu.usb*
```

You want a device that appears when the board is plugged in, often
`/dev/cu.usbmodem*` or `/dev/cu.usbserial*`. That path is `PORT` below.

If nothing appears, try another cable. Charge-only cables are the usual
failure.

## 4. Build, flash, save the log

From the repo, with IDF exported:

```bash
cd firmware
idf.py set-target esp32p4
idf.py build
idf.py -p PORT flash monitor | tee ../hardware-check.log
```

Replace `PORT` with your `/dev/cu...` path. If you omit `-p`, `idf.py`
sometimes finds the board by itself.

Leave the monitor running for about ten seconds so the camera capture
finishes. Quit with `Ctrl+]`.

If flash cannot connect:

1. Hold **BOOT**.
2. Tap **RESET**.
3. Release **BOOT**.
4. Run the flash command again.

If the chip revision does not match the build (`requires chip revision in
range [v3.1 ...] (this chip is revision v1.x)`), do **not** pass `--force`.
Tell me the printed revision. Waveshare boards exist as both engineering
samples and production silicon, and the build defaults have to match.

## 5. How to read the log

`firmware/main/check.c` prints three blocks.

**BOARD** should look like:

```text
target:            esp32p4
CPU cores:         2
silicon revision:  v3.1          # yours may be v0.x / v1.x / v3.0
flash:             16 MB
PSRAM:             32 MB, initialized
```

If PSRAM is missing, stop. Later models will not fit.

**CAMERA** should mention the OV5647 in the IDF log (`Detected Camera sensor
PID=0x5647`) and then:

```text
driver:            MIPI-CSI
frame size:        800 x 800     # default check mode, not the model size
captured:          90 frames in 3 s
FPS:               30            # anything clearly above 0 is a pass
```

**SUMMARY**

```text
BOARD:   OK
CAMERA:  OK
```

That is the pass. The 800×800 capture is only "the sensor streams". The
neural net will use a much smaller luminance crop later.

## 6. What to keep

`hardware-check.log` in the repo root is gitignored. Keep it. It is the
record of:

- silicon revision
- flash and PSRAM sizes
- whether OV5647 was detected
- the format, frame size, bytes per frame, and FPS
- free internal RAM and PSRAM after capture

If something fails, that file is what to send. Re-run the same command after
any cable or IDF change so the log stays current.

## 7. If CAMERA is FAIL

| What you see | Try |
|---|---|
| no `/dev/cu.usb*` | another USB-C cable; the UART Type-C port |
| flash timeout | BOOT+RESET sequence above |
| `failed to initialize video` / no PID 0x5647 | ribbon in **CSI** not DSI; reseat or flip the cable; power-cycle |
| driver opens, 0 frames | leave the monitor up longer; power-cycle with the cable already seated |
| wrong chip revision | do not force-flash; report the printed `vX.Y` |

A later pass can time copies, PSRAM bandwidth, and a resized luminance
frame. That only matters after BOARD and CAMERA are OK.

The firmware you will want to hand-tune lives in this same folder. Start
from `check.c`. Do not add a second project tree for experiments.
