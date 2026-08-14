# Firmware

This folder is the ESP-IDF app for the Waveshare ESP32-P4-Module-DEV-KIT and
the OV5647 MIPI-CSI camera.

Right now it only checks that the board and camera work. The tracker itself is
not here yet. When it is, keep optimized C or `.S` next to a readable C
reference of the same operation.

```text
firmware/
  main/check.c           board + camera check
  sdkconfig.defaults     this board, OV5647, PSRAM
  build/                 generated, not committed
```

Full Mac install, cable, flash, and how to read the log:

[docs/hardware.md](../docs/hardware.md)
