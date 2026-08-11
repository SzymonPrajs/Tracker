# Firmware

This folder will contain the ESP-IDF application for the ESP32-P4-Module-DEV-KIT
and OV5647 camera.

```text
firmware/
  main/          C/C++ and any measured inline or .S assembly
  components/    reusable ESP-IDF components, only when needed
  build/         generated binaries and object files (ignored by Git)
```

Build, flash, monitor, and benchmark commands will live in this README beside
the firmware they operate on. Assembly belongs beside the C code that calls it
and is added only for a measured hot path. Debug and release behavior will use
ESP-IDF build configuration so release code does not carry debug work.

The firmware itself is not implemented during the data-download phase.
