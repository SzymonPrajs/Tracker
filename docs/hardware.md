# Hardware

The target is the ESP32-P4-Module-DEV-KIT and OV5647 MIPI-CSI camera.

The hardware phase will contain terminal-first instructions for installing
ESP-IDF and its compilers, creating the ESP-IDF app under `firmware/`, building,
flashing, monitoring, and using the installed VS Code ESP-IDF extension. It
will also add a very small benchmark app that measures:

- camera formats, including RAW10, ISP luminance, and RGB;
- resize/crop throughput;
- internal SRAM versus PSRAM bandwidth;
- each unavoidable buffer copy;
- model input conversion;
- layer/activation memory and end-to-end inference time.

These measurements decide the input representation and drive the train-fit-
measure loop. RAW10 is a packed sensor representation, not automatically a
ready one-channel neural-network input; it will be tested against ISP luminance
and RGB rather than assumed best.
