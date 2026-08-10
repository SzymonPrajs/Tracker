#!/usr/bin/env bash
set -eu

mkdir -p .tools
if [ ! -d .tools/esp-idf ]; then
    git clone --depth 1 --recursive --branch v6.0.2 \
        https://github.com/espressif/esp-idf.git .tools/esp-idf
fi
IDF_TOOLS_PATH="$PWD/.tools/espressif" .tools/esp-idf/install.sh esp32p4
