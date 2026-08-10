#!/usr/bin/env bash
set -eu

export IDF_TOOLS_PATH="$PWD/.tools/espressif"
. .tools/esp-idf/export.sh
idf.py -C firmware -B "$PWD/.build/firmware" -D IDF_TARGET=esp32p4 build
