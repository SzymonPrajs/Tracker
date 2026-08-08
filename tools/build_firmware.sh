#!/usr/bin/env bash
set -euo pipefail

readonly IDF_COMMIT="7101770dc6db2667b3c477cc31365dd1acd6db4e"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly IDF_PATH_LOCAL="${REPO_ROOT}/.tools/esp-idf"
readonly FIRMWARE_DIR="${REPO_ROOT}/firmware"
readonly SILICON="${1:-rev1}"

case "${SILICON}" in
    rev1|rev3) ;;
    *) echo "usage: $0 [rev1|rev3]" >&2; exit 2 ;;
esac

if [[ ! -f "${IDF_PATH_LOCAL}/export.sh" ]]; then
    echo "ESP-IDF is not installed; run ${REPO_ROOT}/tools/bootstrap_esp_idf.sh" >&2
    exit 1
fi

actual_commit="$(git -C "${IDF_PATH_LOCAL}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${IDF_COMMIT}" ]]; then
    echo "ESP-IDF commit mismatch: expected ${IDF_COMMIT}, found ${actual_commit}" >&2
    exit 1
fi

export IDF_TOOLS_PATH="${REPO_ROOT}/.tools/espressif"
source "${IDF_PATH_LOCAL}/export.sh"

readonly BUILD_DIR="${REPO_ROOT}/.build/firmware-${SILICON}"
readonly SDKCONFIG_PATH="${BUILD_DIR}/sdkconfig"
readonly DEFAULTS="${FIRMWARE_DIR}/sdkconfig.defaults;${FIRMWARE_DIR}/sdkconfig.${SILICON}.defaults"

idf.py \
    -C "${FIRMWARE_DIR}" \
    -B "${BUILD_DIR}" \
    -D "IDF_TARGET=esp32p4" \
    -D "SDKCONFIG=${SDKCONFIG_PATH}" \
    -D "SDKCONFIG_DEFAULTS=${DEFAULTS}" \
    build

"${SCRIPT_DIR}/inspect_target_elf.sh" "${SILICON}" "${BUILD_DIR}/tracker_p4.elf"
