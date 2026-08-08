#!/usr/bin/env bash
set -euo pipefail

readonly IDF_TAG="v6.0.2"
readonly IDF_COMMIT="7101770dc6db2667b3c477cc31365dd1acd6db4e"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly TOOLS_ROOT="${REPO_ROOT}/.tools"
readonly IDF_PATH_LOCAL="${TOOLS_ROOT}/esp-idf"

export IDF_TOOLS_PATH="${TOOLS_ROOT}/espressif"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "Python 3.10 or newer is required" >&2; exit 1; }

python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'
mkdir -p "${TOOLS_ROOT}"

if [[ ! -e "${IDF_PATH_LOCAL}" ]]; then
    git clone --filter=blob:none --shallow-submodules --recursive \
        --branch "${IDF_TAG}" \
        https://github.com/espressif/esp-idf.git \
        "${IDF_PATH_LOCAL}"
elif [[ ! -d "${IDF_PATH_LOCAL}/.git" ]]; then
    echo "${IDF_PATH_LOCAL} exists but is not an ESP-IDF Git checkout" >&2
    exit 1
fi

actual_commit="$(git -C "${IDF_PATH_LOCAL}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${IDF_COMMIT}" ]]; then
    echo "ESP-IDF commit mismatch: expected ${IDF_COMMIT}, found ${actual_commit}" >&2
    exit 1
fi

git -C "${IDF_PATH_LOCAL}" submodule update --init --recursive --depth 1
"${IDF_PATH_LOCAL}/install.sh" esp32p4

echo "ESP-IDF ${IDF_TAG} is ready."
echo "Build with: ${REPO_ROOT}/tools/build_firmware.sh rev1"
