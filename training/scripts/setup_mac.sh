#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TRAINING_ROOT}/.." && pwd)"
TRAIN_VENV="${TRACKER_TRAIN_VENV:-${REPO_ROOT}/.tools/tracker-train}"
QUANT_VENV="${TRACKER_QUANT_VENV:-${REPO_ROOT}/.tools/tracker-quant}"
PYTHON_BIN="${TRACKER_PYTHON:-python3}"
INSTALL_TRAIN=1
INSTALL_QUANT=1

usage() {
    echo "usage: $0 [--training-only|--quantize-only] [--train-venv PATH] [--quantize-venv PATH]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --training-only)
            INSTALL_QUANT=0
            shift
            ;;
        --quantize-only)
            INSTALL_TRAIN=0
            shift
            ;;
        --train-venv)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            TRAIN_VENV="$2"
            shift 2
            ;;
        --quantize-venv)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            QUANT_VENV="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${INSTALL_TRAIN}" -eq 0 && "${INSTALL_QUANT}" -eq 0 ]]; then
    echo "error: --training-only and --quantize-only are mutually exclusive" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: this setup is pinned and verified for Apple-silicon macOS" >&2
    exit 1
fi

"${PYTHON_BIN}" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "use Python 3.11 or 3.12"'

install_environment() {
    local venv_dir="$1"
    local requirements="$2"
    if [[ ! -x "${venv_dir}/bin/python" ]]; then
        mkdir -p "$(dirname "${venv_dir}")"
        "${PYTHON_BIN}" -m venv "${venv_dir}"
    fi
    "${venv_dir}/bin/python" -m pip install --upgrade pip
    "${venv_dir}/bin/python" -m pip install --requirement "${requirements}"
    "${venv_dir}/bin/python" -c 'import platform, torch; print(f"ready: Python {platform.python_version()}, PyTorch {torch.__version__}")'
}

if [[ "${INSTALL_TRAIN}" -eq 1 ]]; then
    install_environment "${TRAIN_VENV}" "${TRAINING_ROOT}/requirements-mac.txt"
    echo "training: source ${TRAIN_VENV}/bin/activate"
fi

if [[ "${INSTALL_QUANT}" -eq 1 ]]; then
    install_environment "${QUANT_VENV}" "${TRAINING_ROOT}/requirements-quantize.txt"
    echo "quantization: source ${QUANT_VENV}/bin/activate"
fi
