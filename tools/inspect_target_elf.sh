#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly SILICON="${1:-rev1}"
readonly ELF_PATH="${2:-${REPO_ROOT}/.build/firmware-${SILICON}/tracker_p4.elf}"

case "${SILICON}" in
    rev1) expected_xespv="xespv2p1" ;;
    rev3) expected_xespv="xespv2p2" ;;
    *) echo "usage: $0 [rev1|rev3] [ELF]" >&2; exit 2 ;;
esac

if [[ ! -f "${ELF_PATH}" ]]; then
    echo "ELF not found: ${ELF_PATH}" >&2
    exit 1
fi

export IDF_TOOLS_PATH="${REPO_ROOT}/.tools/espressif"
if ! command -v riscv32-esp-elf-readelf >/dev/null; then
    source "${REPO_ROOT}/.tools/esp-idf/export.sh"
fi

readonly INSPECTION_DIR="$(dirname "${ELF_PATH}")/inspection"
readonly DISASSEMBLY="${INSPECTION_DIR}/tracker_p4.disasm"
mkdir -p "${INSPECTION_DIR}"

riscv32-esp-elf-readelf -h -A "${ELF_PATH}" | tee "${INSPECTION_DIR}/elf.txt"
riscv32-esp-elf-readelf -Ws "${ELF_PATH}" | tee "${INSPECTION_DIR}/symbols.txt" >/dev/null
riscv32-esp-elf-objdump -drwC -S "${ELF_PATH}" > "${DISASSEMBLY}"

grep -q 'Machine:.*RISC-V' "${INSPECTION_DIR}/elf.txt"
grep -q 'single-float ABI' "${INSPECTION_DIR}/elf.txt"
grep -q 'Tag_RISCV_stack_align: 16-bytes' "${INSPECTION_DIR}/elf.txt"
grep -q 'Tag_RISCV_arch: "rv32i' "${INSPECTION_DIR}/elf.txt"
grep -q "${expected_xespv}" "${INSPECTION_DIR}/elf.txt"

for symbol in \
    tracker_perf_cycles \
    tracker_perf_instructions \
    tracker_argmax_hwc16_s8 \
    tracker_alpha_beta_update \
    tracker_decode_centroid_hwc16 \
    tracker_mailbox_take_latest \
    tracker_rgb888_to_centered_s8 \
    tracker_tensor_view_is_valid
do
    grep -Eq \
        "[[:space:]]FUNC[[:space:]]+GLOBAL[[:space:]]+DEFAULT[[:space:]]+[0-9]+[[:space:]]+${symbol}$" \
        "${INSPECTION_DIR}/symbols.txt"
done

grep -q 'mcycleh' "${DISASSEMBLY}"
grep -q 'mcycle' "${DISASSEMBLY}"
grep -q 'minstreth' "${DISASSEMBLY}"
grep -q 'minstret' "${DISASSEMBLY}"

echo "ELF inspection passed for ${SILICON}: ${ELF_PATH}"
echo "Disassembly: ${DISASSEMBLY}"
