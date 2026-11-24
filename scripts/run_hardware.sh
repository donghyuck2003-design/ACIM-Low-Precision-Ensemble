#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?single or ensemble}; tech=${2:?SRAM RRAM FeFET}; w=${3:?weight bits}; shift 3
case "$mode" in single|ensemble);; *) exit 2;; esac
case "$tech" in SRAM|RRAM|FeFET);; *) exit 2;; esac
cd "$root/hardware/$mode/$tech/Inference_pytorch"
[[ -f log/VGG8.pth ]] || { echo "Missing checkpoint: $PWD/log/VGG8.pth" >&2; exit 1; }
mkdir -p summary layer_record
exec python "HW_${mode}_inference.py" --wl_weight "$w" --wl_activate 8 --vari 0 --inference 1 --batch_size 100 --cellBit 1 "$@"
