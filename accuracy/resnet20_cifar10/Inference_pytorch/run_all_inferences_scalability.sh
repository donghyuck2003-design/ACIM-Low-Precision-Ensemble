#!/bin/bash
set -euo pipefail

#declare -a ws=(3 4 5 6 7 8)
declare -a ws=(5)
declare -a vs=(0.15 0.20 0.25 0.05)
#declare -a vs=(0.25 0.10 0.15 0.20)

mkdir -p ./inference_log
mkdir -p ./Result

for w in "${ws[@]}"; do
  a=8  # a는 항상 8로 고정

  for v in "${vs[@]}"; do
    ts="$(date +%Y%m%d_%H%M%S)"
    echo "[$(date)] 실행: w=$w, a=$a, v=$v"

    log_file="./inference_log/w${w}_a${a}_v${v}_log_${ts}.txt"

    # 순차 실행 (문제 확인 목적: 실패하면 즉시 중단)
    python inference.py \
      --wl_weight "$w" \
      --wl_activate "$a" \
      --vari "$v" \
      --inference 1 \
      --batch_size 200 \
      1>>"$log_file" 2>&1


    echo "[$(date)] 완료: w=$w, a=$a, v=$v"
  done
done

echo "[$(date)] 모든 inference 작업 완료"

python merge_results.py
echo "[$(date)] 결과 CSV 병합 완료"
