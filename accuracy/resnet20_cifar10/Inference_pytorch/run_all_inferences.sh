#!/bin/bash

TARGET_PID=4174324
while ps -p $TARGET_PID > /dev/null; do
    sleep 36000  
done

declare -a ws=(8)
declare -a vs=(0.05 0.25)
#declare -a vs=(0 0.02 0.04 0.06 0.08 0.1)

mkdir -p ./inference_log

for w in "${ws[@]}"; do
  a=8  # a는 항상 8로 고정

  for v in "${vs[@]}"; do
    echo "[$(date)] 실행: w=$w, a=$a, v=$v"

    log_file="./inference_log/ResNet20_w${w},a${a},v${v}_log_$(date +%Y%m%d_%H%M%S).txt"


    nohup python inference.py \
      --dataset cifar10 \
      --model ResNet20_CIFAR10 \
      --mode WAGE \
      --inference 1 \
      --ADCprecision 12 \
      --pretrained ./decay0_W8.pth \
      --act_calib 1 \
      --act_calib_json "w${w},a${a},v${v}_act_alpha_p0.999_b100.json" \
      --act_calib_batch 5 \
      --wl_weight "$w" \
      --wl_activate "$a" \
      --vari "$v" \
      --batch_size 200 \
      >> "$log_file" 2>&1 &
    pid1=$!

    wait $pid1
  done
done

echo "[$(date)] 모든 inference 작업 완료"

# (아래 3)에서 만든 merge 스크립트 실행
python merge_results.py
echo "[$(date)] 결과 CSV 병합 완료"
