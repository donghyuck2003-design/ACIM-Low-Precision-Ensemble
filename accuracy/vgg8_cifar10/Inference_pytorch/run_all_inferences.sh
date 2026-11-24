#!/bin/bash
set -euo pipefail

TARGET_PID=79440
echo "Target PID: $TARGET_PID 프로세스 종료를 대기합니다."
while ps -p $TARGET_PID > /dev/null; do
    # 현재 시각과 함께 대기 문구 출력
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $TARGET_PID 종료 대기 중..."
    # 36시간 대기 (필요에 따라 조절하세요)
    sleep 129600
done
echo "프로세스가 종료되었습니다. 다음 단계를 진행합니다."

declare -a ws=(3 4 5 6 7 8)
#declare -a ws=(4 3)
declare -a vs=(0 0.05 0.1 0.15 0.2 0.25)
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
      --ADCprecision 8 \
      --inference 1 \
      --batch_size 200 \
      1>>"$log_file" 2>&1


    echo "[$(date)] 완료: w=$w, a=$a, v=$v"
  done
done

echo "[$(date)] 모든 inference 작업 완료"

python merge_results.py
echo "[$(date)] 결과 CSV 병합 완료"
