declare -a ws=(3 4 5 6 7 8)
v=0.0

for i in "${!ws[@]}"; do
  w=${ws[$i]}
  #a=$w
  a=8
  echo "[단일 모델 HW 성능 테스트 - W=${w}, a=${a}]"


  nohup python HW_single_inference.py --wl_weight $w --wl_activate $a  --vari $v --inference 1 --batch_size 100 --cellBit 1 > "FeFET_w1.45v_singleHW_w${w},a${a},v${v}_log_$(date +%Y%m%d_%H%M%S).txt" 2>&1 &
  
  wait $pid1
  echo "[$(date)] w=$w 작업 완료"
  
done
