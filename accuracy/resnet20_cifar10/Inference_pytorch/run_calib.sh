#!/bin/bash

(
  # 첫 번째 실행 (w8)
  nohup python inference_test.py \
    --dataset cifar10 --model ResNet20_CIFAR10 --mode WAGE --inference 1 \
    --wl_weight 8 --wl_activate 8 --ADCprecision 16 --batch_size 100 \
    --act_calib 1 --act_calib_batches 50 \
    --act_calib_json w8_act_alpha_p0.999_b100.json \
    --pretrained ./decay0_W8.pth > "w8_calib_$(date +%Y%m%d_%H%M%S).txt" 2>&1 && \
  
  # 두 번째 실행 (w5)
  nohup python inference_test.py \
    --dataset cifar10 --model ResNet20_CIFAR10 --mo/home/ssdl_neurosim1/All_Neurosim_Project/ResNet20t_CIFAR10_ver2/Inference_pytorchde WAGE --inference 1 \
    --wl_weight 5 --wl_activate 8 --ADCprecision 16 --batch_size 100 \
    --act_calib 1 --act_calib_batches 50 \
    --act_calib_json w5_act_alpha_p0.999_b100.json \
    --pretrained ./decay0_W8.pth > "w5_calib_$(date +%Y%m%d_%H%M%S).txt" 2>&1
) &