#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
model=${1:?benchmark}; w=${2:?weight bits}; sigma=${3:?variation}; shift 3
case "$model" in
 vgg8_cifar10) ck=log/VGG8.pth; args=(--dataset cifar10 --ADCprecision 8 --num_models 300 --num_ensembles 6 --x_min 2 --x_max 2);;
 vgg8_cifar100) ck=log/VGG_CIFAR100.pth; args=(--dataset cifar100 --ADCprecision 5 --num_models 300 --num_ensembles 6 --x_min 2 --x_max 2);;
 ensemble_size_scaling) ck=log/VGG8.pth; args=(--dataset cifar10 --ADCprecision 5 --num_models 300 --num_ensembles 3 --x_min 3 --x_max 5);;
 resnet20_cifar10) ck=decay0_W8.pth; args=(--dataset cifar10 --model ResNet20_CIFAR10 --mode WAGE --ADCprecision 12 --pretrained "$ck" --act_calib 1 --act_calib_batches 5 --act_calib_json "w${w},a8,v${sigma}_act_alpha_p0.999_b100.json" --num_models 150 --num_ensembles 3);;
 *) echo 'Unknown benchmark' >&2; exit 2;;
esac
cd "$root/accuracy/$model/Inference_pytorch"
[[ -f "$ck" ]] || { echo "Missing checkpoint: $PWD/$ck" >&2; exit 1; }
mkdir -p Result inference_log
exec python inference.py "${args[@]}" --wl_weight "$w" --wl_activate 8 --vari "$sigma" --inference 1 --batch_size 200 "$@"
