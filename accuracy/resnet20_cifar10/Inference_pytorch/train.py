import argparse
import os
import time
from utee import misc
import torch
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from models import dataset
from utee import wage_util
from datetime import datetime
from utee import wage_quantizer
import torch.nn as nn

# [변경됨] 데이터셋 분리를 위해 라이브러리 추가
from torch.utils.data import DataLoader, random_split
from models.ResNet import calib_enable, calib_disable, calib_finalize, calib_save_json, calib_load_json
from modules import quantization_cpu_np_infer as q  # qmod (전역 _ACT_ALPHA에 주입)

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='imagenet', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='ResNet18',
                    help='VGG8|DenseNet40|ResNet18|ResNet10t|ResNet20_CIFAR10')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='60,120,160', help='decreasing strategy')
parser.add_argument('--wl_weight', type = int, default=8)
parser.add_argument('--wl_grad', type = int, default=8)
parser.add_argument('--wl_activate', type = int, default=8)
parser.add_argument('--wl_error', type = int, default=8)
# Hardware Properties
parser.add_argument('--inference', default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--ADCprecision', default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', default=4, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
parser.add_argument('--vari', default=0, help='conductance variation')
parser.add_argument('--t', default=0, help='retention time')
parser.add_argument('--v', default=0, help='drift coefficient')
parser.add_argument('--detect', default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', default=0, help='drift target for fixed-direction drift')
parser.add_argument('--parallelRead', type=int, default=128)
parser.add_argument('--pretrained', default=None, help='path to pretrained checkpoint (e.g., CIFAR10 VGG8)')
parser.add_argument('--wage_start_epoch', type=int, default=20)
# Activation calibration (per-layer alpha, P99.9)
parser.add_argument('--act_calib', type=int, default=0,
                    help='1: run activation calibration to collect per-layer alpha')
parser.add_argument('--act_calib_batches', type=int, default=200,
                    help='number of batches for calibration')
parser.add_argument('--act_calib_p', type=float, default=0.999,
                    help='percentile for alpha (e.g., 0.999 = P99.9)')
parser.add_argument('--act_calib_stride', type=int, default=16,
                    help='downsample stride when collecting activation samples')
parser.add_argument('--act_calib_max_samples', type=int, default=200000,
                    help='max samples per layer tag')
parser.add_argument('--act_calib_json', type=str, default='',#'training_act_alpha_p0.999_b200.json',
                    help='path to save/load alpha json (default auto under logdir)')


current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs'])
misc.logger.init(args.logdir, 'train_log_' +current_time)
logger = misc.logger.info

import os

# (A) alpha json path 결정 (inference_test.py와 동일)
if args.act_calib_json.strip() == "":
    alpha_json_path = os.path.join(args.logdir, f"act_alpha_p{args.act_calib_p}_b{args.act_calib_batches}.json")
else:
    alpha_json_path = args.act_calib_json



# (C) 2) alpha json 로드해서 qmod에 주입 (학습/평가에서 사용)
if os.path.exists(alpha_json_path):
    alpha_dict = calib_load_json(alpha_json_path)
    q.calib_set_alpha_dict(alpha_dict)
    logger(f"[ACT] loaded alpha json: {alpha_json_path}")
else:
    logger(f"[ACT] alpha json not found: {alpha_json_path} (act_q may fallback / bypass)")


# logger
misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# -------- Data Loader --------
assert args.dataset in ['cifar10', 'cifar100', 'imagenet'], args.dataset

if args.dataset == 'cifar10':
    train_loader, val_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=4)
    test_loader = val_loader  # cifar10은 기존 코드 호환용 (원하면 별도 test로 분리 가능)

elif args.dataset == 'cifar100':
    train_loader, val_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=4)
    test_loader = val_loader  # cifar100도 동일

elif args.dataset == 'imagenet':
    train_loader, val_loader, test_loader = dataset.get_imagenet(
    batch_size=args.batch_size,
    data_root='/ssdl/neurosim_data/ImageNet/extracted_data/ILSVRC',
    loc_root='/ssdl/neurosim_data/ImageNet/extracted_data',
    val_ratio=0.1,
    num_workers=4,
    pin_memory=args.cuda
)     
else:
    raise ValueError("Unknown dataset type")



    
assert args.model in ['VGG8','DenseNet40','ResNet18','ResNet10t','ResNet20_CIFAR10'], args.model
if args.model == 'VGG8':
    from models import VGG
    model = VGG.vgg8(args=args, logger=logger, pretrained=args.pretrained)
    criterion = torch.nn.CrossEntropyLoss()
elif args.model == 'DenseNet40':
    from models import DenseNet
    model = DenseNet.densenet40(args = args, logger=logger)
    criterion = wage_util.SSE()
elif args.model == 'ResNet18':
    from models import ResNet
    model = ResNet.resnet18(pretrained=True, args = args, logger=logger)
    criterion = torch.nn.CrossEntropyLoss()
elif args.model == 'ResNet20_CIFAR10':
    from models import ResNet
    model = ResNet.cifar_resnet20(args=args, logger=logger, num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()

    if args.pretrained is not None:
        ckpt = torch.load(args.pretrained, map_location='cpu')

        # ✅ ckpt가 state_dict 자체인 경우
        if isinstance(ckpt, dict) and all(k.startswith(("conv","layer","bn","fc","module.")) for k in ckpt.keys()):
            state_dict = ckpt
        # ✅ ckpt가 {"state_dict": ...} 형태인 경우
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # ✅ misc.model_save가 다른 키로 저장했을 수도 있어서 방어
            state_dict = ckpt

        # DataParallel 저장이면 module. 제거
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        ret = model.load_state_dict(state_dict, strict=False)
        logger(f"[LOAD] {args.pretrained}")
        logger(f"[LOAD] missing_keys={len(ret.missing_keys)} unexpected_keys={len(ret.unexpected_keys)}")
    else:
        # 없으면 chenyaofo pretrained로 시작(옵션)
        hub_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        ret = model.load_state_dict(hub_model.state_dict(), strict=False)
        logger(f"[LOAD] missing_keys={len(ret.missing_keys)} unexpected_keys={len(ret.unexpected_keys)}")

else:
    raise ValueError("Unknown model type")


import math, numpy as np
from modules.quantization_cpu_np_infer import QConv2d, QLinear

def reset_wage_scale_after_loading_pretrained(model: nn.Module, bits_W: int, verbose: bool = True):
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, (QConv2d, QLinear)):
                new_scale = wage_scale_from_weight_std(m.weight, bits_W)
                old_scale = float(getattr(m, "scale", 1.0))
                m.scale = new_scale
                if verbose:
                    w = m.weight.detach()
                    # print(f"[scale reset] {name:35s} old={old_scale:.3e} new={new_scale:.3e} "
                    #       f"w_std={w.std().item():.3e} w_max={w.abs().max().item():.3e}")

def wage_scale_from_weight_std(weight: torch.Tensor, bits_W: int, eps: float = 1e-12) -> float:
    # quant limit 정의는 기존과 동일하게
    delta = 1.0 / (2 ** (bits_W - 1))
    limit = 1.0 - delta
    if bits_W > 2:
        limit_std = limit / math.sqrt(3)
    else:
        limit_std = 0.75 / math.sqrt(3)

    w_std = float(weight.detach().std().clamp_min(eps).cpu().item())
    # power-of-two scale 유지
    scale = 2.0 ** float(np.ceil(np.log2(limit_std / w_std)))
    return scale

if args.cuda:
    model.cuda()

reset_wage_scale_after_loading_pretrained(model, args.wl_weight, verbose=True)


# (B) 1) calibration 실행해서 저장
if args.act_calib == 1:
    logger(f"[ACT-CALIB] enabled: p={args.act_calib_p} batches={args.act_calib_batches} "
           f"stride={args.act_calib_stride} max_samples={args.act_calib_max_samples}")

    # 중요: act_q가 이 플래그를 보고 calib_update만 수행
    args.act_calib = True

    model.eval()
    calib_enable(p=args.act_calib_p,
                 max_samples=args.act_calib_max_samples,
                 stride=args.act_calib_stride)

    calib_loader = train_loader  # 보통 train 분포가 더 다양해서 안정적 (inference_test도 train_loader 사용) :contentReference[oaicite:6]{index=6}
    with torch.no_grad():
        for b, (data, target) in enumerate(calib_loader):
            if b >= args.act_calib_batches:
                break
            if args.cuda:
                data = data.cuda(non_blocking=True)
            _ = model(data)

            if (b + 1) % 20 == 0:
                logger(f"[ACT-CALIB] collected batches: {b+1}/{args.act_calib_batches}")

    alphas = calib_finalize()
    calib_save_json(alpha_json_path)
    logger(f"[ACT-CALIB] saved {len(alphas)} layer alphas to: {alpha_json_path}")

    # 종료
    args.act_calib = False
    calib_disable()
    
import json, os
from modules import quantization_cpu_np_infer as q  # qmod

def load_alpha_json_to_qmod(path):
    with open(path, "r") as f:
        d = json.load(f)
    d = {str(k): float(v) for k, v in d.items()}
    q.calib_set_alpha_dict(d)   # <- qmod 전역 _ACT_ALPHA 세팅
    return d

# model 만든 다음, training 시작 전에:
if args.act_calib_json and os.path.exists(args.act_calib_json):
    alpha_dict = load_alpha_json_to_qmod(args.act_calib_json)
    logger(f"[ACT] loaded alpha json for training: {args.act_calib_json} (#keys={len(alpha_dict)})")
else:
    logger(f"[ACT] alpha json not loaded. path={args.act_calib_json}")




optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
logger('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()
grad_scale = args.grad_scale

# def print_clip_stats(model, bits):
#     # C()와 동일한 방식으로 lower/upper 계산
#     from utee.wage_quantizer import S  # 혹은 wage_quantizer.S 위치에 맞게 수정
#     if bits > 15 or bits == 1:
#         delta = 0.0
#     else:
#         delta = 1.0 / S(bits)
#     upper = 1.0 - delta
#     lower = -1.0 + delta

#     print(f"[CLIP STAT] bits={bits}, lower={lower:.6f}, upper={upper:.6f}")
#     total_elems = 0
#     total_clipped = 0


#     with torch.no_grad():
#         for name, p in model.named_parameters():
#             if ("bn" in name.lower()) or ("running_" in name.lower()):
#                 continue
#             if "weight" not in name:
#                 continue

#             if p is None or p.data is None:
#                 continue
#             w = p.data
#             # 가중치(Conv/FC) 중심으로 보고 싶으면 bias/BN 제외할 수도 있음
#             # if ("weight" not in name) or ("bn" in name.lower()): continue

#             clipped = ((w < lower) | (w > upper)).sum().item()
#             n = w.numel()
#             ratio = clipped / n
#             print(f"{name:40s}  clipped={clipped:8d}/{n:10d}  ({ratio*100:6.3f}%)  "
#                   f"min={w.min().item():+.4f} max={w.max().item():+.4f}")

#             total_elems += n
#             total_clipped += clipped

#     print(f"[TOTAL] clipped={total_clipped}/{total_elems} ({100*total_clipped/total_elems:.4f}%)")
#     total_clipped = 0

# print("== Before C ==")
# print_clip_stats(model, args.wl_weight)

# print("== After C ==")
# print_clip_stats(model, args.wl_weight)

for name, p in model.named_parameters():
    if "bn" in name.lower():
        continue
    p.data = wage_quantizer.C(p.data, args.wl_weight)

if args.dataset == 'imagenet':
    x, y = next(iter(train_loader))
    print("Sample targets min/max:", int(y.min()), int(y.max()))

#print("Model last layer out_features:", model.fc.out_features)


try:
    # ready to go
    for epoch in range(args.epochs):
        model.train()

        if epoch in decreasing_lr:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1

            if args.mode == 'WAGE':
                grad_scale = grad_scale / 8.0
             

        logger("training phase")
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)

            loss.backward()

            if args.mode == 'WAGE':
                for name, param in list(model.named_parameters())[::-1]:
                    if param.grad is None:
                        continue

                    g = param.grad.data
                    if g is None:
                        continue

                    # grad가 전부 0이면 QG 스킵 (QG blow 방지)
                    if g.abs().max().item() == 0:
                        continue

                    # (권장) BN은 스킵: ResNet에서 BN까지 QG/C 적용하면 잘 깨짐
                    if "bn" in name.lower():
                        continue

                    param.grad.data = wage_quantizer.QG(g, args.wl_grad, grad_scale)


            optimizer.step()
            
            if args.mode == 'WAGE':
                for name, param in list(model.named_parameters())[::-1]:
                    if "bn" in name.lower():
                        continue
                    param.data = wage_quantizer.C(param.data, args.wl_weight)

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) * 1.0 / len(data)
                logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.data, acc, optimizer.param_groups[0]['lr']))
                

            if not torch.isfinite(output).all():
                print("[NAN OUTPUT]", batch_idx, "min/max", float(output.min()), float(output.max()))
                break

            for n,p in model.named_parameters():
                if "weight" in n and not torch.isfinite(p.data).all():
                    print("[NAN WEIGHT]", n, "min/max", float(p.data.min()), float(p.data.max()))
                    break

            for n,p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("[NAN GRAD]", n, "max", float(p.grad.abs().max()))
                    break
                

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        logger("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))

        misc.model_save(model, os.path.join(args.logdir, 'latest.pth'))

        # 여기부터는 test_loader 변수를 사용하지만, 
        # 위에서 Validation Set으로 교체했으므로 실제로는 Validation을 수행합니다.
        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            logger("validation phase") # 로그 메시지를 testing -> validation으로 살짝 변경
            for i, (data, target) in enumerate(val_loader):
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                    output = model(data)
                    test_loss_i = criterion(output, target)
                    test_loss += test_loss_i.data
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.cpu().eq(indx_target).sum()
 

            test_loss = test_loss / len(val_loader)
            acc = 100. * correct / len(val_loader.dataset)
            logger('\tEpoch {} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, test_loss, correct, len(val_loader.dataset), acc))

            if acc > best_acc:
                new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                misc.model_save(model, new_file, old_file=old_file, verbose=True)
                best_acc = acc
                old_file = new_file

except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    logger("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))
    