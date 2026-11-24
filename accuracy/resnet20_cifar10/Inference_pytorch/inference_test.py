import math, numpy as np
import torch
import torch.nn as nn
import timm
import argparse
import os
import time
from utee import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from models import dataset
import torchvision.models as models
from utee import hook
#from IPython import embed
from datetime import datetime
from subprocess import call
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='ResNet20_CIFAR10', help='VGG8|DenseNet40|ResNet18|ResNet50|ResNet10|ResNet20_CIFAR10')
parser.add_argument('--mode', default='FP', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='140,180', help='decreasing strategy')
parser.add_argument('--wl_weight', type=int, default=8)
parser.add_argument('--wl_grad', type=int, default=8)
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)
# Hardware Properties
# if do not consider hardware effects, set inference=0
parser.add_argument('--inference', type=int, default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', type=int, default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--parallelRead', type=int, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
parser.add_argument('--ADCprecision', type=int, default=9, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', type=int, default=1, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', type=float, default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
# if do not run the device retention / conductance variation effects, set vari=0, v=0
parser.add_argument('--vari', type=float, default=0., help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', type=float, default=0, help='retention time')
parser.add_argument('--v', type=float, default=0, help='drift coefficient')
parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', type=float, default=0, help='drift target for fixed-direction drift, range 0-1')

# Activation calibration (per-layer alpha, P99.9)
parser.add_argument('--act_calib', type=int, default=0, help='1: run activation calibration to collect per-layer alpha')
parser.add_argument('--act_calib_batches', type=int, default=200, help='number of batches for calibration')
parser.add_argument('--act_calib_p', type=float, default=0.999, help='percentile for alpha (e.g., 0.999 = P99.9)')
parser.add_argument('--act_calib_stride', type=int, default=16, help='downsample stride when collecting activation samples')
parser.add_argument('--act_calib_max_samples', type=int, default=200000, help='max samples per layer tag')
parser.add_argument('--act_calib_json', type=str, default='act_alpha_p0.999_b200.json', help='path to save/load alpha json (default auto under logdir)')
parser.add_argument('--pretrained', default=None, help='path to pretrained checkpoint (e.g., CIFAR10 VGG8)')

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

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

# data loader and model
assert args.dataset in ['cifar10', 'cifar100', 'imagenet'], args.dataset
if args.dataset == 'cifar10':
    train_loader, test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'cifar100':
    train_loader, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'imagenet':
    train_loader, val_loader, test_loader = dataset.get_imagenet(
    batch_size=args.batch_size,
    val_ratio=0.1,
    num_workers=4,
    pin_memory=args.cuda
)    
else:
    raise ValueError("Unknown dataset type")

#assert args.model in ['VGG8', 'DenseNet40', 'ResNet18'], args.model
if args.model == 'VGG8':
    from models import VGG
    model_path = './log/VGG8.pth'   # WAGE mode pretrained model
    modelCF = VGG.vgg8(args = args, logger=logger, pretrained = model_path)
elif args.model == 'DenseNet40':
    from models import DenseNet
    model_path = './log/DenseNet40.pth'     # WAGE mode pretrained model
    modelCF = DenseNet.densenet40(args = args, logger=logger, pretrained = model_path)
elif args.model == 'ResNet18':
    from models import ResNet
    modelCF = ResNet.resnet18(args=args, logger=logger, pretrained=True)
elif args.model == 'ResNet50':
    from models import ResNet
    modelCF = ResNet.resnet50(args=args, logger=logger, pretrained=True)

elif args.model == 'ResNet10':
    from models import ResNet
    import timm
    
    # 1. 수정된 deep_stem 구조의 모델 생성
    modelCF = ResNet.resnet10t(args=args, logger=logger, pretrained=None)
    
    # 2. timm 가중치 로드
    print("=> Loading 'resnet10t' from timm...")
    timm_model = timm.create_model('resnet10t', pretrained=True)
    t_state = timm_model.state_dict()
    c_state = modelCF.state_dict()
    
    new_state = {}
    
    # --- 핵심 매핑 로직 ---
 
    # 1) Stem 매핑: conv1.{i} -> stem.{i}
    stem_idx = [0, 1, 3, 4, 6, 7]

    for i in stem_idx:
        # 기본 후보 (너가 쓰던 형태)
        cand_t_prefix = [f"conv1.{i}"]

        # ✅ stem.7(BN64)만 이름이 다른 timm 변형을 흡수
        if i == 7:
            cand_t_prefix += ["bn1", "stem.7", "conv1.6.bn", "conv1.7.bn"]

        for t_prefix in cand_t_prefix:
            # weight/bias/running_* 모두 시도
            for suffix in [".weight", ".bias", ".running_mean", ".running_var", ".num_batches_tracked"]:
                tk = t_prefix + suffix
                ck = f"stem.{i}" + suffix
                if tk in t_state and ck in c_state and t_state[tk].shape == c_state[ck].shape:
                    new_state[ck] = t_state[tk]
        

    # 2) 나머지 Layer & FC 매핑 (순서대로 shape 확인하며 매핑)
    # timm과 custom의 구조가 동일하므로 shape 기반으로 주입
    t_keys = [k for k in t_state.keys() if "conv1" not in k]
    c_keys = [k for k in c_state.keys() if "stem" not in k]

    
    # Index 매칭 대신 이름 기반 매칭 (downsample 보정 필요)
    for ck in c_keys:
        # timm의 downsample.1 (Conv) -> custom의 downsample.1 (Conv) 대응 (구조 맞춤)
        # 만약 이름이 완벽히 같다면 그대로 복사
        if ck in t_state and t_state[ck].shape == c_state[ck].shape:
            new_state[ck] = t_state[ck]
        else:
            if "downsample.0" in ck:
                tk_alt = ck.replace("downsample.0", "downsample.1")
            elif "downsample.1" in ck:
                tk_alt = ck.replace("downsample.1", "downsample.2")
            else:
                tk_alt = None

            if tk_alt and tk_alt in t_state and t_state[tk_alt].shape == c_state[ck].shape:
                new_state[ck] = t_state[tk_alt]



    modelCF.load_state_dict(new_state, strict=False)
    print(f"=> Mapped {len(new_state)} tensors. Pretrained weights loaded successfully.")

elif args.model == 'ResNet20_CIFAR10':
    from models import ResNet
    modelCF = ResNet.cifar_resnet20(args=args, logger=logger, num_classes=10)
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

        ret = modelCF.load_state_dict(state_dict, strict=False)
        logger(f"[LOAD] {args.pretrained}")
        logger(f"[LOAD] missing_keys={len(ret.missing_keys)} unexpected_keys={len(ret.unexpected_keys)}")
    else:
        # 없으면 chenyaofo pretrained로 시작(옵션)
        hub_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        modelCF.load_state_dict(hub_model.state_dict(), strict=False)



else:
    raise ValueError("Unknown model type")

if args.cuda:
	modelCF.cuda()

# ret = modelCF.load_state_dict(new_state, strict=False)
# print("missing_keys:", len(ret.missing_keys), ret.missing_keys[:20])
# print("unexpected_keys:", len(ret.unexpected_keys), ret.unexpected_keys[:20])





# ---- import calibration helpers from the model module (ResNet.py) ----
# (ResNet일 때만 써도 되지만, 공통으로 안전하게 try)
calib_enable = calib_disable = calib_finalize = calib_save_json = calib_load_json = None
try:
    from models import ResNet as ResNetMod
    calib_enable = ResNetMod.calib_enable
    calib_disable = ResNetMod.calib_disable
    calib_finalize = ResNetMod.calib_finalize
    calib_save_json = ResNetMod.calib_save_json
    calib_load_json = ResNetMod.calib_load_json
except Exception as e:
    print("[WARN] cannot import calibration helpers from models/ResNet.py:", e)


best_acc, old_file = 0, None
t_begin = time.time()
# ready to go
modelCF.eval()

test_loss = 0
correct = 0
trained_with_quantization = True

criterion = torch.nn.CrossEntropyLoss()

modelCF.eval()
test_loss = 0.0
correct = 0
total = 0
seen_batches = 0

hook_handle_list = None

# def print_conv_fc_weight_range(model, topk=30):
#     rows = []
#     with torch.no_grad():
#         for name, p in model.named_parameters():
#             if not name.endswith(".weight"):
#                 continue
#             # BatchNorm/LayerNorm 제외 (원하면 제거 가능)
#             if "bn" in name.lower() or "norm" in name.lower():
#                 continue
#             w = p.detach().float().cpu()
#             rows.append((w.abs().max().item(), name, w.min().item(), w.max().item(), w.std(unbiased=False).item(), w.numel()))

#     rows.sort(reverse=True, key=lambda x: x[0])
#     print(f"\n[TOP {topk} conv/fc weights by |w|max]")
#     for i, (amax, name, mn, mx, std, n) in enumerate(rows[:topk], 1):
#         print(f"{i:02d} {name:50s} min={mn:+.6g} max={mx:+.6g} std={std:.3g} |w|max={amax:.6g} n={n:,}")

#print_conv_fc_weight_range(modelCF, topk=25)
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

reset_wage_scale_after_loading_pretrained(modelCF, args.wl_weight, verbose=True)

if args.act_calib_json.strip() == "":
    alpha_json_path = os.path.join(args.logdir, f"act_alpha_p{args.act_calib_p}_b{args.act_calib_batches}.json")
else:
    alpha_json_path = args.act_calib_json

# -------------------------
# (1) Activation calibration
# -------------------------
if args.act_calib == 1:
    if calib_enable is None:
        raise RuntimeError("Calibration helpers not available. Did you add calib_* functions into models/ResNet.py?")

    logger(f"[ACT-CALIB] enabled: p={args.act_calib_p} batches={args.act_calib_batches} "
           f"stride={args.act_calib_stride} max_samples={args.act_calib_max_samples}")
    
    # ResNet.py에서 act_q가 args.act_calib or _ACT_CALIB["enabled"]를 보고 calib_update만 하도록 되어 있어야 함
    args.act_calib = True  # 중요: act_q가 이 플래그를 봄

    modelCF.eval()
    calib_enable(p=args.act_calib_p,
                 max_samples=args.act_calib_max_samples,
                 stride=args.act_calib_stride)

    # 통계는 train_loader로 잡는 게 보통 더 안정적(분포 다양)
    calib_loader = train_loader #if args.dataset == "imagenet" else test_loader

    with torch.no_grad():
        for b, (data, target) in enumerate(calib_loader):
            if b >= args.act_calib_batches:
                break
            if args.cuda:
                data = data.cuda(non_blocking=True)

            _ = modelCF(data)

            if (b + 1) % 20 == 0:
                logger(f"[ACT-CALIB] collected batches: {b+1}/{args.act_calib_batches}")

    alphas = calib_finalize()
    calib_save_json(alpha_json_path)
    logger(f"[ACT-CALIB] saved {len(alphas)} layer alphas to: {alpha_json_path}")

    # 캘리브레이션 종료
    args.act_calib = False
    calib_disable()

# -------------------------
# (2) Load alpha and evaluate
# -------------------------

from modules import quantization_cpu_np_infer as q

if calib_load_json is not None and os.path.exists(alpha_json_path):
    alpha_dict = calib_load_json(alpha_json_path)
    q.calib_set_alpha_dict(alpha_dict)
    logger(f"[ACT] loaded alpha json: {alpha_json_path}")
else:
    logger(f"[ACT] alpha json not found (or no loader). path={alpha_json_path}. "
           f"act quant may be bypassed if alpha missing.")


with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        output = modelCF(data)
        loss = criterion(output, target)

        test_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        seen_batches += 1


        if i == 0:   # ✅ shape는 한 번만
            print("output shape:", output.shape)
            print("data stats:", data.min().item(), data.max().item(), data.mean().item(), data.std().item())
            #print("first 10 classes:", test_loader.dataset.classes[:10])
            #print("class_to_idx sample:", list(test_loader.dataset.class_to_idx.items())[:5])
            print("fc has bias param?", hasattr(modelCF.fc, "bias") and modelCF.fc.bias is not None)
            #print("bn1.training:", modelCF.bn1.training)
            print("first logits (top5):", torch.topk(output[0].detach().cpu(), 5))
            print("target min/max:", target.min().item(), target.max().item())
            print("target first10:", target[:10].detach().cpu().tolist())
            #print("conv1 =", modelCF.conv1.weight.shape)

        # (디버깅용) 몇 배치만 돌리고 싶으면 여기서 break
        if i == 10:
            break

avg_loss = test_loss / seen_batches
acc = 100.0 * correct / total
logger(f"Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)")
