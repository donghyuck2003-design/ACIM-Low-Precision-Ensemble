# python inference_comparison.py \
#   --dataset cifar100 --model VGG8 --mode WAGE --batch_size 200 \
#   --analyze_bits 3,3 --analyze_batches 1 \
#   --analyze_vari_list 0,0.05,0.1 --mc_samples 16 \
#   --dump_effw_layers conv1,features.3 \
#   --inference 1 --cellBit 1 --onoffratio 10 --ADCprecision 5

import argparse
import os
import time
import math
import csv
from utee import misc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from models import dataset
import torchvision.models as models
from utee import hook
from utee import wage_quantizer   # ✅ 추가: WAGE quantizer 사용
# from IPython import embed
from datetime import datetime
from subprocess import call

# -----------------------------
# Metrics helpers
# -----------------------------
_EPS = 1e-12

def _to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float()

def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    # float32로 정리 (계산 안정성)
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if x.dtype not in (torch.float16, torch.float32, torch.float64):
        x = x.float()
    else:
        x = x.float()
    return x

@torch.no_grad()
def quantize_like_wage(x: torch.Tensor, bits: int) -> torch.Tensor:
    # Neuorsim/WAGE에서 흔히 쓰는 wage_quantizer.Q를 그대로 사용
    # (bits=3/4)
    return wage_quantizer.Q(x, bits)

@torch.no_grad()
def compute_metrics(x: torch.Tensor, xq: torch.Tensor) -> dict:
    """
    x: original (float)
    xq: quantized
    Returns:
        cosine, l2_rmse, mae, snr_db, zero_ratio,
        x_min/x_max/x_mean/x_std,
        xq_min/xq_max/xq_mean/xq_std
    """
    x = _to_float_tensor(x).detach()
    xq = _to_float_tensor(xq).detach()

    xf = x.reshape(-1)
    xqf = xq.reshape(-1)

    # cosine
    dot = torch.sum(xf * xqf)
    denom = (torch.norm(xf) * torch.norm(xqf)).clamp_min(_EPS)
    cosine = (dot / denom).item()

    # error
    err = xf - xqf
    mse = torch.mean(err * err).item()
    l2_rmse = math.sqrt(max(mse, 0.0))
    mae = torch.mean(torch.abs(err)).item()

    # SNR (dB): 10 log10 (Px / Perr)
    px = torch.mean(xf * xf).item()
    perr = torch.mean(err * err).item()
    snr_db = 10.0 * math.log10((px + _EPS) / (perr + _EPS))

    # zero ratio (exact 0 after quant)
    zero_ratio = torch.mean((xqf == 0).float()).item()

    # stats
    x_min = xf.min().item()
    x_max = xf.max().item()
    x_mean = xf.mean().item()
    x_std = xf.std(unbiased=False).item()

    xq_min = xqf.min().item()
    xq_max = xqf.max().item()
    xq_mean = xqf.mean().item()
    xq_std = xqf.std(unbiased=False).item()

    return dict(
        cosine=cosine,
        l2_rmse=l2_rmse,
        mae=mae,
        snr_db=snr_db,
        zero_ratio=zero_ratio,
        x_min=x_min, x_max=x_max, x_mean=x_mean, x_std=x_std,
        xq_min=xq_min, xq_max=xq_max, xq_mean=xq_mean, xq_std=xq_std,
    )


class RunningAvg:
    def __init__(self):
        self.n = 0
        self.sum = 0.0

    def update(self, v: float):
        self.n += 1
        self.sum += float(v)

    def mean(self) -> float:
        return self.sum / max(self.n, 1)

class ActAgg:
    """
    Activation(레이어 입력)용 누적 집계:
    - 원본 global min/max/mean/std (대략적으로는 batch별 mean/std 평균이 아니라
      전체 flatten 기준을 정확히 하려면 sum/sumsq/count 필요)
    - 여기서는 전체값 기준 mean/std를 위해 sum/sumsq/count를 사용
    - cosine/l2/mae/snr/zero_ratio는 batch별 계산 후 평균
    """
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.minv = None
        self.maxv = None

        self.cos = RunningAvg()
        self.l2 = RunningAvg()
        self.mae = RunningAvg()
        self.snr = RunningAvg()
        self.zr = RunningAvg()

        self.q_min = None
        self.q_max = None
        self.q_sum = 0.0
        self.q_sumsq = 0.0

    def update(self, x: torch.Tensor, xq: torch.Tensor):
        x = _to_float_tensor(x).detach()
        xq = _to_float_tensor(xq).detach()

        xf = x.reshape(-1)
        xqf = xq.reshape(-1)

        # 원본 전체 통계
        self.count += xf.numel()
        self.sum += xf.sum().item()
        self.sumsq += torch.sum(xf * xf).item()

        bmin = xf.min().item()
        bmax = xf.max().item()
        self.minv = bmin if self.minv is None else min(self.minv, bmin)
        self.maxv = bmax if self.maxv is None else max(self.maxv, bmax)

        # 양자화 전체 통계
        self.q_sum += xqf.sum().item()
        self.q_sumsq += torch.sum(xqf * xqf).item()
        qbmin = xqf.min().item()
        qbmax = xqf.max().item()
        self.q_min = qbmin if self.q_min is None else min(self.q_min, qbmin)
        self.q_max = qbmax if self.q_max is None else max(self.q_max, qbmax)

        # batch metrics
        m = compute_metrics(x, xq)
        self.cos.update(m["cosine"])
        self.l2.update(m["l2_rmse"])
        self.mae.update(m["mae"])
        self.snr.update(m["snr_db"])
        self.zr.update(m["zero_ratio"])

    def finalize(self) -> dict:
        if self.count <= 0:
            return dict(
                x_min=None, x_max=None, x_mean=None, x_std=None,
                xq_min=None, xq_max=None, xq_mean=None, xq_std=None,
                cosine=None, l2_rmse=None, mae=None, snr_db=None, zero_ratio=None
            )

        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(var)

        q_mean = self.q_sum / self.count
        q_var = max(self.q_sumsq / self.count - q_mean * q_mean, 0.0)
        q_std = math.sqrt(q_var)

        return dict(
            x_min=self.minv, x_max=self.maxv, x_mean=mean, x_std=std,
            xq_min=self.q_min, xq_max=self.q_max, xq_mean=q_mean, xq_std=q_std,
            cosine=self.cos.mean(),
            l2_rmse=self.l2.mean(),
            mae=self.mae.mean(),
            snr_db=self.snr.mean(),
            zero_ratio=self.zr.mean(),
        )

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='VGG8', help='VGG8|DenseNet40|ResNet18')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
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

# ✅ 추가: 분석 설정
parser.add_argument('--analyze_bits', default='2,3', help='comma-separated bitwidths for analysis (e.g., 3,4)')
parser.add_argument('--analyze_batches', type=int, default=1, help='how many test batches to use for activation analysis')
parser.add_argument('--analyze_save_dir', default='.', help='(optional) override save dir for CSV (default: args.logdir)')
parser.add_argument('--analyze_layers', default='conv,linear', help='which layers to analyze: conv,linear,all')

# Hardware Properties
parser.add_argument('--inference', type=int, default=1, help='run hardware inference simulation')
parser.add_argument('--subArray', type=int, default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--parallelRead', type=int, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
parser.add_argument('--ADCprecision', type=int, default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', type=int, default=1, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', type=float, default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
parser.add_argument('--vari', type=float, default=0., help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', type=float, default=0, help='retention time')
parser.add_argument('--v', type=float, default=0, help='drift coefficient')
parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', type=float, default=0, help='drift target for fixed-direction drift, range 0-1')

parser.add_argument('--analyze_vari_list', default='', help='comma-separated variation std list (e.g., 0,0.02,0.05)')
parser.add_argument('--mc_samples', type=int, default=8, help='MC samples per variation (effective weight)')
parser.add_argument('--dump_effw_layers', default='', help='optional: comma-separated layer name filters (substring match)')


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
    train_loader, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
else:
    raise ValueError("Unknown dataset type")

assert args.model in ['VGG8', 'DenseNet40', 'ResNet18'], args.model
if args.model == 'VGG8':
    from All_Neurosim_Project.Accuracy_Ensemble.VGG8_CIFAR100.Trash import VGG
    model_path = './log/VGG8.pth'
    modelCF = VGG.vgg8(args=args, logger=logger, pretrained=model_path)
elif args.model == 'DenseNet40':
    from models import DenseNet
    model_path = './log/DenseNet40.pth'
    modelCF = DenseNet.densenet40(args=args, logger=logger, pretrained=model_path)
elif args.model == 'ResNet18':
    from models import ResNet
    modelCF = ResNet.resnet18(args=args, logger=logger, pretrained=True)
else:
    raise ValueError("Unknown model type")

w = next(p for n, p in modelCF.named_parameters() if "weight" in n)
print("weight abs max:", w.abs().max().item())

if args.cuda:
    modelCF.cuda()

# ready to go
modelCF.eval()

criterion = torch.nn.CrossEntropyLoss()

# -----------------------------
# ✅ 분석 대상 비트 파싱
# -----------------------------
bits_list = []
for s in args.analyze_bits.split(","):
    s = s.strip()
    if not s:
        continue
    bits_list.append(int(s))
bits_list = sorted(list(set(bits_list)))
assert len(bits_list) > 0, "analyze_bits is empty"

save_dir = args.analyze_save_dir.strip() or args.logdir
os.makedirs(save_dir, exist_ok=True)

weight_csv_path = os.path.join(save_dir, f"quant_compare_{current_time}_weights.csv")
act_csv_path = os.path.join(save_dir, f"quant_compare_{current_time}_activations.csv")

# -----------------------------
# ✅ 레이어 필터 설정
# -----------------------------
layer_mode = args.analyze_layers.lower().strip()
want_conv = ("conv" in layer_mode) or (layer_mode == "all")
want_linear = ("linear" in layer_mode) or (layer_mode == "all")

def is_target_layer(m: nn.Module) -> bool:
    # 커스텀 QConv2d/QLinear도 nn.Conv2d/nn.Linear subclass라면 여기서 잡힘
    if want_conv and isinstance(m, nn.Conv2d):
        return True
    if want_linear and isinstance(m, nn.Linear):
        return True
    # 혹시 커스텀 Linear가 nn.Linear를 상속 안 하면 weight attribute로 fallback
    if layer_mode == "all" and hasattr(m, "weight") and torch.is_tensor(getattr(m, "weight", None)):
        return True
    return False

# -----------------------------
# ✅ 1) Weight 분석 (원본 weight vs 3/4bit quant)
# -----------------------------

import math
import numpy as np

@torch.no_grad()
def neurosim_scale_only(weight_tensor: torch.Tensor, bits_W: int, factor: float = 1.0) -> float:
    """
    wage_initializer.wage_init_()의 'scale 계산'만 그대로 복제.
    ⚠️ 절대 weight_tensor를 수정하지 않음 (uniform_ 호출 없음)
    """
    t = weight_tensor
    dimensions = t.ndimension()
    if dimensions < 2:
        raise ValueError("weight tensor must be at least 2D")

    if dimensions == 2:
        fan_in = t.size(1)
    else:
        num_input_fmaps = t.size(1)
        receptive_field_size = 1
        if t.dim() > 2:
            receptive_field_size = t[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size

    float_limit = math.sqrt(3.0 * factor / fan_in)

    # wage_initializer.scale_limit(float_limit, bits_W)과 동일 로직 :contentReference[oaicite:5]{index=5}
    delta = 1 / (2 ** (bits_W - 1))
    limit = 1 - delta
    if bits_W > 2:
        limit_std = limit / math.sqrt(3)
    else:
        limit_std = 0.75 / math.sqrt(3)

    scale = 2 ** np.ceil(np.log2(limit_std / float_limit))
    return float(scale)

@torch.no_grad()
def neurosim_effective_weight_for_bits(m, bits: int, factor: float = 1.0):
    """
    Neuorsim QConv2d/QLinear forward 의미에 맞춘 '유효 weight' 비교용.
    - forward는 weight1 = w*scale → Q(weight1) → output = output/scale 로 상쇄 :contentReference[oaicite:6]{index=6}
    => effective quant weight = Q(w*scale, bits)/scale
    """
    w = m.weight.detach().float()
    scale = neurosim_scale_only(m.weight, bits_W=bits, factor=factor)

    w1 = w * scale
    wq_internal = wage_quantizer.Q(w1, bits)
    wq_eff = wq_internal / scale
    return w, wq_eff, scale

@torch.no_grad()
def hw_bitserial_effective_weight(
    w_signed: torch.Tensor,
    bitWeight: int,
    cellBit: int,
    onoffratio: float,
    vari: float,
    t: float, v: float, detect: int, target: float,
    mc_samples: int = 8,
    device: str = "cuda",
):
    """
    Neuorsim bit-serial HW path에서 weight에 적용되는 연산을 'weight 관점'으로 역산해
    effective signed weight w_eff를 만든다.

    w_signed: forward에서 사용되는 signed weight (대개 wage_quantizer.Q(...) 결과, 범위 [-1,1] 가정)
    return:
      w_eff_nominal (vari=0),
      w_eff_mc_mean, w_eff_mc_std  (vari>0이면 MC 평균/표준편차)
    """
    w = w_signed.detach().to(device).float()

    cellRange = 2 ** int(cellBit)
    upper = 1.0
    lower = 1.0 / float(onoffratio)

    # (cellRange-1)*(upper+lower)/2 : code의 dummyP와 동일
    dummyP = (cellRange - 1.0) * (upper + lower) / 2.0

    # signed [-1,1] -> decimal [0, 2^bitW - 1]
    X0 = torch.round(((2**bitWeight - 1) / 2.0) * (w + 1.0))

    # ---------- nominal (vari=0) ----------
    X = X0.clone()
    contrib = torch.zeros_like(w)

    nSlices = int(bitWeight // cellBit)
    for k in range(nSlices):
        remainder = torch.fmod(X, cellRange)
        remainder = wage_quantizer.Retention(remainder, t, v, detect, target)
        X = torch.round((X - remainder) / cellRange)

        # map remainder [0,cellRange-1] -> conductance-like [ (cellRange-1)*lower , (cellRange-1)*upper ]
        remainderQ = (upper - lower) * remainder + (cellRange - 1.0) * lower

        # dummy subtraction + per-slice scaling (scaler=cellRange**k, and *2/(1-1/onoffratio))
        scaler = float(cellRange**k)
        contrib = contrib + (remainderQ - dummyP) * scaler * (2.0 / (1.0 - 1.0 / onoffratio))

    # code 끝에서 output = output/(2**bitWeight) 하므로 weight도 동일 정규화
    w_eff_nominal = contrib / float(2**bitWeight)

    # ---------- variation MC ----------
    if vari <= 0 or mc_samples <= 1:
        return w_eff_nominal, w_eff_nominal, torch.zeros_like(w_eff_nominal)

    # MC: remainderQ = remainderQ + remainderQ * N(0, vari)
    eff_list = []
    for _ in range(int(mc_samples)):
        X = X0.clone()
        contrib = torch.zeros_like(w)
        for k in range(nSlices):
            remainder = torch.fmod(X, cellRange)
            remainder = wage_quantizer.Retention(remainder, t, v, detect, target)
            X = torch.round((X - remainder) / cellRange)

            remainderQ = (upper - lower) * remainder + (cellRange - 1.0) * lower
            noise = torch.normal(mean=0.0, std=torch.full_like(remainderQ, float(vari)))
            remainderQ = remainderQ + remainderQ * noise

            scaler = float(cellRange**k)
            contrib = contrib + (remainderQ - dummyP) * scaler * (2.0 / (1.0 - 1.0 / onoffratio))

        w_eff = contrib / float(2**bitWeight)
        eff_list.append(w_eff)

    eff_stack = torch.stack(eff_list, dim=0)  # [mc, ...]
    w_eff_mc_mean = eff_stack.mean(dim=0)
    w_eff_mc_std  = eff_stack.std(dim=0, unbiased=False)

    return w_eff_nominal, w_eff_mc_mean, w_eff_mc_std

# -----------------------------
# ✅ 1.5) HW(bit-serial) effective weight vs variation
# -----------------------------
vari_list = []
if args.analyze_vari_list.strip():
    for s in args.analyze_vari_list.split(","):
        s = s.strip()
        if s:
            vari_list.append(float(s))
else:
    # 기본: args.vari만
    vari_list = [float(args.vari)]

# (optional) layer name filter
layer_filters = [x.strip() for x in args.dump_effw_layers.split(",") if x.strip()]

def layer_ok(name: str) -> bool:
    if not layer_filters:
        return True
    return any(f in name for f in layer_filters)

effw_csv_path = os.path.join(save_dir, f"quant_compare_{current_time}_effw_hw.csv")

eff_rows = []
with torch.no_grad():
    for name, m in modelCF.named_modules():
        if not is_target_layer(m):
            continue
        if not layer_ok(name):
            continue
        if not hasattr(m, "weight") or m.weight is None:
            continue

        # ✅ forward에서 쓰는 것과 동일하게 "디지털 quant weight"부터 시작
        w_fp = m.weight.detach().float()
        # scale은 네 QConv2d/QLinear에서 weight1=w*scale 후 Q를 하므로 동일하게 맞춤
        # (여기선 inference_comparison.py에 이미 있는 neurosim_scale_only 재사용) :contentReference[oaicite:2]{index=2}
        for bW in bits_list:
            scale = neurosim_scale_only(m.weight, bits_W=int(bW), factor=1.0)
            w1 = w_fp * scale
            w_digital = wage_quantizer.Q(w1, int(bW)) / scale   # signed weight ([-1,1] 근처)

            for vv in vari_list:
                w0, wmean, wstd = hw_bitserial_effective_weight(
                    w_signed=w_digital,
                    bitWeight=int(bW),
                    cellBit=int(getattr(m, "cellBit", args.cellBit)),
                    onoffratio=float(getattr(m, "onoffratio", args.onoffratio)),
                    vari=float(vv),
                    t=float(getattr(m, "t", args.t)),
                    v=float(getattr(m, "v", args.v)),
                    detect=int(getattr(m, "detect", args.detect)),
                    target=float(getattr(m, "target", args.target)),
                    mc_samples=int(args.mc_samples),
                    device="cuda" if args.cuda else "cpu",
                )

                # 비교 지표: nominal effw vs variation mean effw
                met = compute_metrics(w0, wmean)

                eff_rows.append(dict(
                    layer=name,
                    layer_class=m.__class__.__name__,
                    bits=int(bW),
                    vari=float(vv),
                    mc_samples=int(args.mc_samples),
                    # w_eff 분포 요약(variation std의 평균 크기)
                    effw_std_mean=float(wstd.abs().mean().item()),
                    effw_std_max=float(wstd.abs().max().item()),
                    # nominal vs mean(vari) similarity
                    cosine=met["cosine"],
                    l2_rmse=met["l2_rmse"],
                    mae=met["mae"],
                    snr_db=met["snr_db"],
                    zero_ratio=met["zero_ratio"],
                    # nominal range
                    w0_min=float(w0.min().item()),
                    w0_max=float(w0.max().item()),
                    # mean range
                    wmean_min=float(wmean.min().item()),
                    wmean_max=float(wmean.max().item()),
                ))

eff_fields = [
    "layer","layer_class","bits","vari","mc_samples",
    "effw_std_mean","effw_std_max",
    "cosine","l2_rmse","mae","snr_db","zero_ratio",
    "w0_min","w0_max","wmean_min","wmean_max",
]
with open(effw_csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=eff_fields)
    w.writeheader()
    for r in eff_rows:
        w.writerow(r)

logger(f"[Quant-Compare] HW effective-weight(vs variation) saved to: {effw_csv_path}")



