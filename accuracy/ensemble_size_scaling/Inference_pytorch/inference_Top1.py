import argparse
import os
import time
import random
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from models import dataset
import torchvision.models as models
from utee import hook
from datetime import datetime
from subprocess import call

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

# Hardware Properties
parser.add_argument('--inference', type=int, default=1, help='run hardware inference simulation')
parser.add_argument('--subArray', type=int, default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--parallelRead', type=int, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
parser.add_argument('--ADCprecision', type=int, default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', type=int, default=1, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', type=float, default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
parser.add_argument('--vari', type=float, default=0., help='conductance variation std (sigma)')

parser.add_argument('--t', type=float, default=0, help='retention time')
parser.add_argument('--v', type=float, default=0, help='drift coefficient')
parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', type=float, default=0, help='drift target for fixed-direction drift, range 0-1')

# ✅ 추가: replica seed 두 개
parser.add_argument('--replica_seeds', default='', help='comma-separated two seeds for replicas (e.g., 117,118)')
# (옵션) 빨리 확인용: 일부 배치만 사용
parser.add_argument('--max_test_batches', type=int, default=-1, help='if >0, evaluate only first N batches')

parser.add_argument('--vari_list', default='', help='comma-separated sigma list (e.g., 0,0.05,0.1,0.2)')


current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
args = parser.parse_args()

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info
misc.ensure_dir(args.logdir)

if args.vari_list.strip():
    vari_list = [float(x.strip()) for x in args.vari_list.split(",") if x.strip()]
else:
    vari_list = [float(args.vari)]

logger(f"[Sigma list] {vari_list}")

logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")


# -----------------------------
# ✅ seed helper (replica용)
# -----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # (선택) 연산 deterministic 강화 (속도 약간 저하 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args.cuda = torch.cuda.is_available()

# data loader
assert args.dataset in ['cifar10', 'cifar100', 'imagenet'], args.dataset
if args.dataset == 'cifar10':
    train_loader, test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'cifar100':
    train_loader, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'imagenet':
    train_loader, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
else:
    raise ValueError("Unknown dataset type")

# model
assert args.model in ['VGG8', 'DenseNet40', 'ResNet18'], args.model
if args.model == 'VGG8':
    from models import VGG
    model_path = './log/VGG8.pth'
    modelCF = VGG.vgg8(args=args, logger=logger, pretrained=model_path)
elif args.model == 'DenseNet40':
    from models import DenseNet
    model_path = './log/DenseNet40.pth'
    modelCF = DenseNet.densenet40(args=args, logger=logger, pretrained=model_path)
elif args.model == 'ResNet18':
    from models import ResNet
    model_path = None
    modelCF = ResNet.resnet18(args=args, logger=logger, pretrained=True)
else:
    raise ValueError("Unknown model type")

print("model_path", model_path)

w = next(p for n,p in modelCF.named_parameters() if "weight" in n)
print("weight abs max:", w.abs().max().item())

if args.cuda:
    modelCF.cuda()

modelCF.eval()
criterion = torch.nn.CrossEntropyLoss()

# -----------------------------
# ✅ replica seed 파싱
# -----------------------------
if args.replica_seeds.strip():
    ss = [int(x.strip()) for x in args.replica_seeds.split(",") if x.strip()]
    if len(ss) != 2:
        raise ValueError("--replica_seeds must contain exactly two integers, e.g., 117,118")
    seed1, seed2 = ss[0], ss[1]
else:
    seed1, seed2 = int(args.seed), int(args.seed) + 1

logger(f"[Replica seeds] seed1={seed1}, seed2={seed2} (sigma/vari={args.vari})")

# -----------------------------
# ✅ 1회 inference 실행 함수
# -----------------------------
@torch.no_grad()
def run_eval(replica_seed: int):
    set_all_seeds(replica_seed)

    test_loss = 0.0
    correct = 0
    total = 0
    seen_batches = 0

    all_preds = []
    all_logits = []
    all_targets = []

    for i, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        output = modelCF(data)  # [B, C]

        # 디버그 (한 번만)
        if i == 0:
            print(f"[seed={replica_seed}] output shape:", output.shape)
            print(f"[seed={replica_seed}] first logits (top5):", torch.topk(output[0].detach().cpu(), 5))

        loss = criterion(output, target)
        test_loss += float(loss.item())

        pred = output.argmax(dim=1)
        correct += int((pred == target).sum().item())
        total += int(target.size(0))
        seen_batches += 1

        all_preds.append(pred.detach().cpu())
        all_logits.append(output.detach().cpu().float())   # ✅ logits 저장
        all_targets.append(target.detach().cpu())          # ✅ target 저장

        if args.max_test_batches > 0 and (i + 1) >= args.max_test_batches:
            break

    all_preds = torch.cat(all_preds, dim=0)        # [N]
    all_logits = torch.cat(all_logits, dim=0)      # [N, C]
    all_targets = torch.cat(all_targets, dim=0)    # [N]

    avg_loss = test_loss / max(seen_batches, 1)
    acc = 100.0 * correct / max(total, 1)

    return {
        "avg_loss": avg_loss,
        "acc": acc,
        "preds": all_preds,
        "logits": all_logits,
        "targets": all_targets,
        "total": total,
    }


def set_sigma_to_model(model, sigma: float, args):
    for m in model.modules():
        # variation
        if hasattr(m, "vari"):
            m.vari = float(sigma)

        # inference/hw params도 모듈이 들고 있는 경우가 많아서 같이 맞춰줌
        if hasattr(m, "inference"):
            m.inference = int(args.inference)
        if hasattr(m, "ADCprecision"):
            m.ADCprecision = int(args.ADCprecision)
        if hasattr(m, "cellBit"):
            m.cellBit = int(args.cellBit)
        if hasattr(m, "onoffratio"):
            m.onoffratio = float(args.onoffratio)
        if hasattr(m, "t"):
            m.t = float(args.t)
        if hasattr(m, "v"):
            m.v = float(args.v)
        if hasattr(m, "detect"):
            m.detect = int(args.detect)
        if hasattr(m, "target"):
            m.target = float(args.target)

def pearson_corr_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """
    a, b: [N, C] logits (cpu float)
    returns: Pearson correlation of flattened logits
    """
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()

    am = a.mean()
    bm = b.mean()
    av = a - am
    bv = b - bm

    denom = (av.pow(2).mean().sqrt() * bv.pow(2).mean().sqrt()).clamp_min(eps)
    corr = (av * bv).mean() / denom
    return float(corr.item())

# -----------------------------
# ✅ vari_list 파싱
# -----------------------------
if args.vari_list.strip():
    vari_list = [float(x.strip()) for x in args.vari_list.split(",") if x.strip()]
else:
    vari_list = [float(args.vari)]

logger(f"[Sigma list] {vari_list}")

# -----------------------------
# ✅ CSV 저장 경로
# -----------------------------
save_dir = args.logdir
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, f"top1_disagreement_{current_time}.csv")

csv_fields = [
    "sigma", "seed1", "seed2",
    "rep1_acc", "rep2_acc",
    "rep1_loss", "rep2_loss",
    "top1_disagree_rate_percent",
    "logit_corr_pearson",
    "rep1_err_rate_percent",
    "rep2_err_rate_percent",
    "disagree_given_rep1_wrong_percent",
    "disagree_given_any_wrong_percent",
    "num_samples"
]


rows = []

# -----------------------------
# ✅ sigma별 반복
# -----------------------------
for sigma in vari_list:
    # ✅ 핵심: 같은 σ에서 두 replica 비교
    args.vari = float(sigma)
    logger(f"\n=== Evaluating sigma={args.vari} (seed1={seed1}, seed2={seed2}) ===")

    set_sigma_to_model(modelCF, sigma, args)

    r1 = run_eval(seed1)
    r2 = run_eval(seed2)

    # 동일 샘플 수 보장
    N = min(r1["preds"].numel(), r2["preds"].numel(), r1["targets"].numel(), r2["targets"].numel())
    p1 = r1["preds"][:N]
    p2 = r2["preds"][:N]
    t1 = r1["targets"][:N]  # 보통 동일해야 함

    # 1) Top-1 disagreement
    disagree = (p1 != p2).float().mean().item() * 100.0

    # 2) Prediction/Logit correlation (Pearson corr over flattened logits)
    l1 = r1["logits"][:N]
    l2 = r2["logits"][:N]
    logit_corr = pearson_corr_flat(l1, l2)

    # 3) Error masks
    err1 = (p1 != t1)
    err2 = (p2 != t1)  # target은 동일하다고 가정(동일 dataloader)

    rep1_err = err1.float().mean().item() * 100.0
    rep2_err = err2.float().mean().item() * 100.0

    # 4) Disagreement on error set
    # (a) Disagree | rep1 wrong
    if err1.any():
        disagree_rep1_wrong = ((p1 != p2) & err1).float().sum().item() / err1.float().sum().item()
        disagree_rep1_wrong *= 100.0
    else:
        disagree_rep1_wrong = 0.0

    # (b) Disagree | any wrong (rep1 or rep2 wrong)
    any_wrong = (err1 | err2)
    if any_wrong.any():
        disagree_any_wrong = ((p1 != p2) & any_wrong).float().sum().item() / any_wrong.float().sum().item()
        disagree_any_wrong *= 100.0
    else:
        disagree_any_wrong = 0.0

    logger(f"[sigma={sigma:.4f}] Replica-1: loss={r1['avg_loss']:.4f}, acc={r1['acc']:.2f}% (err={rep1_err:.2f}%)")
    logger(f"[sigma={sigma:.4f}] Replica-2: loss={r2['avg_loss']:.4f}, acc={r2['acc']:.2f}% (err={rep2_err:.2f}%)")
    logger(f"[sigma={sigma:.4f}] Top-1 disagreement = {disagree:.2f}%  (N={N})")
    logger(f"[sigma={sigma:.4f}] Logit corr (Pearson) = {logit_corr:.6f}")
    logger(f"[sigma={sigma:.4f}] Disagree | rep1 wrong = {disagree_rep1_wrong:.2f}%")
    logger(f"[sigma={sigma:.4f}] Disagree | any wrong = {disagree_any_wrong:.2f}%")

    rows.append({
        "sigma": float(sigma),
        "seed1": int(seed1),
        "seed2": int(seed2),
        "rep1_acc": float(r1["acc"]),
        "rep2_acc": float(r2["acc"]),
        "rep1_loss": float(r1["avg_loss"]),
        "rep2_loss": float(r2["avg_loss"]),
        "top1_disagree_rate_percent": float(disagree),
        "logit_corr_pearson": float(logit_corr),
        "rep1_err_rate_percent": float(rep1_err),
        "rep2_err_rate_percent": float(rep2_err),
        "disagree_given_rep1_wrong_percent": float(disagree_rep1_wrong),
        "disagree_given_any_wrong_percent": float(disagree_any_wrong),
        "num_samples": int(N),
    })

# -----------------------------
# ✅ CSV 기록
# -----------------------------
with open(csv_path, "w", newline="") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

logger(f"[Saved] Top-1 disagreement results saved to: {csv_path}")
