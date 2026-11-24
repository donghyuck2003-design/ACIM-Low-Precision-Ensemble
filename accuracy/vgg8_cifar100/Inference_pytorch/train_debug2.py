import argparse
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.autograd import Variable

from utee import misc, make_path, wage_util, wage_quantizer
from models import dataset


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='cifar100', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='VGG8', help='VGG8|DenseNet40|ResNet56')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)

parser.add_argument('--grad_scale', type=float, default=8)
parser.add_argument('--seed', type=int, default=117)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=1)

parser.add_argument('--logdir', default='log/default')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decreasing_lr', default='60,120,160')

parser.add_argument('--wl_weight', type=int, default=8)
parser.add_argument('--wl_grad', type=int, default=8)
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)

# Hardware Properties
parser.add_argument('--inference', default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', default=128)
parser.add_argument('--ADCprecision', default=5)
parser.add_argument('--cellBit', default=4)
parser.add_argument('--onoffratio', default=10)
parser.add_argument('--vari', default=0)
parser.add_argument('--t', default=0)
parser.add_argument('--v', default=0)
parser.add_argument('--detect', default=0)
parser.add_argument('--target', default=0)
parser.add_argument('--parallelRead', type=int, default=128)

# IMPORTANT:
# - ResNet56: 'hub'면 torch.hub에서 CIFAR100 pretrained를 가져와서 NeuroSim ResNet56(너의 모델)에 로드
# - 그 외: 경로면 해당 ckpt 로드
parser.add_argument('--pretrained', default='hub', help="hub | path/to/checkpoint.pth | None")
parser.add_argument('--wage_start_epoch', type=int, default=20)

args = parser.parse_args()

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args, ['log_interval', 'test_interval', 'logdir', 'epochs'])

misc.logger.init(args.logdir, 'train_log_' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)

logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger(f'{k}: {v}')
logger("========================================")


# -------------------------
# Seed / Device
# -------------------------
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)


# -------------------------
# Data
# -------------------------
assert args.dataset in ['cifar10', 'cifar100', 'imagenet'], args.dataset
if args.dataset == 'cifar10':
    train_loader, test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=1)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_loader, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1)
    num_classes = 100
elif args.dataset == 'imagenet':
    train_loader, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
    num_classes = 1000
else:
    raise ValueError("Unknown dataset type")



# -------------------------
# Utils
# -------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        if device == "cuda":
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        output = model(data)
        loss = criterion(output, target)
        total_loss += float(loss.item())

        pred = output.argmax(dim=1)
        correct += int((pred == target).sum().item())
        total += int(target.size(0))

    avg_loss = total_loss / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


def save_ckpt(path, model, optimizer, epoch, best_acc, grad_scale):
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": float(best_acc),
        "grad_scale": float(grad_scale),
        "args": vars(args),
    }
    torch.save(payload, path)


def load_hub_cifar100_resnet56_state_dict():
    hub_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_resnet56",
        pretrained=True
    )
    return hub_model.state_dict()


def load_state_flexible(model, state_dict, logger_fn):
    model_sd = model.state_dict()
    filtered = {}
    dropped = []
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        if nk in model_sd and v.shape == model_sd[nk].shape:
            filtered[nk] = v
        else:
            dropped.append(nk)

    model.load_state_dict(filtered, strict=False)
    if logger_fn:
        logger_fn(f"==> pretrained keys kept: {len(filtered)} / dropped: {len(dropped)}")
        if len(dropped) > 0:
            logger_fn("dropped examples: " + ", ".join(dropped[:10]))


import math
from utee import wage_initializer
from modules.quantization_cpu_np_infer import QConv2d, QLinear

def compute_wage_scale_from_shape(weight_tensor, bits_W, factor=2.0):
    # wage_init_()의 fan_in 계산 로직 그대로 (weight 값은 사용 안 함)
    dims = weight_tensor.ndimension()
    if dims < 2:
        raise ValueError("weight tensor must be at least 2D")

    if dims == 2:  # Linear [out, in]
        fan_in = weight_tensor.size(1)
    else:          # Conv [out, in, kH, kW]
        num_input_fmaps = weight_tensor.size(1)
        receptive_field_size = weight_tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size

    float_limit = math.sqrt(3 * factor / fan_in)
    _, scale = wage_initializer.scale_limit(float_limit, bits_W)  # scale만 얻기
    return float(scale)
 
def refresh_wage_scales(model):
    for m in model.modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.scale = compute_wage_scale_from_shape(m.weight, m.wl_weight, factor=2.0)


# -------------------------
# Model
# -------------------------
assert args.model in ['VGG8', 'DenseNet40', 'ResNet56'], args.model

if args.model == 'VGG8':
    from models import VGG
    model = VGG.vgg8(args=args, logger=logger, pretrained=args.pretrained)
    criterion = torch.nn.CrossEntropyLoss()

elif args.model == 'DenseNet40':
    from models import DenseNet
    model = DenseNet.densenet40(args=args, logger=logger)
    criterion = wage_util.SSE()

elif args.model == 'ResNet56':
    from models import ResNet  # 너의 NeuroSim용 ResNet.py (CIFAR-ResNet56 포함)

    # 항상 "NeuroSim용 ResNet56" 모델을 생성한다.
    model = ResNet.resnet56(args=args, logger=logger, pretrained=None, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    # pretrained 로드 전략:
    # - args.pretrained == 'hub' : torch.hub CIFAR100 resnet56 weight -> 현재 model에 로드
    # - args.pretrained == None : 랜덤 초기화에서 학습
    # - args.pretrained == 'path' : 해당 ckpt 로드 (state_dict / dict(state_dict=...) 둘 다 처리)
    if args.pretrained is not None:
        if isinstance(args.pretrained, str) and args.pretrained.lower() in ["hub", "torchhub", "cifar100", "true"]:
            logger("==> Loading CIFAR100 pretrained from torch.hub into NeuroSim ResNet56 ...")
            sd = load_hub_cifar100_resnet56_state_dict()
            load_state_flexible(model, sd, logger)
            #refresh_wage_scales(model)
        elif isinstance(args.pretrained, str) and os.path.isfile(args.pretrained):
            logger(f"==> Loading pretrained checkpoint from: {args.pretrained}")
            ckpt = torch.load(args.pretrained, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
            load_state_flexible(model, sd, logger)
            #refresh_wage_scales(model)
        else:
            logger(f"==> pretrained argument ignored (not found): {args.pretrained}")

else:
    raise ValueError("Unknown model type")


if args.cuda:
    model.cuda()
    device = "cuda"
else:
    device = "cpu"


# -------------------------
# Optimizer / LR schedule
# -------------------------
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
logger('decreasing_lr: ' + str(decreasing_lr))

best_acc = 0.0
t_begin = time.time()
grad_scale = args.grad_scale


# -------------------------
# Train
# -------------------------
try:
    for epoch in range(args.epochs):
        model.train()

        # LR decay
        if epoch in decreasing_lr:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1
            if args.mode == 'WAGE':
                grad_scale = grad_scale / 8.0

        logger(f"training phase (epoch {epoch})")

        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()

            if device == "cuda":
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            do_wage = (args.mode == 'WAGE') and (epoch >= args.wage_start_epoch)

            # IMPORTANT: BN 파라미터(1D)는 WAGE 양자화 대상에서 제외(깨지기 쉬움)
            if do_wage:
                for name, param in list(model.named_parameters())[::-1]:
                    if param.grad is None:
                        continue
                    if param.ndim in (2, 4):  # Linear(2D), Conv(4D)만
                        param.grad.data = wage_quantizer.QG(param.grad.data, args.wl_grad, grad_scale)

            optimizer.step()

            if do_wage:
                for name, param in list(model.named_parameters())[::-1]:
                    if param.ndim in (2, 4):  # Linear/Conv weights만 클리핑
                        param.data = wage_quantizer.C(param.data, args.wl_weight)

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) / len(data)
                logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    float(loss.item()), acc, optimizer.param_groups[0]['lr']
                ))

        # speed log
        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        logger("Elapsed {:.2f}s, {:.2f} s/epoch, {:.4f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))

        # -------- Save latest (state_dict + optimizer + meta) --------
        latest_path = os.path.join(args.logdir, 'latest.pth')
        save_ckpt(latest_path, model, optimizer, epoch, best_acc, grad_scale)

        # -------- Evaluate / Save best --------
        if epoch % args.test_interval == 0:
            logger("testing phase (current model)")
            test_loss, acc = evaluate(model, test_loader, criterion, device)
            logger(f'\tEpoch {epoch} Test set: Average loss: {test_loss:.4f}, Accuracy: {acc:.2f}%')

            # ✅ 저장 직후, 같은 프로세스에서 “새 모델 만들어 로드 후” 재검증 (원인 판별용)
            logger("testing phase (reloaded model from latest.pth)")
            #from models import ResNet
            #model_reload = ResNet.resnet56(args=args, logger=logger, pretrained=latest_path, num_classes=num_classes)
            # model_reload = VGG.vgg8(args=args, logger=logger, pretrained=latest_path)
            # refresh_wage_scales(model_reload)
            # if device == "cuda":
            #     model_reload.cuda()
            # reload_loss, reload_acc = evaluate(model_reload, test_loader, criterion, device)
            # logger(f'\t[ReloadCheck] Epoch {epoch} Reloaded: loss {reload_loss:.4f}, acc {reload_acc:.2f}%')

            if acc > best_acc:
                best_acc = acc
                best_path = os.path.join(args.logdir, f'best-epoch{epoch}.pth')
                save_ckpt(best_path, model, optimizer, epoch, best_acc, grad_scale)
                logger(f"==> New best saved: {best_path} (acc={best_acc:.2f}%)")

except Exception:
    import traceback
    traceback.print_exc()
finally:
    logger("Total Elapse: {:.2f}, Best Result: {:.2f}%".format(time.time() - t_begin, best_acc))
