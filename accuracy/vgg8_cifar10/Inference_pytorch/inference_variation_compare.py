# inference_weightbits_compare_neurosim.py
# 예)
# python inference_variation_compare.py \
#   --dataset cifar10 --model VGG8 --mode WAGE \
#   --batch_size 200 --analyze_batches 1 \
#   --pretrained "./log/VGG8.pth" \
#   --analyze_varis 0.05,0.05 --inference 1 --hw_hook 0 --save_dir ./Result

import argparse, os, math, csv, copy
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter

from utee import misc
from models import dataset
from utee import hook  # 뉴로심 방식 dump hook

# hook.py가 modules.quantization_cpu_np_infer를 참조하므로 동일하게 맞춰둠
try:
    from modules.quantization_cpu_np_infer import QConv2d, QLinear
except Exception:
    QConv2d, QLinear = None, None

import modules.quantization_cpu_np_infer as q
print(q.__file__)

_EPS = 1e-12


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.detach().float()


@torch.no_grad()
def compute_metrics(x: torch.Tensor, y: torch.Tensor) -> dict:
    x = _to_float_tensor(x)
    y = _to_float_tensor(y)
    xf = x.reshape(-1)
    yf = y.reshape(-1)

    dot = torch.sum(xf * yf)
    denom = (torch.norm(xf) * torch.norm(yf)).clamp_min(_EPS)
    cosine = (dot / denom).item()

    err = xf - yf
    mse = torch.mean(err * err).item()
    rmse = math.sqrt(max(mse, 0.0))
    mae = torch.mean(torch.abs(err)).item()

    px = torch.mean(xf * xf).item()
    perr = torch.mean(err * err).item()
    snr_db = 10.0 * math.log10((px + _EPS) / (perr + _EPS))

    zero_ratio = torch.mean((yf == 0).float()).item()

    return dict(
        cosine=cosine,
        l2_rmse=rmse,
        mae=mae,
        snr_db=snr_db,
        zero_ratio=zero_ratio,
        x_min=xf.min().item(), x_max=xf.max().item(),
        y_min=yf.min().item(), y_max=yf.max().item(),
    )


class RunningAvg:
    def __init__(self):
        self.n = 0
        self.s = 0.0

    def update(self, v):
        self.n += 1
        self.s += float(v)

    def mean(self):
        return self.s / max(self.n, 1)


class PairAgg:
    def __init__(self):
        self.cos = RunningAvg()
        self.rmse = RunningAvg()
        self.mae = RunningAvg()
        self.snr = RunningAvg()
        self.zr = RunningAvg()
        self.count = 0
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        # ===== 추가: y 기준 quant 통계 =====
        self.sat = RunningAvg()          # batch별 sat_ratio 평균
        self.uniq = RunningAvg()         # batch별 unique_levels 평균(참고용)
        self.hist = Counter()            # 전체 누적 히스토그램 (value -> count)

    @staticmethod
    def _quant_stats_for_y(y: torch.Tensor):
        """
        y 텐서(보통 quant output)의
        - sat_ratio: min/max 코드에 걸린 비율
        - unique_levels: 유니크 값 개수
        - hist Counter: value->count
        를 리턴
        """
        y = _to_float_tensor(y).reshape(-1)
        if y.numel() == 0:
            return 0.0, 0, Counter()

        ymin = y.min()
        ymax = y.max()

        # min/max saturation ratio
        sat_ratio = torch.mean(((y == ymin) | (y == ymax)).float()).item()

        # unique + counts
        # (quant output이면 unique가 적어서 이 방식이 안전/빠름)
        u, c = torch.unique(y, sorted=True, return_counts=True)
        unique_levels = int(u.numel())

        # Counter 누적 (float 그대로 key로 써도 되지만, 안전하게 반올림해 키를 안정화)
        hist = Counter()
        for vv, cc in zip(u.tolist(), c.tolist()):
            key = float(round(vv, 7))  # -0.9921875 같은 값 안정적으로 유지
            hist[key] += int(cc)

        return sat_ratio, unique_levels, hist

    @staticmethod
    def _hist_to_str(hist: Counter, max_items: int = 64) -> str:
        """
        CSV 한 칸에 넣기 위한 문자열 변환:
        "value:count;value:count;..."
        너무 길어지면 max_items까지만 잘라 저장.
        """
        if not hist:
            return ""
        items = sorted(hist.items(), key=lambda kv: kv[0])  # value 기준 정렬
        if len(items) > max_items:
            items = items[:max_items]
        return ";".join([f"{v}:{c}" for v, c in items])

    def update(self, x, y):
        m = compute_metrics(x, y)
        self.cos.update(m["cosine"])
        self.rmse.update(m["l2_rmse"])
        self.mae.update(m["mae"])
        self.snr.update(m["snr_db"])
        self.zr.update(m["zero_ratio"])
        self.count += 1

        self.x_min = m["x_min"] if self.x_min is None else min(self.x_min, m["x_min"])
        self.x_max = m["x_max"] if self.x_max is None else max(self.x_max, m["x_max"])
        self.y_min = m["y_min"] if self.y_min is None else min(self.y_min, m["y_min"])
        self.y_max = m["y_max"] if self.y_max is None else max(self.y_max, m["y_max"])

        # ===== 추가: y 기반 sat/unique/hist 누적 =====
        sat_ratio, unique_levels, hist = self._quant_stats_for_y(y)
        self.sat.update(sat_ratio)
        self.uniq.update(unique_levels)
        self.hist.update(hist)

    def finalize(self):
        return dict(
            cosine=self.cos.mean(),
            l2_rmse=self.rmse.mean(),
            mae=self.mae.mean(),
            snr_db=self.snr.mean(),
            zero_ratio=self.zr.mean(),
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            batches=self.count,

            # ===== 추가 컬럼 =====
            sat_ratio=self.sat.mean(),                    # batch 평균
            unique_levels=len(self.hist) if self.hist else 0,  # 누적 unique 개수(권장)
            histogram=self._hist_to_str(self.hist),       # 누적 histogram 문자열
        )


def register_activation_capture_hooks(model: nn.Module):
    """
    뉴로심 forward에서 "다음 레이어로 흘러가는 activation(=각 레이어 forward output)"을 캡처.
    기본은 QConv2d/QLinear만 대상으로 함.
    """
    handles = []
    layer_meta = []
    capture = {"buf": None}

    def is_target(m: nn.Module):
        if QConv2d is not None and isinstance(m, QConv2d):
            return True
        if QLinear is not None and isinstance(m, QLinear):
            return True
        # 원하면 아래를 True로 열어 nn.Conv2d/nn.Linear까지 캡처 가능
        # if isinstance(m, (nn.Conv2d, nn.Linear)): return True
        return False

    def make_hook(key):
        def _hook(m, inp, out):
            if capture["buf"] is None:
                return
            if torch.is_tensor(out):
                capture["buf"][key] = out.detach().cpu()
        return _hook

    for name, m in model.named_modules():
        if not is_target(m):
            continue
        layer_meta.append((name, m.__class__.__name__))
        handles.append(m.register_forward_hook(make_hook(name)))

    return handles, capture, layer_meta


def build_model(args, logger, pretrained_path: str):
    assert args.model == "VGG8", "현재 스크립트는 VGG8 기준"
    from models import VGG
    model = VGG.vgg8(args=args, logger=logger, pretrained=pretrained_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="VGG8")
    parser.add_argument("--mode", default="WAGE")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=117)

    # compare bits
    #parser.add_argument("--analyze_bits", default="3,4")
    parser.add_argument("--analyze_varis", default="0.0,0.05")
    parser.add_argument("--analyze_batches", type=int, default=5)

    # pretrained
    parser.add_argument("--pretrained", required=True)

    # outputs
    parser.add_argument("--save_dir", default="./Result")

    # ===== 매우 중요 =====
    # inference=0 : 디지털 WAGE 경로로 정확도 비교 (뉴로심에서 accuracy 볼 때 보통 이쪽)
    # inference=1 : VGG8 전용 bit-serial + ADC 근사 경로 -> 정확도 더 떨어질 수 있음
    parser.add_argument("--inference", type=int, default=1)

    # 아래 hw_hook은 "뉴로심 입력/가중치 덤프용 hook" (forward 결과를 바꾸지 않는 forward_hook)
    parser.add_argument("--hw_hook", type=int, default=1)

    # quant params (VGG.py에서 QConv/QLinear init에 들어감)
    parser.add_argument("--wl_weight", type=int, default=8)    # 여기 값은 analyze_bits로 덮임
    parser.add_argument("--wl_activate", type=int, default=8)
    parser.add_argument("--wl_error", type=int, default=8)

    # HW-related params (quantization_cpu_np_infer가 참조)
    parser.add_argument("--subArray", type=int, default=128)
    parser.add_argument("--parallelRead", type=int, default=128)
    parser.add_argument("--ADCprecision", type=int, default=5)
    parser.add_argument("--cellBit", type=int, default=1)
    parser.add_argument("--onoffratio", type=float, default=10)
    parser.add_argument("--vari", type=float, default=0.0)
    parser.add_argument("--t", type=float, default=0.0)
    parser.add_argument("--v", type=float, default=0.0)
    parser.add_argument("--detect", type=int, default=0)
    parser.add_argument("--target", type=float, default=0.0)

    args = parser.parse_args()

    var_list = [float(x.strip()) for x in args.analyze_varis.split(",") if x.strip()]
    assert len(var_list) == 2, "예: --analyze_varis 0.0,0.05 형태로 2개만"
    vA, vB = var_list
    bW = int(args.wl_weight)

    os.makedirs(args.save_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logdir = os.path.join(args.save_dir, f"run_{current_time}")
    os.makedirs(logdir, exist_ok=True)

    misc.logger.init(logdir, "compare_log")
    logger = misc.logger.info
    logger(f"bits: {vA} vs {vB}, inference={args.inference}, hw_hook={args.hw_hook}")

    # seed/cuda
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # dataloader
    if args.dataset == "cifar10":
        _, test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=1)
    elif args.dataset == "cifar100":
        _, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1)
    elif args.dataset == "imagenet":
        _, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
    else:
        raise ValueError("Unknown dataset")

    # ===== 뉴로심 방식: bit별로 모델을 새로 생성 (init부터 wl_weight 반영) =====
    argsA = copy.deepcopy(args)
    argsB = copy.deepcopy(args)

    # bit 고정
    argsA.wl_weight = bW
    argsB.wl_weight = bW

    # vari만 다르게
    argsA.vari = vA
    argsB.vari = vB

    modelA = build_model(argsA, logger, args.pretrained)
    modelB = build_model(argsB, logger, args.pretrained)

    if args.cuda:
        modelA.cuda()
        modelB.cuda()
    modelA.eval()
    modelB.eval()

    # activation capture hooks
    handlesA, capA, metaA = register_activation_capture_hooks(modelA)
    handlesB, capB, metaB = register_activation_capture_hooks(modelB)

    # layer set match
    meta_map = {n: cls for (n, cls) in metaA}

    act_aggs = {}  # layer -> PairAgg (A vs B)
    logit_agg = PairAgg()

    correctA = 0
    correctB = 0
    total = 0

    for i, (data, target) in enumerate(test_loader):
        if i >= int(args.analyze_batches):
            break

        indx_target = target.clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # ===== (옵션) 뉴로심 dump hook: i==0에서만 걸고 바로 제거 =====
        hookA = hookB = None
        if i == 0 and args.hw_hook:
            # 덮어쓰기 방지: model_name에 bit를 넣어서 폴더 분리
            hookA = hook.hardware_evaluation(
                modelA, argsA.wl_weight, argsA.wl_activate,
                argsA.subArray, argsA.parallelRead,
                f"{args.model}_w{vA}", args.mode
            )
            hookB = hook.hardware_evaluation(
                modelB, argsB.wl_weight, argsB.wl_activate,
                argsB.subArray, argsB.parallelRead,
                f"{args.model}_w{vB}", args.mode
            )

        with torch.no_grad():
            capA["buf"] = {}
            outA = modelA(Variable(data))
            actsA = capA["buf"]

            capB["buf"] = {}
            outB = modelB(Variable(data))
            actsB = capB["buf"]

        if i == 0 and args.hw_hook:
            hook.remove_hook_list(hookA)
            hook.remove_hook_list(hookB)

        # accuracy
        predA = outA.data.max(1)[1]
        predB = outB.data.max(1)[1]
        correctA += predA.cpu().eq(indx_target).sum().item()
        correctB += predB.cpu().eq(indx_target).sum().item()
        total += target.size(0)

        # logits compare
        logit_agg.update(outA.detach().cpu(), outB.detach().cpu())

        # activation compare
        for lname, cls in meta_map.items():
            if (lname in actsA) and (lname in actsB):
                if lname not in act_aggs:
                    act_aggs[lname] = {"layer_class": cls, "agg": PairAgg()}
                act_aggs[lname]["agg"].update(actsA[lname], actsB[lname])

    # remove hooks
    for h in handlesA + handlesB:
        try:
            h.remove()
        except Exception:
            pass

    # save CSVs
    act_csv = os.path.join(args.save_dir, f"act_compare_{vA}vs{vB}_{current_time}.csv")
    with open(act_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "layer", "layer_class", "variA", "variB",
                "cosine", "l2_rmse", "mae", "snr_db", "zero_ratio",
                "x_min", "x_max", "y_min", "y_max",
                "sat_ratio", "unique_levels", "histogram",   # <= 추가
                "batches",
            ]

        )
        w.writeheader()
        for lname, obj in sorted(act_aggs.items()):
            fin = obj["agg"].finalize()
            w.writerow(
    dict(
        layer=lname,
        layer_class=obj["layer_class"],
        variA=vA,
        variB=vB,
        cosine=fin["cosine"],
        l2_rmse=fin["l2_rmse"],
        mae=fin["mae"],
        snr_db=fin["snr_db"],
        zero_ratio=fin["zero_ratio"],
        x_min=fin["x_min"],
        x_max=fin["x_max"],
        y_min=fin["y_min"],
        y_max=fin["y_max"],

        sat_ratio=fin["sat_ratio"],
        unique_levels=fin["unique_levels"],
        histogram=fin["histogram"],

        batches=fin["batches"],
            )
        )
            
    
    logit_csv = os.path.join(args.save_dir, f"logits_compare_{vA}vs{vB}_{current_time}.csv")
    with open(logit_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variA","variB","cosine","l2_rmse","mae","snr_db","zero_ratio","batches"
        ])
        w.writeheader()
        fin = logit_agg.finalize()
        w.writerow(dict(
            variA=vA, variB=vB,
            cosine=fin["cosine"], l2_rmse=fin["l2_rmse"], mae=fin["mae"],
            snr_db=fin["snr_db"], zero_ratio=fin["zero_ratio"],
            batches=fin["batches"]
        ))
            
        

    accA_pct = 100.0 * correctA / max(total, 1)
    accB_pct = 100.0 * correctB / max(total, 1)
    logger(f"[DONE] batches={args.analyze_batches}, samples={total}")
    logger(f"ACC({vA}bit)={accA_pct:.3f}%, ACC({vB}bit)={accB_pct:.3f}%")
    logger(f"Saved: {act_csv}")
    logger(f"Saved: {logit_csv}")


if __name__ == "__main__":
    main()
