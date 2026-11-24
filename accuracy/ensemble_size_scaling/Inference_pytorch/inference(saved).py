import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import VGG
import argparse
import os
from utee import misc
import random
import numpy as np
from models import dataset
from utee import make_path
from datetime import datetime
import time
import csv


# =============================
# FIXED SEED (NO CLI ARG)
# =============================
FIXED_BASE_SEED = 1  # 고정 seed


# -----------------------------
# Utils
# -----------------------------
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def run_inference_get_probs(model, test_loader, device):
    """
    Returns:
        probs_all: torch.FloatTensor [num_samples, num_classes] on CPU
        targets_all: torch.LongTensor [num_samples] on CPU
        acc: float
        correct: int
        total: int
    """
    model.eval()

    probs_chunks = []
    targets_chunks = []

    for _, (data, target) in enumerate(test_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(data)
        probs = F.softmax(output, dim=1)

        probs_chunks.append(probs.detach().cpu())
        targets_chunks.append(target.detach().cpu())

    probs_all = torch.cat(probs_chunks, dim=0)
    targets_all = torch.cat(targets_chunks, dim=0)

    pred = probs_all.argmax(dim=1)
    correct = int((pred == targets_all).sum().item())
    total = int(targets_all.numel())
    acc = correct / max(1, total)

    return probs_all, targets_all, acc, correct, total


def build_model(args, logger, model_path):
    if args.model.upper() == "VGG8":
        return VGG.vgg8(args=args, logger=logger, pretrained=model_path)
    else:
        raise ValueError(f"Unsupported model: {args.model}. Please extend build_model().")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def split_indices_evenly(num_samples: int, k: int):
    idx = np.arange(num_samples)
    chunks = np.array_split(idx, k)
    return [c.astype(np.int64).tolist() for c in chunks]


def make_subset_loader(test_dataset, indices, batch_size, num_workers, pin_memory):
    subset = Subset(test_dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,     # deterministic order for ensemble
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )


def stats(arr):
    if len(arr) == 0:
        return None
    a = np.array(arr, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


# def weighted_mean_from_counts(correct_list, total_list):
#     if len(correct_list) == 0:
#         return None
#     c = int(np.sum(correct_list))
#     t = int(np.sum(total_list))
#     return (c / max(1, t))


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description='NeuroSim CIM Accuracy Inference (k=2 ensemble only, M per group)')

parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='VGG8', help='VGG8|DenseNet40|ResNet18')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')

parser.add_argument('--batch_size', type=int, default=200)

parser.add_argument('--wl_weight', type=int, default=8)
parser.add_argument('--wl_grad', type=int, default=8)
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)

# Hardware Properties
parser.add_argument('--inference', type=int, default=1)
parser.add_argument('--subArray', type=int, default=128)
parser.add_argument('--parallelRead', type=int, default=128)
parser.add_argument('--ADCprecision', type=int, default=5)
parser.add_argument('--cellBit', type=int, default=1)
parser.add_argument('--onoffratio', type=float, default=10)
parser.add_argument('--vari', type=float, default=0.)
parser.add_argument('--t', type=float, default=0)
parser.add_argument('--v', type=float, default=0)
parser.add_argument('--detect', type=int, default=0)
parser.add_argument('--target', type=float, default=0)

# Logging (force to ./inference_log)
parser.add_argument('--logdir', default='./inference_log')

# N, M (k=2 only)
parser.add_argument('--num_models', type=int, default=150, help='N: number of individual models')
parser.add_argument('--num_ensembles', type=int, default=3, help='M: ensembles PER GROUP (k=2 only)')

# split testset into k chunks
parser.add_argument('--num_test_groups', type=int, default=50, help='k: split test dataset into k groups')

args = parser.parse_args()

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

run_tag = f"w{args.wl_weight}_a{args.wl_activate}_v{args.vari}_{current_time}"
args.logdir = os.path.join(args.logdir, run_tag)

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args, ['logdir'])

# -----------------------------
# Logger
# -----------------------------
misc.logger.init(args.logdir, 'test_log_' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger(f'{k}: {v}')
logger("========================================")
logger(f"FIXED_BASE_SEED (internal): {FIXED_BASE_SEED}")

# -----------------------------
# Device
# -----------------------------
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
logger(f"Device: {device}")

# -----------------------------
# Data
# -----------------------------
num_workers = 1
pin_memory = bool(args.cuda)

if args.dataset.lower() == "cifar10":
    train_loader, test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=num_workers)
elif args.dataset.lower() == "cifar100":
    train_loader, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=num_workers)
elif args.dataset.lower() == "imagenet":
    train_loader, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=4)
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

test_dataset = test_loader.dataset
num_test = len(test_dataset)

# -----------------------------
# Split test dataset into k groups
# -----------------------------
k_groups = int(args.num_test_groups)
k_groups = max(1, k_groups)
if k_groups > num_test:
    k_groups = num_test

group_indices = split_indices_evenly(num_test, k_groups)
group_sizes = [len(g) for g in group_indices]
logger(f"Split testset: total={num_test}, k={k_groups}, sizes={group_sizes}")

group_loaders = [
    make_subset_loader(
        test_dataset,
        indices=group_indices[g],
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    for g in range(k_groups)
]

# -----------------------------
# Main
# -----------------------------
t_begin = time.time()

N = int(args.num_models)
M = int(args.num_ensembles)  # PER GROUP

model_path = './log/VGG8.pth'

# per model
all_model_probs = [None] * N
all_model_targets = [None] * N
all_model_seeds = [None] * N
all_model_group = [None] * N
individual_accs = [None] * N
individual_correct = [None] * N
individual_total = [None] * N

group_to_models = {g: [] for g in range(k_groups)}

logger(f"========== Running {N} individual models (each on 1 group only) ==========")

for i in range(N):
    grp = i % k_groups
    group_to_models[grp].append(i)

    model_seed = int(FIXED_BASE_SEED) + i
    all_model_seeds[i] = model_seed
    all_model_group[i] = grp

    set_all_seeds(model_seed)

    model = build_model(args, logger, model_path)
    model.to(device)
    model.eval()

    probs_i, targets_i, acc_i, corr_i, tot_i = run_inference_get_probs(model, group_loaders[grp], device)

    all_model_probs[i] = probs_i
    all_model_targets[i] = targets_i
    individual_accs[i] = acc_i
    individual_correct[i] = corr_i
    individual_total[i] = tot_i

    logger(f"[Individual {i+1}/{N}] group={grp} size={tot_i} seed={model_seed} acc={acc_i:.6f}")

# -----------------------------
# Ensembles: k=2 only, "M per group"
#   if vari == 0, skip ensemble inference
# -----------------------------
ensemble2_accs = []
ensemble2_choices = []
ensemble2_correct = []
ensemble2_total = []

do_ensemble = (float(args.vari) > 0.0)

if not do_ensemble:
    logger(f"[Skip] Ensemble inference because vari={args.vari} (==0).")
else:
    valid_groups = [g for g in range(k_groups) if group_sizes[g] > 0 and len(group_to_models[g]) >= 2]
    logger(f"Valid groups for k=2 ensemble: {valid_groups} (each will sample M={M})")

    for g in valid_groups:
        candidates = group_to_models[g]
        rng_g = np.random.default_rng(int(FIXED_BASE_SEED) + 1000 * g)

        n = len(candidates)
        max_unique = n * (n - 1) // 2
        if M > max_unique:
            raise RuntimeError(
                f"[Group {g}] Cannot create M={M} unique pairs with only {n} models. "
                f"Max unique pairs = C({n},2) = {max_unique}. "
                f"Reduce M or increase num_models per group."
            )

        all_pairs = []
        for a in range(n):
            for b in range(a + 1, n):
                i1 = candidates[a]
                i2 = candidates[b]
                if i1 > i2:
                    i1, i2 = i2, i1
                all_pairs.append((i1, i2))

        perm = rng_g.permutation(len(all_pairs))

        logger(f"========== Ensembles (k=2) group={g} candidates={n} : unique_pairs={max_unique}, M={M} ==========")

        for m, pidx in enumerate(perm[:M]):
            i1, i2 = all_pairs[int(pidx)]

            avg_prob = (all_model_probs[i1] + all_model_probs[i2]) / 2.0
            targets_ref = all_model_targets[i1]
            pred = avg_prob.argmax(dim=1)

            corr = int((pred == targets_ref).sum().item())
            tot = int(targets_ref.numel())
            acc = corr / max(1, tot)

            ensemble2_choices.append((g, m, i1, i2))
            ensemble2_accs.append(acc)
            ensemble2_correct.append(corr)
            ensemble2_total.append(tot)

            if (m + 1) % max(1, (M // 5)) == 0:
                logger(f"[Ensemble2 g={g} {m+1}/{M}] choice=({i1},{i2}) acc={acc:.6f}")

# -----------------------------
# Timing + Summary
# -----------------------------
t_end = time.time()
elapsed = t_end - t_begin
elapsed_hms = str(datetime.fromtimestamp(elapsed) - datetime.fromtimestamp(0))

ind_stats = stats(individual_accs)
e2_stats = stats(ensemble2_accs)
#e2_weighted_mean = weighted_mean_from_counts(ensemble2_correct, ensemble2_total) if ensemble2_accs else None

logger("========== Summary ==========")
logger(f"Total inference time: {elapsed_hms} (hh:mm:ss)")
logger(f"Individual acc: mean={ind_stats['mean']:.6f}, std={ind_stats['std']:.6f}")

if e2_stats:
    #logger(f"Ensemble2 acc: mean={e2_stats['mean']:.6f}, std={e2_stats['std']:.6f}, weighted_mean={e2_weighted_mean:.6f}")
    logger(f"Ensemble2 acc: mean={e2_stats['mean']:.6f}, std={e2_stats['std']:.6f}")
    logger(f"Total ensembles generated (k=2): {len(ensemble2_accs)}")
else:
    logger("Ensemble2 acc: skipped or empty.")

# -----------------------------
# CSV Save
# -----------------------------
result_dir = "./Result"
ensure_dir(result_dir)

csv_name = (
    f"accuracy_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    f"_w{args.wl_weight}_a{args.wl_activate}_v{args.vari}"
    f"_N{N}_K{k_groups}_MperG{M}.csv"
)
csv_path = os.path.join(result_dir, csv_name)

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)

    # meta
    w.writerow(["# total_inference_time", elapsed_hms])
    w.writerow(["# --model", args.model])
    w.writerow(["# --dataset", args.dataset])
    w.writerow(["# FIXED_BASE_SEED", FIXED_BASE_SEED])
    w.writerow(["# --wl_weight", args.wl_weight])
    w.writerow(["# --wl_activate", args.wl_activate])
    w.writerow(["# --vari", args.vari])
    w.writerow(["# --num_models(N)", N])
    w.writerow(["# --num_test_groups(k)", k_groups])
    w.writerow(["# --num_ensembles_per_group(M)", M])
    valid_groups_str = "|".join(map(str, valid_groups)) if (do_ensemble and 'valid_groups' in locals()) else ""
    w.writerow(["# valid_groups_for_ensemble", valid_groups_str])
    w.writerow(["# total_ensembles_generated", str(len(ensemble2_accs))])

    if e2_stats:
        w.writerow(["# ensemble_k2_mean", f"{e2_stats['mean']:.8f}"])
        #w.writerow(["# ensemble_k2_weighted_mean", f"{e2_weighted_mean:.8f}"])

    w.writerow([])

    # summary
    w.writerow(["section", "mean", "std", "min", "max"])#, "weighted_mean"])
    w.writerow(["individual",
                f"{ind_stats['mean']:.8f}", f"{ind_stats['std']:.8f}",
                f"{ind_stats['min']:.8f}", f"{ind_stats['max']:.8f}"])
    if e2_stats:
        w.writerow(["ensemble_k2",
                    f"{e2_stats['mean']:.8f}", f"{e2_stats['std']:.8f}",
                    f"{e2_stats['min']:.8f}", f"{e2_stats['max']:.8f}"]) #,f"{e2_weighted_mean:.8f}"])

    w.writerow([])

    # detailed rows
    w.writerow(["type", "id", "group", "group_size", "ensemble_size",
                "model_indices", "model_seeds", "accuracy", "correct", "total", "m_in_group"])

    # individuals
    for i in range(N):
        g = all_model_group[i]
        w.writerow([
            "individual", i, g, group_sizes[g], 1,
            str(i), str(all_model_seeds[i]),
            f"{individual_accs[i]:.8f}", individual_correct[i], individual_total[i], ""
        ])

    # ensembles (k=2)
    for eid, (choice, acc, corr, tot) in enumerate(zip(ensemble2_choices, ensemble2_accs, ensemble2_correct, ensemble2_total)):
        g, m_in_group, i1, i2 = choice
        seeds = (all_model_seeds[i1], all_model_seeds[i2])
        w.writerow([
            "ensemble", eid, g, group_sizes[g], 2,
            f"{i1}|{i2}", f"{seeds[0]}|{seeds[1]}",
            f"{acc:.8f}", corr, tot, m_in_group
        ])

logger(f"CSV saved: {csv_path}")
logger("Done.")
