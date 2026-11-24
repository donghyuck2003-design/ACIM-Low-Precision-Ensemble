import os
import csv
import math
import re
from datetime import datetime

RESULT_DIR = "./Result"

# meta에서 기본적으로 뽑고 싶은 키 (새 inference.py가 실제로 쓰는 키들 기준)
META_KEYS = [
    "total_inference_time",
    "--model",
    "--dataset",
    "FIXED_BASE_SEED",
    "--wl_weight",
    "--wl_activate",
    "--vari",
    "--num_models(N)",
    "--num_test_groups(k)",
    "--num_ensembles_per_group(M)",
    "valid_groups_for_ensemble",
    "total_ensembles_generated",
]

# meta에 자동으로 생기는 ensemble summary 키 패턴:
#   ensemble_pool6_x3_mean, ensemble_pool6_x3_std ...
ENSEMBLE_META_RE = re.compile(r"^ensemble_pool(?P<pool>\d+)_x(?P<x>\d+)_(?P<stat>mean|std)$")

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def basic_stats(vals):
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    return {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "count": n}

def top10_mean(vals):
    if not vals:
        return None
    s = sorted(vals, reverse=True)
    k = max(1, int(math.ceil(0.10 * len(s))))
    return sum(s[:k]) / k

def parse_run_csv(csv_path):
    """
    새 inference.py 포맷 지원:
    - meta: # key, value
    - detail header: type,id,group,group_size,pool_size,ensemble_size,...
      ensemble_size == x (3/4/5)
    Returns:
      meta: dict
      individuals: list[float]
      ens_by_x: dict[int, list[float]]
      derived_pool: pool_size (int or None)
      x_values_seen: set[int]
    """
    meta = {"source_file": os.path.basename(csv_path)}
    individuals = []
    ens_by_x = {}     # x -> [accs]
    pool_size_seen = set()
    x_seen = set()

    in_meta = True
    in_detail = False
    header = None
    idx = {}

    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                if in_meta:
                    in_meta = False
                continue

            # meta
            if in_meta and row[0].startswith("#"):
                key = row[0].lstrip("#").strip()
                val = row[1].strip() if len(row) > 1 else ""
                meta[key] = val
                continue

            # detail header
            if row and row[0] == "type":
                in_detail = True
                header = row
                idx = {name: i for i, name in enumerate(header)}
                continue

            # detail rows
            if in_detail and header is not None:
                typ = row[idx["type"]].strip() if "type" in idx and len(row) > idx["type"] else ""
                if typ == "":
                    continue

                # acc column
                acc = safe_float(row[idx["accuracy"]], default=None) if "accuracy" in idx and len(row) > idx["accuracy"] else None
                if acc is None:
                    continue

                if typ == "individual":
                    individuals.append(acc)
                    continue

                if typ == "ensemble":
                    # ensemble_size == x
                    x = None
                    if "ensemble_size" in idx and len(row) > idx["ensemble_size"]:
                        x = safe_float(row[idx["ensemble_size"]], default=None)
                    if x is None:
                        continue
                    x = int(x)
                    x_seen.add(x)

                    # pool_size
                    if "pool_size" in idx and len(row) > idx["pool_size"]:
                        ps = safe_float(row[idx["pool_size"]], default=None)
                        if ps is not None:
                            pool_size_seen.add(int(ps))

                    ens_by_x.setdefault(x, []).append(acc)

    derived_pool = None
    if len(pool_size_seen) == 1:
        derived_pool = list(pool_size_seen)[0]

    # meta에 있는 ensemble_pool*_x*_mean/std도 자동 수집(있으면)
    ensemble_meta = {}
    for k, v in meta.items():
        m = ENSEMBLE_META_RE.match(k)
        if m:
            pool = int(m.group("pool"))
            x = int(m.group("x"))
            stat = m.group("stat")
            ensemble_meta.setdefault(pool, {}).setdefault(x, {})[stat] = v

    return meta, individuals, ens_by_x, derived_pool, x_seen, ensemble_meta

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def main():
    if not os.path.isdir(RESULT_DIR):
        raise RuntimeError(f"Result dir not found: {RESULT_DIR}")

    files = sorted([
        os.path.join(RESULT_DIR, fn)
        for fn in os.listdir(RESULT_DIR)
        if fn.endswith(".csv") and fn.startswith("accuracy_")
    ])

    if not files:
        print("No accuracy_*.csv found in ./Result")
        return

    merged_rows = []
    all_x_values = set()  # across runs
    all_pools = set()

    for fp in files:
        meta, individuals, ens_by_x, derived_pool, x_seen, ensemble_meta = parse_run_csv(fp)

        # run-level row
        row = {"source_file": meta.get("source_file", os.path.basename(fp))}

        # 기본 meta keys
        for k in META_KEYS:
            row[k] = meta.get(k, "")

        if derived_pool is not None:
            row["derived_pool_size"] = str(derived_pool)
            all_pools.add(derived_pool)
        else:
            row["derived_pool_size"] = ""

        # individual stats
        ind_st = basic_stats(individuals)
        row["individual_mean"] = f"{ind_st['mean']:.8f}" if ind_st else ""
        row["individual_std"]  = f"{ind_st['std']:.8f}"  if ind_st else ""
        row["individual_min"]  = f"{ind_st['min']:.8f}"  if ind_st else ""
        row["individual_max"]  = f"{ind_st['max']:.8f}"  if ind_st else ""
        row["individual_count"] = str(ind_st["count"]) if ind_st else "0"

        # ensemble stats by x (detail rows 기준)
        for x, vals in ens_by_x.items():
            all_x_values.add(x)
            st = basic_stats(vals)
            row[f"ensemble_x{x}_mean"] = f"{st['mean']:.8f}" if st else ""
            row[f"ensemble_x{x}_std"]  = f"{st['std']:.8f}"  if st else ""
            row[f"ensemble_x{x}_min"]  = f"{st['min']:.8f}"  if st else ""
            row[f"ensemble_x{x}_max"]  = f"{st['max']:.8f}"  if st else ""
            row[f"ensemble_x{x}_top10_mean"] = f"{top10_mean(vals):.8f}" if st else ""
            row[f"ensemble_x{x}_count"] = str(st["count"]) if st else "0"

        # (옵션) meta summary도 같이 보관하고 싶으면: ensemble_meta를 flatten
        # 예: meta summary가 존재하면 row["meta_ensemble_pool6_x3_mean"]=...
        for pool, xdict in ensemble_meta.items():
            all_pools.add(pool)
            for x, sdict in xdict.items():
                all_x_values.add(x)
                if "mean" in sdict:
                    row[f"meta_ensemble_pool{pool}_x{x}_mean"] = sdict["mean"]
                if "std" in sdict:
                    row[f"meta_ensemble_pool{pool}_x{x}_std"] = sdict["std"]

        merged_rows.append(row)

    # merged header 구성: x가 가변이므로 동적으로 컬럼 생성
    base_cols = ["source_file"] + META_KEYS + ["derived_pool_size",
                                              "individual_mean","individual_std","individual_min","individual_max","individual_count"]

    x_cols = []
    for x in sorted(all_x_values):
        x_cols += [
            f"ensemble_x{x}_mean", f"ensemble_x{x}_std", f"ensemble_x{x}_min", f"ensemble_x{x}_max",
            f"ensemble_x{x}_top10_mean", f"ensemble_x{x}_count",
        ]

    meta_ens_cols = []
    for pool in sorted(all_pools):
        for x in sorted(all_x_values):
            meta_ens_cols += [f"meta_ensemble_pool{pool}_x{x}_mean", f"meta_ensemble_pool{pool}_x{x}_std"]

    merged_header = base_cols + x_cols + meta_ens_cols

    merged_out = os.path.join(
        RESULT_DIR,
        f"merged_results_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
    )
    write_csv(merged_out, merged_header, merged_rows)

    # Pivot by (w,a,v): x별 mean을 평균(여러 run이 있을 때)
    def key_wav(r):
        return (str(r.get("--wl_weight","")), str(r.get("--wl_activate","")), str(r.get("--vari","")))

    pivot_metrics = ["individual_mean", "individual_std"] + [f"ensemble_x{x}_mean" for x in sorted(all_x_values)]
    agg = {}
    for r in merged_rows:
        wav = key_wav(r)
        agg.setdefault(wav, {m: [] for m in pivot_metrics})
        for m in pivot_metrics:
            val = safe_float(r.get(m, ""), default=None)
            if val is not None:
                agg[wav][m].append(val)

    pivot_rows = []
    for (w, a, v), md in sorted(
        agg.items(),
        key=lambda x: (safe_float(x[0][0],0), safe_float(x[0][1],0), safe_float(x[0][2],0))
    ):
        out = {"wl_weight": w, "wl_activate": a, "vari": v}
        for m in pivot_metrics:
            vals = md[m]
            out[m] = f"{(sum(vals)/len(vals)):.8f}" if vals else ""
        out["num_runs"] = str(max(len(md["individual_mean"]), 0))
        pivot_rows.append(out)

    pivot_header = ["wl_weight", "wl_activate", "vari", "num_runs"] + pivot_metrics
    pivot_out = os.path.join(RESULT_DIR, f"pivot_by_wav_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
    write_csv(pivot_out, pivot_header, pivot_rows)

    # Long format
    long_rows = []
    for pr in pivot_rows:
        for m in pivot_metrics:
            long_rows.append({
                "wl_weight": pr["wl_weight"],
                "wl_activate": pr["wl_activate"],
                "vari": pr["vari"],
                "metric": m,
                "value": pr.get(m, ""),
                "num_runs": pr.get("num_runs", "0")
            })
    long_header = ["wl_weight", "wl_activate", "vari", "metric", "value", "num_runs"]
    long_out = os.path.join(RESULT_DIR, f"pivot_by_wav_multi_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
    write_csv(long_out, long_header, long_rows)

    print(f"Saved merged CSV: {merged_out} (rows={len(merged_rows)})")
    print(f"Saved pivot  CSV: {pivot_out} (rows={len(pivot_rows)})")
    print(f"Saved long   CSV: {long_out} (rows={len(long_rows)})")

if __name__ == "__main__":
    main()
