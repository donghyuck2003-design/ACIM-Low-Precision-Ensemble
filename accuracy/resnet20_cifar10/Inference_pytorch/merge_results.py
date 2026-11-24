import os
import csv
import math
from datetime import datetime

RESULT_DIR = "./Result"

META_KEYS = [
    "total_inference_time",
    "--model",
    "--dataset",
    "FIXED_BASE_SEED",
    "--wl_weight",
    "--wl_activate",
    "--vari",
    "--num_models(N)",
    "--num_ensembles(M)",
    "--num_test_groups(k)",
    "ensemble_k2_mean",
    "ensemble_k2_weighted_mean",
]

DERIVED_KEYS = [
    "individual_mean", "individual_std", "individual_min", "individual_max",
    "ensemble2_min", "ensemble2_max", "ensemble2_top10_mean", "ensemble2_count",
]

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

def parse_run_csv(csv_path):
    meta = {"source_file": os.path.basename(csv_path)}
    individuals = []
    ens2 = []

    in_meta = True
    in_detail = False
    header = None

    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                if in_meta:
                    in_meta = False
                continue

            if in_meta and row[0].startswith("#"):
                key = row[0].lstrip("#").strip()
                val = row[1].strip() if len(row) > 1 else ""
                meta[key] = val
                continue

            if row and row[0] == "type":
                in_detail = True
                header = row
                continue

            if in_detail and header is not None:
                if len(row) < 8:
                    continue
                typ = row[0].strip()
                ens_size = row[4].strip() if len(row) > 4 else ""
                acc = safe_float(row[7], default=None)
                if acc is None:
                    continue

                if typ == "individual":
                    individuals.append(acc)
                elif typ == "ensemble" and ens_size == "2":
                    ens2.append(acc)

    return meta, individuals, ens2

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

    for fp in files:
        meta, individuals, ens2 = parse_run_csv(fp)
        row = {"source_file": meta.get("source_file", "")}

        for k in META_KEYS:
            row[k] = meta.get(k, "")

        ind_st = basic_stats(individuals)
        if ind_st:
            row["individual_mean"] = f"{ind_st['mean']:.8f}"
            row["individual_std"] = f"{ind_st['std']:.8f}"
            row["individual_min"] = f"{ind_st['min']:.8f}"
            row["individual_max"] = f"{ind_st['max']:.8f}"
        else:
            row["individual_mean"] = ""
            row["individual_std"] = ""
            row["individual_min"] = ""
            row["individual_max"] = ""

        e2_st = basic_stats(ens2)
        if e2_st:
            row["ensemble2_min"] = f"{e2_st['min']:.8f}"
            row["ensemble2_max"] = f"{e2_st['max']:.8f}"
            row["ensemble2_top10_mean"] = f"{top10_mean(ens2):.8f}"
            row["ensemble2_count"] = str(e2_st["count"])
        else:
            row["ensemble2_min"] = ""
            row["ensemble2_max"] = ""
            row["ensemble2_top10_mean"] = ""
            row["ensemble2_count"] = "0"

        merged_rows.append(row)

    merged_header = ["source_file"] + META_KEYS + DERIVED_KEYS
    merged_out = os.path.join(RESULT_DIR, f"merged_results_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
    write_csv(merged_out, merged_header, merged_rows)

    # Pivot by (w,a,v)
    def key_wav(r):
        return (str(r.get("--wl_weight","")), str(r.get("--wl_activate","")), str(r.get("--vari","")))

    METRICS = [
        "individual_mean",
        "individual_std",
        "ensemble_k2_mean",
        "ensemble_k2_weighted_mean",
        "ensemble2_top10_mean",
    ]

    agg = {}
    for r in merged_rows:
        wav = key_wav(r)
        if wav not in agg:
            agg[wav] = {m: [] for m in METRICS}
        for m in METRICS:
            val = safe_float(r.get(m, ""), default=None)
            if val is not None:
                agg[wav][m].append(val)

    pivot_rows = []
    for (w, a, v), md in sorted(agg.items(), key=lambda x: (safe_float(x[0][0],0), safe_float(x[0][1],0), safe_float(x[0][2],0))):
        out = {"wl_weight": w, "wl_activate": a, "vari": v}
        for m in METRICS:
            vals = md[m]
            out[m] = f"{(sum(vals)/len(vals)):.8f}" if vals else ""
        out["num_runs"] = str(len(md["individual_mean"])) if md["individual_mean"] else "0"
        pivot_rows.append(out)

    pivot_header = ["wl_weight", "wl_activate", "vari", "num_runs"] + METRICS
    pivot_out = os.path.join(RESULT_DIR, f"pivot_by_wav_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
    write_csv(pivot_out, pivot_header, pivot_rows)

    # Long format
    long_rows = []
    for pr in pivot_rows:
        for m in METRICS:
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
