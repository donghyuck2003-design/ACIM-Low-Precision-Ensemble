#!/usr/bin/env python3

import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPO = "donghyuck2003-design/ACIM-Low-Precision-Ensemble"
TAG = "v1.0.0"

parser = argparse.ArgumentParser()
parser.add_argument("--local-dir", type=Path)
args = parser.parse_args()

CACHE = (
    args.local_dir.resolve()
    if args.local_dir
    else ROOT / ".checkpoint_cache"
)

CACHE.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

with (ROOT / "docs/checkpoints.csv").open() as f:
    rows = list(csv.DictReader(f))

verified = {}

for row in rows:
    relative = row["expected_relative_path"]
    expected = row["sha256"]

    dst = ROOT / relative
    name = dst.name
    src = CACHE / name

    if name in verified and verified[name] != expected:
        raise RuntimeError(f"Conflicting hashes: {name}")

    if name not in verified:
        if not src.exists():
            if args.local_dir:
                raise FileNotFoundError(src)

            subprocess.run([
                "gh", "release", "download", TAG,
                "--repo", REPO,
                "--pattern", name,
                "--dir", str(CACHE)
            ], check=True)

        if sha256(src) != expected:
            raise RuntimeError(f"Checksum failed: {src}")

        verified[name] = expected

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if sha256(dst) != expected:
            raise RuntimeError(
                f"Refusing to overwrite: {dst}"
            )
    else:
        shutil.copy2(src, dst)

    print(f"[PASS] {relative}")

print(f"\nInstalled {len(rows)} checkpoint paths.")
