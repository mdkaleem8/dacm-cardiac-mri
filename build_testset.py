"""Build a fixed, reproducible degraded test set.

Conditions: clean, and each degradation type at target severities
LEVELS (all slices of both ED and ES frames of each test subject degraded with
the same type and target severity, independent random draws per slice).
Optionally, unseen three-type combinations for the transfer experiment.
Each condition is saved as one .npz with float16 unit-range images and the
measured per-type severity of every slice (used for oracle conditioning).
"""
import argparse
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm import degradations as deg  # noqa: E402
from dacm.data import load_splits, load_subject, to_unit  # noqa: E402

LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
COMBOS = [("motion", "noise", "bias"), ("motion", "blur", "undersample"), ("noise", "blur", "bias")]


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def degrade_subject(args):
    cache, pid, cond, spec_targets, calib = args
    sub = load_subject(Path(cache), pid)
    out = {}
    for ph in ("ED", "ES"):
        img, gt, lohi = sub[f"img_{ph}"], sub[f"gt_{ph}"], sub[f"range_{ph}"]
        xs, ts, ss = [], [], []
        for z in range(img.shape[0]):
            x = to_unit(img[z], lohi)
            rng = np.random.default_rng(seed_of(cond, pid, ph, z))
            spec = [(k, deg.r_for_severity(calib, k, st)) for k, st in spec_targets]
            xd, t, s = deg.apply_with_severity(x, spec, rng, calib, gt[z])
            xs.append(xd.astype(np.float16)); ts.append(t); ss.append(s)
        out[f"{pid}_{ph}_x"] = np.stack(xs)
        out[f"{pid}_{ph}_t"] = np.stack(ts)
        out[f"{pid}_{ph}_s"] = np.stack(ss)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--combos", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    calib = json.loads(Path(a.calib).read_text())
    pids = load_splits(Path(a.cache))[a.split]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conds = {"clean_0.0": []}
    for k in deg.TYPES:
        for lv in LEVELS:
            conds[f"{k}_{lv:.1f}"] = [(k, lv)]
    if a.combos:
        for c in COMBOS:
            conds["combo-" + "+".join(c) + "_0.6"] = [(k, 0.6) for k in c]
    with ProcessPoolExecutor(a.workers) as ex:
        for cond, targets in conds.items():
            f = out / f"{cond}.npz"
            if f.exists():
                continue
            parts = ex.map(degrade_subject, [(a.cache, pid, cond, targets, calib) for pid in pids])
            merged = {}
            for p in parts:
                merged.update(p)
            np.savez(f, **merged)
            print("wrote", f.name, flush=True)
    (out / "conditions.json").write_text(json.dumps({"pids": pids, "conditions": list(conds)}, indent=1))


if __name__ == "__main__":
    main()
