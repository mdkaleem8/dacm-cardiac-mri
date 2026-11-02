"""Estimate kappa_k and the monotone map r -> median severity for each type.

Uses training slices only. Writes calib.json used by all other scripts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm import degradations as deg  # noqa: E402
from dacm.data import load_splits, load_subject, to_unit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_r", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cache = Path(a.cache)
    pids = load_splits(cache)["train"]
    slices = []
    for pid in pids:
        sub = load_subject(cache, pid)
        for ph in ("ED", "ES"):
            img, gt, lohi = sub[f"img_{ph}"], sub[f"gt_{ph}"], sub[f"range_{ph}"]
            for z in range(img.shape[0]):
                if gt[z].any():
                    slices.append((to_unit(img[z], lohi), gt[z]))
    print(f"{len(slices)} labelled training slices")
    r_grid = np.linspace(0, 1, 21)
    calib = {}
    for k in deg.TYPES:
        raw = np.zeros((len(r_grid), a.n_per_r))
        for i, r in enumerate(r_grid):
            for j in range(a.n_per_r):
                x, y = slices[rng.integers(len(slices))]
                xd = deg.FUNCS[k](x, float(r), rng)
                raw[i, j] = deg.distortion(xd, x, deg.heart_box(y, x.shape))
        kappa = float(np.median(raw[-1]))  # median distortion at maximum strength -> s = 1
        s_med = np.minimum(1.0, np.median(raw, axis=1) / kappa)
        s_med = np.maximum.accumulate(s_med)  # enforce monotonicity
        s_med = s_med + np.arange(len(s_med)) * 1e-6  # strictly increasing for interp
        calib[k] = {"kappa": kappa, "r_grid": r_grid.tolist(), "s_median": s_med.tolist()}
        print(f"{k:12s} kappa={kappa:.4f}  s(r=0.25,0.5,1)={s_med[5]:.2f},{s_med[10]:.2f},{s_med[20]:.2f}", flush=True)
    Path(a.out).write_text(json.dumps(calib, indent=1))


if __name__ == "__main__":
    main()
