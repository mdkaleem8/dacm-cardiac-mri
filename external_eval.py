"""Shared evaluation of trained models on an external, already-cached dataset.

The index (a DataFrame) has one row per volume with columns case, scan, phase (ED/ES),
labelled, and any metadata columns (listed in `meta`) that are copied into the outputs.
Each cached volume <case>.npz holds x (unit-range slices, 224 x 224), spacing
(z, y, x) and, when labelled, gt in the ACDC coding (1 RV, 2 Myo, 3 LV).

Outputs in `out`:
  <run>.csv        per volume and structure: Dice and HD95 (DACM runs also with z = 0)
  <run>_clin.csv   per scan with both phases labelled: predicted and reference indices
  estimator.csv    per volume: mean (and max) estimated probability and severity of each type
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dacm import degradations as deg  # noqa: E402
from dacm.data import zscore  # noqa: E402
from dacm.metrics import CLASSES, clinical, dice, hd95  # noqa: E402
from dacm.models import DegradationEstimator  # noqa: E402
from evaluate import load_run, predict  # noqa: E402


def run_eval(idx, cache, est_dir, runs, out, meta, dev, clin_meta=None):
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    est = DegradationEstimator(deg.K).to(dev).eval()
    est.load_state_dict(torch.load(Path(est_dir) / "estimator.pt", map_location=dev))
    vols = {}
    for r in idx.itertuples():
        d = np.load(cache / f"{r.case}.npz")
        X = np.stack([zscore(s) for s in d["x"]])[:, None]
        with torch.no_grad():
            Z = torch.cat([est.conditioning(torch.from_numpy(X[i:i + 64]).to(dev)).float()
                           for i in range(0, len(X), 64)])
        vols[r.case] = (X, Z, d["gt"] if "gt" in d.files else None, tuple(float(v) for v in d["spacing"]))

    ef = out / "estimator.csv"
    if not ef.exists():
        rows = []
        for r in idx.itertuples():
            Z = vols[r.case][1].cpu().numpy()
            rows.append({"case": r.case, **{c: getattr(r, c) for c in meta},
                         **{f"p_{k}": float(Z[:, i].mean()) for i, k in enumerate(deg.TYPES)},
                         **{f"sev_{k}": float(Z[:, deg.K + i].mean()) for i, k in enumerate(deg.TYPES)},
                         **{f"p_{k}_max": float(Z[:, i].max()) for i, k in enumerate(deg.TYPES)},
                         **{f"sev_{k}_max": float(Z[:, deg.K + i].max()) for i, k in enumerate(deg.TYPES)}})
        pd.DataFrame(rows).to_csv(ef, index=False)

    lab = idx[idx.labelled]
    by_case = lab.set_index("case")
    for run in map(Path, runs):
        f = out / f"{run.name}.csv"
        if f.exists() or not (run / "model.pt").exists():
            continue
        model, args = load_run(run, dev)
        variants = ["est"]
        if args["mode"] == "dacm" and not args.get("zero_cond"):
            variants.append("zero")
        rows, crow, segs = [], [], {}
        for v in variants:
            for r in lab.itertuples():
                X, Z, gt, sp = vols[r.case]
                z = torch.zeros_like(Z) if (v == "zero" or args.get("zero_cond")) else Z
                pred = predict(model, args, X, z, dev)
                segs[(v, r.case)] = (pred, gt, sp)
                for c, cname in CLASSES.items():
                    rows.append({"run": run.name, "variant": v, "case": r.case, "phase": r.phase,
                                 **{m: getattr(r, m) for m in meta}, "cls": cname,
                                 "dice": dice(pred == c, gt == c), "hd95": hd95(pred == c, gt == c, sp)})
            for scan in sorted(set(lab.scan)):
                ed, es = (v, f"{scan}-ED"), (v, f"{scan}-ES")
                if ed in segs and es in segs:
                    sp = segs[ed][2]
                    m = {c: by_case.loc[ed[1], c] for c in meta}
                    if clin_meta:
                        m.update(clin_meta(by_case.loc[ed[1]], by_case.loc[es[1]]))
                    crow.append({"run": run.name, "variant": v, "scan": scan, **m,
                                 **{f"pred_{k}": x for k, x in clinical(segs[ed][0], segs[es][0], sp).items()},
                                 **{f"ref_{k}": x for k, x in clinical(segs[ed][1], segs[es][1], sp).items()}})
        pd.DataFrame(crow).to_csv(out / f"{run.name}_clin.csv", index=False)
        pd.DataFrame(rows).to_csv(f, index=False)
        print("done", run.name, flush=True)
        del model
        torch.cuda.empty_cache()
