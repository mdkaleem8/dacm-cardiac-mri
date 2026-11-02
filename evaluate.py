"""Evaluate trained runs on the fixed degraded test set.

For each run and condition: per-volume Dice and HD95 (RV, Myo, LV) for ED and
ES, and clinical indices (LV/RV volumes, EF, LV mass) from predicted and
reference segmentations. DACM runs are additionally evaluated with
alternative conditioning inputs (E2): oracle, shuffled, and noisy estimates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm import degradations as deg  # noqa: E402
from dacm.conditioning import DACM  # noqa: E402
from dacm.data import load_subject, zscore  # noqa: E402
from dacm.metrics import CLASSES, clinical, dice, hd95  # noqa: E402
from dacm.models import DegradationEstimator, build_backbone  # noqa: E402


def load_run(run: Path, dev):
    args = json.loads((run / "args.json").read_text())
    backbone, stages = build_backbone(args["backbone"], norm_affine=args.get("norm_affine", False))
    if args["mode"] == "dacm":
        sel = stages if args.get("stages", "all") == "all" else \
            [n for n in stages if n.startswith(("enc", "encoder")) == (args["stages"] == "encoder")]
        if args.get("placement", "stage") == "norm":
            sel = "norm"
        model = DACM(backbone, sel, cond_dim=2 * deg.K, mechanism=args.get("mechanism", "film"),
                     cond_part=args.get("cond_part", "both"))
    else:
        model = backbone
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.to(dev).eval(), args


@torch.no_grad()
def predict(model, args, x, z, dev, bs=32):
    preds = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(xb, z[i:i + bs]) if args["mode"] == "dacm" else model(xb)
        preds.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--e2", action="store_true", help="alternative conditioning inputs for DACM runs")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--tag", default="", help="suffix of the output files, e.g. _e2 -> eval_seg_e2.csv")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = Path(a.testset)
    meta = json.loads((ts / "conditions.json").read_text())
    pids, conds = meta["pids"], a.conditions or meta["conditions"]
    est = DegradationEstimator(deg.K).to(dev).eval()
    est.load_state_dict(torch.load(Path(a.estimator) / "estimator.pt", map_location=dev))
    subs = {pid: load_subject(Path(a.cache), pid) for pid in pids}

    for run in map(Path, a.runs):
        model, args = load_run(run, dev)
        rows, crow = [], []
        for cond in conds:
            data = np.load(ts / f"{cond}.npz")
            dtype, level = cond.rsplit("_", 1)
            # Gather all slices of this condition to compute estimates once.
            keys = [(pid, ph) for pid in pids for ph in ("ED", "ES")]
            X = {k: np.stack([zscore(s.astype(np.float32)) for s in data[f"{k[0]}_{k[1]}_x"]])[:, None] for k in keys}
            Z_est = {k: torch.cat([est.conditioning(torch.from_numpy(X[k][i:i + 64]).to(dev)).float()
                                   for i in range(0, len(X[k]), 64)]) for k in keys}
            if args.get("zero_cond"):  # control run: always condition on zeros
                Z_est = {k: torch.zeros_like(v) for k, v in Z_est.items()}
            variants = {"est": Z_est}
            if args["mode"] == "dacm" and a.e2 and not args.get("zero_cond"):
                Z_or = {k: torch.from_numpy(np.concatenate([data[f"{k[0]}_{k[1]}_t"],
                                                            data[f"{k[0]}_{k[1]}_t"] * data[f"{k[0]}_{k[1]}_s"]], 1)).to(dev)
                        for k in keys}
                allz = torch.cat([Z_est[k] for k in keys])
                perm = torch.from_numpy(np.random.default_rng(0).permutation(len(allz))).to(dev)
                sh, i0 = allz[perm], 0
                Z_sh = {}
                for k in keys:
                    n = len(Z_est[k]); Z_sh[k] = sh[i0:i0 + n]; i0 += n
                variants.update({"oracle": Z_or, "shuffled": Z_sh})
                for sig in (0.1, 0.2, 0.3):
                    g = torch.Generator(device=dev).manual_seed(int(sig * 100))
                    Z_n = {}
                    for k in keys:
                        z = Z_est[k].clone()
                        z[:, deg.K:] = (z[:, deg.K:] + sig * torch.randn(z[:, deg.K:].shape, device=dev, generator=g)).clamp(0, 1)
                        Z_n[k] = z
                    variants[f"noise{sig:.1f}"] = Z_n
            if args["mode"] != "dacm":
                variants = {"est": Z_est}  # unconditioned models ignore z
            for vname, Z in variants.items():
                for pid in pids:
                    sub, spacing = subs[pid], tuple(float(v) for v in subs[pid]["spacing"])
                    segs = {}
                    for ph in ("ED", "ES"):
                        pred = predict(model, args, X[(pid, ph)], Z[(pid, ph)], dev)
                        gt = sub[f"gt_{ph}"]
                        segs[ph] = pred
                        for c, cname in CLASSES.items():
                            rows.append({"run": run.name, "cond": cond, "type": dtype, "level": float(level),
                                         "variant": vname, "pid": pid, "phase": ph, "cls": cname,
                                         "dice": dice(pred == c, gt == c), "hd95": hd95(pred == c, gt == c, spacing)})
                    cp = clinical(segs["ED"], segs["ES"], spacing)
                    cg = clinical(sub["gt_ED"], sub["gt_ES"], spacing)
                    crow.append({"run": run.name, "cond": cond, "type": dtype, "level": float(level),
                                 "variant": vname, "pid": pid,
                                 **{f"pred_{k}": v for k, v in cp.items()}, **{f"ref_{k}": v for k, v in cg.items()}})
            print(run.name, cond, flush=True)
        pd.DataFrame(rows).to_csv(run / f"eval_seg{a.tag}.csv", index=False)
        pd.DataFrame(crow).to_csv(run / f"eval_clinical{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
