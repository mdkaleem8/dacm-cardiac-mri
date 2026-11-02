"""Test-time adaptation baseline (Tent; Wang et al., ICLR 2021) for augmentation-only models.

For every test volume (one cardiac phase of one subject) the affine parameters
of all normalisation layers are adapted by minimising the mean pixel-wise
entropy of the softmax prediction for a few Adam steps, then the volume is
segmented; parameters are reset before the next volume (episodic adaptation).
Learning rate and number of steps are chosen on the degraded validation set
(--select); the chosen setting is then applied once to the test set.
Writes <runs_root>/<backbone>_augtent<seed>/eval_seg.csv and eval_clinical.csv.
"""
import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dacm.data import load_subject, zscore  # noqa: E402
from dacm.metrics import CLASSES, clinical, dice, hd95  # noqa: E402
from evaluate import load_run  # noqa: E402

NORMS = (nn.InstanceNorm2d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)
TYPES = ["motion", "noise", "blur", "bias", "undersample"]


def norm_params(model):
    ps = []
    for m in model.modules():
        if isinstance(m, NORMS):
            for p in (getattr(m, "weight", None), getattr(m, "bias", None)):
                if p is not None:
                    ps.append(p)
    return ps


def segment(model, x, dev, lr, steps, init):
    """x: (Z, 1, H, W) z-scored slices of one volume."""
    params = norm_params(model)
    with torch.no_grad():
        for p, p0 in zip(params, init):
            p.copy_(p0)
    xb = torch.from_numpy(x).to(dev)
    if steps > 0:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(xb)
            prob = torch.softmax(logits.float(), 1)
            loss = -(prob * torch.log(prob.clamp(min=1e-8))).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        return model(xb).argmax(1).cpu().numpy().astype(np.uint8)


def run_set(model, dev, ts, cache, conds, lr, steps, name, full=True):
    meta = json.loads((ts / "conditions.json").read_text())
    pids = meta["pids"]
    init = [p.detach().clone() for p in norm_params(model)]
    rows, crow = [], []
    for cond in conds:
        data = np.load(ts / f"{cond}.npz")
        dtype, level = cond.rsplit("_", 1)
        for pid in pids:
            sub = load_subject(cache, pid)
            spacing = tuple(float(v) for v in sub["spacing"])
            segs = {}
            for ph in ("ED", "ES"):
                x = np.stack([zscore(s.astype(np.float32)) for s in data[f"{pid}_{ph}_x"]])[:, None]
                seg = segment(model, x, dev, lr, steps, init)
                gt = sub[f"gt_{ph}"]
                segs[ph] = seg
                for c, cname in CLASSES.items():
                    rows.append({"run": name, "cond": cond, "type": dtype, "level": float(level), "variant": "est",
                                 "pid": pid, "phase": ph, "cls": cname, "dice": dice(seg == c, gt == c),
                                 "hd95": hd95(seg == c, gt == c, spacing) if full else np.nan})
            if full:
                cp, cg = clinical(segs["ED"], segs["ES"], spacing), clinical(sub["gt_ED"], sub["gt_ES"], spacing)
                crow.append({"run": name, "cond": cond, "type": dtype, "level": float(level), "variant": "est",
                             "pid": pid, **{f"pred_{k}": v for k, v in cp.items()},
                             **{f"ref_{k}": v for k, v in cg.items()}})
        print(name, cond, lr, steps, flush=True)
    with torch.no_grad():
        for p, p0 in zip(norm_params(model), init):
            p.copy_(p0)
    return pd.DataFrame(rows), pd.DataFrame(crow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--valset", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="augmentation-only run folders (one backbone)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, ts, vs = Path(a.cache), Path(a.testset), Path(a.valset)
    runs = [Path(r) for r in a.runs]
    backbone = json.loads((runs[0] / "args.json").read_text())["backbone"]
    sel_file = runs[0].parent / f"tent_selection_{backbone}.json"

    if not sel_file.exists():  # choose lr and steps on validation data, first seed only
        model, _ = load_run(runs[0], dev)
        vconds = ["clean_0.0"] + [f"{t}_1.0" for t in TYPES]
        grid = [(0.0, 0)] + list(itertools.product((1e-4, 1e-3), (3, 10)))
        scores = {}
        for lr, st in grid:
            d, _ = run_set(model, dev, vs, cache, vconds, lr, st, "select", full=False)
            scores[f"{lr}_{st}"] = float(d.dice.mean())
            print("validation", lr, st, scores[f"{lr}_{st}"], flush=True)
        best = max(scores, key=scores.get)
        sel_file.write_text(json.dumps({"scores": scores, "best": best}, indent=1))
        del model
    sel = json.loads(sel_file.read_text())
    lr, steps = float(sel["best"].split("_")[0]), int(sel["best"].split("_")[1])
    print("selected", lr, steps, flush=True)

    meta = json.loads((ts / "conditions.json").read_text())
    for run in runs:
        name = run.name.replace("_aug", "_augtent", 1)
        out = run.parent / name
        if (out / "eval_seg.csv").exists():
            continue
        out.mkdir(exist_ok=True)
        shutil.copy(run / "args.json", out / "args.json")
        (out / "tent.json").write_text(json.dumps({"source_run": run.name, "lr": lr, "steps": steps}, indent=1))
        model, _ = load_run(run, dev)
        d, c = run_set(model, dev, ts, cache, meta["conditions"], lr, steps, name)
        c.to_csv(out / "eval_clinical.csv", index=False)
        d.to_csv(out / "eval_seg.csv", index=False)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
