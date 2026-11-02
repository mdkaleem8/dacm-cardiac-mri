"""Evaluate an official nnU-Net 2D model on the fixed degraded test set.

The predictor receives exactly the unit-range slices that our own models see
(before z-scoring), with spacing (slice spacing, 1.25, 1.25), and applies the
official nnU-Net preprocessing, sliding-window inference and mirroring. Dice,
HD95 and clinical indices are computed with the same code as evaluate.py and
written to <out>/eval_seg.csv and eval_clinical.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dacm.data import load_subject  # noqa: E402
from dacm.metrics import CLASSES, clinical, dice, hd95  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="nnU-Net trainer output folder")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--conditions", nargs="*", default=None)
    a = ap.parse_args()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           device=torch.device("cuda"), verbose=False, verbose_preprocessing=False,
                           allow_tqdm=False)
    pred.initialize_from_trained_model_folder(a.model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    ts = Path(a.testset)
    meta = json.loads((ts / "conditions.json").read_text())
    pids, conds = meta["pids"], a.conditions or meta["conditions"]
    subs = {pid: load_subject(Path(a.cache), pid) for pid in pids}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, crow = [], []
    for cond in conds:
        data = np.load(ts / f"{cond}.npz")
        dtype, level = cond.rsplit("_", 1)
        for pid in pids:
            sub = subs[pid]
            spacing = tuple(float(v) for v in sub["spacing"])
            segs = {}
            for ph in ("ED", "ES"):
                x = data[f"{pid}_{ph}_x"].astype(np.float32)[None]  # (1, Z, H, W), unit range
                seg = pred.predict_single_npy_array(x, {"spacing": list(spacing)}, None, None, False)
                seg = np.asarray(seg).astype(np.uint8)
                gt = sub[f"gt_{ph}"]
                segs[ph] = seg
                for c, cname in CLASSES.items():
                    rows.append({"run": a.name, "cond": cond, "type": dtype, "level": float(level), "variant": "est",
                                 "pid": pid, "phase": ph, "cls": cname,
                                 "dice": dice(seg == c, gt == c), "hd95": hd95(seg == c, gt == c, spacing)})
            cp, cg = clinical(segs["ED"], segs["ES"], spacing), clinical(sub["gt_ED"], sub["gt_ES"], spacing)
            crow.append({"run": a.name, "cond": cond, "type": dtype, "level": float(level), "variant": "est",
                         "pid": pid, **{f"pred_{k}": v for k, v in cp.items()},
                         **{f"ref_{k}": v for k, v in cg.items()}})
        print(a.name, cond, flush=True)
    # written only when complete, so that downstream scripts never read a partial evaluation
    pd.DataFrame(crow).to_csv(out / "eval_clinical.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "eval_seg.csv", index=False)


if __name__ == "__main__":
    main()
