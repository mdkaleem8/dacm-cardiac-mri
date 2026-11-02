"""External test on real respiratory motion: CMRxMotion training release (never used for training).

160 volumes (20 volunteers x 4 breath-hold instructions x ED/ES), each with a
radiologist image-quality grade (1 good, 2 intermediate, 3 non-diagnostic);
segmentation labels exist for grades 1-2 only (139 volumes). CMRxMotion codes
1 LV, 2 Myo, 3 RV; they are mapped to the ACDC coding (1 RV, 2 Myo, 3 LV).
Preprocessing is identical to ACDC: in-plane resampling to 1.25 mm, centre
crop/pad to 224 x 224, scaling to [0, 1] with the volume's 0.5/99.5 percentiles,
z-scoring per slice.

Outputs (work/cmrx_eval):
  <run>.csv          per volume, phase and structure: Dice and HD95 (DACM runs also with z = 0)
  <run>_clin.csv     per scan with both phases labelled: LV/RV volumes, EF, LV mass
  estimator.csv      per volume: mean estimated motion probability and motion severity, and grade
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dacm import degradations as deg  # noqa: E402
from dacm.data import CROP, TARGET_SPACING, _center_crop_pad, zscore  # noqa: E402
from dacm.metrics import CLASSES, clinical, dice, hd95  # noqa: E402
from dacm.models import DegradationEstimator  # noqa: E402
from evaluate import load_run, predict  # noqa: E402

MAP = np.array([0, 3, 2, 1], np.uint8)  # CMRxMotion -> ACDC label coding


def load_vol(path, order):
    n = nib.load(str(path))
    a = np.asarray(n.dataobj)
    a = a.reshape(a.shape[:3]).astype(np.float32 if order else np.uint8)
    sx, sy, sz = [float(v) for v in n.header.get_zooms()[:3]]
    a = zoom(a, (sx / TARGET_SPACING, sy / TARGET_SPACING, 1), order=order)
    return _center_crop_pad(np.transpose(a, (2, 0, 1)), CROP), (sz, TARGET_SPACING, TARGET_SPACING)


def build_cache(data_root, cache):
    cache.mkdir(parents=True, exist_ok=True)
    grades = {r["Image"]: int(r["Label"]) for r in csv.DictReader(open(data_root / "IQA.csv"))}
    index = []
    for case, g in sorted(grades.items()):
        f = cache / f"{case}.npz"
        scan = case.rsplit("-", 1)[0]
        d = data_root / "data" / scan
        lab_path = d / f"{case}-label.nii.gz"
        if not f.exists():
            img, sp = load_vol(d / f"{case}.nii.gz", 1)
            lo, hi = np.percentile(img, [0.5, 99.5])
            x = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
            out = {"x": x, "spacing": np.array(sp, np.float32)}
            if lab_path.exists():
                out["gt"] = MAP[load_vol(lab_path, 0)[0]]
            np.savez_compressed(f, **out)
        index.append({"case": case, "scan": scan, "volunteer": case.split("-")[0], "phase": case.rsplit("-", 1)[1],
                      "grade": g, "labelled": lab_path.exists()})
    pd.DataFrame(index).to_csv(cache / "index.csv", index=False)
    return pd.DataFrame(index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="CMRxMotion_Training folder (IQA.csv, data/)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = build_cache(Path(a.data_root), Path(a.cache))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    est = DegradationEstimator(deg.K).to(dev).eval()
    est.load_state_dict(torch.load(Path(a.estimator) / "estimator.pt", map_location=dev))
    vols = {}
    for r in idx.itertuples():
        d = np.load(Path(a.cache) / f"{r.case}.npz")
        X = np.stack([zscore(s) for s in d["x"]])[:, None]
        with torch.no_grad():
            Z = torch.cat([est.conditioning(torch.from_numpy(X[i:i + 64]).to(dev)).float() for i in range(0, len(X), 64)])
        vols[r.case] = (X, Z, d["gt"] if "gt" in d.files else None, tuple(float(v) for v in d["spacing"]))

    ef = out / "estimator.csv"
    if not ef.exists():
        rows = []
        for r in idx.itertuples():
            Z = vols[r.case][1].cpu().numpy()
            rows.append({"case": r.case, "volunteer": r.volunteer, "grade": r.grade,
                         **{f"p_{k}": float(Z[:, i].mean()) for i, k in enumerate(deg.TYPES)},
                         **{f"sev_{k}": float(Z[:, deg.K + i].mean()) for i, k in enumerate(deg.TYPES)},
                         "p_motion_max": float(Z[:, 0].max()), "sev_motion_max": float(Z[:, deg.K].max())})
        pd.DataFrame(rows).to_csv(ef, index=False)

    lab = idx[idx.labelled]
    for run in map(Path, a.runs):
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
                    rows.append({"run": run.name, "variant": v, "case": r.case, "volunteer": r.volunteer,
                                 "phase": r.phase, "grade": r.grade, "cls": cname,
                                 "dice": dice(pred == c, gt == c), "hd95": hd95(pred == c, gt == c, sp)})
            for scan in sorted(set(lab.scan)):
                ed, es = (v, f"{scan}-ED"), (v, f"{scan}-ES")
                if ed in segs and es in segs:
                    sp = segs[ed][2]
                    cp = clinical(segs[ed][0], segs[es][0], sp)
                    cg = clinical(segs[ed][1], segs[es][1], sp)
                    g = int(max(idx[idx.case == ed[1]].grade.iloc[0], idx[idx.case == es[1]].grade.iloc[0]))
                    crow.append({"run": run.name, "variant": v, "scan": scan, "volunteer": scan.split("-")[0],
                                 "grade": g, **{f"pred_{k}": x for k, x in cp.items()},
                                 **{f"ref_{k}": x for k, x in cg.items()}})
        pd.DataFrame(crow).to_csv(out / f"{run.name}_clin.csv", index=False)
        pd.DataFrame(rows).to_csv(f, index=False)
        print("done", run.name, flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
