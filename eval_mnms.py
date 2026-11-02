"""External test on multi-vendor, multi-centre data: the M&Ms open dataset (never used for training).

All subjects with reference labels are used (Training/Labeled, Training/Unlabeled, Validation,
Testing; 344 subjects from Siemens, Philips, GE and Canon scanners in 5 centres). The ED and ES
frames come from the dataset information file. M&Ms codes 1 LV, 2 Myo, 3 RV; they are mapped
to the ACDC coding (1 RV, 2 Myo, 3 LV). Preprocessing is identical to ACDC.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dacm.data import CROP, TARGET_SPACING, _center_crop_pad  # noqa: E402
from external_eval import run_eval  # noqa: E402

MAP = np.array([0, 3, 2, 1], np.uint8)  # M&Ms -> ACDC label coding
META = ["subject", "vendor", "centre", "pathology", "sex", "age", "split"]


def frame(a, t, sx, sy, order):
    v = a[..., t]
    v = np.rint(v).astype(np.uint8) if order == 0 else v.astype(np.float32)
    v = zoom(v, (sx / TARGET_SPACING, sy / TARGET_SPACING, 1), order=order)
    return _center_crop_pad(np.transpose(v, (2, 0, 1)), CROP)


def frames(gt, r):
    """ED and ES frame indices. The information file is used when both of its frames carry labels.
    For some subjects it lists ED = ES = 0 while the labels sit on two other frames; then the two
    labelled frames are used and the one with the larger LV volume (M&Ms label 1) is taken as ED."""
    lab = [t for t in range(gt.shape[-1]) if gt[..., t].any()]
    if int(r["ED"]) in lab and int(r["ES"]) in lab and int(r["ED"]) != int(r["ES"]):
        return {"ED": int(r["ED"]), "ES": int(r["ES"])}, "info"
    assert len(lab) == 2, lab
    lv = [(gt[..., t] == 1).sum() for t in lab]
    ed = lab[int(np.argmax(lv))]
    return {"ED": ed, "ES": lab[1 - lab.index(ed)]}, "labels"


def build_cache(root, cache):
    cache.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(glob.glob(str(root / "*.csv"))[0], index_col=0).set_index("External code")
    index = []
    for g in sorted(glob.glob(str(root / "**" / "*_sa_gt.nii.gz"), recursive=True)):
        code = os.path.basename(g)[:6]
        split = Path(g).relative_to(root).parts[0]
        r = info.loc[code]
        img_path = g.replace("_sa_gt.nii.gz", "_sa.nii.gz")
        fr, src = frames(np.asarray(nib.load(g).dataobj), r)
        loaded = None
        for ph in ("ED", "ES"):
            case = f"{code}-{ph}"
            f = cache / f"{case}.npz"
            if not f.exists():
                if loaded is None:
                    ni, ng = nib.load(img_path), nib.load(g)
                    sx, sy, sz = [float(v) for v in ni.header.get_zooms()[:3]]
                    loaded = (np.asarray(ni.dataobj), np.asarray(ng.dataobj), sx, sy, sz)
                a, b, sx, sy, sz = loaded
                t = fr[ph]
                x = frame(a, t, sx, sy, 1)
                lo, hi = np.percentile(x, [0.5, 99.5])
                x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
                np.savez_compressed(f, x=x, gt=MAP[frame(b, t, sx, sy, 0)],
                                    spacing=np.array((sz, TARGET_SPACING, TARGET_SPACING), np.float32))
            index.append({"case": case, "scan": code, "phase": ph, "labelled": True, "subject": code,
                          "vendor": r["VendorName"], "centre": int(r["Centre"]), "pathology": r["Pathology"],
                          "sex": r["Sex"], "age": r["Age"], "split": split, "frame_src": src})
    idx = pd.DataFrame(index)
    idx.to_csv(cache / "index.csv", index=False)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="M&Ms OpenDataset folder")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--redo_label_frames", action="store_true",
                    help="re-evaluate only subjects whose frames come from the labels and merge them into --out")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = build_cache(Path(a.data_root), Path(a.cache))
    print(f"{idx.scan.nunique()} subjects, {len(idx)} volumes, "
          f"{(idx.frame_src == 'labels').sum() // 2} with frames taken from the labels", flush=True)
    if not a.redo_label_frames:
        run_eval(idx, a.cache, a.estimator, a.runs, a.out, META, dev)
        return
    sub = idx[idx.frame_src == "labels"]
    out, tmp = Path(a.out), Path(a.out) / "_redo"
    run_eval(sub, a.cache, a.estimator, a.runs, tmp, META, dev)
    for f in sorted(tmp.glob("*.csv")):
        key, keep = ("scan", sub.scan) if f.name.endswith("_clin.csv") else ("case", sub.case)
        old = pd.read_csv(out / f.name)
        new = pd.concat([old[~old[key].isin(set(keep))], pd.read_csv(f)], ignore_index=True)
        new.to_csv(out / f.name, index=False)
        f.unlink()
        print("merged", f.name, len(old), "->", len(new), flush=True)
    tmp.rmdir()


if __name__ == "__main__":
    main()
