"""Export the cached ACDC split to nnU-Net v2 raw format (Dataset501_ACDCdacm).

Each training or validation subject gives two cases (ED, ES): the cached
image (resampled in-plane to 1.25 mm, 224 x 224) scaled to [0, 1] with the
volume's 0.5/99.5 percentiles, exactly the input our own pipeline degrades,
and its label map. A splits_final.json with one fold reproduces our 80/20
train/validation split, so the official nnU-Net is trained on the same data.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dacm.data import load_splits, load_subject, to_unit  # noqa: E402

DS = "Dataset501_ACDCdacm"


def save(arr_zyx, spacing_zyx, path, dtype):
    # nibabel expects (x, y, z); keep 1.25 mm in-plane and the native slice spacing
    a = np.transpose(arr_zyx, (2, 1, 0)).astype(dtype)
    aff = np.diag([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0], 1.0])
    nib.save(nib.Nifti1Image(a, aff), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nnunet_raw", required=True)
    ap.add_argument("--nnunet_preprocessed", required=True)
    a = ap.parse_args()
    cache = Path(a.cache)
    sp = load_splits(cache)
    root = Path(a.nnunet_raw) / DS
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = {"train": [], "val": []}
    for split in ("train", "val"):
        for pid in sp[split]:
            sub = load_subject(cache, pid)
            sp_zyx = [float(v) for v in sub["spacing"]]
            for ph in ("ED", "ES"):
                case = f"{pid}_{ph}"
                img = to_unit(sub[f"img_{ph}"], sub[f"range_{ph}"])
                save(img, sp_zyx, root / "imagesTr" / f"{case}_0000.nii.gz", np.float32)
                save(sub[f"gt_{ph}"], sp_zyx, root / "labelsTr" / f"{case}.nii.gz", np.uint8)
                cases[split].append(case)
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "cineMRI"},
        "labels": {"background": 0, "RV": 1, "MYO": 2, "LV": 3},
        "numTraining": len(cases["train"]) + len(cases["val"]),
        "file_ending": ".nii.gz"}, indent=1))
    pre = Path(a.nnunet_preprocessed) / DS
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "splits_final.json").write_text(json.dumps([cases], indent=1))
    print(f"exported {len(cases['train'])} training and {len(cases['val'])} validation cases to {root}")


if __name__ == "__main__":
    main()
