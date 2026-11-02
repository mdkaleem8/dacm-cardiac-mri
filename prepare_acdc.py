"""Preprocess ACDC into one .npz per subject and write stratified splits.

Usage: python scripts/prepare_acdc.py --acdc <folder containing patientXXX dirs> --out <cache dir>
Training subjects 001-100 -> 80 train / 20 val (4 per diagnostic group);
testing subjects 101-150 -> test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm.data import GROUPS, prepare_subject  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pdirs = sorted({p.parent for p in Path(a.acdc).rglob("Info.cfg")}, key=lambda p: p.name)
    print(f"found {len(pdirs)} subjects")
    meta = {}
    for pdir in pdirs:
        sub = prepare_subject(pdir)
        np.savez_compressed(out / f"{pdir.name}.npz", **{k: v for k, v in sub.items() if isinstance(v, np.ndarray)})
        meta[pdir.name] = {"group": sub["group"], "n_slices": int(sub["img_ED"].shape[0]),
                           "spacing": [float(v) for v in sub["spacing"]]}
        print(pdir.name, sub["group"], sub["img_ED"].shape, flush=True)
    num = lambda pid: int(pid.replace("patient", ""))
    rng = np.random.default_rng(a.seed)
    train_all = [p for p in meta if num(p) <= 100]
    test = sorted(p for p in meta if num(p) > 100)
    val = []
    for g in GROUPS:
        members = sorted(p for p in train_all if meta[p]["group"] == g)
        val += list(rng.choice(members, size=min(4, len(members)), replace=False))
    train = sorted(p for p in train_all if p not in val)
    splits = {"train": train, "val": sorted(val), "test": test}
    (out / "splits.json").write_text(json.dumps(splits, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
