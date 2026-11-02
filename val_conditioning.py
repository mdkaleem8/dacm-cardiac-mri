"""Validation-only check used to choose the FiLM placement (no test data).

For each DACM run, compare validation Dice with the estimated conditioning
and with the conditioning shuffled across slices. A run that ignores its
conditioning shows no gap. Degraded validation slices use fixed seeds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm import degradations as deg  # noqa: E402
from dacm.data import SliceDataset, load_splits  # noqa: E402
from dacm.models import DegradationEstimator  # noqa: E402
from evaluate import load_run  # noqa: E402

cache, calib_p, est_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
runs = [Path(r) for r in sys.argv[4:]]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
calib = json.loads(calib_p.read_text())
ds = SliceDataset(cache, load_splits(cache)["val"], "aug", calib, train_aug=False)
ds._rng = np.random.default_rng(123)
items = [ds[i] for i in range(len(ds))] + [ds[i] for i in range(len(ds))]
X = torch.stack([t[0] for t in items]).to(dev)
Y = torch.stack([t[1] for t in items]).to(dev)
est = DegradationEstimator(deg.K).to(dev).eval()
est.load_state_dict(torch.load(est_dir / "estimator.pt", map_location=dev))
with torch.no_grad():
    Z = torch.cat([est.conditioning(X[i:i + 64]) for i in range(0, len(X), 64)]).float()
perm = torch.from_numpy(np.random.default_rng(0).permutation(len(X))).to(dev)


def dice_all(model, z):
    inter, den = torch.zeros(4, device=dev), torch.zeros(4, device=dev)
    with torch.no_grad():
        for i in range(0, len(X), 32):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                p = model(X[i:i + 32], z[i:i + 32]).argmax(1)
            y = Y[i:i + 32]
            for c in range(1, 4):
                inter[c] += ((p == c) & (y == c)).sum()
                den[c] += (p == c).sum() + (y == c).sum()
    return float((2 * inter[1:] / den[1:]).mean())


for r in runs:
    model, _ = load_run(r, dev)
    a, b = dice_all(model, Z), dice_all(model, Z[perm])
    print(f"{r.name:18s} val Dice estimated {a:.4f}  shuffled {b:.4f}  gap {a - b:+.4f}  (n={len(X)} slices)")
