"""How strongly does a trained DACM run modulate its backbone?

For each hooked stage, report the weight norm of the FiLM head and the spread
of (gamma, beta) across conditioning vectors from validation slices.
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

run, cache, calib_p, est_dir = map(Path, sys.argv[1:5])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, args = load_run(run, dev)
est = DegradationEstimator(deg.K).to(dev).eval()
est.load_state_dict(torch.load(est_dir / "estimator.pt", map_location=dev))
calib = json.loads(calib_p.read_text())
ds = SliceDataset(cache, load_splits(cache)["val"], "aug", calib, train_aug=False)
xs = torch.stack([ds[i][0] for i in range(0, len(ds), 2)]).to(dev)
with torch.no_grad():
    z = torch.cat([est.conditioning(xs[i:i + 64]) for i in range(0, len(xs), 64)]).float()
    e = model.embed(z)
    print(f"z: mean {z.mean().item():.3f}, std over slices {z.std(0).mean().item():.3f}")
    for name, head in model.heads.items():
        gb = head(e)
        g, b = gb.chunk(2, 1)
        print(f"{name:10s} |W|={head.weight.norm().item():.4f}  "
              f"|gamma| mean {g.abs().mean().item():.4f}  gamma std over slices {g.std(0).mean().item():.4f}  "
              f"|beta| mean {b.abs().mean().item():.4f}")
