"""Quick checks before training: degradation examples, DACM wrapping, loader speed."""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm import degradations as deg  # noqa: E402
from dacm.conditioning import DACM  # noqa: E402
from dacm.data import SliceDataset, load_splits, load_subject, to_unit  # noqa: E402
from dacm.models import build_backbone, count_params  # noqa: E402

cache, out = Path(sys.argv[1]), Path(sys.argv[2])
calib_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)
sp = load_splits(cache)
sub = load_subject(cache, sp["train"][0])
z = sub["img_ED"].shape[0] // 2
x = to_unit(sub["img_ED"][z], sub["range_ED"])
fig, ax = plt.subplots(len(deg.TYPES), 4, figsize=(8, 10))
for i, k in enumerate(deg.TYPES):
    for j, r in enumerate([0.0, 0.33, 0.66, 1.0]):
        xd = deg.FUNCS[k](x, r, np.random.default_rng(1))
        ax[i, j].imshow(xd, cmap="gray", vmin=0, vmax=1)
        ax[i, j].set_title(f"{k} r={r:.2f}  D={deg.distortion(xd, x, deg.heart_box(sub['gt_ED'][z], x.shape)):.3f}", fontsize=6)
        ax[i, j].axis("off")
plt.tight_layout()
plt.savefig(out / "degradations.png", dpi=110)
print("saved", out / "degradations.png")

for b in ("unet", "swinunetr"):
    bb, stages = build_backbone(b)
    n_bb = count_params(bb)
    m = DACM(bb, stages)
    xx = torch.randn(2, 1, 224, 224)
    zz = torch.rand(2, 10)
    with torch.no_grad():
        y0 = m.backbone(xx)
        y1 = m(xx, zz)
    print(f"{b}: backbone {n_bb/1e6:.2f}M, added {m.added_parameters()/1e3:.1f}k "
          f"({100*m.added_parameters()/n_bb:.2f}%), out {tuple(y1.shape)}, "
          f"identity-at-init max|diff|={float((y0 - y1).abs().max()):.2e}")

if calib_path and calib_path.exists():
    calib = json.loads(calib_path.read_text())
    ds = SliceDataset(cache, sp["train"][:10], "aug", calib)
    t0 = time.time()
    for i in range(200):
        ds[i % len(ds)]
    print(f"aug sample time: {(time.time() - t0) / 200 * 1000:.1f} ms")
