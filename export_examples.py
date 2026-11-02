"""Export a heart-centred crop of one test slice, clean and degraded, as raw
8-bit grayscale tiles. The paper arranges and labels them with pgfplots.
Severity targets use the calibrated map.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm import degradations as deg  # noqa: E402
from dacm.data import load_splits, load_subject, to_unit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pid", default="")
    ap.add_argument("--crop", type=int, default=96)
    a = ap.parse_args()
    calib = json.loads(Path(a.calib).read_text())
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pid = a.pid or load_splits(cache)["test"][0]
    sub = load_subject(cache, pid)
    gt = sub["gt_ED"]
    zs = np.flatnonzero(gt.reshape(gt.shape[0], -1).any(1))
    z = int(zs[len(zs) // 2])
    x = to_unit(sub["img_ED"][z], sub["range_ED"])
    cy, cx = [int(round(c)) for c in center_of_mass(gt[z] > 0)]
    h = a.crop // 2
    cy, cx = np.clip(cy, h, x.shape[0] - h), np.clip(cx, h, x.shape[1] - h)
    box = deg.heart_box(gt[z], x.shape)

    def save(name, img):
        # Raw pixel tile only (8-bit grayscale, no annotation); layout and
        # labels are drawn by pgfplots in the paper.
        c = np.clip(img[cy - h:cy + h, cx - h:cx + h], 0, 1)
        Image.fromarray((c * 255).round().astype(np.uint8), mode="L").save(out / f"{name}.png")

    save("clean", x)
    meta = {"pid": pid, "slice": z}
    for k in deg.TYPES:
        for st in (0.4, 1.0):
            r = deg.r_for_severity(calib, k, st)
            xd = deg.FUNCS[k](x, r, np.random.default_rng(7))
            save(f"{k}_{st:.1f}", xd)
            meta[f"{k}_{st:.1f}"] = round(deg.severity_from_image(calib, k, xd, x, box), 2)
    (out / "examples.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
