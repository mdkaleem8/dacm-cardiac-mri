"""Stage 1: train the degradation estimator (type + severity heads)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm import degradations as deg  # noqa: E402
from dacm.data import SliceDataset, load_splits, subset_per_group  # noqa: E402
from dacm.metrics import auc  # noqa: E402
from dacm.models import DegradationEstimator, count_params  # noqa: E402


def run_epoch(model, loader, opt, dev, lam, train):
    model.train(train)
    tot, n = 0.0, 0
    all_p, all_t, all_s, all_sh = [], [], [], []
    for x, _, t, s in loader:
        x, t, s = x.to(dev, non_blocking=True), t.to(dev), s.to(dev)
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits, sh = model(x)
        logits, sh = logits.float(), sh.float()
        loss = F.binary_cross_entropy_with_logits(logits, t) + \
            lam * ((sh - s).abs() * t).sum() / t.sum().clamp(min=1)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        tot += loss.item() * x.shape[0]
        n += x.shape[0]
        if not train:
            all_p.append(torch.sigmoid(logits).cpu()); all_t.append(t.cpu())
            all_s.append(s.cpu()); all_sh.append(sh.cpu())
    out = {"loss": tot / n}
    if not train:
        p, t, s, sh = (torch.cat(v).numpy() for v in (all_p, all_t, all_s, all_sh))
        out["auc"] = {k: auc(p[:, i], t[:, i]) for i, k in enumerate(deg.TYPES)}
        out["mae"] = {k: float(np.abs(sh[t[:, i] > 0, i] - s[t[:, i] > 0, i]).mean()) if (t[:, i] > 0).any() else float("nan")
                      for i, k in enumerate(deg.TYPES)}
        out["mae_all"] = float(np.abs(sh - s)[t > 0].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train_per_group", type=int, default=0,
                    help="stress test: train on this many subjects per diagnostic group (0 = all)")
    ap.add_argument("--max_sev", type=float, default=1.0,
                    help="stress test: upper bound of training and validation severities")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    calib = json.loads(Path(a.calib).read_text())
    sp = load_splits(Path(a.cache))
    train_pids = sp["train"] if a.train_per_group <= 0 else subset_per_group(a.cache, sp["train"], a.train_per_group)
    print(f"training subjects: {len(train_pids)}  max_sev: {a.max_sev}", flush=True)
    tr = SliceDataset(a.cache, train_pids, "aug", calib, p_types=(0.2, 0.5, 0.3), max_sev=a.max_sev)
    va = SliceDataset(a.cache, sp["val"], "aug", calib, p_types=(0.2, 0.5, 0.3), train_aug=False,
                      max_sev=a.max_sev)
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                    persistent_workers=a.workers > 0)
    vl = DataLoader(va, a.batch, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)
    model = DegradationEstimator(deg.K).to(dev)
    print(f"estimator parameters: {count_params(model):,}  device: {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))
    best, log = 1e9, []
    for ep in range(a.epochs):
        t0 = time.time()
        trm = run_epoch(model, tl, opt, dev, a.lam, True)
        for _ in range(len(tl)):
            sched.step()
        vam = run_epoch(model, vl, None, dev, a.lam, False)
        log.append({"epoch": ep, "train_loss": trm["loss"], **{f"val_{k}": v for k, v in vam.items()}})
        print(f"ep {ep:3d} train {trm['loss']:.4f} val {vam['loss']:.4f} mae {vam['mae_all']:.3f} "
              f"auc {np.nanmean(list(vam['auc'].values())):.3f} ({time.time() - t0:.0f}s)", flush=True)
        if vam["loss"] < best:
            best = vam["loss"]
            torch.save(model.state_dict(), out / "estimator.pt")
            (out / "estimator_val.json").write_text(json.dumps(
                {"epoch": ep, "sigma_est": vam["mae_all"], "auc": vam["auc"], "mae": vam["mae"]}, indent=1))
    (out / "estimator_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
