"""Stage 2: train a segmentation backbone in one of three versions.

  --mode default : backbone, standard augmentation, no degradation
  --mode aug     : backbone, same degradation simulator as DACM, no conditioning
  --mode dacm    : backbone wrapped by DACM; conditioning from the frozen
                   estimator (prob 0.5) or from true type/severity + noise
"""
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
from dacm.conditioning import DACM  # noqa: E402
from dacm.data import SliceDataset, load_splits, subset_per_group  # noqa: E402
from dacm.models import DegradationEstimator, build_backbone, count_params  # noqa: E402


def dice_loss(logits, y, n_classes=4, eps=1e-5):
    p = torch.softmax(logits, 1)
    oh = F.one_hot(y, n_classes).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps))[1:].mean()


def make_z(t, s):
    return torch.cat([t, t * s], 1)


def build(a, dev):
    backbone, stages = build_backbone(a.backbone, norm_affine=a.norm_affine)
    if a.mode == "dacm":
        if a.placement == "norm":
            sel = "norm"
        else:
            sel = stages if a.stages == "all" else \
                [n for n in stages if n.startswith(("enc", "encoder")) == (a.stages == "encoder")]
        model = DACM(backbone, sel, cond_dim=2 * deg.K, mechanism=a.mechanism, cond_part=a.cond_part)
    else:
        model = backbone
    return model.to(dev)


@torch.no_grad()
def validate(model, loader, dev, mode, est, zero=False):
    model.eval()
    inter = torch.zeros(4, device=dev)
    den = torch.zeros(4, device=dev)
    for x, y, t, s in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            if mode == "dacm":
                z = est.conditioning(x).float()
                logits = model(x, torch.zeros_like(z) if zero else z)
            else:
                logits = model(x)
        pred = logits.argmax(1)
        for c in range(1, 4):
            inter[c] += ((pred == c) & (y == c)).sum()
            den[c] += (pred == c).sum() + (y == c).sum()
    d = (2 * inter[1:] / den[1:].clamp(min=1)).cpu().numpy()
    return {"RV": float(d[0]), "Myo": float(d[1]), "LV": float(d[2]), "mean": float(d.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="unet", choices=["unet", "swinunetr"])
    ap.add_argument("--mode", default="dacm", choices=["default", "aug", "dacm"])
    ap.add_argument("--estimator", default="")
    ap.add_argument("--stages", default="all", choices=["all", "encoder", "decoder"])
    ap.add_argument("--mechanism", default="film", choices=["film", "adain"])
    ap.add_argument("--placement", default="stage", choices=["stage", "norm"],
                    help="modulate stage outputs, or the output of every normalisation layer")
    ap.add_argument("--p_est", type=float, default=0.5)
    ap.add_argument("--no_noise", action="store_true", help="ablation: no noise on true severity")
    ap.add_argument("--zero_cond", action="store_true",
                    help="control: same module and parameters, but z is always zero (no degradation information)")
    ap.add_argument("--shuffle_cond", action="store_true",
                    help="control: z is permuted across the batch during training (realistic but uninformative)")
    ap.add_argument("--norm_affine", action="store_true",
                    help="control: give instance-norm layers without one a learnable scale and shift")
    ap.add_argument("--cond_part", default="both", choices=["both", "type"],
                    help="variant: condition on type and severity, or on type probabilities only")
    ap.add_argument("--max_sev", type=float, default=1.0,
                    help="upper bound of training (and validation) severities")
    ap.add_argument("--train_per_group", type=int, default=0,
                    help="use only this many training subjects per diagnostic group (0 = all)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    calib = json.loads(Path(a.calib).read_text())
    sp = load_splits(Path(a.cache))
    data_mode = "default" if a.mode == "default" else "aug"
    train_pids = sp["train"] if a.train_per_group <= 0 else subset_per_group(a.cache, sp["train"], a.train_per_group)
    print(f"training subjects: {len(train_pids)}", flush=True)
    tr = SliceDataset(a.cache, train_pids, data_mode, calib, max_sev=a.max_sev)
    # Validation always sees degraded images so that model selection is comparable;
    # it uses the same severity range as training, so capped runs never see high severities.
    va = SliceDataset(a.cache, sp["val"], "aug", calib, train_aug=False, max_sev=a.max_sev)
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                    persistent_workers=a.workers > 0)
    vl = DataLoader(va, 32, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)

    est, sigma_est = None, 0.0
    if a.mode == "dacm":
        if not a.estimator:
            raise SystemExit("--estimator is required for --mode dacm")
        est = DegradationEstimator(deg.K).to(dev).eval()
        est.load_state_dict(torch.load(Path(a.estimator) / "estimator.pt", map_location=dev))
        for p in est.parameters():
            p.requires_grad_(False)
        sigma_est = json.loads((Path(a.estimator) / "estimator_val.json").read_text())["sigma_est"]

    model = build(a, dev)
    info = {"params_total": count_params(model)}
    if a.mode == "dacm":
        info["params_added"] = model.added_parameters()
        info["params_backbone"] = count_params(model.backbone)
    print(info, f"sigma_est={sigma_est:.3f}", flush=True)
    (out / "params.json").write_text(json.dumps(info, indent=1))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl), pct_start=0.05)
    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, t, s in tl:
            x, y, t, s = x.to(dev, non_blocking=True), y.to(dev), t.to(dev), s.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                if a.mode == "dacm":
                    use_est = torch.rand(x.shape[0], 1, device=dev) < a.p_est
                    z_est = est.conditioning(x).float()
                    s_noisy = s if a.no_noise else (s + sigma_est * torch.randn_like(s)).clamp(0, 1)
                    z = torch.where(use_est, z_est, make_z(t, s_noisy))
                    if a.zero_cond:
                        z = torch.zeros_like(z)
                    elif a.shuffle_cond:
                        z = z[torch.randperm(z.shape[0], device=dev)]
                    logits = model(x, z)
                else:
                    logits = model(x)
            logits = logits.float()
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step()
            sched.step()
            tot += loss.item()
        row = {"epoch": ep, "loss": tot / len(tl), "time": time.time() - t0}
        if ep % 5 == 4 or ep == a.epochs - 1:
            row["val"] = validate(model, vl, dev, a.mode, est, a.zero_cond)
            if row["val"]["mean"] > best:
                best = row["val"]["mean"]
                torch.save(model.state_dict(), out / "model.pt")
        log.append(row)
        print(f"ep {ep:3d} loss {row['loss']:.4f} " + (f"val {row['val']['mean']:.4f} " if 'val' in row else "") +
              f"({row['time']:.0f}s)", flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))
    (out / "done.json").write_text(json.dumps({"best_val_mean_dice": best}, indent=1))


if __name__ == "__main__":
    main()
