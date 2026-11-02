"""Linear probes: is the degradation already encoded in the segmenter's features?

For trained segmentation networks (no conditioning), features are taken at an
early encoder stage, the bottleneck and the last decoder stage, summarised per
slice by the channel-wise mean and standard deviation. Linear probes are then
fitted on test subjects 1-25 and evaluated on test subjects 26-50:
  - type presence: one logistic regression per type -> AUC
  - severity: ridge regression on slices where the type is present -> MAE
Regularisation strengths are chosen on a held-out fifth of the fitting
subjects. The dedicated estimator is scored on the same held-out subjects.
Writes table_probe.tex and numbers_probe.tex.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacm import degradations as deg  # noqa: E402
from dacm.data import zscore  # noqa: E402
from dacm.metrics import auc  # noqa: E402
from dacm.models import SWIN_STAGES, UNET_STAGES, DegradationEstimator  # noqa: E402

ALL_STAGES = {"unet": UNET_STAGES, "swinunetr": SWIN_STAGES}
from evaluate import load_run  # noqa: E402

LAYERS = {"unet": [("enc.1", "Early encoder"), ("enc.5", "Bottleneck"), ("dec.4", "Last decoder")],
          "swinunetr": [("encoder2", "Early encoder"), ("encoder10", "Bottleneck"), ("decoder1", "Last decoder")]}
RUNS = [("unet_default", "U-Net, default"), ("unet_aug", "U-Net, augmentation-only"),
        ("swinunetr_default", "Swin-UNETR, default"), ("swinunetr_aug", "Swin-UNETR, augmentation-only")]


def iter_slices(ts, pids, conds):
    """Yield (X, T, S, P) one test condition at a time (all slices of all subjects)."""
    for cond in conds:
        d = np.load(ts / f"{cond}.npz")
        X, T, S, P = [], [], [], []
        for i, pid in enumerate(pids):
            for ph in ("ED", "ES"):
                x = d[f"{pid}_{ph}_x"]
                X += [zscore(s.astype(np.float32)) for s in x]
                T.append(d[f"{pid}_{ph}_t"]); S.append(d[f"{pid}_{ph}_s"]); P += [i] * len(x)
        yield np.stack(X)[:, None], np.concatenate(T), np.concatenate(S), np.array(P)


@torch.no_grad()
def features(model, names, X, dev, bs=64):
    mods = dict(model.named_modules())
    store, hooks = {}, []
    for n in names:
        def hook(_m, _i, o, n=n):
            store[n] = torch.cat([o.float().mean((2, 3)), o.float().std((2, 3))], 1).cpu()
        hooks.append(mods[n].register_forward_hook(hook))
    out = {n: [] for n in names}
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            model(torch.from_numpy(X[i:i + bs]).to(dev))
        for n in names:
            out[n].append(store[n])
    for h in hooks:
        h.remove()
    return {n: torch.cat(v).numpy() for n, v in out.items()}


def fit_logreg(F, y, wd, dev, steps=300):
    F, y = torch.from_numpy(F).to(dev), torch.from_numpy(y).float().to(dev)
    w = torch.zeros(F.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(F @ w + b, y) + wd * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def ridge(F, y, lam):
    A = F.T @ F + lam * len(F) * np.eye(F.shape[1])
    w = np.linalg.solve(A, F.T @ (y - y.mean()))
    return w, y.mean()


def probe(Ftr, Ttr, Str, ptr, Fte, Tte, Ste, dev):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    inner = ptr >= np.quantile(ptr, 0.8)  # last fifth of fitting subjects for model selection
    aucs, maes = [], []
    for k in range(deg.K):
        best = max((auc((torch.from_numpy(Ftr[inner]).to(dev) @ w + b).cpu().numpy(), Ttr[inner, k]), wd)
                   for wd in (1e-4, 1e-3, 1e-2)
                   for w, b in [fit_logreg(Ftr[~inner], Ttr[~inner, k], wd, dev)])[1]
        w, b = fit_logreg(Ftr, Ttr[:, k], best, dev)
        aucs.append(auc((torch.from_numpy(Fte).to(dev) @ w + b).cpu().numpy(), Tte[:, k]))
        m_tr, m_te = Ttr[:, k] > 0, Tte[:, k] > 0
        sel = m_tr & ~inner, m_tr & inner
        lam = min((np.abs(np.clip(Ftr[sel[1]] @ ridge(Ftr[sel[0]], Str[sel[0], k], l)[0]
                                  + ridge(Ftr[sel[0]], Str[sel[0], k], l)[1], 0, 1) - Str[sel[1], k]).mean(), l)
                  for l in (1e-3, 1e-2, 1e-1, 1.0))[1]
        w, c = ridge(Ftr[m_tr], Str[m_tr, k], lam)
        maes.append(float(np.abs(np.clip(Fte[m_te] @ w + c, 0, 1) - Ste[m_te, k]).mean()))
    return aucs, maes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--paper_results", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = Path(a.testset)
    meta = json.loads((ts / "conditions.json").read_text())
    pids = meta["pids"]
    conds = [c for c in meta["conditions"] if not c.startswith("combo")]
    T, S, P = [], [], []
    for _, t, s, p in iter_slices(ts, pids, conds):
        T.append(t); S.append(s); P.append(p)
    T, S, P = np.concatenate(T), np.concatenate(S), np.concatenate(P)
    tr, te = P < len(pids) // 2, P >= len(pids) // 2
    print(f"{len(P)} slices, {tr.sum()} for fitting, {te.sum()} held out", flush=True)

    rows = []
    est = DegradationEstimator(deg.K).to(dev).eval()
    est.load_state_dict(torch.load(Path(a.estimator) / "estimator.pt", map_location=dev))
    lg, sv = [], []
    with torch.no_grad():
        for X, _, _, _ in iter_slices(ts, pids, conds):
            for i in range(0, len(X), 128):
                o = est(torch.from_numpy(X[i:i + 128]).to(dev))
                lg.append(o[0].float().cpu()); sv.append(o[1].float().cpu())
    lg, sv = torch.cat(lg).numpy(), torch.cat(sv).numpy()
    e_auc = [auc(lg[te, k], T[te, k]) for k in range(deg.K)]
    e_mae = [float(np.abs(sv[te & (T[:, k] > 0), k] - S[te & (T[:, k] > 0), k]).mean()) for k in range(deg.K)]
    rows.append(("Dedicated estimator", "--", e_auc, e_mae))

    depth = {}  # backbone -> {stage: {version: (mean AUC, mean MAE)}}
    for run, label in RUNS:
        rdir = Path(a.runs_root) / run
        if not (rdir / "model.pt").exists():
            continue
        model, args = load_run(rdir, dev)
        b, ver = args["backbone"], run.split("_", 1)[1]
        stages = ALL_STAGES[b]
        parts = [features(model, stages, X, dev) for X, _, _, _ in iter_slices(ts, pids, conds)]
        F = {n: np.concatenate([p[n] for p in parts]) for n in stages}
        named = dict(LAYERS[b])
        for n in stages:
            aucs, maes = probe(F[n][tr], T[tr], S[tr], P[tr], F[n][te], T[te], S[te], dev)
            depth.setdefault(b, {}).setdefault(n, {})[ver] = (float(np.nanmean(aucs)), float(np.mean(maes)))
            if n in named:
                rows.append((label, named[n], aucs, maes))
            print(label, n, np.round(aucs, 3), np.round(maes, 3), flush=True)
        del model, F, parts
        torch.cuda.empty_cache()
    out = Path(a.paper_results)
    for b, st in depth.items():
        with open(out / f"probe_depth_{b}.csv", "w") as f:
            f.write("idx,stage,auc_default,auc_aug,mae_default,mae_aug\n")
            for i, n in enumerate(ALL_STAGES[b]):
                d = st.get(n, {})
                g = lambda v, j: f"{d[v][j]:.4f}" if v in d else "nan"  # noqa: E731
                f.write(f"{i},{n},{g('default', 0)},{g('aug', 0)},{g('default', 1)},{g('aug', 1)}\n")

    lines = [r"\begin{tabular}{@{}ll" + "c" * deg.K + "cc@{}}", r"\toprule",
             r" & & \multicolumn{" + str(deg.K) + r"}{c}{Type detection (AUC)} & & Severity \\",
             r"\cmidrule(lr){3-" + str(2 + deg.K) + "}",
             "Features & Layer & " + " & ".join(k.capitalize()[:4] + "." if len(k) > 5 else k.capitalize()
                                                for k in deg.TYPES) + r" & Mean & MAE \\", r"\midrule"]
    macros = ["% generated by probe.py"]
    last = None
    for label, nl, aucs, maes in rows:
        if last is not None and label.split(",")[0] != last:
            lines.append(r"\midrule")
        last = label.split(",")[0]
        lines.append(f"{label} & {nl} & " + " & ".join(f"{v:.2f}" for v in aucs) +
                     f" & {np.nanmean(aucs):.2f} & {np.mean(maes):.3f} \\\\")
        key = (label.replace("Dedicated estimator", "Est").replace("U-Net, ", "unet").replace("Swin-UNETR, ", "swin")
               .replace("augmentation-only", "aug").replace("default", "default"))
        lk = {"--": "", "Early encoder": "Early", "Bottleneck": "Bottleneck", "Last decoder": "Decoder"}[nl]
        macros += [f"\\newcommand{{\\Probe{key}{lk}AUC}}{{{np.nanmean(aucs):.2f}}}",
                   f"\\newcommand{{\\Probe{key}{lk}MAE}}{{{np.mean(maes):.3f}}}"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_probe.tex").write_text("\n".join(lines))
    (out / "numbers_probe.tex").write_text("\n".join(macros) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
