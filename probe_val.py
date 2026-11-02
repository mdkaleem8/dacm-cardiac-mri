"""Linear probes, revised design (replaces the test-subject split of probe.py).

Probes are fitted on the 20 validation subjects (all single-type conditions and
clean) and evaluated on all 50 test subjects, so no test subject is used for
fitting. Regularisation is chosen on a held-out fifth of the validation
subjects. Besides the unconditioned segmenters (default, augmentation-only),
the conditioned DACM (norm) and the shuffled-training module are probed, with
their conditioning taken from the estimator as at test time. The dedicated
estimator is scored on the same test slices.
Writes table_probe.tex, numbers_probe.tex and probe_depth_<backbone>.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dacm import degradations as deg  # noqa: E402
from dacm.metrics import auc  # noqa: E402
from dacm.models import SWIN_STAGES, UNET_STAGES, DegradationEstimator  # noqa: E402
from evaluate import load_run  # noqa: E402
from probe import iter_slices, probe  # noqa: E402

ALL_STAGES = {"unet": UNET_STAGES, "swinunetr": SWIN_STAGES}
LAYERS = {"unet": [("enc.1", "Early encoder"), ("enc.5", "Bottleneck"), ("dec.4", "Last decoder")],
          "swinunetr": [("encoder2", "Early encoder"), ("encoder10", "Bottleneck"), ("decoder1", "Last decoder")]}
RUNS = [("unet_default", "U-Net, default"), ("unet_aug", "U-Net, augmentation-only"),
        ("unet_dacmnorm", "U-Net, DACM (norm)"), ("unet_dacmshuf", "U-Net, shuffled-training"),
        ("swinunetr_default", "Swin-UNETR, default"), ("swinunetr_aug", "Swin-UNETR, augmentation-only"),
        ("swinunetr_dacmnorm", "Swin-UNETR, DACM (norm)"), ("swinunetr_dacmshuf", "Swin-UNETR, shuffled-training")]
IN_TABLE = {"default", "aug", "dacmnorm"}  # shuffled-training appears in the depth figure only
VER_KEY = {"default": "default", "aug": "aug", "dacmnorm": "norm", "dacmshuf": "shuf"}


@torch.no_grad()
def features(model, args, est, names, X, dev, bs=64):
    """Channel mean and SD of each named backbone module, for unconditioned or DACM models."""
    is_dacm = args["mode"] == "dacm"
    net = model.backbone if is_dacm else model
    mods = dict(net.named_modules())
    store, hooks = {}, []
    for n in names:
        def hook(_m, _i, o, n=n):
            store[n] = torch.cat([o.float().mean((2, 3)), o.float().std((2, 3))], 1).cpu()
        hooks.append(mods[n].register_forward_hook(hook))
    out = {n: [] for n in names}
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            if is_dacm:
                model(xb, est.conditioning(xb).float())
            else:
                model(xb)
        for n in names:
            out[n].append(store[n])
    for h in hooks:
        h.remove()
    return {n: torch.cat(v).numpy() for n, v in out.items()}


def load_set(ts):
    meta = json.loads((ts / "conditions.json").read_text())
    conds = [c for c in meta["conditions"] if not c.startswith("combo")]
    T, S, P = [], [], []
    for _, t, s, p in iter_slices(ts, meta["pids"], conds):
        T.append(t); S.append(s); P.append(p)
    return meta["pids"], conds, np.concatenate(T), np.concatenate(S), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valset", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--paper_results", required=True)
    ap.add_argument("--suffix", default="", help="run-name suffix of the seed to probe, e.g. _s1")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vs, ts = Path(a.valset), Path(a.testset)
    vp, vc, Tv, Sv, Pv = load_set(vs)
    tp, tc, Tt, St, _ = load_set(ts)
    print(f"fit: {len(Pv)} validation slices ({len(vp)} subjects); test: {len(Tt)} slices ({len(tp)} subjects)",
          flush=True)

    est = DegradationEstimator(deg.K).to(dev).eval()
    est.load_state_dict(torch.load(Path(a.estimator) / "estimator.pt", map_location=dev))
    lg, sv = [], []
    with torch.no_grad():
        for X, _, _, _ in iter_slices(ts, tp, tc):
            for i in range(0, len(X), 128):
                o = est(torch.from_numpy(X[i:i + 128]).to(dev))
                lg.append(o[0].float().cpu()); sv.append(o[1].float().cpu())
    lg, sv = torch.cat(lg).numpy(), torch.cat(sv).numpy()
    rows = [("Dedicated estimator", "--", [auc(lg[:, k], Tt[:, k]) for k in range(deg.K)],
             [float(np.abs(sv[Tt[:, k] > 0, k] - St[Tt[:, k] > 0, k]).mean()) for k in range(deg.K)])]

    depth = {}
    for run, label in RUNS:
        rdir = Path(a.runs_root) / (run + a.suffix)
        if not (rdir / "model.pt").exists():
            print("missing", run, flush=True)
            continue
        model, args = load_run(rdir, dev)
        b, ver = args["backbone"], run.split("_", 1)[1]
        stages = ALL_STAGES[b]
        Fv = [features(model, args, est, stages, X, dev) for X, _, _, _ in iter_slices(vs, vp, vc)]
        Ft = [features(model, args, est, stages, X, dev) for X, _, _, _ in iter_slices(ts, tp, tc)]
        Fv = {n: np.concatenate([p[n] for p in Fv]) for n in stages}
        Ft = {n: np.concatenate([p[n] for p in Ft]) for n in stages}
        named = dict(LAYERS[b])
        for n in stages:
            aucs, maes = probe(Fv[n], Tv, Sv, Pv, Ft[n], Tt, St, dev)
            depth.setdefault(b, {}).setdefault(n, {})[ver] = (float(np.nanmean(aucs)), float(np.mean(maes)))
            if n in named:
                rows.append((label, named[n], aucs, maes, ver))
            print(label, n, np.round(aucs, 3), np.round(maes, 3), flush=True)
        del model, Fv, Ft
        torch.cuda.empty_cache()

    out = Path(a.paper_results)
    out.mkdir(parents=True, exist_ok=True)
    vers = ["default", "aug", "dacmnorm", "dacmshuf"]
    for b, st in depth.items():
        with open(out / f"probe_depth_{b}.csv", "w") as f:
            f.write("idx,stage," + ",".join(f"auc_{VER_KEY[v]}" for v in vers) + ","
                    + ",".join(f"mae_{VER_KEY[v]}" for v in vers) + "\n")
            for i, n in enumerate(ALL_STAGES[b]):
                d = st.get(n, {})
                g = lambda v, j: f"{d[v][j]:.4f}" if v in d else "nan"  # noqa: E731
                f.write(f"{i},{n}," + ",".join(g(v, 0) for v in vers) + "," + ",".join(g(v, 1) for v in vers) + "\n")

    lines = [r"\begin{tabular}{@{}ll" + "c" * deg.K + "cc@{}}", r"\toprule",
             r" & & \multicolumn{" + str(deg.K) + r"}{c}{Type detection (AUC)} & & Severity \\",
             r"\cmidrule(lr){3-" + str(2 + deg.K) + "}",
             "Features & Layer & " + " & ".join(k.capitalize()[:4] + "." if len(k) > 5 else k.capitalize()
                                                for k in deg.TYPES) + r" & Mean & MAE \\", r"\midrule"]
    macros = ["% generated by probe_val.py (probes fitted on validation subjects, evaluated on test subjects)"]
    last = None
    for row in rows:
        label, nl, aucs, maes = row[:4]
        ver = row[4] if len(row) > 4 else None
        bkey = "Est" if ver is None else ("unet" if label.startswith("U-Net") else "swin") + VER_KEY[ver]
        lk = {"--": "", "Early encoder": "Early", "Bottleneck": "Bottleneck", "Last decoder": "Decoder"}[nl]
        macros += [f"\\newcommand{{\\Probe{bkey}{lk}AUC}}{{{np.nanmean(aucs):.2f}}}",
                   f"\\newcommand{{\\Probe{bkey}{lk}MAE}}{{{np.mean(maes):.3f}}}"]
        if ver is not None and ver not in IN_TABLE:
            continue
        if last is not None and label.split(",")[0] != last:
            lines.append(r"\midrule")
        last = label.split(",")[0]
        lines.append(f"{label} & {nl} & " + " & ".join(f"{v:.2f}" for v in aucs) +
                     f" & {np.nanmean(aucs):.2f} & {np.mean(maes):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_probe.tex").write_text("\n".join(lines))
    (out / "numbers_probe.tex").write_text("\n".join(macros) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
