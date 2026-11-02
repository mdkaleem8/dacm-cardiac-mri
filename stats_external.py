"""Statistics for the external tests (CMRxMotion and M&Ms); no model was trained or tuned on them.

Effects are differences in mean foreground Dice (points) over all labelled volumes and all seeds.
95% CIs (90% for equivalence) come from a two-level bootstrap: clusters (CMRxMotion volunteers with
all their scans; M&Ms subjects with both phases) and, independently per version, training seeds.
For DACM (norm) with z = 0 at test time the seeds are paired, because both use the same models.
Holm correction is applied within each dataset and backbone. Equivalence margin as in the main study.

The estimator's motion output (maximum over slices) is compared with the radiologists' grades by
ROC AUC and Spearman correlation, with CIs from resampling volunteers.
Writes numbers_external.tex and table_external.tex into --paper_results.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from stats_revision import MARGIN, holm, pfmt

B = 10000
RNG = np.random.default_rng(2024)
SEEDS = ["", "_s1", "_s2", "_s3", "_s4"]
BACKBONES = {"unet": "unet", "swinunetr": "swin"}
VERSIONS = ["default", "aug", "dacmnorm", "dacmzero", "dacmshuf", "augaff", "dacmtype"]
LABEL = {"default": "Default", "aug": "Augmentation-only", "dacmnorm": "DACM (norm)",
         "dacmzero": r"Module, $\mathbf{z}{=}0$", "dacmshuf": r"Module, $\mathbf{z}$ shuffled",
         "normtz": r"DACM (norm), $\mathbf{z}{=}0$ at test",
         "augaff": "Aug.-only, affine norms", "dacmtype": "DACM (norm), type only"}
SHORT = {"aug": "Aug", "dacmnorm": "Norm", "dacmzero": "Zero", "dacmshuf": "Shuf", "normtz": "NormTZ",
         "augaff": "Aff", "dacmtype": "Type"}
PAIRS = [("dacmnorm", "aug"), ("dacmzero", "aug"), ("dacmshuf", "aug"), ("dacmnorm", "dacmzero"),
         ("dacmnorm", "dacmshuf"), ("dacmnorm", "normtz"), ("augaff", "aug"), ("dacmtype", "aug")]
DATASETS = {"Cmrx": ("cmrx_eval", "volunteer"), "Mnms": ("mnms_eval", "subject")}


def load(ev, run, variant, cluster):
    f = ev / f"{run}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.variant == variant]
    return d.groupby(cluster).dice.agg(["sum", "count"])


def version(ev, bb, v, cluster):
    """Cluster sums (clusters x seeds) and counts for one version; normtz = DACM (norm) with z = 0."""
    name, var = ("dacmnorm", "zero") if v == "normtz" else (v, "est")
    parts = {s or "_s0": load(ev, f"{bb}_{name}{s}", var, cluster) for s in SEEDS}
    parts = {k: x for k, x in parts.items() if x is not None}
    if not parts:
        return None
    S = pd.DataFrame({k: x["sum"] for k, x in parts.items()})
    c = next(iter(parts.values()))["count"]
    return S, c


def boot(A, Bv, c, paired):
    """Bootstrap of the difference in mean Dice (points) between A and Bv (clusters x seeds sums)."""
    n = len(c)
    out = np.empty(B)
    for i in range(B):
        k = RNG.integers(0, n, n)
        sa = RNG.integers(0, A.shape[1], A.shape[1])
        sb = sa if paired else RNG.integers(0, Bv.shape[1], Bv.shape[1])
        tot = c[k].sum()
        out[i] = 100 * (A[k][:, sa].sum() / (tot * len(sa)) - Bv[k][:, sb].sum() / (tot * len(sb)))
    return out


def mean_of(S, c):
    return 100 * S.to_numpy().sum() / (c.to_numpy().sum() * S.shape[1])


def compare(data, dkey, bkey, macros):
    rows = []
    for v1, v2 in PAIRS:
        if data.get(v1) is None or data.get(v2) is None:
            continue
        (S1, c1), (S2, _) = data[v1], data[v2]
        S1, S2 = S1.align(S2, join="inner", axis=0)
        c = c1.loc[S1.index].to_numpy()
        a, b = S1.to_numpy(), S2.to_numpy()
        d = 100 * (a.sum() / (c.sum() * a.shape[1]) - b.sum() / (c.sum() * b.shape[1]))
        bs = boot(a, b, c, paired=(v2 == "normtz"))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lo90, hi90 = np.percentile(bs, [5, 95])
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / B)
        rows.append(dict(v1=v1, v2=v2, delta=d, lo=lo, hi=hi, p=p,
                         equiv=bool(lo90 > -MARGIN and hi90 < MARGIN)))
    # Holm within the original comparisons; the review controls (augaff, dacmtype) form their own family
    new = {"augaff", "dacmtype"}
    fam_a = [r for r in rows if r["v1"] not in new]
    fam_b = [r for r in rows if r["v1"] in new]
    adj = {}
    for fam in (fam_a, fam_b):
        if fam:
            adj.update({id(r): h for r, h in zip(fam, holm([r["p"] for r in fam]))})
    for r in rows:
        h = r["holm"] = adj[id(r)]
        key = f"Ext{dkey}{bkey}{SHORT[r['v1']]}Vs{SHORT[r['v2']]}"
        macros += [f"\\newcommand{{\\{key}Delta}}{{{r['delta']:.2f}}}",
                   f"\\newcommand{{\\{key}Lo}}{{{r['lo']:.2f}}}",
                   f"\\newcommand{{\\{key}Hi}}{{{r['hi']:.2f}}}",
                   f"\\newcommand{{\\{key}Holm}}{{{pfmt(h)}}}",
                   f"\\newcommand{{\\{key}Equiv}}{{{'yes' if r['equiv'] else 'no'}}}"]
    return rows


def subgroup(ev, bb, cluster, col, dkey, bkey, macros):
    """Mean Dice per subgroup (grade or vendor) for augmentation-only and DACM (norm), with the difference."""
    out = {}
    for v in ("aug", "dacmnorm", "dacmzero"):
        frames = []
        for s in SEEDS:
            f = ev / f"{bb}_{v}{s}.csv"
            if f.exists():
                d = pd.read_csv(f)
                frames.append(d[d.variant == "est"])
        if frames:
            d = pd.concat(frames)
            out[v] = 100 * d.groupby(col).dice.mean()
    for g in out["aug"].index:
        gk = {1: "GradeOne", 2: "GradeTwo", 3: "GradeThree"}.get(g, str(g).replace(" ", ""))
        for v, m in out.items():
            macros.append(f"\\newcommand{{\\Ext{dkey}{bkey}{SHORT[v]}{gk}}}{{{m[g]:.1f}}}")
        macros.append(f"\\newcommand{{\\Ext{dkey}{bkey}NormVsAug{gk}}}{{{out['dacmnorm'][g] - out['aug'][g]:.2f}}}")
    return out


def estimator(ev, macros):
    e = pd.read_csv(ev / "estimator.csv")
    vols = e.volunteer.to_numpy()
    uv = np.unique(vols)
    res = {}
    for name, y in (("Two", e.grade >= 2), ("Three", e.grade == 3)):
        res[name] = roc_auc_score(y, e.p_motion_max)
    res["Rho"] = spearmanr(e.grade, e.p_motion_max).statistic
    bs = {k: [] for k in res}
    for _ in range(2000):
        pick = RNG.choice(uv, len(uv))
        s = pd.concat([e[vols == v] for v in pick])
        for name, y in (("Two", s.grade >= 2), ("Three", s.grade == 3)):
            if y.nunique() == 2:
                bs[name].append(roc_auc_score(y, s.p_motion_max))
        bs["Rho"].append(spearmanr(s.grade, s.p_motion_max).statistic)
    for k, v in res.items():
        lo, hi = np.percentile(bs[k], [2.5, 97.5])
        nm = "Rho" if k == "Rho" else f"AUC{k}"
        macros += [f"\\newcommand{{\\Cmrx{nm}}}{{{v:.2f}}}", f"\\newcommand{{\\Cmrx{nm}Lo}}{{{lo:.2f}}}",
                   f"\\newcommand{{\\Cmrx{nm}Hi}}{{{hi:.2f}}}"]
    g = e.grade.value_counts()
    macros += [f"\\newcommand{{\\CmrxVols}}{{{len(e)}}}", f"\\newcommand{{\\CmrxVolunteers}}{{{len(uv)}}}"]
    macros += [f"\\newcommand{{\\CmrxGrade{w}}}{{{int(g.get(i, 0))}}}" for i, w in ((1, "One"), (2, "Two"), (3, "Three"))]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--paper_results", required=True)
    ap.add_argument("--mnms_cache", default="work/mnms_cache")
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.paper_results)
    macros = ["% generated by stats_external.py"]

    idx = pd.read_csv(Path(a.mnms_cache) / "index.csv").drop_duplicates("subject")
    macros.append(f"\\newcommand{{\\MnmsN}}{{{len(idx)}}}")
    macros += [f"\\newcommand{{\\Mnms{v}}}{{{n}}}" for v, n in idx.vendor.value_counts().items()]
    macros.append(f"\\newcommand{{\\MnmsLabelFrames}}{{{int((idx.frame_src == 'labels').sum())}}}")
    macros.append(f"\\newcommand{{\\MnmsLabelFramesVendor}}{{{', '.join(sorted(idx[idx.frame_src == 'labels'].vendor.unique()))}}}")

    est = estimator(work / "cmrx_eval", macros)
    print("estimator vs grade:", {k: round(v, 3) for k, v in est.items()})

    table, nseeds = {}, {}
    for dkey, (folder, cluster) in DATASETS.items():
        ev = work / folder
        for bb, bkey in BACKBONES.items():
            data = {v: version(ev, bb, v, cluster) for v in VERSIONS + ["normtz"]}
            means = {}
            for v, x in data.items():
                if x is None:
                    continue
                S, c = x
                seedmeans = 100 * S.sum() / c.sum()
                means[v] = (mean_of(S, c), seedmeans.std(ddof=1) if S.shape[1] > 1 else np.nan, S.shape[1])
                macros.append(f"\\newcommand{{\\Ext{dkey}{bkey}{SHORT.get(v, v.capitalize())}Mean}}{{{means[v][0]:.1f}}}")
            nseeds[(dkey, bkey)] = means["aug"][2]
            macros.append(f"\\newcommand{{\\Ext{dkey}{bkey}Seeds}}{{{means['aug'][2]}}}")
            rows = compare(data, dkey, bkey, macros)
            col = "grade" if dkey == "Cmrx" else "vendor"
            sub = subgroup(ev, bb, cluster, col, dkey, bkey, macros)
            table[(dkey, bkey)] = (means, rows, sub)
            print(dkey, bkey, {v: round(m[0], 2) for v, m in means.items()})
            for r in rows:
                print(f"   {r['v1']} vs {r['v2']}: {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                      f"holm {pfmt(r['holm'])} equiv {r['equiv']}")

    cols = [("Cmrx", "unet"), ("Cmrx", "swin"), ("Mnms", "unet"), ("Mnms", "swin")]
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{CMRxMotion} & \multicolumn{2}{c}{M\&Ms} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r" & U-Net & Swin-UNETR & U-Net & Swin-UNETR \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\emph{Mean foreground Dice (\%), SD over seeds}} \\"]
    for v in VERSIONS + ["normtz"]:
        cells = []
        for k in cols:
            m = table[k][0].get(v)
            cells.append("--" if m is None else (f"{m[0]:.1f}" if np.isnan(m[1]) else f"{m[0]:.1f} ({m[1]:.2f})"))
        L.append(f"{LABEL[v]} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{Difference in points [95\% CI]}} \\"]
    for v1, v2 in PAIRS:
        cells = []
        for k in cols:
            r = next((r for r in table[k][1] if r["v1"] == v1 and r["v2"] == v2), None)
            if r is None:
                cells.append("--")
                continue
            mark = "$^{*}$" if r["holm"] < 0.05 else ""
            mark += "$^{\\dagger}$" if r["equiv"] else ""
            cells.append(f"{r['delta']:+.2f}{mark} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
        name = f"{LABEL[v1]} vs {LABEL[v2][0].lower() + LABEL[v2][1:] if v2 == 'aug' else LABEL[v2]}"
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_external.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "numbers_external.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print("seeds:", nseeds)
    print("wrote", out / "table_external.tex", "and numbers_external.tex")


if __name__ == "__main__":
    main()
