"""Revision statistics over all available seeds (seed 0 = "", then _s1.._s4).

For every pairwise comparison of versions:
  * effect: difference in mean foreground Dice (points), averaged over seeds and subjects;
  * 95% CI (and 90% CI for equivalence) from a two-level bootstrap that resamples test
    subjects and, independently for each version, training seeds;
  * two-sided p-value from the same bootstrap (consistent with the CI), Holm-corrected within a
    stated family; a Welch t-test on per-seed mean Dice is kept as a sensitivity check;
  * equivalence (TOST at the 5% level, margin +-0.5 Dice points): the 90% CI lies inside the margin.
Families: main experiment = all comparisons of one backbone at one severity level;
stress tests = all comparisons of both stress tests at s=0.6 and s=1.0 together.

Also: five-seed clinical errors with bootstrap CIs, Bland-Altman limits for repeated
measurements per subject, per-pathology Dice, and seed bands for the severity curves.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

TYPES = ["motion", "noise", "blur", "bias", "undersample"]
SEEDS = ["", "_s1", "_s2", "_s3", "_s4"]
LEVELS = [(0.0, "Clean"), (0.6, "Six"), (1.0, "Ten")]
SHORT = {"aug": "Aug", "dacmnorm": "Norm", "dacmnormr": "NormR", "dacmzero": "Zero", "dacmshuf": "Shuf",
         "augaff": "Aff", "dacmtype": "Type"}
LABEL = {"aug": "Augmentation-only", "dacmnorm": "DACM (norm)", "dacmnormr": "DACM (norm), matched estimator",
         "dacmzero": r"Module, $\mathbf{z}{=}0$", "dacmshuf": r"Module, $\mathbf{z}$ shuffled",
         "augaff": "Aug.-only, affine norms", "dacmtype": "DACM (norm), type only"}
# Review controls: one extra version per backbone, compared with the versions it is meant to explain
EXTRA = {"unet": ("dacmtype", [("dacmtype", "aug"), ("dacmnorm", "dacmtype")]),
         "swinunetr": ("augaff", [("augaff", "aug"), ("dacmzero", "augaff")])}
MARGIN = 0.5
MARGIN_Q = 0.25  # sensitivity check of the equivalence margin
B = 10000
RNG = np.random.default_rng(12345)


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def subject_matrix(root, name, level, col="dice"):
    """Subjects x seeds matrix of mean foreground Dice (or HD95) at one level, averaged over types."""
    cols = {}
    for s in SEEDS:
        f = root / f"{name}{s}" / "eval_seg.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.variant == "est"]
        d = d[d.type == "clean"] if level == 0 else d[d.type.isin(TYPES) & (d.level == level)]
        cols[s or "_s0"] = d.groupby("pid")[col].mean()
    return pd.DataFrame(cols)


def boot_diff(m1, m2, b=B):
    """Two-level bootstrap of mean(m1) - mean(m2); m: subjects x seeds arrays on the same subjects."""
    n = m1.shape[0]
    subj = RNG.integers(0, n, (b, n))
    s1 = RNG.integers(0, m1.shape[1], (b, m1.shape[1]))
    s2 = RNG.integers(0, m2.shape[1], (b, m2.shape[1]))
    out = np.empty(b)
    for i in range(b):
        out[i] = m1[subj[i]][:, s1[i]].mean() - m2[subj[i]][:, s2[i]].mean()
    return out


def compare(mats, pairs):
    rows = []
    for v1, v2 in pairs:
        m1, m2 = mats[v1].align(mats[v2], join="inner", axis=0)
        a1, a2 = 100 * m1.to_numpy(), 100 * m2.to_numpy()
        d = a1.mean() - a2.mean()
        bs = boot_diff(a1, a2)
        lo95, hi95 = np.percentile(bs, [2.5, 97.5])
        lo90, hi90 = np.percentile(bs, [5, 95])
        seed1, seed2 = a1.mean(0), a2.mean(0)
        # two-sided bootstrap p-value (consistent with the CI); seed-level Welch test as a sensitivity check
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / len(bs))
        p_seed = ttest_ind(seed1, seed2, equal_var=False).pvalue
        k = min(len(seed1), len(seed2))
        wins = int((seed1[:k] > seed2[:k]).sum())
        rows.append(dict(v1=v1, v2=v2, delta=d, lo=lo95, hi=hi95, lo90=lo90, hi90=hi90, p=p, p_seed=p_seed,
                         wins=wins, nseeds=k, equiv=bool(lo90 > -MARGIN and hi90 < MARGIN),
                         equivq=bool(lo90 > -MARGIN_Q and hi90 < MARGIN_Q)))
    return rows


def ordered_pairs(versions):
    pairs = []
    for v1, v2 in combinations(versions, 2):
        pairs.append((v2, v1) if v1 == "aug" else (v1, v2))  # always "X vs Aug"
    return pairs


def write_compare(rows_by_level, key, out_tex, macros, label_rows):
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Comparison & $\Delta$ (points) & 95\% CI & $p$ (Holm) & Seeds ahead & Equivalent$^\dagger$ \\",
             r"\midrule"]
    for lt, rows in rows_by_level.items():
        for r in rows:
            name = f"{SHORT[r['v1']]}Vs{SHORT[r['v2']]}"
            suf = "" if lt == "Ten" else lt
            macros += [f"\\newcommand{{\\Cmp{key}{name}{suf}Delta}}{{{r['delta']:.2f}}}",
                       f"\\newcommand{{\\Cmp{key}{name}{suf}Lo}}{{{r['lo']:.2f}}}",
                       f"\\newcommand{{\\Cmp{key}{name}{suf}Hi}}{{{r['hi']:.2f}}}",
                       f"\\newcommand{{\\Cmp{key}{name}{suf}P}}{{{pfmt(r['p'])}}}",
                       f"\\newcommand{{\\Cmp{key}{name}{suf}Holm}}{{{pfmt(r['holm'])}}}",
                       f"\\newcommand{{\\Cmp{key}{name}{suf}PSeed}}{{{pfmt(r['p_seed'])}}}",
                       f"\\newcommand{{\\Cmp{key}{name}{suf}Wins}}{{{r['wins']}}}",
                       f"\\newcommand{{\\Cmp{key}{name}{suf}Equiv}}{{{'yes' if r['equiv'] else 'no'}}}",
                       f"\\newcommand{{\\Cmp{key}{name}{suf}EquivQ}}{{{'yes' if r['equivq'] else 'no'}}}"]
            if lt == label_rows:
                lines.append(f"{LABEL[r['v1']]} vs {LABEL[r['v2']].replace('Augmentation-only', 'aug.-only')} & "
                             f"{r['delta']:+.2f} & [{r['lo']:+.2f}, {r['hi']:+.2f}] & {pfmt(r['holm'])} & "
                             f"{r['wins']}/{r['nseeds']} & {'yes' if r['equiv'] else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex.write_text("\n".join(lines))


def seed_table(mats_by_level, versions, key, out_tex, macros):
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Version & Clean & $s{=}0.6$ & $s{=}1.0$ \\", r"\midrule"]
    n = None
    for v in versions:
        cells = []
        for _, lt in LEVELS:
            per_seed = 100 * mats_by_level[lt][v].mean(axis=0).to_numpy()
            n = len(per_seed)
            m, sd = per_seed.mean(), per_seed.std(ddof=1)
            cells.append(f"{m:.1f} $\\pm$ {sd:.2f}")
            macros += [f"\\newcommand{{\\Seed{key}{v}{lt}}}{{{m:.1f}}}",
                       f"\\newcommand{{\\Seed{key}{v}{lt}SD}}{{{sd:.2f}}}"]
        lines.append(f"{LABEL[v]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex.write_text("\n".join(lines))
    macros.append(f"\\newcommand{{\\Seed{key}N}}{{{n}}}")


def experiment(root, out, backbone, tag, versions, macros, extra_pairs=()):
    key = backbone.replace("swinunetr", "swin") + {"": "", "_low": "Low", "_cap": "Cap"}[tag]
    versions = [v for v in versions if (root / f"{backbone}_{v}{tag}" / "eval_seg.csv").exists()]
    mats = {lt: {v: subject_matrix(root, f"{backbone}_{v}{tag}", lv) for v in versions} for lv, lt in LEVELS}
    common = sorted(set.intersection(*[set(m.columns) for d in mats.values() for m in d.values()]))
    mats = {lt: {v: m[common] for v, m in d.items()} for lt, d in mats.items()}
    seed_table(mats, versions, key, out / f"table_seeds_{backbone}{tag}.tex", macros)
    base = [v for v in versions if v not in {e for e, _ in EXTRA.values()}]
    pairs = ordered_pairs(base) + [p for p in extra_pairs if p[0] in versions and p[1] in versions]
    return key, {lt: compare(mats[lt], pairs) for _, lt in LEVELS}, mats


def clinical(root, out, backbone, versions, macros):
    """Five-seed mean absolute errors at s=1.0 and clean, with bootstrap CIs of DACM (norm) minus control."""
    key = backbone.replace("swinunetr", "swin")
    idx = ["LV_EF", "LV_EDV", "LV_ESV", "LV_mass", "RV_EF"]
    res = {}
    for v in versions:
        per_seed = []
        for s in SEEDS:
            f = root / f"{backbone}_{v}{s}" / "eval_clinical.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[d.variant == "est"]
            for lv, lt in [(0.0, "Clean"), (1.0, "Ten")]:
                dd = d[d.type == "clean"] if lv == 0 else d[d.type.isin(TYPES) & (d.level == lv)]
                err = pd.DataFrame({k: (dd[f"pred_{k}"] - dd[f"ref_{k}"]).abs().to_numpy() for k in idx})
                err["pid"] = dd.pid.to_numpy()
                per_seed.append((s or "_s0", lt, err.groupby("pid").mean()))
        res[v] = per_seed
    lines = [r"\begin{tabular}{@{}ll" + "c" * (2 * len(idx)) + "@{}}", r"\toprule"]
    head = {"LV_EF": r"LV EF (\%)", "LV_EDV": "LV EDV (mL)", "LV_ESV": "LV ESV (mL)",
            "LV_mass": "LV mass (g)", "RV_EF": r"RV EF (\%)"}
    lines.append(" & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{head[k]}}}" for k in idx) + r" \\")
    lines.append("Backbone & Version & " + " & ".join(["Clean & $s{=}1.0$"] * len(idx)) + r" \\")
    lines.append(r"\midrule")
    first = True
    mats = {}
    for v in versions:
        cells = []
        for k in idx:
            for lt in ("Clean", "Ten"):
                m = pd.DataFrame({s: e[k] for s, l, e in res[v] if l == lt})
                mats[(v, lt, k)] = m
                cells.append(f"{m.to_numpy().mean():.1f}")
                macros.append(f"\\newcommand{{\\Clin{key}{v}{k.replace('_', '')}{lt}}}{{{m.to_numpy().mean():.1f}}}")
        lines.append(f"{'U-Net' if backbone == 'unet' else 'Swin-UNETR'} & {LABEL[v]} & " + " & ".join(cells) + r" \\"
                     if first else f" & {LABEL[v]} & " + " & ".join(cells) + r" \\")
        first = False
    # DACM (norm) minus augmentation-only, s = 1.0, for every index
    for k in idx:
        a1, a2 = mats[("dacmnorm", "Ten", k)].align(mats[("aug", "Ten", k)], join="inner", axis=0)
        bs = boot_diff(a1.to_numpy(), a2.to_numpy(), 4000)
        d = a1.to_numpy().mean() - a2.to_numpy().mean()
        lo, hi = np.percentile(bs, [2.5, 97.5])
        kk = k.replace("_", "")
        macros += [f"\\newcommand{{\\Clin{key}NormVsAug{kk}Delta}}{{{d:.2f}}}",
                   f"\\newcommand{{\\Clin{key}NormVsAug{kk}Lo}}{{{lo:.2f}}}",
                   f"\\newcommand{{\\Clin{key}NormVsAug{kk}Hi}}{{{hi:.2f}}}"]
    return lines


def bland_altman(root, out, macros):
    """LV EF limits of agreement at s=1.0 for repeated measurements per subject
    (five degradation types per subject), seed 0 (Bland and Altman, 2007)."""
    for b in ("unet", "swinunetr"):
        key = b.replace("swinunetr", "swin")
        for v in ("aug", "dacmnorm"):
            f = root / f"{b}_{v}" / "eval_clinical.csv"
            d = pd.read_csv(f)
            d = d[(d.variant == "est") & d.type.isin(TYPES) & (d.level == 1.0)]
            diff = d.pred_LV_EF - d.ref_LV_EF
            g = pd.DataFrame({"pid": d.pid, "diff": diff}).dropna()
            m = g.groupby("pid")["diff"].agg(["mean", "var", "count"])
            mbar = m["count"].mean()
            s2_between = m["mean"].var(ddof=1)
            s2_within = m["var"].mean()
            sd = np.sqrt(s2_between + (1 - 1 / mbar) * s2_within)
            bias = g["diff"].mean()
            macros += [f"\\newcommand{{\\BArm{key}{v}Bias}}{{{bias:.1f}}}",
                       f"\\newcommand{{\\BArm{key}{v}Lo}}{{{bias - 1.96 * sd:.1f}}}",
                       f"\\newcommand{{\\BArm{key}{v}Hi}}{{{bias + 1.96 * sd:.1f}}}"]


def pathology(root, out, cache, macros):
    meta = json.loads((cache / "meta.json").read_text())
    groups = ["NOR", "MINF", "DCM", "HCM", "RV"]
    gname = {"NOR": "Normal", "MINF": "Previous MI", "DCM": "Dilated CM", "HCM": "Hypertrophic CM", "RV": "Abnormal RV"}
    lines = [r"\begin{tabular}{@{}l" + "cccc" * 2 + "@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{U-Net (nnU-Net configuration)} & \multicolumn{4}{c}{Swin-UNETR} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             r"Group & Default & Aug.-only & DACM (norm) & $\Delta$ [95\% CI] & Default & Aug.-only & DACM (norm) & $\Delta$ [95\% CI] \\",
             r"\midrule"]
    cells = {g: [] for g in groups}
    for b in ("unet", "swinunetr"):
        key = b.replace("swinunetr", "swin")
        mats = {v: subject_matrix(root, f"{b}_{v}", 1.0) for v in ("default", "aug", "dacmnorm")}
        for g in groups:
            pids = [p for p in mats["aug"].index if meta[p]["group"] == g]
            m = {v: 100 * mats[v].loc[pids] for v in mats}
            a1, a2 = m["dacmnorm"].to_numpy(), m["aug"].to_numpy()
            bs = boot_diff(a1, a2, 4000)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            d = a1.mean() - a2.mean()
            cells[g] += [f"{m['default'].to_numpy().mean():.1f}", f"{a2.mean():.1f}", f"{a1.mean():.1f}",
                         f"${d:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$]"]
            macros += [f"\\newcommand{{\\Path{key}{g}Aug}}{{{a2.mean():.1f}}}",
                       f"\\newcommand{{\\Path{key}{g}Norm}}{{{a1.mean():.1f}}}",
                       f"\\newcommand{{\\Path{key}{g}Default}}{{{m['default'].to_numpy().mean():.1f}}}",
                       f"\\newcommand{{\\Path{key}{g}Delta}}{{{d:.2f}}}",
                       f"\\newcommand{{\\Path{key}{g}Lo}}{{{lo:.2f}}}",
                       f"\\newcommand{{\\Path{key}{g}Hi}}{{{hi:.2f}}}",
                       f"\\newcommand{{\\Path{key}{g}N}}{{{len(pids)}}}"]
    for g in groups:
        lines.append(f"{gname[g]} & " + " & ".join(cells[g]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_pathology.tex").write_text("\n".join(lines))


def baselines(root, out, macros):
    """E8: official nnU-Net (with and without our simulator) and Tent, next to our models.
    Mean over available seeds; differences to our augmentation-only model with two-level bootstrap CIs."""
    rows_def = [("unet", "U-Net (nnU-Net configuration)", [
                    ("unet_default", "Ours, default"), ("unet_aug", "Ours, augmentation-only"),
                    ("unet_dacmnorm", "Ours, DACM (norm)"), ("unet_augtent", "Ours, aug.-only + Tent"),
                    ("nnunet_default", "Official nnU-Net, default augm."),
                    ("nnunet_deg", "Official nnU-Net + our simulator")]),
                ("swinunetr", "Swin-UNETR", [
                    ("swinunetr_aug", "Ours, augmentation-only"), ("swinunetr_dacmnorm", "Ours, DACM (norm)"),
                    ("swinunetr_augtent", "Ours, aug.-only + Tent")])]
    lines = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r"Backbone & Model & Seeds & Clean & $s{=}0.6$ & $s{=}1.0$ & HD95 $s{=}1.0$ & $\Delta$ vs aug.-only, $s{=}1.0$ [95\% CI] \\",
             r"\midrule"]
    for b, bl, models in rows_def:
        ref = subject_matrix(root, f"{b}_aug", 1.0)
        first = True
        for name, label in models:
            if not (root / name / "eval_seg.csv").exists():
                continue
            m = {lt: subject_matrix(root, name, lv) for lv, lt in LEVELS}
            hd = subject_matrix(root, name, 1.0, "hd95")
            key = "Base" + name.replace("swinunetr", "swin").replace("_", "")
            vals = {lt: 100 * m[lt].to_numpy().mean() for lt in m}
            cell = "--"
            if not name.endswith("_aug"):
                a1, a2 = m["Ten"].align(ref, join="inner", axis=0)
                bs = boot_diff(100 * a1.to_numpy(), 100 * a2.to_numpy(), 4000)
                d = 100 * (a1.to_numpy().mean() - a2.to_numpy().mean())
                lo, hi = np.percentile(bs, [2.5, 97.5])
                cell = f"${d:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$]"
                macros += [f"\\newcommand{{\\{key}Delta}}{{{d:.2f}}}", f"\\newcommand{{\\{key}Lo}}{{{lo:.2f}}}",
                           f"\\newcommand{{\\{key}Hi}}{{{hi:.2f}}}"]
            macros += [f"\\newcommand{{\\{key}{lt}}}{{{v:.1f}}}" for lt, v in vals.items()]
            macros += [f"\\newcommand{{\\{key}HD}}{{{hd.to_numpy().mean():.1f}}}",
                       f"\\newcommand{{\\{key}N}}{{{m['Ten'].shape[1]}}}"]
            lines.append(f"{bl if first else ''} & {label} & {m['Ten'].shape[1]} & {vals['Clean']:.1f} & "
                         f"{vals['Six']:.1f} & {vals['Ten']:.1f} & {hd.to_numpy().mean():.1f} & {cell} \\\\")
            first = False
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
    (out / "table_baselines.tex").write_text("\n".join(lines))


def curves(root, out, backbone, versions):
    rows = []
    for lv in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = {"level": lv}
        for v in versions:
            m = 100 * subject_matrix(root, f"{backbone}_{v}", lv).mean(axis=0)
            r[f"{v}_mean"], r[f"{v}_sd"] = m.mean(), (m.std(ddof=1) if len(m) > 1 else 0.0)
            r[f"{v}_lo"], r[f"{v}_hi"] = r[f"{v}_mean"] - r[f"{v}_sd"], r[f"{v}_mean"] + r[f"{v}_sd"]
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / f"curves_seeds_{backbone}.csv", index=False, float_format="%.3f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--paper_results", required=True)
    a = ap.parse_args()
    root, out, cache = Path(a.runs_root), Path(a.paper_results), Path(a.cache)
    macros = ["% generated by stats_revision.py", f"\\newcommand{{\\EquivMargin}}{{{MARGIN}}}",
              f"\\newcommand{{\\EquivMarginQ}}{{{MARGIN_Q}}}", f"\\newcommand{{\\BootN}}{{{B}}}"]
    main_versions = ["aug", "dacmnorm", "dacmzero", "dacmshuf"]
    for b in ("unet", "swinunetr"):
        ev, ep = EXTRA[b]
        key, res, _ = experiment(root, out, b, "", main_versions + [ev], macros, ep)
        macros.append(f"\\newcommand{{\\Cmp{key}NPairs}}{{{len(res['Ten'])}}}")
        for lt, rows in res.items():  # family: one backbone, one level
            for r, h in zip(rows, holm([r["p"] for r in rows])):
                r["holm"] = h
        write_compare(res, key, out / f"table_compare_{b}.tex", macros, "Ten")
        curves(root, out, b, ["default"] + main_versions)

    stress = {}
    for tag in ("_low", "_cap"):
        stress[tag] = experiment(root, out, "unet", tag, ["aug", "dacmnormr", "dacmzero"], macros)
    # family: all stress-test comparisons at s=0.6 and s=1.0
    fam = [r for _, res, _ in stress.values() for lt in ("Six", "Ten") for r in res[lt]]
    for r, h in zip(fam, holm([r["p"] for r in fam])):
        r["holm"] = h
    for tag, (key, res, _) in stress.items():
        for r in res["Clean"]:
            r["holm"] = r["p"]  # clean level is descriptive only
        write_compare(res, key, out / f"table_compare_unet{tag}.tex", macros, "Ten")
    # the earlier full-range-estimator DACM in the stress tests, for reference (descriptive, uncorrected)
    for tag in ("_low", "_cap"):
        key = "unet" + {"_low": "Low", "_cap": "Cap"}[tag]
        m1 = subject_matrix(root, f"unet_dacmnorm{tag}", 1.0)
        m2 = subject_matrix(root, f"unet_aug{tag}", 1.0)[m1.columns.intersection(
            subject_matrix(root, f"unet_aug{tag}", 1.0).columns)]
        r = compare({"dacmnorm": m1, "aug": m2}, [("dacmnorm", "aug")])[0]
        macros += [f"\\newcommand{{\\Cmp{key}FullEstNormVsAugDelta}}{{{r['delta']:.2f}}}",
                   f"\\newcommand{{\\Cmp{key}FullEstNormVsAugLo}}{{{r['lo']:.2f}}}",
                   f"\\newcommand{{\\Cmp{key}FullEstNormVsAugHi}}{{{r['hi']:.2f}}}",
                   f"\\newcommand{{\\Cmp{key}FullEstN}}{{{m1.shape[1]}}}"]

    clin = []
    for b in ("unet", "swinunetr"):
        clin += clinical(root, out, b, ["aug", "dacmnorm", "dacmzero", "dacmshuf"], macros)
        clin.append(r"\midrule")
    clin = clin[:-1]
    head = [r"\begin{tabular}{@{}ll" + "c" * 10 + "@{}}", r"\toprule",
            r" & & \multicolumn{2}{c}{LV EF (\%)} & \multicolumn{2}{c}{LV EDV (mL)} & \multicolumn{2}{c}{LV ESV (mL)}"
            r" & \multicolumn{2}{c}{LV mass (g)} & \multicolumn{2}{c}{RV EF (\%)} \\",
            "Backbone & Version & " + " & ".join(["Clean & $s{=}1.0$"] * 5) + r" \\", r"\midrule"]
    body = [l for l in clin if not l.startswith((r"\begin", r"\toprule", " & & ", "Backbone &"))]
    body = [l for i, l in enumerate(body) if not (l == r"\midrule" and (i == 0 or body[i - 1] == r"\midrule"))]
    (out / "table_clinical_seeds.tex").write_text("\n".join(head + body + [r"\bottomrule", r"\end{tabular}"]))

    bland_altman(root, out, macros)
    pathology(root, out, cache, macros)
    baselines(root, out, macros)
    (out / "numbers_revision.tex").write_text("\n".join(macros) + "\n")
    for f in sorted(out.glob("table_compare_*.tex")):
        print("=====", f.name)
        print(f.read_text())


if __name__ == "__main__":
    main()
