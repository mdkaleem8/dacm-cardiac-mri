"""Volume metrics: Dice, HD95 and clinical indices. Labels: 1 RV, 2 Myo, 3 LV."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

CLASSES = {1: "RV", 2: "Myo", 3: "LV"}


def dice(pred, gt):
    a, b = pred.sum(), gt.sum()
    if a + b == 0:
        return np.nan
    return 2.0 * np.logical_and(pred, gt).sum() / (a + b)


def hd95(pred, gt, spacing):
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    ps = pred ^ binary_erosion(pred)
    gs = gt ^ binary_erosion(gt)
    d_to_g = distance_transform_edt(~gs, sampling=spacing)[ps]
    d_to_p = distance_transform_edt(~ps, sampling=spacing)[gs]
    return float(np.percentile(np.concatenate([d_to_g, d_to_p]), 95))


def volume_ml(mask, spacing):
    return float(mask.sum() * np.prod(spacing) / 1000.0)


def clinical(seg_ed, seg_es, spacing):
    """LV/RV volumes (mL), ejection fractions (%) and LV mass (g)."""
    lv_edv, lv_esv = volume_ml(seg_ed == 3, spacing), volume_ml(seg_es == 3, spacing)
    rv_edv, rv_esv = volume_ml(seg_ed == 1, spacing), volume_ml(seg_es == 1, spacing)
    ef = lambda edv, esv: 100.0 * (edv - esv) / edv if edv > 0 else np.nan
    return {"LV_EDV": lv_edv, "LV_ESV": lv_esv, "LV_EF": ef(lv_edv, lv_esv),
            "RV_EDV": rv_edv, "RV_ESV": rv_esv, "RV_EF": ef(rv_edv, rv_esv),
            "LV_mass": 1.05 * volume_ml(seg_ed == 2, spacing)}


def auc(scores, labels):
    """Area under the ROC curve via the rank statistic."""
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
