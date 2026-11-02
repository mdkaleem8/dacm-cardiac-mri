"""ACDC loading, resampling and slice datasets.

Preprocessing (prepare_acdc.py) stores one .npz per subject with the ED and ES
frames resampled in-plane to TARGET_SPACING mm and centre-cropped/padded to
CROP x CROP pixels. Intensities are kept raw; each volume's 0.5/99.5
percentiles are stored so that slices can be scaled to [0, 1] before
degradation and z-scored afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from . import degradations as deg

TARGET_SPACING = 1.25  # mm, in-plane
CROP = 224             # pixels (280 mm field of view)
GROUPS = ["DCM", "HCM", "MINF", "NOR", "RV"]


# ---------------------------------------------------------------------------
# Raw ACDC reading
# ---------------------------------------------------------------------------
def read_info(path: Path) -> dict:
    info = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def _center_crop_pad(a: np.ndarray, size: int) -> np.ndarray:
    """Crop or zero-pad the last two axes of (Z, H, W) to size x size."""
    z, h, w = a.shape
    out = np.zeros((z, size, size), dtype=a.dtype)
    sh, sw = max(0, (h - size) // 2), max(0, (w - size) // 2)
    dh, dw = max(0, (size - h) // 2), max(0, (size - w) // 2)
    ch, cw = min(h, size), min(w, size)
    out[:, dh:dh + ch, dw:dw + cw] = a[:, sh:sh + ch, sw:sw + cw]
    return out


def load_frame(img_path: Path, gt_path: Path):
    """Return image (Z,H,W) float32, label (Z,H,W) uint8 and spacing (dz, dy, dx)."""
    img_nii, gt_nii = nib.load(str(img_path)), nib.load(str(gt_path))
    img = np.asarray(img_nii.dataobj, dtype=np.float32)
    gt = np.asarray(gt_nii.dataobj).astype(np.uint8)
    sx, sy, sz = [float(v) for v in img_nii.header.get_zooms()[:3]]
    fx, fy = sx / TARGET_SPACING, sy / TARGET_SPACING
    img = zoom(img, (fx, fy, 1), order=1)
    gt = zoom(gt, (fx, fy, 1), order=0)
    img = _center_crop_pad(np.transpose(img, (2, 0, 1)), CROP)
    gt = _center_crop_pad(np.transpose(gt, (2, 0, 1)), CROP)
    return img, gt, (sz, TARGET_SPACING, TARGET_SPACING)


def prepare_subject(pdir: Path) -> dict:
    info = read_info(pdir / "Info.cfg")
    pid = pdir.name
    out = {"group": info.get("Group", ""), "height": float(info.get("Height", "nan")),
           "weight": float(info.get("Weight", "nan"))}
    for phase in ("ED", "ES"):
        fr = int(info[phase])
        img, gt, spacing = load_frame(pdir / f"{pid}_frame{fr:02d}.nii.gz",
                                      pdir / f"{pid}_frame{fr:02d}_gt.nii.gz")
        lo, hi = np.percentile(img, [0.5, 99.5])
        out[f"img_{phase}"] = img
        out[f"gt_{phase}"] = gt
        out[f"range_{phase}"] = np.array([lo, hi], dtype=np.float32)
        out["spacing"] = np.array(spacing, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# Splits and caching
# ---------------------------------------------------------------------------
def load_subject(cache_dir: Path, pid: str) -> dict:
    with np.load(cache_dir / f"{pid}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def load_splits(cache_dir: Path) -> dict:
    return json.loads((cache_dir / "splits.json").read_text())


def subset_per_group(cache_dir: Path, pids: list, n_per_group: int) -> list:
    """First n_per_group subjects of each diagnostic group (in split order)."""
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    out, count = [], {}
    for pid in pids:
        g = meta[pid]["group"]
        if count.get(g, 0) < n_per_group:
            out.append(pid)
            count[g] = count.get(g, 0) + 1
    return out


def to_unit(img: np.ndarray, rng_lohi: np.ndarray) -> np.ndarray:
    lo, hi = float(rng_lohi[0]), float(rng_lohi[1])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


# ---------------------------------------------------------------------------
# Training dataset (on-the-fly degradation)
# ---------------------------------------------------------------------------
class SliceDataset(Dataset):
    """2D slices from ED and ES frames.

    mode:
      'default' - spatial/intensity augmentation only, no degradation
      'aug'     - plus random degradation (used by augmentation-only and DACM)
    Returns x (1,H,W) z-scored, y (H,W), t (K,) presence, s (K,) severity.
    """

    def __init__(self, cache_dir, pids, mode="aug", calib=None, p_types=(0.2, 0.5, 0.3),
                 train_aug=True, max_sev=1.0):
        self.mode, self.calib, self.p_types, self.train_aug = mode, calib, p_types, train_aug
        self.max_sev = max_sev
        self.items, self.vols = [], {}
        for pid in pids:
            sub = load_subject(Path(cache_dir), pid)
            for ph in ("ED", "ES"):
                key = (pid, ph)
                self.vols[key] = (sub[f"img_{ph}"], sub[f"gt_{ph}"], sub[f"range_{ph}"])
                for z in range(sub[f"img_{ph}"].shape[0]):
                    self.items.append((pid, ph, z))
        self._rng = None

    def __len__(self):
        return len(self.items)

    @property
    def rng(self):
        if self._rng is None:  # one independent stream per worker process
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._rng

    def _spatial(self, x, y):
        r = self.rng
        k = int(r.integers(0, 4))
        x, y = np.rot90(x, k), np.rot90(y, k)
        if r.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        if r.random() < 0.3:  # gamma
            x = np.clip(x, 0, 1) ** float(r.uniform(0.7, 1.5))
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def __getitem__(self, i):
        pid, ph, z = self.items[i]
        img, gt, rng_lohi = self.vols[(pid, ph)]
        x, y = to_unit(img[z], rng_lohi), gt[z].astype(np.int64)
        if self.train_aug:
            x, y = self._spatial(x, y)
        t = np.zeros(deg.K, np.float32)
        s = np.zeros(deg.K, np.float32)
        if self.mode == "aug":
            spec = deg.random_spec(self.rng, self.calib, self.p_types, self.max_sev)
            x, t, s = deg.apply_with_severity(x, spec, self.rng, self.calib, y)
        return (torch.from_numpy(zscore(x))[None], torch.from_numpy(y),
                torch.from_numpy(t), torch.from_numpy(s))
