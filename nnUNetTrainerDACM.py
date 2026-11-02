"""nnU-Net v2 trainers for the official-pipeline baselines (copied into the
nnunetv2 package by install_trainers.py so that nnUNetv2_train finds them).

  nnUNetTrainer_250epochs  : official nnU-Net (built in), 250 epochs
  nnUNetTrainerDeg_250epochs: the same, plus our degradation simulator as the
                              first augmentation step (identical type mix and
                              severity range as Augmentation-only and DACM)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from dacm import degradations as deg  # requires the dacm project on PYTHONPATH


class DegradationTransform(BasicTransform):
    """Apply 0, 1 or 2 simulated degradations (p = 0.2, 0.5, 0.3) to a 2D training sample.

    nnU-Net hands over z-scored images; each slice is mapped to [0, 1] with its
    0.5/99.5 percentiles (the range our simulator was calibrated on), degraded,
    and z-scored again, as in our own pipeline.
    """

    def __init__(self, calib, p_types=(0.2, 0.5, 0.3), max_sev=1.0):
        super().__init__()
        self.calib, self.p_types, self.max_sev = calib, p_types, max_sev
        self._rng = None

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng((os.getpid() * 7919 + int.from_bytes(os.urandom(4), "little")) % 2 ** 32)
        return self._rng

    def apply(self, data_dict, **params):
        img = data_dict["image"]  # (C, X, Y) torch tensor for 2D
        arr = img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
        out = np.empty_like(arr, dtype=np.float32)
        for c in range(arr.shape[0]):
            x = arr[c].astype(np.float32)
            lo, hi = np.percentile(x, [0.5, 99.5])
            u = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
            spec = deg.random_spec(self.rng, self.calib, self.p_types, self.max_sev)
            if spec:
                u = deg.apply_spec(u, spec, self.rng)
            out[c] = (u - u.mean()) / (u.std() + 1e-6)
        data_dict["image"] = torch.from_numpy(out) if isinstance(img, torch.Tensor) else out
        return data_dict


class nnUNetTrainerDeg_250epochs(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None):
        base = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label)
        calib = json.loads(Path(os.environ["DACM_CALIB"]).read_text())
        return ComposeTransforms([DegradationTransform(calib)] + list(base.transforms))
