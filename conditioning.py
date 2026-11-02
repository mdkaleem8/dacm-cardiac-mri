"""Backbone-agnostic feature-wise conditioning through forward hooks.

DACM attaches to named modules of any backbone (the outputs of its resolution
stages) and modulates each output h as h' = (1 + gamma) * h + beta, with
(gamma, beta) predicted from the conditioning vector z. The prediction layers
are zero-initialised, so the wrapped network starts exactly as the backbone.
No layer of the backbone is replaced.
"""
from __future__ import annotations

import torch
import torch.nn as nn


NORM_TYPES = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
              nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)


def norm_layer_names(backbone: nn.Module):
    """Names of all channel-first normalisation layers of a backbone.

    Modulating right after normalisation (conditional normalisation) keeps the
    FiLM shift and scale from being removed by the next normalisation layer,
    which happens when stage outputs are modulated instead.
    """
    return [n for n, m in backbone.named_modules() if isinstance(m, NORM_TYPES)]


class DACM(nn.Module):
    def __init__(self, backbone: nn.Module, stage_names, cond_dim=10, emb_dim=64,
                 probe_shape=(1, 1, 224, 224), mechanism="film", cond_part="both"):
        super().__init__()
        self.backbone = backbone
        # "type": keep only the type probabilities, z = [p ; 0] (severity removed)
        self.cond_part = cond_part
        if stage_names == "norm":
            stage_names = norm_layer_names(backbone)
        self.stage_names = list(stage_names)
        self.mechanism = mechanism
        mods = dict(backbone.named_modules())
        missing = [n for n in self.stage_names if n not in mods]
        if missing:
            raise KeyError(f"stage modules not found in backbone: {missing}")

        chans, probes = {}, []
        for n in self.stage_names:
            probes.append(mods[n].register_forward_hook(
                lambda m, i, o, n=n: chans.__setitem__(n, o.shape[1])))
        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            backbone(torch.zeros(probe_shape))
        backbone.train(was_training)
        for h in probes:
            h.remove()

        self.embed = nn.Sequential(nn.Linear(cond_dim, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim), nn.SiLU())
        self.heads = nn.ModuleDict()
        for n in self.stage_names:
            lin = nn.Linear(emb_dim, 2 * chans[n])
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.heads[n.replace(".", "_")] = lin
        self._e = None
        for n in self.stage_names:
            mods[n].register_forward_hook(self._hook(n))

    def _hook(self, name):
        key = name.replace(".", "_")

        def hook(module, inputs, out):
            if self._e is None:
                return out
            g, b = self.heads[key](self._e).chunk(2, dim=1)
            shape = (out.shape[0], out.shape[1]) + (1,) * (out.dim() - 2)
            g, b = g.view(shape).to(out.dtype), b.view(shape).to(out.dtype)
            if self.mechanism == "adain":  # renormalise, then apply predicted affine
                dims = tuple(range(2, out.dim()))
                mu = out.mean(dims, keepdim=True)
                sd = out.std(dims, keepdim=True) + 1e-5
                return (out - mu) / sd * (sd * (1 + g)) + mu + b
            return out * (1 + g) + b
        return hook

    def forward(self, x, z):
        if self.cond_part == "type":
            half = z.shape[1] // 2
            z = torch.cat([z[:, :half], torch.zeros_like(z[:, half:])], 1)
        self._e = self.embed(z)
        try:
            return self.backbone(x)
        finally:
            self._e = None

    def added_parameters(self) -> int:
        return sum(p.numel() for p in self.embed.parameters()) + \
            sum(p.numel() for p in self.heads.parameters())
