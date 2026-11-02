"""Five MRI degradation types with a common severity scale.

Each degradation takes an image scaled to [0, 1] and a strength r in [0, 1]
(the physical parameter as a fraction of its maximum). Severity is defined
from the image itself (methodology, Eq. severity):

    s_k = min(1, D(x_k, x) / kappa_k),
    D(x_k, x) = RMS(x_k - x) / SD(x)   within a box around the heart,

where x_k is the image degraded by type k alone and kappa_k is the median of
D for type k at maximum strength over training slices. (SSIM was tried
first but is almost insensitive to smooth bias fields.) calibrate.py estimates
kappa_k and a monotone map r -> median s, which is inverted to sample r for a
target severity.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.metrics import structural_similarity

TYPES = ["motion", "noise", "blur", "bias", "undersample"]
K = len(TYPES)
# Order in which combined degradations are applied (image-domain bias first,
# k-space effects next, blur, and thermal noise last).
APPLY_ORDER = ["bias", "motion", "undersample", "blur", "noise"]

# Maximum physical parameters (r = 1). Pixel size is 1.25 mm.
MOTION_MAX_PX = 10.0      # 12.5 mm peak breathing displacement
NOISE_MAX_SIGMA = 0.12    # Rician sigma, fraction of [0, 1] range
BLUR_MAX_SIGMA = 2.5      # Gaussian sigma in pixels
BIAS_MAX_LOG = 0.8        # peak log-amplitude of the bias field
US_MAX_R = 4.0            # Cartesian acceleration factor
US_CENTER = 0.12          # fully sampled central fraction of k-space


def _fft(x):
    return np.fft.fftshift(np.fft.fft2(x))


def _ifft_mag(X):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(X))).astype(np.float32)


def motion(x, r, rng):
    """Respiratory motion in k-space: per-heartbeat in-plane translation.

    Phase-encoding lines are split into B contiguous segments (one per
    heartbeat). Segment b is acquired with the heart displaced by
    d_b = A sin(2 pi b / P + phi) along a random direction, applied exactly
    through the Fourier shift theorem. Displacements are demeaned with respect
    to the segment containing the k-space centre so that the anatomy stays
    registered with the clean label map.
    """
    if r <= 0:
        return x.copy()
    H, W = x.shape
    X = _fft(x)
    pe_axis = int(rng.integers(0, 2))
    n = H if pe_axis == 0 else W
    B = int(rng.integers(8, 17))
    seg = (np.arange(n) * B) // n
    P = rng.uniform(3.0, 8.0)
    phi = rng.uniform(0, 2 * np.pi)
    theta = rng.uniform(0, 2 * np.pi)
    A = r * MOTION_MAX_PX
    d = A * np.sin(2 * np.pi * np.arange(B) / P + phi)
    d = d - d[seg[n // 2]]
    dy_line, dx_line = d[seg] * np.sin(theta), d[seg] * np.cos(theta)
    ky = np.fft.fftshift(np.fft.fftfreq(H))
    kx = np.fft.fftshift(np.fft.fftfreq(W))
    if pe_axis == 0:
        phase = ky[:, None] * dy_line[:, None] + kx[None, :] * dx_line[:, None]
    else:
        phase = ky[:, None] * dy_line[None, :] + kx[None, :] * dx_line[None, :]
    return _ifft_mag(X * np.exp(-2j * np.pi * phase))


def noise(x, r, rng):
    """Rician noise on the magnitude image."""
    if r <= 0:
        return x.copy()
    sig = r * NOISE_MAX_SIGMA
    n1 = rng.normal(0, sig, x.shape)
    n2 = rng.normal(0, sig, x.shape)
    return np.sqrt((x + n1) ** 2 + n2 ** 2).astype(np.float32)


def blur(x, r, rng):
    if r <= 0:
        return x.copy()
    return gaussian_filter(x, sigma=r * BLUR_MAX_SIGMA).astype(np.float32)


def bias(x, r, rng):
    """Smooth multiplicative field from a random polynomial of order <= 3.

    The log-field is scaled so that its peak magnitude over the central half of
    the image (where the heart lies) equals r * BIAS_MAX_LOG, and clipped
    elsewhere, so the field actually varies across the heart.
    """
    if r <= 0:
        return x.copy()
    H, W = x.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    f = np.zeros_like(yy)
    for i in range(4):
        for j in range(4 - i):
            if i + j == 0:
                continue
            f += rng.uniform(-1, 1) * (yy ** i) * (xx ** j)
    f = f - f[H // 4:3 * H // 4, W // 4:3 * W // 4].mean()
    f = f / (np.abs(f[H // 4:3 * H // 4, W // 4:3 * W // 4]).max() + 1e-6)
    f = np.clip(f, -1.5, 1.5)
    return (x * np.exp(r * BIAS_MAX_LOG * f)).astype(np.float32)


def undersample(x, r, rng):
    """Cartesian undersampling with a fully sampled centre, zero-filled."""
    if r <= 0:
        return x.copy()
    H, W = x.shape
    X = _fft(x)
    pe_axis = int(rng.integers(0, 2))
    n = H if pe_axis == 0 else W
    R = 1.0 + r * (US_MAX_R - 1.0)
    n_keep = max(4, int(round(n / R)))
    n_c = min(n_keep, max(4, int(round(US_CENTER * n))))
    c0 = n // 2 - n_c // 2
    keep = np.zeros(n, bool)
    keep[c0:c0 + n_c] = True
    outer = np.flatnonzero(~keep)
    extra = n_keep - n_c
    if extra > 0:
        keep[rng.choice(outer, size=min(extra, outer.size), replace=False)] = True
    mask = keep[:, None] if pe_axis == 0 else keep[None, :]
    return _ifft_mag(X * mask)


FUNCS = {"motion": motion, "noise": noise, "blur": blur, "bias": bias, "undersample": undersample}


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
def heart_box(label: np.ndarray | None, shape, margin=12):
    """Bounding box around the heart (dilated labels), or the central half."""
    H, W = shape
    if label is not None and label.any():
        m = binary_dilation(label > 0, iterations=margin)
        ys, xs = np.nonzero(m)
        return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)
    return slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4)


def one_minus_ssim(xd, x, box):
    """Kept for reporting only; not used for severity."""
    a, b = x[box], xd[box]
    if min(a.shape) < 7:
        return 0.0
    return float(1.0 - structural_similarity(a, b, data_range=1.0, win_size=7))


def distortion(xd, x, box):
    """Normalised RMS error in the heart box: RMS(xd - x) / SD(x)."""
    a, b = x[box], xd[box]
    return float(np.sqrt(np.mean((b - a) ** 2)) / (a.std() + 1e-6))


def r_for_severity(calib, k_name, s_target):
    c = calib[k_name]
    return float(np.interp(s_target, c["s_median"], c["r_grid"]))


def severity_from_image(calib, k_name, xd, x, box):
    return min(1.0, distortion(xd, x, box) / calib[k_name]["kappa"])


# ---------------------------------------------------------------------------
# Random specifications and application
# ---------------------------------------------------------------------------
def random_spec(rng, calib, p_types=(0.2, 0.5, 0.3), max_sev=1.0):
    """List of (type_name, r). 0, 1 or 2 types with target severity ~ U(0.05, max_sev)."""
    n = int(rng.choice(len(p_types), p=np.asarray(p_types) / np.sum(p_types)))
    names = list(rng.choice(TYPES, size=n, replace=False)) if n else []
    return [(k, r_for_severity(calib, k, rng.uniform(0.05, max_sev))) for k in names]


def apply_spec(x, spec, rng):
    d = dict(spec)
    out = x
    for k in APPLY_ORDER:
        if k in d:
            out = FUNCS[k](out, d[k], rng)
    return out.astype(np.float32)


def apply_with_severity(x, spec, rng, calib, label=None):
    """Apply a combined degradation and measure per-type severity.

    Severity of type k is measured on x degraded by type k alone, with the same
    random draws as in the combined image (state re-seeded per type).
    """
    t = np.zeros(K, np.float32)
    s = np.zeros(K, np.float32)
    if not spec:
        return x.astype(np.float32), t, s
    seeds = {k: int(rng.integers(0, 2 ** 31)) for k, _ in spec}
    d = dict(spec)
    out = x
    for k in APPLY_ORDER:
        if k in d:
            out = FUNCS[k](out, d[k], np.random.default_rng(seeds[k]))
    box = heart_box(label, x.shape)
    for k, r in spec:
        xk = FUNCS[k](x, r, np.random.default_rng(seeds[k]))
        i = TYPES.index(k)
        t[i] = 1.0
        s[i] = severity_from_image(calib, k, xk, x, box)
    return out.astype(np.float32), t, s
