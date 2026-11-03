"""
Noise, striping and bad-pixel handling for pushbroom and frame sensors.

For pushbroom sensors the detector array maps to ``(cols, bands)`` of the image
(each image row is one frame / scan line), so most detector artefacts appear as
along-track stripes that are constant per column and band.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .image import HSImage


def estimate_noise(image: HSImage, axis: int = 1) -> np.ndarray:
    """Per-band noise standard deviation from differences of adjacent pixels (shift-difference method)."""
    d = np.diff(image.data.astype(float), axis=axis)
    return np.nanstd(d, axis=(0, 1)) / np.sqrt(2)


def snr(image: HSImage) -> np.ndarray:
    """Per-band signal-to-noise ratio (median signal / shift-difference noise)."""
    return np.nanmedian(image.data, axis=(0, 1)) / np.maximum(estimate_noise(image), 1e-12)


# ------------------------------------------------------------- bad pixels
def detect_bad_detector_pixels(image: HSImage, z_thresh: float = 6.0) -> np.ndarray:
    """Find dead / hot detector elements of a pushbroom image.

    The column profile of each band (median over all lines) is compared against a
    smoothed version of itself; columns deviating more than ``z_thresh`` robust
    standard deviations, or that are constant along-track (dead), are flagged.

    Returns:
        ``(cols, bands)`` boolean array, ``True`` where the detector element is bad.
    """
    from scipy.ndimage import median_filter

    d = image.data.astype(float)
    prof = np.nanmedian(d, axis=0)  # (cols, bands)
    smooth = median_filter(np.nan_to_num(prof), size=(7, 1), mode='nearest')
    resid = prof - smooth
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)), axis=0) * 1.4826
    z = np.abs(resid) / np.maximum(mad, 1e-12)
    dead = np.nanstd(d, axis=0) == 0
    return (z > z_thresh) | dead | ~np.isfinite(prof)


def repair_detector_pixels(image: HSImage, bad: Optional[np.ndarray] = None, **kwds) -> HSImage:
    """Replace bad detector elements (``(cols, bands)`` mask) by interpolating neighbouring columns.

    This is the pure-numpy equivalent of the ``bpr=True`` (bad pixel replacement)
    option of the hylite sensor corrections.
    """
    if bad is None:
        bad = detect_bad_detector_pixels(image, **kwds)
    d = image.data.astype(np.float32).copy()
    cols = np.arange(image.cols)
    for b in range(image.bands):
        badc = bad[:, b]
        if not badc.any() or badc.all():
            continue
        good = ~badc
        for r in range(image.rows):
            d[r, badc, b] = np.interp(cols[badc], cols[good], d[r, good, b])
    return image.copy(d)


def remove_spikes(image: HSImage, z_thresh: float = 5.0, size: int = 3) -> HSImage:
    """Remove spectral spikes (e.g. cosmic rays, saturated elements) with a spectral median filter."""
    from scipy.ndimage import median_filter

    d = image.data.astype(np.float32)
    med = median_filter(np.nan_to_num(d), size=(1, 1, size), mode='nearest')
    resid = d - med
    mad = np.nanmedian(np.abs(resid), axis=(0, 1), keepdims=True) * 1.4826
    spikes = np.abs(resid) > z_thresh * np.maximum(mad, 1e-12)
    out = np.where(spikes, med, d)
    return image.copy(out)


# -------------------------------------------------------------- destriping
def destripe(image: HSImage, method: str = 'moment', axis: int = 0, window: Optional[int] = None) -> HSImage:
    """Remove along-track stripes caused by detector non-uniformity.

    Args:
        method: ``'moment'`` (moment matching: every column gets the global mean and
            std per band) or ``'ratio'`` (divide by column-mean / global-mean, i.e. a
            scene-derived flat field; preserves zero offsets).
        axis: the along-track axis (0 = rows are scan lines, the default for pushbroom).
        window: if given, the target statistics vary smoothly across track (moving
            average over ``window`` columns) instead of using the global value. This
            protects genuine cross-track gradients (e.g. BRDF) from being removed.
    """
    from scipy.ndimage import uniform_filter1d

    d = image.data.astype(float)
    if axis == 1:
        d = np.transpose(d, (1, 0, 2))
    mu = np.nanmean(d, axis=0, keepdims=True)
    sd = np.nanstd(d, axis=0, keepdims=True)
    if window:
        tmu = uniform_filter1d(np.nan_to_num(mu), window, axis=1, mode='nearest')
        tsd = uniform_filter1d(np.nan_to_num(sd), window, axis=1, mode='nearest')
    else:
        tmu = np.nanmean(mu, axis=1, keepdims=True)
        tsd = np.nanmean(sd, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        if method == 'moment':
            out = (d - mu) / np.where(sd > 0, sd, 1) * tsd + tmu
        elif method == 'ratio':
            out = d * (tmu / np.where(mu != 0, mu, 1))
        else:
            raise ValueError(method)
    if axis == 1:
        out = np.transpose(out, (1, 0, 2))
    return image.copy(out.astype(np.float32))


# -------------------------------------------------------------------- MNF
def mnf(image: HSImage, n_components: Optional[int] = None):
    """Minimum Noise Fraction transform.

    Returns:
        ``(components, transform)`` where components is an :class:`HSImage` and
        ``transform`` is a dict with the matrices needed to invert it.
    """
    d = image.data.astype(float)
    X = d.reshape(-1, image.bands)
    valid = np.isfinite(X).all(axis=1)
    Xv = X[valid]
    mean = Xv.mean(axis=0)
    N = np.diff(d, axis=1).reshape(-1, image.bands)
    N = N[np.isfinite(N).all(axis=1)] / np.sqrt(2)
    Cn = np.cov(N, rowvar=False) + np.eye(image.bands) * 1e-10
    Cs = np.cov(Xv, rowvar=False)
    # noise whitening
    ev, evec = np.linalg.eigh(Cn)
    W = evec / np.sqrt(np.maximum(ev, 1e-12))
    Cw = W.T @ Cs @ W
    ev2, evec2 = np.linalg.eigh(Cw)
    order = np.argsort(ev2)[::-1]
    A = W @ evec2[:, order]  # forward transform
    n = n_components or image.bands
    comps = np.full((X.shape[0], n), np.nan)
    comps[valid] = (Xv - mean) @ A[:, :n]
    out = image.copy(comps.reshape(image.rows, image.cols, n).astype(np.float32))
    out.wavelengths = np.arange(n, dtype=float)
    out.fwhm = None
    out.bbl = None
    return out, {'A': A, 'Ainv': np.linalg.pinv(A), 'mean': mean, 'eigenvalues': ev2[order]}


def mnf_denoise(image: HSImage, n_components: int = 10) -> HSImage:
    """Denoise by forward MNF, keeping ``n_components`` high-SNR components, and inverting."""
    comps, T = mnf(image)
    C = comps.data.reshape(-1, image.bands).astype(float)
    C[:, n_components:] = 0
    X = C @ T['Ainv'] + T['mean']
    return image.copy(X.reshape(image.shape).astype(np.float32))


def percentile_brightness_filter(image: HSImage, low: float = 0.0, high: float = 100.0) -> HSImage:
    """Mask pixels darker/brighter than brightness percentiles (``low_thresh`` / ``high_thresh`` in the
    hypercloud notebook: removes residual shadows and false highlights after topographic correction)."""
    b = np.nanmean(image.data, axis=-1)
    lo, hi = np.nanpercentile(b, (low, high))
    d = image.data.astype(np.float32).copy()
    d[(b < lo) | (b > hi)] = np.nan
    return image.copy(d)
