"""
Spectral-domain pre-processing shared by all platforms.

Includes bad-band (water vapour) removal, spectral resampling to a target
sensor, Savitzky-Golay smoothing, continuum (hull) removal, derivatives,
normalisation and band binning.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .image import HSImage

#: Default atmospheric absorption windows (nm) removed before analysis. These are
#: the windows used in the original hypercloud notebook, extended with the O2-A band.
DEFAULT_BAD_WINDOWS: Tuple[Tuple[float, float], ...] = (
    (0.0, 400.0),
    (757.0, 768.0),       # O2-A
    (900.0, 1000.0),      # H2O
    (1100.0, 1180.0),     # H2O
    (1300.0, 1500.0),     # H2O (strong)
    (1750.0, 2000.0),     # H2O (strong)
    (2450.0, 2600.0),     # detector edge / H2O
)

#: The stricter windows used in ``1b_Build_and_correct_hypercloud_ambient.ipynb``.
HYPERCLOUD_BAD_WINDOWS: Tuple[Tuple[float, float], ...] = (
    (0.0, 500.0), (900.0, 1000.0), (1100.0, 1180.0), (1300.0, 1550.0), (1750.0, 2020.0), (2400.0, 2600.0),
)


def bad_band_mask(wavelengths: Sequence[float], windows: Iterable[Tuple[float, float]] = DEFAULT_BAD_WINDOWS) -> np.ndarray:
    """Boolean mask that is ``True`` for *good* bands (outside every window)."""
    wav = np.asarray(wavelengths, dtype=float)
    good = np.ones(wav.shape[0], dtype=bool)
    for lo, hi in windows:
        good[(wav > lo) & (wav < hi)] = False
    return good


def remove_bad_bands(image: HSImage, windows=DEFAULT_BAD_WINDOWS, use_bbl: bool = True,
                     drop: bool = True) -> HSImage:
    """Remove (or NaN out) bands within absorption windows and/or flagged in the image BBL.

    Args:
        image: input image.
        windows: list of ``(min_nm, max_nm)`` windows to remove. ``None`` to use only the BBL.
        use_bbl: also honour the image's bad band list if present.
        drop: if True the bands are removed; otherwise they are set to NaN and the BBL updated.
    """
    good = bad_band_mask(image.wavelengths, windows or ())
    if use_bbl and image.bbl is not None:
        good &= image.bbl.astype(bool)
    if drop:
        out = image.subset_bands(good)
        out.bbl = None
        return out
    out = image.copy()
    out.data = out.data.astype(np.float32, copy=False)
    out.data[..., ~good] = np.nan
    out.bbl = good
    return out


def detect_noisy_bands(image: HSImage, snr_thresh: float = 10.0) -> np.ndarray:
    """Flag bands with a low spatial signal-to-noise ratio (returns ``True`` for good bands).

    SNR is estimated per band as ``median / std(horizontal pixel differences / sqrt(2))``.
    """
    d = image.data.astype(float)
    noise = np.nanstd(np.diff(d, axis=1), axis=(0, 1)) / np.sqrt(2)
    signal = np.nanmedian(d, axis=(0, 1))
    snr = np.abs(signal) / np.maximum(noise, 1e-12)
    return snr >= snr_thresh


# ----------------------------------------------------------------- resampling
def gaussian_srf(src_wav, dst_wav, dst_fwhm, src_fwhm=None) -> np.ndarray:
    """Build a ``(n_dst, n_src)`` Gaussian spectral response matrix (rows sum to 1)."""
    src_wav = np.asarray(src_wav, float)
    dst_wav = np.asarray(dst_wav, float)
    dst_fwhm = np.broadcast_to(np.asarray(dst_fwhm, float), dst_wav.shape)
    if src_fwhm is not None:  # deconvolve approximately: target width relative to source
        src_fwhm = np.interp(dst_wav, src_wav, np.broadcast_to(np.asarray(src_fwhm, float), src_wav.shape))
        dst_fwhm = np.sqrt(np.maximum(dst_fwhm ** 2 - src_fwhm ** 2, 1e-6))
    sigma = dst_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    W = np.exp(-0.5 * ((src_wav[None, :] - dst_wav[:, None]) / sigma[:, None]) ** 2)
    W[W < 1e-6] = 0
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return W / s


def resample_spectra(X: np.ndarray, src_wav, dst_wav, dst_fwhm=None, src_fwhm=None) -> np.ndarray:
    """Resample spectra (last axis) to new band centres.

    Uses Gaussian spectral response functions if ``dst_fwhm`` is provided (recommended
    for convolving to a coarser sensor), otherwise linear interpolation. NaNs in the
    input are ignored for the Gaussian method.
    """
    X = np.asarray(X, dtype=float)
    src_wav = np.asarray(src_wav, float)
    dst_wav = np.asarray(dst_wav, float)
    shp = X.shape
    X2 = X.reshape(-1, shp[-1])
    if dst_fwhm is None:
        order = np.argsort(src_wav)
        out = np.empty((X2.shape[0], dst_wav.shape[0]))
        for i in range(X2.shape[0]):
            out[i] = np.interp(dst_wav, src_wav[order], X2[i, order], left=np.nan, right=np.nan)
    else:
        W = gaussian_srf(src_wav, dst_wav, dst_fwhm, src_fwhm)
        valid = np.isfinite(X2)
        num = np.nan_to_num(X2) @ W.T
        den = valid.astype(float) @ W.T
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(den > 0.5, num / den, np.nan)
        out[:, (dst_wav < src_wav.min()) | (dst_wav > src_wav.max())] = np.nan
    return out.reshape(shp[:-1] + (dst_wav.shape[0],))


def resample_image(image: HSImage, dst_wav, dst_fwhm=None) -> HSImage:
    """Resample an image to new wavelengths (e.g. to merge sensors with different binning,
    as done for the 2016/2019 FENIX scenes in the hypercloud notebook)."""
    out = image.copy(resample_spectra(image.data, image.wavelengths, dst_wav, dst_fwhm, image.fwhm).astype(np.float32))
    out.wavelengths = np.asarray(dst_wav, float)
    out.fwhm = None if dst_fwhm is None else np.broadcast_to(np.asarray(dst_fwhm, float), out.wavelengths.shape).copy()
    out.bbl = None
    return out


def bin_bands(image: HSImage, factor: int = 2) -> HSImage:
    """Average groups of ``factor`` adjacent bands (spectral binning)."""
    n = (image.bands // factor) * factor
    d = image.data[..., :n].reshape(image.rows, image.cols, n // factor, factor)
    out = image.copy(np.nanmean(d, axis=-1).astype(np.float32))
    out.wavelengths = image.wavelengths[:n].reshape(-1, factor).mean(axis=1)
    if image.fwhm is not None:
        out.fwhm = image.fwhm[:n].reshape(-1, factor).sum(axis=1)
    out.bbl = None
    return out


# ------------------------------------------------------------------ smoothing
def savgol(image: HSImage, window: int = 7, polyorder: int = 2, deriv: int = 0) -> HSImage:
    """Savitzky-Golay smoothing (or derivative) along the spectral axis; NaN-safe."""
    from scipy.signal import savgol_filter

    d = image.data.astype(float)
    nan = ~np.isfinite(d)
    if nan.any():  # fill NaNs along spectra by linear interpolation before filtering
        d = _fill_nan_spectral(d, image.wavelengths)
    out = savgol_filter(d, window_length=window, polyorder=polyorder, deriv=deriv, axis=-1)
    out[nan] = np.nan
    return image.copy(out.astype(np.float32))


def _fill_nan_spectral(d, wav):
    flat = d.reshape(-1, d.shape[-1]).copy()
    bad_rows = np.where(~np.isfinite(flat).all(axis=1))[0]
    for i in bad_rows:
        ok = np.isfinite(flat[i])
        if ok.sum() >= 2:
            flat[i, ~ok] = np.interp(wav[~ok], wav[ok], flat[i, ok])
        else:
            flat[i] = 0
    return flat.reshape(d.shape)


def spatial_smooth(image: HSImage, size: int = 3) -> HSImage:
    """Spatial (box) averaging of each band, ignoring NaNs (``HyImage.blur`` equivalent)."""
    from scipy.ndimage import uniform_filter

    d = image.data.astype(float)
    valid = np.isfinite(d).astype(float)
    num = uniform_filter(np.nan_to_num(d), size=(size, size, 1))
    den = uniform_filter(valid, size=(size, size, 1))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den > 0, num / den, np.nan)
    return image.copy(out.astype(np.float32))


# ---------------------------------------------------------- continuum removal
def _upper_hull(x, y):
    """Indices of the upper convex hull of (x, y), x sorted ascending (monotone chain)."""
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            cross = (x[i2] - x[i1]) * (y[i] - y[i1]) - (y[i2] - y[i1]) * (x[i] - x[i1])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def continuum_removal(X: np.ndarray, wavelengths, method: str = 'div') -> np.ndarray:
    """Upper-convex-hull continuum removal on spectra stored along the last axis.

    Args:
        method: ``'div'`` (ratio, standard) or ``'sub'`` (difference).
    """
    X = np.asarray(X, dtype=float)
    wav = np.asarray(wavelengths, dtype=float)
    shp = X.shape
    X2 = X.reshape(-1, shp[-1])
    out = np.full_like(X2, np.nan)
    for i in range(X2.shape[0]):
        y = X2[i]
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        xv, yv = wav[ok], y[ok]
        h = _upper_hull(xv, yv)
        cont = np.interp(xv, xv[h], yv[h])
        res = np.full(y.shape, np.nan)
        res[ok] = yv / cont if method == 'div' else cont - yv
        out[i] = res
    return out.reshape(shp)


def hull_correct(image: HSImage, band_range: Optional[Tuple[float, float]] = None, method: str = 'div') -> HSImage:
    """Continuum-removed copy of an image, optionally restricted to a wavelength range
    (equivalent to ``hylite.correct.get_hull_corrected`` used in the FENIX notebook)."""
    img = image.subset_bands(band_range) if band_range is not None else image
    return img.copy(continuum_removal(img.data, img.wavelengths, method).astype(np.float32))


# ---------------------------------------------------------------- normalising
def normalise(image: HSImage, method: str = 'brightness') -> HSImage:
    """Spectral normalisation.

    ``'brightness'``: divide by spectrum L2 norm; ``'max'``: divide by maximum;
    ``'snv'``: standard normal variate; ``'minmax'``: scale each spectrum to 0-1.
    """
    d = image.data.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        if method == 'brightness':
            out = d / np.sqrt(np.nansum(d ** 2, axis=-1, keepdims=True))
        elif method == 'max':
            out = d / np.nanmax(d, axis=-1, keepdims=True)
        elif method == 'snv':
            out = (d - np.nanmean(d, axis=-1, keepdims=True)) / np.nanstd(d, axis=-1, keepdims=True)
        elif method == 'minmax':
            mn, mx = np.nanmin(d, axis=-1, keepdims=True), np.nanmax(d, axis=-1, keepdims=True)
            out = (d - mn) / (mx - mn)
        else:
            raise ValueError("Unknown normalisation %r" % method)
    return image.copy(out.astype(np.float32))


def clip_range(image: HSImage, vmin: float = 0.0, vmax: float = 1.5, action: str = 'nan_pixel') -> HSImage:
    """Handle out-of-range values (e.g. reflectance < 0 or > 2.5 after topographic correction).

    ``action='nan_pixel'`` sets the whole pixel to NaN if any band is out of range (as in the
    hypercloud notebook); ``'nan'`` masks only the offending values; ``'clip'`` clips values.
    """
    d = image.data.astype(np.float32).copy()
    if action == 'clip':
        d = np.clip(d, vmin, vmax)
    elif action == 'nan':
        d[(d < vmin) | (d > vmax)] = np.nan
    elif action == 'nan_pixel':
        with np.errstate(invalid='ignore'):
            bad = ((d < vmin) | (d > vmax)).any(axis=-1)
        d[bad] = np.nan
    else:
        raise ValueError(action)
    return image.copy(d)
