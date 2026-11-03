"""
Satellite radiometry: TOA radiance, TOA reflectance, atmospheric correction and
detector-geometry corrections (smile, VNIR/SWIR overlap, spatial subsetting).

Atmospheric correction here is scene-based (dark object subtraction, empirical
line with field spectra, IARR / flat field). For physically based correction
use the agencies' L2A products (EnMAP L2A, PRISMA L2D, EMIT L2A) or ISOFIT, and
load them with :mod:`hyperprep.satellite.readers`.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..image import HSImage
from ..solar import earth_sun_distance, solar_irradiance


def toa_radiance(image: HSImage, gain, offset=0.0) -> HSImage:
    """DN -> top-of-atmosphere radiance with per-band (or per-detector) gain/offset."""
    from ..calibration.radiometric import dn_to_radiance
    return dn_to_radiance(image, gain, offset)


def toa_reflectance(image: HSImage, sun_zenith: Optional[float] = None, time=None,
                    esun: Optional[np.ndarray] = None, esun_path: Optional[str] = None,
                    radiance_scale: float = 1.0) -> HSImage:
    """Top-of-atmosphere reflectance ``rho = pi * L * d^2 / (E_sun * cos(theta_s))``.

    Args:
        sun_zenith: solar zenith in degrees (default ``image.meta['sun_zenith']``); may also be a
            ``(rows, cols)`` array for per-pixel geometry.
        time: acquisition time for the earth-sun distance (default ``image.meta['acquisition_time']``;
            1 AU if unknown).
        esun: band-averaged solar irradiance in W m-2 nm-1 (default from :func:`solar_irradiance`).
        esun_path: two-column solar spectrum file convolved to the bands (recommended).
        radiance_scale: factor converting the image radiance to W m-2 sr-1 nm-1
            (e.g. 0.01 for uW cm-2 sr-1 nm-1, 0.001 for mW m-2 sr-1 nm-1).
    """
    sz = sun_zenith if sun_zenith is not None else image.meta.get('sun_zenith')
    if sz is None:
        raise ValueError("Solar zenith angle required (argument or meta['sun_zenith'])")
    sz = np.asarray(sz, float)
    t = time if time is not None else image.meta.get('acquisition_time')
    d = earth_sun_distance(t) if t is not None else 1.0
    E = np.asarray(esun, float) if esun is not None else solar_irradiance(image.wavelengths, image.fwhm, esun_path)
    cos = np.cos(np.deg2rad(sz))
    if cos.ndim == 2:
        cos = cos[..., None]
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = np.pi * image.data.astype(float) * radiance_scale * d ** 2 / (E * cos)
    out = image.copy(rho.astype(np.float32))
    out.meta['units'] = 'TOA reflectance'
    out.meta['earth_sun_distance'] = d
    return out


def dark_object_subtraction(image: HSImage, percentile: float = 0.1, mask: Optional[np.ndarray] = None) -> HSImage:
    """DOS: subtract a per-band dark value (low percentile over valid / masked pixels) to remove
    additive path radiance (haze). Works on radiance or TOA reflectance."""
    X = image.spectra(mask)
    dark = np.nanpercentile(X, percentile, axis=0)
    out = image.copy((image.data - dark).astype(np.float32))
    out.meta['dos_dark'] = list(np.round(dark, 6))
    return out


def merge_overlap(image: HSImage, n_vnir: Optional[int] = None, split: Optional[float] = None,
                  method: str = 'cut') -> HSImage:
    """Resolve the VNIR / SWIR detector overlap (EnMAP ~900-1000 nm, PRISMA ~920-1010 nm).

    Args:
        n_vnir: number of VNIR bands (the first ``n_vnir`` bands); default from ``meta['n_vnir']``
            or detected as the first wavelength decrease.
        split: wavelength where the output switches from VNIR to SWIR (default: middle of overlap).
        method: ``'cut'`` keeps VNIR below ``split`` and SWIR above; ``'mean'`` additionally averages
            the overlap after interpolating SWIR onto the VNIR bands.
    """
    wav = image.wavelengths
    if n_vnir is None:
        n_vnir = image.meta.get('n_vnir')
    if n_vnir is None:
        dec = np.where(np.diff(wav) < 0)[0]
        if len(dec) == 0:
            return image  # already monotonic
        n_vnir = int(dec[0]) + 1
    wv, ws = wav[:n_vnir], wav[n_vnir:]
    lo, hi = ws.min(), wv.max()
    split = split if split is not None else 0.5 * (lo + hi)
    kv = wv < split
    ks = ws >= split
    d = image.data
    if method == 'mean':
        from ..spectral import resample_spectra
        ov = (wv >= lo) & (wv <= hi)
        sw_on_v = resample_spectra(d[..., n_vnir:], ws, wv[ov])
        vn = d[..., :n_vnir].astype(float).copy()
        vn[..., ov] = np.nanmean(np.stack([vn[..., ov], sw_on_v]), axis=0)
        kv = np.ones_like(wv, bool)
        ks = ws > hi
        arr = np.concatenate([vn[..., kv], d[..., n_vnir:][..., ks]], axis=-1)
    else:
        arr = np.concatenate([d[..., :n_vnir][..., kv], d[..., n_vnir:][..., ks]], axis=-1)
    out = image.copy(arr.astype(np.float32))
    out.wavelengths = np.concatenate([wv[kv], ws[ks]])
    if image.fwhm is not None:
        out.fwhm = np.concatenate([image.fwhm[:n_vnir][kv], image.fwhm[n_vnir:][ks]])
    if image.bbl is not None:
        out.bbl = np.concatenate([image.bbl[:n_vnir][kv], image.bbl[n_vnir:][ks]])
    order = np.argsort(out.wavelengths)
    out = out.subset_bands(order)
    out.meta.pop('n_vnir', None)
    return out


def smile_correction(image: HSImage, column_wavelengths: np.ndarray) -> HSImage:
    """Correct spectral smile: resample every across-track column from its actual band centres
    (``(cols, bands)``, from the sensor's spectral calibration) to the nominal ``image.wavelengths``."""
    cw = np.asarray(column_wavelengths, float)
    if cw.shape != (image.cols, image.bands):
        raise ValueError("column_wavelengths must have shape (cols, bands) = %s" % ((image.cols, image.bands),))
    d = image.data.astype(float)
    out = np.empty_like(d)
    nominal = image.wavelengths
    eye = np.eye(image.bands)
    for c in range(image.cols):
        # linear interpolation matrix from this column's band centres to the nominal ones
        W = np.array([np.interp(nominal, cw[c], eye[k], left=np.nan, right=np.nan) for k in range(image.bands)])
        out[:, c, :] = d[:, c, :] @ W
    return image.copy(out.astype(np.float32))


def estimate_smile(image: HSImage, feature: Tuple[float, float] = (750.0, 775.0), max_shift: float = 3.0,
                   step: float = 0.05, degree: int = 2) -> np.ndarray:
    """Estimate the across-track spectral shift (nm) using an atmospheric absorption feature.

    Each column's mean spectrum in the window (default the O2-A band at 760 nm) is shifted against the
    scene-mean spectrum; the shift minimising the squared difference of the continuum-normalised spectra is
    taken and a polynomial is fitted across track. Returns per-column shifts (add to nominal wavelengths
    to get ``column_wavelengths`` for :func:`smile_correction`).
    """
    sl = image.band_range(*feature)
    w = image.wavelengths[sl]
    col = np.nanmean(image.data[:, :, sl].astype(float), axis=0)  # (cols, nb)
    ref = np.nanmean(col, axis=0)
    norm = lambda s: s / np.linspace(s[0], s[-1], s.size)
    refn = norm(ref)
    shifts = np.arange(-max_shift, max_shift + step, step)
    best = np.zeros(image.cols)
    for c in range(image.cols):
        if not np.isfinite(col[c]).all():
            best[c] = np.nan
            continue
        cn = norm(col[c])
        errs = [np.nanmean((np.interp(w, w + s, cn) - refn) ** 2) for s in shifts]
        best[c] = shifts[int(np.argmin(errs))]
    x = np.arange(image.cols)
    ok = np.isfinite(best)
    return np.polyval(np.polyfit(x[ok], best[ok], degree), x)


def subset_bbox(image: HSImage, bbox: Tuple[float, float, float, float]) -> HSImage:
    """Crop a north-up map-geometry image to ``(xmin, ymin, xmax, ymax)`` in its CRS."""
    if image.transform is None:
        raise ValueError("Image has no geotransform")
    x0, dx, _, y0, _, dy = image.transform
    xmin, ymin, xmax, ymax = bbox
    c0 = max(0, int(np.floor((xmin - x0) / dx)))
    c1 = min(image.cols, int(np.ceil((xmax - x0) / dx)))
    r0 = max(0, int(np.floor((ymax - y0) / dy)))
    r1 = min(image.rows, int(np.ceil((ymin - y0) / dy)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("Bounding box does not intersect the image")
    out = image.copy(image.data[r0:r1, c0:c1])
    out.transform = (x0 + c0 * dx, dx, 0.0, y0 + r0 * dy, 0.0, dy)
    return out


def reflectance_scale_check(image: HSImage) -> HSImage:
    """Detect integer-scaled reflectance (0-10000) and rescale to 0-1."""
    p99 = np.nanpercentile(image.data, 99)
    if p99 > 100:
        out = image.copy((image.data / 10000.0).astype(np.float32))
        out.meta['reflectance rescaled'] = 1e-4
        return out
    return image
