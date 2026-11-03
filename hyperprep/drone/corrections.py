"""
Radiometric corrections specific to drone / airborne flight lines.

* :func:`irradiance_normalisation` - reflectance from a downwelling irradiance
  sensor (DLS / Spectralon-mounted upward spectrometer), per line.
* :func:`sun_angle_normalisation` - compensate changing solar elevation between flights.
* :func:`cross_track_correction` - empirical BRDF / vignetting normalisation
  across track (view-angle dependent brightness).
* :func:`match_flightlines` - relative radiometric normalisation of overlapping
  orthorectified flight lines before mosaicking (pixel-wise analogue of
  ``HyScene.match_colour_to`` used for terrestrial scenes).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..image import HSImage


def irradiance_normalisation(image: HSImage, irradiance: np.ndarray, irradiance_wavelengths=None) -> HSImage:
    """Convert radiance to reflectance with measured downwelling irradiance: ``R = pi * L / E``.

    Args:
        irradiance: ``(bands,)`` (constant) or ``(lines, bands)`` (one spectrum per scan line, e.g. an
            upward-looking spectrometer interpolated to line times). Units must match radiance
            (e.g. W m-2 nm-1 vs W m-2 sr-1 nm-1).
        irradiance_wavelengths: if given, the irradiance is linearly resampled to the image bands.
    """
    E = np.asarray(irradiance, float)
    if irradiance_wavelengths is not None:
        from ..spectral import resample_spectra
        E = resample_spectra(E, irradiance_wavelengths, image.wavelengths)
    if E.ndim == 2:
        E = E[:, None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.pi * image.data.astype(float) / E
    out = image.copy(R.astype(np.float32))
    out.meta['units'] = 'reflectance'
    return out


def sun_angle_normalisation(image: HSImage, sun_zenith: float, reference_zenith: float = 45.0) -> HSImage:
    """Scale radiance/reflectance acquired at ``sun_zenith`` to a common reference zenith
    (``cos(ref) / cos(sz)``), used when flights span several hours."""
    f = np.cos(np.deg2rad(reference_zenith)) / np.cos(np.deg2rad(sun_zenith))
    out = image.copy((image.data * f).astype(np.float32))
    out.meta['sun angle normalised to'] = reference_zenith
    return out


def cross_track_correction(image: HSImage, view_angles: Optional[Sequence[float]] = None, degree: int = 2,
                           mask: Optional[np.ndarray] = None, reference_angle: float = 0.0,
                           smooth_spectral: int = 0) -> HSImage:
    """Empirical cross-track brightness (BRDF / vignetting) normalisation for raw-geometry pushbroom data.

    For each band, the along-track mean of every column is modelled as a polynomial of the view
    angle; the image is divided by the model and multiplied by its value at ``reference_angle``
    (nadir). Use ``mask`` (``(rows, cols)`` bool, ``True`` = use) to restrict the fit to one
    land-cover class for heterogeneous scenes.

    Args:
        view_angles: per-column signed view angle (degrees); defaults to column index centred on nadir.
        smooth_spectral: optional moving-average window across bands for the model coefficients.
    """
    d = image.data.astype(float)
    if mask is not None:
        d = np.where(np.asarray(mask)[..., None], d, np.nan)
    va = np.asarray(view_angles, float) if view_angles is not None else np.arange(image.cols) - (image.cols - 1) / 2
    prof = np.nanmean(d, axis=0)  # (cols, bands)
    ok = np.isfinite(prof).all(axis=1)
    coef = np.polyfit(va[ok], prof[ok], degree)  # (degree+1, bands)
    if smooth_spectral and smooth_spectral > 1:
        from scipy.ndimage import uniform_filter1d
        coef = uniform_filter1d(coef, smooth_spectral, axis=1, mode='nearest')
    model = np.array([np.polyval(coef[:, b], va) for b in range(image.bands)]).T  # (cols, bands)
    ref = np.array([np.polyval(coef[:, b], reference_angle) for b in range(image.bands)])
    with np.errstate(invalid='ignore', divide='ignore'):
        out = image.data.astype(float) * (ref / model)[None, :, :]
    res = image.copy(out.astype(np.float32))
    res.meta['cross-track correction'] = 'polynomial degree %d' % degree
    return res


def _overlap_slices(a: HSImage, b: HSImage):
    res = a.transform[1]
    if not np.isclose(res, b.transform[1]):
        raise ValueError("Flight lines must share the same pixel size")
    xmin = max(a.transform[0], b.transform[0])
    xmax = min(a.transform[0] + a.cols * res, b.transform[0] + b.cols * res)
    ymax = min(a.transform[3], b.transform[3])
    ymin = max(a.transform[3] - a.rows * res, b.transform[3] - b.rows * res)
    if xmax <= xmin or ymax <= ymin:
        return None

    def sl(im):
        c0 = int(round((xmin - im.transform[0]) / res))
        r0 = int(round((im.transform[3] - ymax) / res))
        return slice(r0, r0 + int(round((ymax - ymin) / res))), slice(c0, c0 + int(round((xmax - xmin) / res)))
    return sl(a), sl(b)


def match_flightlines(target: HSImage, reference: HSImage, uniform: bool = False, min_pixels: int = 100) -> HSImage:
    """Linearly adjust ``target`` so its overlap with ``reference`` has the same per-band mean and std.

    Both images must be on the same map grid (same CRS and pixel size). If ``uniform`` is True a single
    gain/offset (from the band-averaged brightness) is used for all bands to avoid spectral artefacts.
    """
    ov = _overlap_slices(target, reference)
    if ov is None:
        raise ValueError("Flight lines do not overlap")
    t = target.data[ov[0]].reshape(-1, target.bands).astype(float)
    r = reference.data[ov[1]].reshape(-1, reference.bands).astype(float)
    ok = np.isfinite(t).all(1) & np.isfinite(r).all(1)
    if ok.sum() < min_pixels:
        raise ValueError("Only %d overlapping valid pixels" % ok.sum())
    t, r = t[ok], r[ok]
    if uniform:
        tb, rb = t.mean(1), r.mean(1)
        gain = np.full(target.bands, rb.std() / tb.std())
        off = np.full(target.bands, rb.mean() - gain[0] * tb.mean())
    else:
        gain = r.std(0) / np.maximum(t.std(0), 1e-12)
        off = r.mean(0) - gain * t.mean(0)
    out = target.copy((target.data * gain + off).astype(np.float32))
    out.meta['flightline matching'] = 'uniform' if uniform else 'per-band'
    return out
