"""
Pixel masks: no-data, saturation, vegetation, water, cloud, cirrus and shadow.

Spectral-threshold masks are intentionally simple and transparent; they are meant
for screening before analysis, not as a replacement for product cloud masks
(e.g. EnMAP/EMIT L2 quality layers), which should be preferred when available.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .image import HSImage


def nodata_mask(image: HSImage, nodata_value: Optional[float] = 0.0) -> np.ndarray:
    """``True`` where a pixel is no-data: all-NaN, or equal to ``nodata_value`` in every band
    (the ``image.data[image.data == 0] = np.nan`` step of the notebooks)."""
    d = image.data
    m = ~np.isfinite(d).any(axis=-1)
    if nodata_value is not None:
        m |= (d == nodata_value).all(axis=-1)
    return m


def saturation_mask(image: HSImage, saturation_value: float, fraction: float = 0.0) -> np.ndarray:
    """``True`` where more than ``fraction`` of bands are at/above the saturation value."""
    sat = (image.data >= saturation_value).mean(axis=-1)
    return sat > fraction


def apply_mask(image: HSImage, mask: np.ndarray, value=np.nan) -> HSImage:
    """Set all bands of masked pixels to ``value`` (returns a float32 copy)."""
    d = image.data.astype(np.float32).copy()
    d[np.asarray(mask, bool)] = value
    return image.copy(d)


def mask_nodata(image: HSImage, nodata_value: Optional[float] = 0.0) -> HSImage:
    return apply_mask(image, nodata_mask(image, nodata_value))


def _nd(image, b1, b2):
    a, b = image.get_band(b1).astype(float), image.get_band(b2).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a - b) / (a + b)


def ndvi(image: HSImage, nir: float = 860.0, red: float = 660.0) -> np.ndarray:
    return _nd(image, nir, red)


def ndwi(image: HSImage, green: float = 560.0, nir: float = 860.0) -> np.ndarray:
    """McFeeters NDWI (water > 0)."""
    return _nd(image, green, nir)


def vegetation_mask(image: HSImage, thresh: float = 0.4) -> np.ndarray:
    return ndvi(image) > thresh


def water_mask(image: HSImage, thresh: float = 0.0) -> np.ndarray:
    return ndwi(image) > thresh


def cloud_mask(image: HSImage, blue_thresh: float = 0.25, whiteness_thresh: float = 0.7,
               cirrus_thresh: Optional[float] = 0.01, swir_thresh: float = 0.2) -> np.ndarray:
    """Simple spectral-threshold cloud mask for *reflectance* data.

    A pixel is cloudy if it is bright in the blue (> ``blue_thresh``), spectrally flat in the
    visible (whiteness: min/max of 470/560/660 nm > ``whiteness_thresh``) and bright in the
    SWIR (> ``swir_thresh`` at 1600 nm, separating clouds from snow). Thin cirrus is flagged
    when 1380 nm reflectance exceeds ``cirrus_thresh`` (only if the sensor covers 1380 nm).
    """
    wav = image.wavelengths
    vis = np.stack([image.get_band(w) for w in (470., 560., 660.)], axis=-1).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        white = np.nanmin(vis, axis=-1) / np.nanmax(vis, axis=-1)
        m = (vis[..., 0] > blue_thresh) & (white > whiteness_thresh)
        if wav.max() > 1600:
            m &= image.get_band(1600.) > swir_thresh
        if cirrus_thresh is not None and wav.min() < 1380 < wav.max() and np.min(np.abs(wav - 1380)) < 15:
            m |= image.get_band(1380.) > cirrus_thresh
    return m


def shadow_mask(image: HSImage, nir_thresh: float = 0.08, vis_thresh: float = 0.06) -> np.ndarray:
    """Dark-pixel shadow mask for reflectance data (excludes water via NDWI)."""
    with np.errstate(invalid='ignore'):
        dark = (image.get_band(860.) < nir_thresh) & (np.nanmean(image.data[..., image.band_range(450., 700.)], -1) < vis_thresh)
        return dark & ~(ndwi(image) > 0.2)


def dilate(mask: np.ndarray, pixels: int = 2) -> np.ndarray:
    """Grow a mask by ``pixels`` (useful as a buffer around clouds)."""
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=pixels)
