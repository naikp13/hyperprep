"""
Radiometric calibration: raw DN -> radiance -> reflectance.

Covers the steps that hylite's sensor classes perform internally for lab /
core-scanner data (dark subtraction, white-reference normalisation) and the
equivalent steps for airborne / drone sensors (gain-offset radiance
calibration, HySpex RAD scale-factor decompression, panel-based reflectance).
"""
from __future__ import annotations

import re
from typing import Optional, Union

import numpy as np

from ..image import HSImage

ArrayLike = Union[float, np.ndarray]


def frame_mean(ref: Union[HSImage, np.ndarray], axis: int = 0) -> np.ndarray:
    """Average a reference recording (dark or white frames) along-track to a ``(cols, bands)`` array."""
    d = ref.data if isinstance(ref, HSImage) else np.asarray(ref)
    return np.nanmean(d.astype(float), axis=axis)


def dark_subtract(image: HSImage, dark: Union[HSImage, np.ndarray]) -> HSImage:
    """Subtract a dark reference (an image recorded with the shutter closed, averaged along-track)."""
    dk = frame_mean(dark) if (isinstance(dark, HSImage) or np.ndim(dark) == 3) else np.asarray(dark, float)
    return image.copy((image.data.astype(np.float32) - dk[None, ...]).astype(np.float32))


def white_reference_reflectance(image: HSImage, white: Union[HSImage, np.ndarray],
                                dark: Optional[Union[HSImage, np.ndarray]] = None,
                                panel_reflectance: ArrayLike = 1.0,
                                white_dark: Optional[Union[HSImage, np.ndarray]] = None,
                                integration_ratio: float = 1.0) -> HSImage:
    """Convert raw DN to reflectance using white and dark references (core scanners / lab scans).

    ``R = (DN - dark) / (white - white_dark) * panel_reflectance * integration_ratio``

    Args:
        image: raw image (rows = scan lines).
        white: white reference recording (or ``(cols, bands)`` mean frame).
        dark: dark current recording for the image. If None no dark subtraction is done.
        panel_reflectance: reflectance of the white panel per band (e.g. a calibrated
            Spectralon R90 curve resampled to the sensor) or a scalar.
        white_dark: dark recording for the white reference if it differs from ``dark``.
        integration_ratio: ``t_white / t_image`` if the integration times differ.
    """
    def _mean(x):
        if x is None:
            return 0.0
        return frame_mean(x) if (isinstance(x, HSImage) or np.ndim(x) == 3) else np.asarray(x, float)

    dk = _mean(dark)
    wdk = _mean(white_dark) if white_dark is not None else dk
    wt = _mean(white) - wdk
    with np.errstate(invalid='ignore', divide='ignore'):
        R = (image.data.astype(float) - dk) / np.where(wt != 0, wt, np.nan)
    R = R * np.asarray(panel_reflectance, float) * integration_ratio
    out = image.copy(R.astype(np.float32))
    out.meta['units'] = 'reflectance'
    return out


def dn_to_radiance(image: HSImage, gain: ArrayLike, offset: ArrayLike = 0.0,
                   dark: Optional[np.ndarray] = None, integration_time: Optional[float] = None) -> HSImage:
    """Linear radiometric calibration ``L = gain * (DN - dark) + offset`` (optionally ``/ t_int``).

    ``gain`` and ``offset`` may be scalars, per-band ``(bands,)`` or per-detector ``(cols, bands)``
    arrays (as delivered in sensor calibration files).
    """
    d = image.data.astype(float)
    if dark is not None:
        d = d - np.asarray(dark, float)
    L = d * np.asarray(gain, float) + np.asarray(offset, float)
    if integration_time:
        L = L / integration_time
    out = image.copy(L.astype(np.float32))
    out.meta['units'] = 'radiance'
    return out


def hyspex_scale_factor(image) -> float:
    """Return the scale factor stored by HySpex RAD in the header description (``... Scaling = 1000``).

    ``image`` may be an :class:`HSImage`, a ``hylite.HyImage`` or the description string itself.
    """
    if isinstance(image, str):
        desc = image
    elif isinstance(image, HSImage):
        desc = str(image.meta.get('description', ''))
    else:  # hylite image
        desc = str(image.header.get('description', ''))
    m = re.search(r'[Ss]cal\w*\s*(?:factor)?\s*=\s*([0-9.eE+-]+)', desc)
    if m is None:
        raise ValueError("No HySpex scaling factor found in header description: %r" % desc[:200])
    return float(m.group(1))


def apply_scale_factor(image: HSImage, factor: Optional[float] = None, mask_zero: bool = True) -> HSImage:
    """Decompress integer-scaled radiance (e.g. HySpex RAD output) to physical units.

    Ported from ``2_Projection_mapping.ipynb``: divides by the scale factor read from the
    header and flags under-exposed pixels (zero in every band) as NaN.
    """
    sf = hyspex_scale_factor(image) if factor is None else float(factor)
    d = image.data.astype(np.float32) / sf
    if mask_zero:
        d[(image.data == 0).all(axis=-1)] = np.nan
    out = image.copy(d)
    out.meta['scale factor applied'] = sf
    return out


def panel_reflectance(image: HSImage, panel_radiance: np.ndarray, panel_reflectance: ArrayLike = 1.0) -> HSImage:
    """Single-panel reflectance: ``R = L / L_panel * R_panel`` (assumes zero path radiance)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        R = image.data.astype(float) / np.asarray(panel_radiance, float) * np.asarray(panel_reflectance, float)
    out = image.copy(R.astype(np.float32))
    out.meta['units'] = 'reflectance'
    return out


def flat_field(image: HSImage, region=None) -> HSImage:
    """Flat-field correction: divide every spectrum by the mean spectrum of a bright, spectrally
    flat region ``(row0, row1, col0, col1)`` (or the whole image if None). Produces relative reflectance."""
    if region is None:
        ref = np.nanmean(image.data.reshape(-1, image.bands), axis=0)
    else:
        r0, r1, c0, c1 = region
        ref = np.nanmean(image.data[r0:r1, c0:c1].reshape(-1, image.bands), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = image.copy((image.data / ref).astype(np.float32))
    out.meta['units'] = 'relative reflectance'
    return out


def iarr(image: HSImage) -> HSImage:
    """Internal Average Relative Reflectance: divide each spectrum by the scene-average spectrum."""
    return flat_field(image, None)
