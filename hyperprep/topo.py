"""
Topographic (illumination geometry) corrections, independent of platform.

All functions take per-pixel unit surface normals (``(rows, cols, 3)``, ENU) or
slope/aspect, and an illumination vector (direction light travels, as returned
by :func:`hyperprep.solar.sun_vector`). Normals come from:

* a hylite ``HyScene`` (terrestrial / drone data projected onto a point cloud),
  see :func:`hyperprep.terrestrial.projection.scene_geometry`;
* a DEM for nadir drone or satellite data, see :func:`normals_from_dem`.

Methods implemented: ``cosine``, ``c`` (C-correction / "cfac"), ``scs+c``,
``minnaert`` and ``ambient`` (sky-view-aware C-correction, the default used in
the hypercloud notebook).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .image import HSImage


def normals_from_dem(dem: np.ndarray, res_x: float, res_y: Optional[float] = None) -> np.ndarray:
    """Unit surface normals (ENU) from a north-up DEM with pixel sizes ``res_x``, ``res_y`` (metres)."""
    res_y = res_x if res_y is None else abs(res_y)
    dzdy, dzdx = np.gradient(np.asarray(dem, float), res_y, abs(res_x))
    # rows increase southward in a north-up raster -> flip the y gradient to get d/dNorth
    n = np.dstack([-dzdx, dzdy, np.ones_like(dzdx)])
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def slope_aspect(normals: np.ndarray):
    """Slope and aspect (degrees, aspect clockwise from north) from unit normals."""
    slope = np.rad2deg(np.arccos(np.clip(np.abs(normals[..., 2]), 0, 1)))
    aspect = (np.rad2deg(np.arctan2(normals[..., 0], normals[..., 1])) + 360) % 360
    return slope, aspect


def skyview_from_normals(normals: np.ndarray) -> np.ndarray:
    """Sky-view factor approximation for an unobstructed tilted plane: ``(1 + cos(slope)) / 2``."""
    return (1 + np.abs(normals[..., 2])) / 2


def incidence(normals: np.ndarray, sunvec: np.ndarray) -> np.ndarray:
    """Cosine of the solar incidence angle (clipped at 0 for self-shadowed facets)."""
    return np.clip(np.einsum('...k,k->...', normals, -np.asarray(sunvec, float)), 0, 1)


def oren_nayar(normals: np.ndarray, view: np.ndarray, sunvec: np.ndarray, roughness: float = 0.3) -> np.ndarray:
    """Oren-Nayar diffuse reflection factor (relative radiance towards the sensor).

    ``view`` is the per-pixel (or global) unit vector from the surface to the sensor.
    A pure-numpy equivalent of ``hylite.correct.illumination.calcOrenNayar``.
    """
    n = np.asarray(normals, float)
    v = np.broadcast_to(np.asarray(view, float), n.shape)
    s = -np.asarray(sunvec, float)
    cos_i = np.clip(np.einsum('...k,k->...', n, s), 0, 1)
    cos_r = np.clip(np.einsum('...k,...k->...', n, v), 0, 1)
    ti, tr = np.arccos(cos_i), np.arccos(cos_r)
    alpha, beta = np.maximum(ti, tr), np.minimum(ti, tr)
    # azimuth difference via projections onto the tangent plane
    sp = s - cos_i[..., None] * n
    vp = v - cos_r[..., None] * n
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_phi = np.einsum('...k,...k->...', sp, vp) / (np.linalg.norm(sp, axis=-1) * np.linalg.norm(vp, axis=-1))
    cos_phi = np.nan_to_num(cos_phi)
    s2 = roughness ** 2
    A = 1 - 0.5 * s2 / (s2 + 0.33)
    B = 0.45 * s2 / (s2 + 0.09)
    return cos_i * (A + B * np.maximum(0, cos_phi) * np.sin(alpha) * np.tan(beta))


def _regress_c(R, cos_i, valid):
    """Per-band regression ``R = a + b cos_i`` -> c = a / b."""
    x = cos_i[valid]
    Y = R[valid]
    xm = x.mean()
    b = ((x - xm)[:, None] * (Y - Y.mean(axis=0))).sum(axis=0) / ((x - xm) ** 2).sum()
    a = Y.mean(axis=0) - b * xm
    with np.errstate(invalid='ignore', divide='ignore'):
        c = a / b
    return np.where(np.isfinite(c) & (b > 0), c, 1e6)  # no dependence -> no correction


def topographic_correction(image: HSImage, cos_i: np.ndarray, method: str = 'c', cos_ref: float = 1.0,
                           slope: Optional[np.ndarray] = None, sun_zenith: Optional[float] = None,
                           skyview: Optional[np.ndarray] = None, min_cos: float = 0.05,
                           sample: int = 200000) -> HSImage:
    """Normalise illumination differences caused by surface orientation.

    Args:
        image: radiance or reflectance image.
        cos_i: ``(rows, cols)`` cosine of solar incidence (see :func:`incidence`).
        method: ``'cosine'``, ``'c'``/``'cfac'``, ``'scs+c'``, ``'minnaert'`` or ``'ambient'``.
        cos_ref: reference illumination to normalise to. For nadir data use ``cos(sun_zenith)``
            (flat terrain); for terrestrial outcrop scans the default 1 (normal incidence).
        slope: slope in degrees (needed for ``scs+c``).
        sun_zenith: solar zenith in degrees (needed for ``scs+c``).
        skyview: optional sky-view factor ``(rows, cols)`` used by the ``'ambient'`` method (pixels are
            normalised to ``cos_ref`` and the scene-median sky view).
        min_cos: pixels with ``cos_i`` below this are treated as shadow and set to NaN.
        sample: max number of pixels used for the per-band regressions.
    """
    d = image.data.astype(float)
    ci = np.asarray(cos_i, float)
    valid = (ci > min_cos) & np.isfinite(d).all(axis=-1)
    idx = np.flatnonzero(valid)
    if idx.size > sample:
        sel = np.zeros(valid.size, bool)
        sel[np.random.default_rng(0).choice(idx, sample, replace=False)] = True
        fit_mask = sel.reshape(valid.shape)
    else:
        fit_mask = valid
    m = method.lower()
    with np.errstate(invalid='ignore', divide='ignore'):
        if m == 'cosine':
            out = d * (cos_ref / ci)[..., None]
        elif m in ('c', 'cfac', 'c-correction'):
            c = _regress_c(d, ci, fit_mask)
            out = d * ((cos_ref + c) / (ci[..., None] + c))
        elif m == 'scs+c':
            if slope is None or sun_zenith is None:
                raise ValueError("scs+c requires slope and sun_zenith")
            c = _regress_c(d, ci, fit_mask)
            num = np.cos(np.deg2rad(slope)) * np.cos(np.deg2rad(sun_zenith))
            out = d * ((num[..., None] + c) / (ci[..., None] + c))
        elif m == 'minnaert':
            x = np.log(ci[fit_mask])
            Y = np.log(np.clip(d[fit_mask], 1e-9, None))
            xm = x.mean()
            k = ((x - xm)[:, None] * (Y - Y.mean(0))).sum(0) / ((x - xm) ** 2).sum()
            k = np.clip(k, 0, 1.5)
            out = d * (cos_ref / ci[..., None]) ** k
        elif m == 'ambient':
            skv = np.ones_like(ci) if skyview is None else np.asarray(skyview, float)
            c = _regress_c(d, ci, fit_mask)  # ratio of diffuse (sky) to direct (sun) illumination
            skv_ref = float(np.nanmedian(skv[valid])) if valid.any() else 1.0  # keep overall brightness
            out = d * ((cos_ref + c * skv_ref) / (ci[..., None] + c * skv[..., None]))
        else:
            raise ValueError("Unknown topographic correction %r" % method)
    out[~valid] = np.nan
    res = image.copy(out.astype(np.float32))
    res.meta['topographic correction'] = m
    return res
