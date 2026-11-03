"""
Resampling of irregularly georeferenced pixels onto a regular map grid.

Used by the drone direct-georeferencing workflow and by satellite products that
ship a geographic lookup table (GLT, e.g. EMIT). A GLT stores, for each output
map cell, the (row, col) of the raw-geometry pixel that fills it (-1 = empty),
so the expensive nearest-neighbour search is done once and can be applied to
every product (radiance, reflectance, masks, view angles, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .image import HSImage


@dataclass
class GLT:
    rows: np.ndarray          # (out_rows, out_cols) int32 source row index, -1 = no data
    cols: np.ndarray          # (out_rows, out_cols) int32 source col index, -1 = no data
    transform: Tuple[float, ...]
    crs: object = None

    @property
    def shape(self):
        return self.rows.shape


def build_glt(X: np.ndarray, Y: np.ndarray, res: float, max_dist: Optional[float] = None,
              bounds: Optional[Tuple[float, float, float, float]] = None, crs=None) -> GLT:
    """Build a nearest-neighbour GLT from per-pixel map coordinates.

    Args:
        X, Y: ``(rows, cols)`` map coordinates of each raw pixel centre (NaN = invalid).
        res: output pixel size (map units).
        max_dist: output cells further than this from any pixel are left empty
            (default ``1.5 * res``; increase for sparse along-track sampling).
        bounds: ``(xmin, ymin, xmax, ymax)``; defaults to the extent of valid pixels.
    """
    from scipy.spatial import cKDTree

    valid = np.isfinite(X) & np.isfinite(Y)
    if not valid.any():
        raise ValueError("No valid georeferenced pixels")
    xs, ys = X[valid], Y[valid]
    rr, cc = np.nonzero(valid)
    if bounds is None:
        bounds = (xs.min(), ys.min(), xs.max(), ys.max())
    xmin, ymin, xmax, ymax = bounds
    ncol = int(np.ceil((xmax - xmin) / res)) + 1
    nrow = int(np.ceil((ymax - ymin) / res)) + 1
    gx = xmin + np.arange(ncol) * res  # cell centres
    gy = ymax - np.arange(nrow) * res
    GX, GY = np.meshgrid(gx, gy)
    tree = cKDTree(np.column_stack([xs, ys]))
    dist, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]), k=1,
                           distance_upper_bound=max_dist if max_dist is not None else 1.5 * res)
    ok = np.isfinite(dist)
    gr = np.full(GX.size, -1, np.int32)
    gc = np.full(GX.size, -1, np.int32)
    gr[ok] = rr[idx[ok]]
    gc[ok] = cc[idx[ok]]
    transform = (xmin - res / 2, res, 0.0, ymax + res / 2, 0.0, -res)
    return GLT(gr.reshape(nrow, ncol), gc.reshape(nrow, ncol), transform, crs)


def apply_glt(image: HSImage, glt: GLT, fill=np.nan) -> HSImage:
    """Resample a raw-geometry image to map geometry with a :class:`GLT`."""
    out = np.full(glt.shape + (image.bands,), fill, dtype=np.float32)
    ok = (glt.rows >= 0) & (glt.cols >= 0)
    out[ok] = image.data[glt.rows[ok], glt.cols[ok]]
    res = image.copy(out)
    res.transform = tuple(glt.transform)
    res.crs = glt.crs
    return res


def apply_glt_array(arr: np.ndarray, glt: GLT, fill=np.nan) -> np.ndarray:
    """Apply a GLT to a 2-D or 3-D array in raw geometry."""
    shp = glt.shape + arr.shape[2:]
    out = np.full(shp, fill, dtype=np.result_type(arr.dtype, np.float32) if np.isnan(fill) else arr.dtype)
    ok = (glt.rows >= 0) & (glt.cols >= 0)
    out[ok] = arr[glt.rows[ok], glt.cols[ok]]
    return out


def mosaic(images, method: str = 'first', weights=None) -> HSImage:
    """Mosaic map-geometry images that share the same CRS, pixel size and bands.

    Args:
        images: list of :class:`HSImage` with geotransforms (north-up).
        method: ``'first'`` (earlier images take priority), ``'last'``, ``'mean'``, or
            ``'weighted'`` (per-pixel ``weights`` list, e.g. inverse view-zenith angle so the
            most nadir observation wins; use ``'best'`` to pick the max-weight pixel).
        weights: list of ``(rows, cols)`` arrays (required for ``'weighted'`` and ``'best'``).
    """
    res = images[0].transform[1]
    xs0 = [im.transform[0] for im in images]
    ys0 = [im.transform[3] for im in images]
    xs1 = [im.transform[0] + im.cols * res for im in images]
    ys1 = [im.transform[3] - im.rows * res for im in images]
    xmin, ymax = min(xs0), max(ys0)
    ncol = int(round((max(xs1) - xmin) / res))
    nrow = int(round((ymax - min(ys1)) / res))
    nb = images[0].bands
    acc = np.zeros((nrow, ncol, nb), np.float64)
    wsum = np.zeros((nrow, ncol), np.float64)
    best = np.full((nrow, ncol), -np.inf)
    out = np.full((nrow, ncol, nb), np.nan, np.float32)
    order = list(range(len(images)))
    if method == 'first':
        order = order[::-1]  # later writes overwrite, so write first image last
    for i in order:
        im = images[i]
        c0 = int(round((im.transform[0] - xmin) / res))
        r0 = int(round((ymax - im.transform[3]) / res))
        sl = (slice(r0, r0 + im.rows), slice(c0, c0 + im.cols))
        d = im.data
        valid = np.isfinite(d).all(axis=-1)
        if method in ('first', 'last'):
            tgt = out[sl]
            tgt[valid] = d[valid]
        elif method in ('mean', 'weighted'):
            w = valid.astype(float) if method == 'mean' else np.where(valid, np.nan_to_num(weights[i]), 0)
            acc[sl] += np.nan_to_num(d) * w[..., None]
            wsum[sl] += w
        elif method == 'best':
            w = np.where(valid, np.nan_to_num(weights[i], nan=-np.inf), -np.inf)
            better = w > best[sl]
            best[sl][better] = w[better]
            tgt = out[sl]
            tgt[better] = d[better]
        else:
            raise ValueError(method)
    if method in ('mean', 'weighted'):
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(wsum[..., None] > 0, acc / wsum[..., None], np.nan).astype(np.float32)
    res_img = images[0].copy(out)
    res_img.transform = (xmin, res, 0.0, ymax, 0.0, -res)
    return res_img
