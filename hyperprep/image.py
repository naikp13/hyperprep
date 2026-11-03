"""
A light-weight hyperspectral image container used throughout hyperprep.

Arrays are stored as ``(rows, cols, bands)`` (i.e. ``(y, x, b)``), which is the
convention used by GDAL/rasterio and most satellite products. hylite stores
images as ``(x, y, b)``; use :meth:`HSImage.to_hylite` / :meth:`HSImage.from_hylite`
to move between the two without worrying about the transpose.
"""
from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class HSImage:
    """Hyperspectral image cube with spectral and spatial metadata.

    Attributes:
        data: float array of shape ``(rows, cols, bands)``.
        wavelengths: band centre wavelengths in nanometres, shape ``(bands,)``.
        fwhm: optional full-width-half-maximum per band (nm).
        transform: optional 6-element GDAL-style geotransform
            ``(x0, dx, rx, y0, ry, dy)``.
        crs: optional coordinate reference system (WKT, EPSG code or proj string).
        meta: free-form metadata (acquisition time, sensor name, sun angles, ...).
        bbl: optional bad band list (1 = good band, 0 = bad band).
    """

    data: np.ndarray
    wavelengths: Optional[np.ndarray] = None
    fwhm: Optional[np.ndarray] = None
    transform: Optional[Sequence[float]] = None
    crs: Any = None
    meta: dict = field(default_factory=dict)
    bbl: Optional[np.ndarray] = None

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim == 2:
            self.data = self.data[..., None]
        if self.data.ndim != 3:
            raise ValueError("HSImage data must be 2-D or 3-D, got shape %s" % (self.data.shape,))
        if self.wavelengths is None:
            self.wavelengths = np.arange(self.data.shape[-1], dtype=float)
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        if self.wavelengths.shape[0] != self.data.shape[-1]:
            raise ValueError("Got %d wavelengths for %d bands" % (self.wavelengths.shape[0], self.data.shape[-1]))
        if self.fwhm is not None:
            self.fwhm = np.asarray(self.fwhm, dtype=float)
        if self.bbl is not None:
            self.bbl = np.asarray(self.bbl).astype(bool)

    # ------------------------------------------------------------------ shape
    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    @property
    def bands(self) -> int:
        return self.data.shape[2]

    @property
    def shape(self):
        return self.data.shape

    def copy(self, data: Optional[np.ndarray] = None) -> "HSImage":
        """Deep copy, optionally replacing the data array (metadata is copied).

        If ``data`` has a different number of bands, band metadata (wavelengths, fwhm, bbl)
        is reset and must be set by the caller.
        """
        if data is None:
            return _copy.deepcopy(self)
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[..., None]
        same = data.shape[-1] == self.bands
        cp = lambda a: None if (a is None or not same) else a.copy()
        return HSImage(data, cp(self.wavelengths), cp(self.fwhm),
                       None if self.transform is None else tuple(self.transform), self.crs,
                       _copy.deepcopy(self.meta), cp(self.bbl))

    # ------------------------------------------------------------- spectral
    def band_index(self, b) -> int:
        """Return the index of a band given an int index or a float wavelength (nm)."""
        if isinstance(b, (int, np.integer)):
            return int(b) if b >= 0 else self.bands + int(b)
        return int(np.argmin(np.abs(self.wavelengths - float(b))))

    def band_range(self, start, end) -> slice:
        """Slice selecting bands between ``start`` and ``end`` (indices or wavelengths, inclusive)."""
        i0, i1 = self.band_index(start), self.band_index(end)
        return slice(min(i0, i1), max(i0, i1) + 1)

    def subset_bands(self, selector) -> "HSImage":
        """Return a copy containing only the selected bands (slice, bool mask or index list)."""
        if isinstance(selector, tuple) and len(selector) == 2:
            selector = self.band_range(*selector)
        out = self.copy(self.data[..., selector])
        out.wavelengths = self.wavelengths[selector]
        if self.fwhm is not None:
            out.fwhm = self.fwhm[selector]
        if self.bbl is not None:
            out.bbl = self.bbl[selector]
        return out

    def get_band(self, b) -> np.ndarray:
        return self.data[..., self.band_index(b)]

    def rgb(self, bands=(640.0, 550.0, 460.0), percentiles=(2, 98)) -> np.ndarray:
        """Return a percentile-stretched uint8 RGB composite for previews."""
        idx = [self.band_index(b) for b in bands]
        arr = self.data[..., idx].astype(float)
        lo, hi = np.nanpercentile(arr, percentiles)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-12), 0, 1)
        return (np.nan_to_num(arr) * 255).astype(np.uint8)

    def spectra(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return valid pixel spectra as an ``(n, bands)`` array."""
        X = self.data.reshape(-1, self.bands)
        if mask is not None:
            X = X[np.asarray(mask).ravel()]
        return X[np.isfinite(X).all(axis=1)]

    # ------------------------------------------------------------- spatial
    def pixel_to_map(self, row, col):
        """Convert pixel (row, col) centres to map coordinates using the geotransform."""
        if self.transform is None:
            raise ValueError("Image has no geotransform")
        x0, dx, rx, y0, ry, dy = self.transform
        col = np.asarray(col) + 0.5
        row = np.asarray(row) + 0.5
        return x0 + col * dx + row * rx, y0 + col * ry + row * dy

    # ------------------------------------------------------------- interop
    def to_hylite(self):
        """Convert to a ``hylite.HyImage`` (which uses ``(x, y, b)`` ordering)."""
        import hylite

        img = hylite.HyImage(np.transpose(self.data, (1, 0, 2)).copy())
        img.set_wavelengths(self.wavelengths)
        if self.fwhm is not None:
            img.header['fwhm'] = list(self.fwhm)
        if self.bbl is not None:
            img.header['bbl'] = list(self.bbl.astype(int))
        for k, v in self.meta.items():
            if isinstance(v, (str, int, float)):
                img.header[k] = v
        return img

    @classmethod
    def from_hylite(cls, img) -> "HSImage":
        """Create an HSImage from a ``hylite.HyImage``."""
        wav = img.get_wavelengths() if img.has_wavelengths() else None
        meta = {}
        try:
            meta = {k: v for k, v in dict(img.header).items() if isinstance(v, (str, int, float))}
        except Exception:  # pragma: no cover - header implementation details vary
            pass
        return cls(np.transpose(img.data, (1, 0, 2)).astype(np.float32), wav, meta=meta)

    def __repr__(self):  # pragma: no cover - cosmetic
        wl = "%.1f-%.1f nm" % (np.nanmin(self.wavelengths), np.nanmax(self.wavelengths)) if self.bands else "-"
        return "HSImage(rows=%d, cols=%d, bands=%d, %s)" % (self.rows, self.cols, self.bands, wl)
