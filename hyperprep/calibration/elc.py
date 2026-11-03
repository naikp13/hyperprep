"""
Empirical Line Calibration (ELC) and calibration panels.

A pure-numpy implementation of the panel-based correction used in the
hypercloud and FENIX notebooks (``hylite.correct.ELC`` / ``Panel``), usable
for terrestrial, drone and satellite (field-spectra) data alike.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..image import HSImage


@dataclass
class Panel:
    """A calibration target observed in an image.

    Attributes:
        name: label (e.g. ``'white'``, ``'R90'``, ``'black'``).
        radiance: measured at-sensor signal of the panel, shape ``(bands,)``.
        reflectance: known reflectance of the panel at the image band centres, ``(bands,)`` or scalar.
        normal: optional unit normal of the panel (ENU) for illumination-aware corrections.
        skyview: optional sky-view factor of the panel (0-1).
    """

    name: str
    radiance: np.ndarray
    reflectance: Union[float, np.ndarray]
    normal: Optional[np.ndarray] = None
    skyview: Optional[float] = None

    def __post_init__(self):
        self.radiance = np.asarray(self.radiance, float)
        self.reflectance = np.broadcast_to(np.asarray(self.reflectance, float), self.radiance.shape).copy()

    @classmethod
    def from_region(cls, name: str, image: HSImage, region: Tuple[int, int, int, int],
                    reflectance, trim: float = 10.0) -> "Panel":
        """Build a panel from an image region ``(row0, row1, col0, col1)`` using a trimmed mean
        (``trim`` percent removed from each tail to suppress edge pixels)."""
        r0, r1, c0, c1 = region
        X = image.data[r0:r1, c0:c1].reshape(-1, image.bands).astype(float)
        X = X[np.isfinite(X).all(axis=1)]
        b = X.mean(axis=1)
        lo, hi = np.percentile(b, (trim, 100 - trim))
        sel = (b >= lo) & (b <= hi)
        return cls(name, X[sel].mean(axis=0) if sel.any() else X.mean(axis=0),
                   resample_reflectance(reflectance, image.wavelengths))

    @classmethod
    def from_file(cls, name: str, path: str, image_wavelengths, reflectance) -> "Panel":
        """Load a panel radiance spectrum from a text export (e.g. ``White_Panel_RadAverage.txt``)."""
        from ..io import read_spectrum_txt
        from ..spectral import resample_spectra

        wav, rad = read_spectrum_txt(path)
        rad = np.atleast_2d(rad.T)[0] if rad.ndim > 1 else rad
        rad = resample_spectra(rad[None, :], wav, image_wavelengths)[0]
        return cls(name, rad, resample_reflectance(reflectance, image_wavelengths))


def resample_reflectance(reflectance, wavelengths) -> np.ndarray:
    """Accept a scalar, a ``(bands,)`` array, or a ``(wavelengths, values)`` tuple and return
    reflectance at ``wavelengths``."""
    wav = np.asarray(wavelengths, float)
    if isinstance(reflectance, tuple) and len(reflectance) == 2:
        w, r = np.asarray(reflectance[0], float), np.asarray(reflectance[1], float)
        if np.nanmax(w) < 30:
            w = w * 1000
        return np.interp(wav, w, r)
    r = np.asarray(reflectance, float)
    return np.broadcast_to(r, wav.shape).copy() if r.ndim == 0 else r


class ELC:
    """Empirical line correction ``R = (L - offset) / gain`` fitted per band from panels.

    * one panel: offset fixed to zero (or to a dark-object estimate passed as ``dark``);
    * two or more panels: least-squares slope and intercept per band.
    """

    def __init__(self, panels: Sequence[Panel], dark: Optional[np.ndarray] = None):
        if len(panels) == 0:
            raise ValueError("At least one panel is required")
        self.panels = list(panels)
        L = np.array([p.radiance for p in panels])          # (n, bands)
        R = np.array([p.reflectance for p in panels])       # (n, bands)
        if len(panels) == 1:
            off = np.zeros(L.shape[1]) if dark is None else np.asarray(dark, float)
            with np.errstate(invalid='ignore', divide='ignore'):
                gain = (L[0] - off) / R[0]
        else:
            Rm, Lm = R.mean(axis=0), L.mean(axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                gain = ((R - Rm) * (L - Lm)).sum(axis=0) / ((R - Rm) ** 2).sum(axis=0)
            off = Lm - gain * Rm
        self.gain, self.offset = gain, off

    def residuals(self) -> np.ndarray:
        """Reflectance residuals of each panel after correction, ``(n_panels, bands)``."""
        return np.array([self.apply_spectra(p.radiance) - p.reflectance for p in self.panels])

    def bad_bands(self, thresh: float = 0.0) -> np.ndarray:
        """Bands where the fit is invalid (non-positive gain), useful to update the BBL."""
        return ~(np.isfinite(self.gain) & (self.gain > thresh))

    def apply_spectra(self, L):
        with np.errstate(invalid='ignore', divide='ignore'):
            return (np.asarray(L, float) - self.offset) / self.gain

    def apply(self, image: HSImage) -> HSImage:
        out = image.copy(self.apply_spectra(image.data).astype(np.float32))
        out.meta['units'] = 'reflectance'
        out.meta['correction'] = 'ELC (%d panels)' % len(self.panels)
        bb = self.bad_bands()
        if bb.any():
            out.bbl = ~bb if out.bbl is None else (out.bbl & ~bb)
        return out


def empirical_line(image: HSImage, panels: Sequence[Panel], dark=None) -> HSImage:
    """Convenience wrapper: fit an :class:`ELC` and apply it."""
    return ELC(panels, dark).apply(image)
