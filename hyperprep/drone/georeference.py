"""
Direct georeferencing and orthorectification of pushbroom drone imagery.

For each scan line the sensor's position and attitude (from a
:class:`~hyperprep.drone.navigation.NavTrack`) define one look ray per
across-track pixel. Rays are intersected with a flat plane or a DEM, giving the
ground coordinates of every pixel, which are then resampled to a regular grid
with a GLT (:mod:`hyperprep.ortho`).

This complements the point-cloud projection in
:mod:`hyperprep.terrestrial.projection`, which is preferred for steep terrain
(cliffs, open pits) where a 2.5-D DEM is insufficient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..geo import NED_TO_ENU, rotation_matrix
from ..image import HSImage
from ..ortho import GLT, apply_glt, build_glt
from .navigation import NavTrack


@dataclass
class DEM:
    """A north-up raster DEM in the same projected CRS as the navigation track."""

    z: np.ndarray
    transform: Tuple[float, ...]  # GDAL geotransform

    def sample(self, x, y):
        """Bilinear interpolation of heights at map coordinates (NaN outside)."""
        from scipy.ndimage import map_coordinates

        x0, dx, _, y0, _, dy = self.transform
        col = (np.asarray(x) - x0) / dx - 0.5
        row = (np.asarray(y) - y0) / dy - 0.5
        return map_coordinates(self.z, [row.ravel(), col.ravel()], order=1, cval=np.nan, mode='constant').reshape(np.shape(x))

    @classmethod
    def from_image(cls, image: HSImage, band: int = 0) -> "DEM":
        return cls(image.data[..., band].astype(float), tuple(image.transform))


def pixel_look_angles(n_pixels: int, ifov_deg: Optional[float] = None, fov_deg: Optional[float] = None,
                      angles_deg: Optional[Sequence[float]] = None, flip: bool = False) -> np.ndarray:
    """Across-track look angle of each detector pixel (degrees, positive to the right).

    Provide either the per-pixel IFOV, the full FOV, or explicit per-pixel angles (from the
    sensor's geometric calibration file, which accounts for lens distortion).
    """
    if angles_deg is not None:
        a = np.asarray(angles_deg, float)
    else:
        if ifov_deg is None:
            if fov_deg is None:
                raise ValueError("Provide ifov_deg, fov_deg or angles_deg")
            ifov_deg = fov_deg / n_pixels
        a = (np.arange(n_pixels) - (n_pixels - 1) / 2) * ifov_deg
    return -a[::-1] if flip else a


def look_vectors(track: NavTrack, angles_deg: np.ndarray, boresight: Sequence[float] = (0, 0, 0)) -> np.ndarray:
    """Unit look vectors (ENU) for each line and pixel, shape ``(lines, pixels, 3)``."""
    th = np.deg2rad(angles_deg)
    body = np.stack([np.zeros_like(th), np.sin(th), np.cos(th)], axis=-1)  # forward, right, down
    Rb = rotation_matrix(*boresight)
    body = body @ Rb.T
    out = np.empty((len(track), len(th), 3))
    for i in range(len(track)):
        R = NED_TO_ENU @ rotation_matrix(track.roll[i], track.pitch[i], track.yaw[i])
        out[i] = body @ R.T
    return out


def georeference(track: NavTrack, angles_deg: np.ndarray, ground: Union[float, DEM] = 0.0,
                 boresight: Sequence[float] = (0, 0, 0), iterations: int = 10, tol: float = 0.01):
    """Ground coordinates of every pixel by ray casting.

    Args:
        track: navigation track (projected x/y/z), one record per image line.
        angles_deg: across-track look angles from :func:`pixel_look_angles`.
        ground: a constant ground height or a :class:`DEM`.
        boresight: (roll, pitch, yaw) misalignment between IMU and sensor, degrees.
        iterations: maximum ray-DEM intersection iterations.

    Returns:
        ``(X, Y, Z, view)`` where X/Y/Z are ``(lines, pixels)`` ground coordinates and ``view`` the
        unit look vectors ``(lines, pixels, 3)``.
    """
    v = look_vectors(track, angles_deg, boresight)
    cp = track.xyz[:, None, :]
    down = -v[..., 2]
    down = np.where(down > 1e-6, down, np.nan)  # rays pointing at/above the horizon never hit
    h = np.full(v.shape[:2], float(ground) if not isinstance(ground, DEM) else np.nanmean(ground.z))
    for _ in range(iterations if isinstance(ground, DEM) else 1):
        t = (cp[..., 2] - h) / down
        X = cp[..., 0] + t * v[..., 0]
        Y = cp[..., 1] + t * v[..., 1]
        if not isinstance(ground, DEM):
            break
        hn = ground.sample(X, Y)
        hn = np.where(np.isfinite(hn), hn, h)
        if np.nanmax(np.abs(hn - h)) < tol:
            h = hn
            break
        h = 0.5 * (h + hn)  # damped update for robustness on steep slopes
    Z = h
    t = (cp[..., 2] - Z) / down
    X = cp[..., 0] + t * v[..., 0]
    Y = cp[..., 1] + t * v[..., 1]
    return X, Y, Z, v


def view_zenith(view: np.ndarray) -> np.ndarray:
    """View zenith angle (degrees) of each pixel from the look vectors."""
    return np.rad2deg(np.arccos(np.clip(-view[..., 2], -1, 1)))


def orthorectify(image: HSImage, track: NavTrack, angles_deg: np.ndarray, res: float,
                 ground: Union[float, DEM] = 0.0, boresight: Sequence[float] = (0, 0, 0),
                 max_dist: Optional[float] = None, crs=None, return_glt: bool = False):
    """Direct-georeference and resample a pushbroom image to a north-up map grid.

    The image must be in raw geometry with rows = scan lines (matching ``track``) and
    cols = across-track pixels.
    """
    if image.rows != len(track):
        raise ValueError("Image has %d lines but track has %d records" % (image.rows, len(track)))
    X, Y, Z, v = georeference(track, angles_deg, ground, boresight)
    glt = build_glt(X, Y, res, max_dist=max_dist, crs=crs if crs is not None else track.crs)
    out = apply_glt(image, glt)
    out.meta['georeferencing'] = 'direct (ray casting, %s)' % ('DEM' if isinstance(ground, DEM) else 'plane z=%.2f' % ground)
    if return_glt:
        return out, glt, {'X': X, 'Y': Y, 'Z': Z, 'view_zenith': view_zenith(v)}
    return out
