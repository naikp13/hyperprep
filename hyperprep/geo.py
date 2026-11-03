"""
Coordinate helpers: WGS84 <-> UTM (vectorised, no dependencies), rotation matrices.

The UTM conversion follows the same series expansion as the ``utm`` package used in
the original projection-mapping notebook (sub-millimetre agreement), so the
``utm`` dependency is no longer required.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_K0 = 0.9996
_E = 0.00669438
_E2, _E3 = _E * _E, _E * _E * _E
_EP2 = _E / (1 - _E)
_SQRT_E = np.sqrt(1 - _E)
_E_ = (1 - _SQRT_E) / (1 + _SQRT_E)
_R = 6378137.0
_M1 = 1 - _E / 4 - 3 * _E2 / 64 - 5 * _E3 / 256
_M2 = 3 * _E / 8 + 3 * _E2 / 32 + 45 * _E3 / 1024
_M3 = 15 * _E2 / 256 + 45 * _E3 / 1024
_M4 = 35 * _E3 / 3072
_P2 = 3. / 2 * _E_ - 27. / 32 * _E_ ** 3 + 269. / 512 * _E_ ** 5
_P3 = 21. / 16 * _E_ ** 2 - 55. / 32 * _E_ ** 4
_P4 = 151. / 96 * _E_ ** 3 - 417. / 128 * _E_ ** 5
_P5 = 1097. / 512 * _E_ ** 4


def utm_zone(lat: float, lon: float) -> Tuple[int, str]:
    """UTM zone number and hemisphere (``'N'`` or ``'S'``) for a location."""
    zone = int((lon + 180) // 6) + 1
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat <= 84 and lon >= 0:
        for lim, z in ((9, 31), (21, 33), (33, 35), (42, 37)):
            if lon < lim:
                zone = z
                break
    return zone, 'N' if lat >= 0 else 'S'


def latlon_to_utm(lat, lon, zone: Optional[int] = None, hemisphere: Optional[str] = None):
    """Convert WGS84 lat/lon (degrees, scalars or arrays) to UTM easting/northing.

    Returns:
        ``(easting, northing, zone, hemisphere)``. A single zone is used for all points
        (the zone of the first point unless forced), as required for a consistent track.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if zone is None or hemisphere is None:
        z, h = utm_zone(float(np.ravel(lat)[0]), float(np.ravel(lon)[0]))
        zone = zone or z
        hemisphere = hemisphere or h
    lat_r = np.deg2rad(lat)
    lat_sin, lat_cos = np.sin(lat_r), np.cos(lat_r)
    lat_tan = lat_sin / lat_cos
    lat_tan2, lat_tan4 = lat_tan ** 2, lat_tan ** 4
    central_lon = (zone - 1) * 6 - 180 + 3
    lon_r = np.deg2rad(lon)
    dlon = np.mod(lon_r - np.deg2rad(central_lon) + np.pi, 2 * np.pi) - np.pi
    n = _R / np.sqrt(1 - _E * lat_sin ** 2)
    c = _EP2 * lat_cos ** 2
    a = lat_cos * dlon
    a2, a3, a4, a5, a6 = a ** 2, a ** 3, a ** 4, a ** 5, a ** 6
    m = _R * (_M1 * lat_r - _M2 * np.sin(2 * lat_r) + _M3 * np.sin(4 * lat_r) - _M4 * np.sin(6 * lat_r))
    easting = _K0 * n * (a + a3 / 6 * (1 - lat_tan2 + c) + a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * _EP2)) + 500000
    northing = _K0 * (m + n * lat_tan * (a2 / 2 + a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c ** 2)
                                         + a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * _EP2)))
    if str(hemisphere).upper().startswith('S'):
        northing = northing + 10000000
    return easting, northing, zone, hemisphere


def utm_to_latlon(easting, northing, zone: int, hemisphere: str = 'N'):
    """Inverse of :func:`latlon_to_utm`."""
    x = np.asarray(easting, float) - 500000
    y = np.asarray(northing, float)
    if str(hemisphere).upper().startswith('S'):
        y = y - 10000000
    m = y / _K0
    mu = m / (_R * _M1)
    p_rad = mu + _P2 * np.sin(2 * mu) + _P3 * np.sin(4 * mu) + _P4 * np.sin(6 * mu) + _P5 * np.sin(8 * mu)
    p_sin, p_cos = np.sin(p_rad), np.cos(p_rad)
    p_tan = p_sin / p_cos
    p_tan2, p_tan4 = p_tan ** 2, p_tan ** 4
    ep_sin = 1 - _E * p_sin ** 2
    ep_sin_sqrt = np.sqrt(ep_sin)
    n = _R / ep_sin_sqrt
    r = (1 - _E) / ep_sin
    c = _EP2 * p_cos ** 2
    d = x / (n * _K0)
    d2, d3, d4, d5, d6 = d ** 2, d ** 3, d ** 4, d ** 5, d ** 6
    lat = (p_rad - (p_tan / r) * (d2 / 2 - d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c ** 2 - 9 * _EP2))
           + d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * _EP2 - 3 * c ** 2))
    lon = (d - d3 / 6 * (1 + 2 * p_tan2 + c) + d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c ** 2 + 8 * _EP2 + 24 * p_tan4)) / p_cos
    central_lon = (zone - 1) * 6 - 180 + 3
    return np.rad2deg(lat), np.mod(np.rad2deg(lon) + central_lon + 180, 360) - 180


def utm_epsg(zone: int, hemisphere: str) -> int:
    """EPSG code of a WGS84 UTM zone."""
    return (32600 if str(hemisphere).upper().startswith('N') else 32700) + int(zone)


def rotation_matrix(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """Body (forward-right-down) to North-East-Down rotation, ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    This is the standard aerospace convention used by IMU/GNSS post-processing software
    (e.g. PosPac / HySpexNav exports): yaw is the heading clockwise from north, pitch is
    nose-up positive and roll is right-wing-down positive.
    """
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


#: Matrix converting North-East-Down vectors to East-North-Up.
NED_TO_ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
