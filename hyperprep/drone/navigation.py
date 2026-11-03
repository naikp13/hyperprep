"""
GNSS / IMU navigation tracks for pushbroom (drone and airborne) sensors.

Generalises the IMU-loading cell of ``2_Projection_mapping.ipynb``: the track is
read from HySpexNav / PosPac style text exports (or any CSV with a column
mapping), converted to UTM with an automatically detected zone (the notebook
hard-coded zone 33S), and checked against the image line count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from ..geo import latlon_to_utm, rotation_matrix, utm_epsg

HYSPEXNAV_COLUMNS = ('frame', 'lon', 'lat', 'z', 'roll', 'pitch', 'yaw', 'time')


@dataclass
class NavTrack:
    """Per-scan-line platform position and attitude.

    Attributes:
        time: time stamps (seconds; any consistent epoch).
        lat, lon: WGS84 degrees (may be None if only projected coordinates are known).
        x, y, z: projected coordinates (e.g. UTM easting, northing, ellipsoidal/orthometric height).
        roll, pitch, yaw: attitude in degrees (aerospace convention, yaw = heading).
        frame: optional frame / line numbers.
        crs: EPSG code of x/y (set by :meth:`to_utm`).
    """

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    yaw: np.ndarray
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    frame: Optional[np.ndarray] = None
    crs: Optional[int] = None
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.time)

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack([self.x, self.y, self.z])

    @property
    def rpy(self) -> np.ndarray:
        return np.column_stack([self.roll, self.pitch, self.yaw])

    # --------------------------------------------------------------- ops
    def to_utm(self, zone: Optional[int] = None, hemisphere: Optional[str] = None) -> "NavTrack":
        """Fill x / y with UTM coordinates from lat / lon (zone detected from the first record)."""
        if self.lat is None or self.lon is None:
            raise ValueError("Track has no lat/lon")
        e, n, zone, hemi = latlon_to_utm(self.lat, self.lon, zone, hemisphere)
        self.x, self.y = np.asarray(e), np.asarray(n)
        self.crs = utm_epsg(zone, hemi)
        self.meta['utm_zone'] = '%d%s' % (zone, hemi)
        return self

    def subset(self, start: int, end: int) -> "NavTrack":
        sl = slice(start, end)
        g = lambda a: None if a is None else np.asarray(a)[sl]
        return NavTrack(g(self.time), g(self.x), g(self.y), g(self.z), g(self.roll), g(self.pitch), g(self.yaw),
                        g(self.lat), g(self.lon), g(self.frame), self.crs, dict(self.meta))

    def interpolate(self, times: Sequence[float]) -> "NavTrack":
        """Interpolate the track to new time stamps (e.g. scan-line exposure times from the
        sensor log when the nav file has a higher/lower rate). Angles are unwrapped first."""
        t = np.asarray(times, float)
        li = lambda a: None if a is None else np.interp(t, self.time, a)
        ang = lambda a: (np.rad2deg(np.interp(t, self.time, np.unwrap(np.deg2rad(a)))) + 180) % 360 - 180
        return NavTrack(t, li(self.x), li(self.y), li(self.z), ang(self.roll), ang(self.pitch),
                        ang(self.yaw) % 360, li(self.lat), li(self.lon), None, self.crs, dict(self.meta))

    def apply_lever_arm(self, offset_body: Sequence[float]) -> "NavTrack":
        """Shift positions from the IMU/GNSS reference point to the sensor's optical centre.

        ``offset_body`` is the (forward, right, down) offset in metres in the body frame.
        x/y/z are assumed to be (east, north, up).
        """
        off = np.asarray(offset_body, float)
        x, y, z = self.x.copy(), self.y.copy(), self.z.copy()
        for i in range(len(self)):
            ned = rotation_matrix(self.roll[i], self.pitch[i], self.yaw[i]) @ off
            x[i] += ned[1]
            y[i] += ned[0]
            z[i] -= ned[2]
        out = self.subset(0, len(self))
        out.x, out.y, out.z = x, y, z
        return out

    def apply_boresight(self, d_roll: float, d_pitch: float, d_yaw: float) -> "NavTrack":
        """Add (small) boresight misalignment angles in degrees."""
        out = self.subset(0, len(self))
        out.roll = self.roll + d_roll
        out.pitch = self.pitch + d_pitch
        out.yaw = (self.yaw + d_yaw) % 360
        return out

    def smooth(self, window: int = 5) -> "NavTrack":
        """Moving-average smoothing of positions and attitude (reduces IMU jitter)."""
        from scipy.ndimage import uniform_filter1d

        f = lambda a: uniform_filter1d(np.asarray(a, float), window, mode='nearest')
        out = self.subset(0, len(self))
        out.x, out.y, out.z = f(self.x), f(self.y), f(self.z)
        out.roll, out.pitch = f(self.roll), f(self.pitch)
        out.yaw = (np.rad2deg(f(np.unwrap(np.deg2rad(self.yaw))))) % 360
        return out

    def check_lines(self, n_lines: int, allow_extra: int = 1) -> int:
        """Validate the track against the number of image lines.

        Returns the number of trailing image lines to drop (0 or up to ``allow_extra``),
        raising if the mismatch is larger, mirroring the notebook's assertion."""
        extra = n_lines - len(self)
        if 0 <= extra <= allow_extra:
            return extra
        raise ValueError("IMU data has %d frames, image has %d." % (len(self), n_lines))


def read_hyspexnav(path: str, columns: Sequence[str] = HYSPEXNAV_COLUMNS, utm: bool = True) -> NavTrack:
    """Read a whitespace-delimited HySpexNav / PosPac export (default columns:
    ``frame lon lat z roll pitch yaw time``)."""
    arr = np.loadtxt(path, ndmin=2)
    cols = {c: arr[:, i] for i, c in enumerate(columns) if i < arr.shape[1]}
    return _from_columns(cols, utm)


def read_nav_csv(path: str, mapping: Optional[Dict[str, str]] = None, delimiter: str = ',', utm: bool = True) -> NavTrack:
    """Read a navigation CSV with a header row.

    Args:
        mapping: maps hyperprep names (``time, lat, lon, z, roll, pitch, yaw, x, y, frame``) to the
            CSV column names, e.g. ``{'time': 'GPS_Time', 'z': 'Altitude', 'yaw': 'Heading'}``.
            Unmapped names are matched case-insensitively.
    """
    import csv

    with open(path, newline='') as f:
        rows = list(csv.DictReader(f, delimiter=delimiter))
    if not rows:
        raise ValueError("Empty navigation file %s" % path)
    header = {k.strip().lower(): k for k in rows[0].keys()}
    mapping = {k: v for k, v in (mapping or {}).items()}
    cols = {}
    for name in ('time', 'lat', 'lon', 'z', 'roll', 'pitch', 'yaw', 'x', 'y', 'frame'):
        src = mapping.get(name, header.get(name))
        if src is not None and src in rows[0]:
            cols[name] = np.array([float(r[src]) for r in rows])
    return _from_columns(cols, utm)


def _from_columns(cols: Dict[str, np.ndarray], utm: bool) -> NavTrack:
    n = len(next(iter(cols.values())))
    zeros = np.zeros(n)
    tr = NavTrack(time=cols.get('time', np.arange(n, dtype=float)), x=cols.get('x', zeros.copy()),
                  y=cols.get('y', zeros.copy()), z=cols.get('z', zeros.copy()), roll=cols.get('roll', zeros.copy()),
                  pitch=cols.get('pitch', zeros.copy()), yaw=cols.get('yaw', zeros.copy()), lat=cols.get('lat'),
                  lon=cols.get('lon'), frame=cols.get('frame'))
    if utm and tr.lat is not None and 'x' not in cols:
        tr.to_utm()
    return tr
