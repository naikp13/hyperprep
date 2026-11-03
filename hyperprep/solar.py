"""
Solar geometry and irradiance utilities (no external dependencies).

The solar position uses the NOAA General Solar Position algorithm (accuracy
better than ~0.1 degree between 1950 and 2050), which is sufficient for
illumination, topographic and TOA-reflectance corrections.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional, Tuple, Union

import numpy as np

TimeLike = Union[_dt.datetime, str, Tuple[str, str, str]]


def _to_utc(time: TimeLike) -> _dt.datetime:
    """Accept a datetime, an ISO string, or the hylite style tuple ``(timestring, format, timezone)``."""
    if isinstance(time, tuple):
        tstr, fmt = time[0], time[1]
        tz = time[2] if len(time) > 2 else 'UTC'
        t = _dt.datetime.strptime(tstr, fmt)
        if tz and tz.upper() != 'UTC':
            try:
                from zoneinfo import ZoneInfo
                t = t.replace(tzinfo=ZoneInfo(tz))
            except Exception as e:  # pragma: no cover
                raise ValueError("Unknown time zone %s" % tz) from e
        else:
            t = t.replace(tzinfo=_dt.timezone.utc)
    elif isinstance(time, str):
        t = _dt.datetime.fromisoformat(time.replace('Z', '+00:00'))
    else:
        t = time
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone(_dt.timezone.utc)


def solar_position(lat: float, lon: float, time: TimeLike) -> Tuple[float, float]:
    """Solar azimuth (degrees clockwise from north) and elevation (degrees above horizon).

    Args:
        lat, lon: position in decimal degrees (east positive).
        time: datetime (naive = UTC), ISO string, or ``(timestring, format, tz)`` tuple.
    """
    t = _to_utc(time)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + t.second / 3600
    ndays = 366 if (t.year % 4 == 0 and (t.year % 100 != 0 or t.year % 400 == 0)) else 365
    g = 2 * np.pi / ndays * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon
    ha = np.deg2rad(tst / 4 - 180)
    phi = np.deg2rad(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    zen = np.arccos(np.clip(cos_zen, -1, 1))
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(decl) * np.cos(phi))
    az = (np.rad2deg(az) + 180) % 360
    return float(az), float(90 - np.rad2deg(zen))


def sph2cart(az: float, el: float, r: float = 1.0) -> np.ndarray:
    """Convert azimuth (from north, clockwise) / elevation (degrees) to an ENU unit vector."""
    az, el = np.deg2rad(az), np.deg2rad(el)
    return r * np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])


def sun_vector(lat: float, lon: float, time: TimeLike):
    """Illumination vector (direction light travels, i.e. pointing *from* the sun, ENU) plus azimuth and
    elevation. Same convention as ``hylite.correct.illumination.estimate_sun_vec``, so the cosine of the
    incidence angle on a surface with normal ``n`` is ``dot(n, -sunvec)``."""
    az, el = solar_position(lat, lon, time)
    return -sph2cart(az, el), az, el


def earth_sun_distance(time: TimeLike) -> float:
    """Earth-sun distance in astronomical units."""
    doy = _to_utc(time).timetuple().tm_yday
    return float(1 - 0.01672 * np.cos(np.deg2rad(0.9856 * (doy - 4))))


def solar_irradiance(wavelengths, fwhm=None, path: Optional[str] = None) -> np.ndarray:
    """Exo-atmospheric solar spectral irradiance (W m-2 nm-1) at 1 AU for the given band centres.

    If ``path`` is given it must point to a two-column text file (wavelength nm, irradiance in
    W m-2 nm-1, e.g. the Thuillier 2003 or TSIS-1 HSRS spectrum) which is convolved to the band
    centres. Otherwise a 5778 K black-body scaled to the 1361 W m-2 solar constant is used: this
    is within ~5-10 % across the VNIR-SWIR but misses Fraunhofer lines, so supply a measured
    spectrum (or the sensor's own ESUN values) for quantitative TOA reflectance.
    """
    wav = np.asarray(wavelengths, float)
    if path is not None:
        from .io import read_spectrum_txt
        from .spectral import resample_spectra
        sw, si = read_spectrum_txt(path)
        if fwhm is None:
            return np.interp(wav, sw, si)
        return resample_spectra(si[None, :], sw, wav, fwhm)[0]
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam = wav * 1e-9
    B = 2 * h * c ** 2 / lam ** 5 / (np.exp(h * c / (lam * k * 5778.0)) - 1)  # W m-2 sr-1 m-1
    # scale to the solar constant: integrate full blackbody analytically (sigma T^4) * (R_sun / AU)^2
    sigma = 5.670374419e-8
    E = np.pi * B * 1e-9  # W m-2 nm-1 at the solar surface
    return E * 1361.0 / (sigma * 5778.0 ** 4)
