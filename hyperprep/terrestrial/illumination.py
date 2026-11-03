"""
Illumination correction for terrestrial (and drone) scenes with known 3-D geometry.

Refactored from ``3_Illumination_corrections_FENIX.ipynb`` (method of Thiele et
al., 2021, https://doi.org/10.1109/TGRS.2021.3098725). Measured radiance is modelled as

    L = R * (skv * S + ilf * I) + P

with ``S`` the diffuse skylight, ``I`` direct sunlight, ``skv`` the per-pixel
sky-view factor, ``ilf`` the direct illumination factor (Lambert or Oren-Nayar
using surface normals and the sun vector) and ``P`` path radiance.

Changes relative to the notebook:

* sun vector derived from header date/time *or* explicit arguments, with a clear
  error instead of silently overriding the time;
* the sky / sun spectra are solved in closed form from a sunlit panel plus the
  median of shaded vs sunlit pixels (pure numpy, :func:`estimate_sky_sun`) with
  the hylite ``estimate_illu`` solver available as an alternative;
* reflectance inversion is pure numpy (:func:`illumination_reflectance`);
* ``hylite.correct.illumination.estimate_illu`` returns ``(sun, sky)``; the notebook
  assigned the result to ``S.skylight, S.sunlight`` (swapped).
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional, Sequence, Tuple

import numpy as np

from ..solar import sun_vector
from ..topo import incidence, oren_nayar, skyview_from_normals


def parse_header_time(meta: dict) -> Optional[_dt.datetime]:
    """Parse acquisition date/time from ENVI header fields (``acquisition date``, ``start time``,
    ``acquisition time``), accepting the ``'DATE: 2021-12-03'`` style used by Specim."""
    def clean(v):
        return str(v).split(': ')[-1].strip() if v is not None else None

    date = clean(meta.get('acquisition date'))
    time = clean(meta.get('start time') or meta.get('acquisition time'))
    if date is None:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%d-%m-%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S'):
        try:
            return _dt.datetime.strptime('%s %s' % (date, time or '12:00:00'), fmt).replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            continue
    raise ValueError("Could not parse date/time %r %r" % (date, time))


def scene_sun_vector(lat: float, lon: float, time=None, meta: Optional[dict] = None):
    """Sun illumination vector for a scene: uses ``time`` if given, otherwise the header in ``meta``
    (times are assumed UTC). Returns ``(sunvec, azimuth, elevation)``."""
    if time is None:
        if meta is None:
            raise ValueError("Provide a time or header metadata")
        time = parse_header_time(meta)
        if time is None:
            raise ValueError("No acquisition time in header; pass time explicitly")
    return sun_vector(lat, lon, time)


def panel_illumination_factor(panel_normal: Sequence[float], sunvec: Sequence[float]) -> float:
    """Direct (Lambertian) illumination factor of a calibration panel: ``max(0, n . -sunvec)``.

    ``panel_normal`` must point *out* of the panel face (towards the sky / sensor). The notebook
    negated one of the two normals by hand; flip the normal you pass instead if needed.
    """
    n = np.asarray(panel_normal, float)
    n = n / np.linalg.norm(n)
    return float(max(0.0, np.dot(n, -np.asarray(sunvec, float))))


def illumination_factors(normals: np.ndarray, sunvec: np.ndarray, view: Optional[np.ndarray] = None,
                         model: str = 'oren-nayar', roughness: float = 0.3) -> np.ndarray:
    """Per-pixel direct illumination factor ``ilf`` from surface normals.

    ``model`` is ``'lambert'`` or ``'oren-nayar'`` (requires the per-pixel view vectors from
    the surface towards the sensor).
    """
    if model == 'lambert' or view is None:
        return incidence(normals, sunvec)
    return np.clip(oren_nayar(normals, view, sunvec, roughness), 0, None)


def estimate_sky_sun(radiance: np.ndarray, ilf: np.ndarray, skv, panel_radiance: np.ndarray,
                     panel_reflectance, panel_ilf: float, panel_skv: float, shade_thresh: float = 0.01,
                     sun_thresh: float = 0.5, clip: Optional[Tuple[int, int, int, int]] = None):
    """Closed-form estimate of skylight ``S`` and sunlight ``I`` spectra.

    Assumes shaded (``ilf < shade_thresh``) and sunlit (``ilf > sun_thresh``) pixels in the clip
    region share the same median reflectance, and that a calibration panel of known reflectance
    with known ``panel_ilf`` / ``panel_skv`` was measured::

        Ls = Rm * skv_s * S
        Ll = Rm * (skv_l * S + ilf_l * I)      ->  I / S = (Ll / Ls * skv_s - skv_l) / ilf_l
        Lp = Rp * (pskv * S + pilf * I)        ->  S = Lp / (Rp * (pskv + pilf * I/S))

    Args:
        radiance: ``(..., bands)`` image radiance (any 2-D spatial orientation).
        ilf: direct illumination factors with the image's spatial shape.
        skv: sky-view factors (array or scalar).
        clip: optional ``(a0, a1, b0, b1)`` subset of the first two image axes used for estimation.

    Returns:
        ``(S, I)`` spectra, shape ``(bands,)``.
    """
    L = np.asarray(radiance, float)
    ilf = np.asarray(ilf, float)
    skv = np.broadcast_to(np.asarray(skv, float), ilf.shape)
    if clip is not None:
        a0, a1, b0, b1 = clip
        L, ilf, skv = L[a0:a1, b0:b1], ilf[a0:a1, b0:b1], skv[a0:a1, b0:b1]
    valid = np.isfinite(L).all(axis=-1) & np.isfinite(ilf)
    shade = valid & (ilf < shade_thresh)
    lit = valid & (ilf > sun_thresh)
    if shade.sum() < 10 or lit.sum() < 10:
        raise ValueError("Need >= 10 shaded and sunlit pixels (got %d, %d); adjust thresholds or clip"
                         % (shade.sum(), lit.sum()))
    Ls, Ll = np.median(L[shade], axis=0), np.median(L[lit], axis=0)
    skv_s, skv_l, ilf_l = np.median(skv[shade]), np.median(skv[lit]), np.median(ilf[lit])
    k = np.clip((Ll / Ls * skv_s - skv_l) / ilf_l, 0, None)  # I / S
    Rp = np.asarray(panel_reflectance, float)
    S = np.asarray(panel_radiance, float) / (Rp * (panel_skv + panel_ilf * k))
    return S, k * S


def illumination_reflectance(radiance: np.ndarray, sun: np.ndarray, sky: np.ndarray, ilf: np.ndarray,
                             skv=1.0, path=0.0, occlusion=None, strict: bool = False) -> np.ndarray:
    """Invert the illumination model: ``R = (L - P) / (skv * S + (1 - oc) * ilf * I)``
    (pure-numpy ``hylite.correct.illumination.IlluModel.getReflectance``)."""
    ilf = np.asarray(ilf, float)
    skv = np.broadcast_to(np.asarray(skv, float), ilf.shape)
    if occlusion is not None:
        ilf = ilf * (1 - np.asarray(occlusion, float))
    denom = skv[..., None] * np.asarray(sky, float) + ilf[..., None] * np.asarray(sun, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        R = (np.asarray(radiance, float) - path) / denom
    return np.clip(R, 0, 1) if strict else R


def correct_scene(scene, panel_radiance, panel_reflectance, panel_normal, panel_skv: float,
                  lat: float, lon: float, time=None, skyview=None, roughness: float = 0.3,
                  clip=None, method: str = 'numpy', **kwds):
    """End-to-end illumination correction of a hylite ``HyScene`` (FENIX notebook workflow).

    Args:
        scene: hylite scene (from :func:`hyperprep.terrestrial.projection.build_scene`) holding
            radiance in ``scene.image``.
        panel_radiance: mean radiance of the sunlit panel at the image bands.
        panel_reflectance: panel reflectance at the image bands (array or scalar).
        panel_normal: outward unit normal of the panel (ENU).
        panel_skv: sky-view factor of the panel.
        lat, lon, time: location / time for the sun vector (time from header if None).
        skyview: per-pixel sky view (hylite orientation ``(x, y)``), an int to read band ``n``
            of the scene cloud, or None to approximate from the normals.
        method: ``'numpy'`` (closed form, :func:`estimate_sky_sun`) or ``'hylite'`` (``estimate_illu``).

    Returns:
        dict with ``reflectance`` (hylite ``HyImage``), ``sun``, ``sky``, ``ilf``, ``skyview``, ``sunvec``.
    """
    import hylite

    sunvec, az, el = scene_sun_vector(lat, lon, time, dict(scene.image.header) if time is None else None)
    normals = scene.normals
    ilf = illumination_factors(normals, sunvec, scene.view, roughness=roughness)
    if skyview is None:
        skv = skyview_from_normals(normals)
    elif isinstance(skyview, (int, np.integer)):
        skv = scene.push_to_image(int(skyview), method='average').data[..., 0]
    else:
        skv = np.asarray(skyview, float)
    pilf = panel_illumination_factor(panel_normal, sunvec)
    if method == 'hylite':
        from hylite.correct import Panel
        from hylite.correct.illumination import estimate_illu
        from hylite.reference.spectra import Target
        wav = scene.image.get_wavelengths()
        target = Target(wav, np.broadcast_to(np.asarray(panel_reflectance, float), wav.shape), name='panel')
        panel = Panel(target, np.asarray(panel_radiance)[None, :], wavelengths=wav)
        # NB: current hylite returns (sun, sky); the original notebook unpacked them the other way round
        sun, sky = estimate_illu(scene.image, panel=panel, pilf=pilf, pskv=panel_skv, ilf=ilf, skv=skv,
                                 clip=clip, **kwds)
    else:
        sky, sun = estimate_sky_sun(scene.image.data, ilf, skv, panel_radiance, panel_reflectance, pilf,
                                    panel_skv, clip=clip, **kwds)
    R = illumination_reflectance(scene.image.data, sun, sky, ilf, skv)
    refl = hylite.HyImage(R.astype(np.float32))
    refl.set_wavelengths(scene.image.get_wavelengths())
    return {'reflectance': refl, 'sun': sun, 'sky': sky, 'ilf': ilf, 'skyview': skv, 'sunvec': sunvec,
            'sun_azimuth': az, 'sun_elevation': el}
