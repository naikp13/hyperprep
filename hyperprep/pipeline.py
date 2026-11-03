"""
Config-driven pre-processing pipelines.

A pipeline is an ordered list of named steps, each a function
``step(image, **params) -> image``. Pipelines can be built in Python or loaded
from YAML/JSON::

    input: data/EMIT_L1B_RAD_001.nc
    output: out/emit_refl.hdr
    steps:
      - read: {sensor: emit, ortho: true}
      - toa_reflectance: {}
      - remove_bad_bands: {}
      - savgol: {window: 7}

Run with ``hyperprep run config.yaml`` or :func:`run_config`. Every step is
logged with timing and written into ``image.meta['history']``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import calibration, masks, noise, spectral, topo
from .image import HSImage

STEPS: Dict[str, Callable[..., HSImage]] = {}


def step(name: str):
    """Decorator registering a pipeline step."""
    def deco(fn):
        STEPS[name] = fn
        return fn
    return deco


# --------------------------------------------------------------- I/O steps
@step('read')
def _read(image, path: Optional[str] = None, sensor: Optional[str] = None, **kw):
    from . import io
    if path is None:
        raise ValueError("read step needs a path (or set 'input' in the config)")
    if sensor:
        from .satellite.readers import read_product
        return read_product(path, sensor=sensor, **kw)
    return io.load(path, **kw)


@step('write')
def _write(image, path: str, **kw):
    from . import io
    io.save(path, image, **kw)
    return image


@step('sensor_correction')
def _sensor(image, path: str, sensor: str = 'fenix', white_panel: str = 'R90', **kw):
    from .calibration.sensors import correct_sensor_folder
    return correct_sensor_folder(path, sensor, white_panel, as_hsimage=True, **kw)


# ----------------------------------------------------------- generic steps
@step('mask_nodata')
def _mask_nodata(image, nodata_value: float = 0.0):
    return masks.mask_nodata(image, nodata_value)


@step('hyspex_scale')
def _hyspex(image, factor: Optional[float] = None):
    return calibration.apply_scale_factor(image, factor)


@step('dn_to_radiance')
def _dn(image, gain=1.0, offset=0.0):
    return calibration.dn_to_radiance(image, np.asarray(gain), np.asarray(offset))


@step('white_reference')
def _white(image, white: str, dark: Optional[str] = None, panel_reflectance=1.0):
    from . import io
    return calibration.white_reference_reflectance(image, io.load(white), io.load(dark) if dark else None,
                                                   panel_reflectance)


@step('elc')
def _elc(image, panels: List[dict]):
    """``panels``: list of ``{name, region: [r0, r1, c0, c1] | radiance_file, reflectance}``."""
    objs = []
    for p in panels:
        refl = p.get('reflectance', 1.0)
        if isinstance(refl, str):
            from .io import read_spectrum_txt
            refl = read_spectrum_txt(refl)
        if 'region' in p:
            objs.append(calibration.Panel.from_region(p.get('name', 'panel'), image, tuple(p['region']), refl))
        else:
            objs.append(calibration.Panel.from_file(p.get('name', 'panel'), p['radiance_file'], image.wavelengths, refl))
    return calibration.empirical_line(image, objs)


@step('flat_field')
def _ff(image, region=None):
    return calibration.flat_field(image, region)


@step('iarr')
def _iarr(image):
    return calibration.iarr(image)


@step('remove_bad_bands')
def _rbb(image, windows=None, use_bbl: bool = True, drop: bool = True, preset: str = 'default'):
    if windows is None:
        windows = spectral.HYPERCLOUD_BAD_WINDOWS if preset == 'hypercloud' else spectral.DEFAULT_BAD_WINDOWS
    return spectral.remove_bad_bands(image, [tuple(w) for w in windows], use_bbl, drop)


@step('resample')
def _resample(image, wavelengths=None, fwhm=None, like: Optional[str] = None):
    if like is not None:
        from . import io
        ref = io.load(like)
        wavelengths, fwhm = ref.wavelengths, ref.fwhm
    return spectral.resample_image(image, wavelengths, fwhm)


@step('bin_bands')
def _bin(image, factor: int = 2):
    return spectral.bin_bands(image, factor)


@step('savgol')
def _savgol(image, window: int = 7, polyorder: int = 2, deriv: int = 0):
    return spectral.savgol(image, window, polyorder, deriv)


@step('spatial_smooth')
def _ss(image, size: int = 3):
    return spectral.spatial_smooth(image, size)


@step('hull_correct')
def _hull(image, band_range=None, method: str = 'div'):
    return spectral.hull_correct(image, tuple(band_range) if band_range else None, method)


@step('normalise')
def _norm(image, method: str = 'brightness'):
    return spectral.normalise(image, method)


@step('clip_range')
def _clip(image, vmin: float = 0.0, vmax: float = 1.5, action: str = 'nan_pixel'):
    return spectral.clip_range(image, vmin, vmax, action)


@step('repair_bad_pixels')
def _bpr(image, z_thresh: float = 6.0):
    return noise.repair_detector_pixels(image, z_thresh=z_thresh)


@step('remove_spikes')
def _spikes(image, z_thresh: float = 5.0, size: int = 3):
    return noise.remove_spikes(image, z_thresh, size)


@step('destripe')
def _destripe(image, method: str = 'moment', axis: int = 0, window: Optional[int] = None):
    return noise.destripe(image, method, axis, window)


@step('mnf_denoise')
def _mnf(image, n_components: int = 10):
    return noise.mnf_denoise(image, n_components)


@step('brightness_filter')
def _bf(image, low: float = 0.0, high: float = 100.0):
    return noise.percentile_brightness_filter(image, low, high)


@step('mask_clouds')
def _clouds(image, buffer: int = 2, **kw):
    m = masks.cloud_mask(image, **kw)
    return masks.apply_mask(image, masks.dilate(m, buffer) if buffer else m)


@step('mask_water')
def _water(image, thresh: float = 0.0):
    return masks.apply_mask(image, masks.water_mask(image, thresh))


@step('mask_shadows')
def _shadows(image, **kw):
    return masks.apply_mask(image, masks.shadow_mask(image, **kw))


@step('topographic_correction')
def _topo(image, dem: str, method: str = 'c', sun_azimuth: Optional[float] = None,
          sun_zenith: Optional[float] = None):
    """DEM-based topographic correction for map-geometry images (DEM on the same grid)."""
    from . import io
    from .solar import sph2cart
    d = io.load(dem)
    if d.shape[:2] != image.shape[:2]:
        raise ValueError("DEM grid %s does not match image grid %s" % (d.shape[:2], image.shape[:2]))
    az = float(sun_azimuth if sun_azimuth is not None else image.meta['sun_azimuth'])
    sz = float(sun_zenith if sun_zenith is not None else image.meta['sun_zenith'])
    n = topo.normals_from_dem(d.data[..., 0], d.transform[1], d.transform[5])
    sunvec = -sph2cart(az, 90 - sz)
    slope, _ = topo.slope_aspect(n)
    return topo.topographic_correction(image, topo.incidence(n, sunvec), method, np.cos(np.deg2rad(sz)),
                                       slope=slope, sun_zenith=sz, skyview=topo.skyview_from_normals(n))


# ------------------------------------------------------------- drone steps
@step('orthorectify')
def _ortho(image, nav: str, res: float, ifov_deg: Optional[float] = None, fov_deg: Optional[float] = None,
           ground: float = 0.0, dem: Optional[str] = None, boresight=(0, 0, 0), lever_arm=None,
           nav_format: str = 'hyspexnav', flip: bool = False, max_dist: Optional[float] = None):
    from .drone import DEM, orthorectify, pixel_look_angles, read_hyspexnav, read_nav_csv
    tr = read_hyspexnav(nav) if nav_format == 'hyspexnav' else read_nav_csv(nav)
    extra = tr.check_lines(image.rows)
    if extra:
        image = image.copy(image.data[:-extra])
    if lever_arm is not None:
        tr = tr.apply_lever_arm(lever_arm)
    g = ground
    if dem is not None:
        from . import io
        g = DEM.from_image(io.load(dem))
    ang = pixel_look_angles(image.cols, ifov_deg, fov_deg, flip=flip)
    return orthorectify(image, tr, ang, res, g, tuple(boresight), max_dist)


@step('cross_track_correction')
def _xtrack(image, degree: int = 2, ifov_deg: Optional[float] = None):
    from .drone import cross_track_correction
    va = None if ifov_deg is None else (np.arange(image.cols) - (image.cols - 1) / 2) * ifov_deg
    return cross_track_correction(image, va, degree)


@step('irradiance_normalisation')
def _irr(image, irradiance_file: str):
    from .drone import irradiance_normalisation
    from .io import read_spectrum_txt
    w, e = read_spectrum_txt(irradiance_file)
    return irradiance_normalisation(image, e, w)


# --------------------------------------------------------- satellite steps
@step('toa_reflectance')
def _toa(image, sun_zenith=None, time=None, esun_path: Optional[str] = None, radiance_scale: float = 1.0):
    from .satellite import toa_reflectance
    return toa_reflectance(image, sun_zenith, time, esun_path=esun_path, radiance_scale=radiance_scale)


@step('dos')
def _dos(image, percentile: float = 0.1):
    from .satellite import dark_object_subtraction
    return dark_object_subtraction(image, percentile)


@step('merge_overlap')
def _merge(image, split: Optional[float] = None, method: str = 'cut'):
    from .satellite import merge_overlap
    return merge_overlap(image, split=split, method=method)


@step('smile_correction')
def _smile(image, feature=(750.0, 775.0), degree: int = 2):
    from .satellite import estimate_smile, smile_correction
    shift = estimate_smile(image, tuple(feature), degree=degree)
    return smile_correction(image, image.wavelengths[None, :] + shift[:, None])


@step('subset_bbox')
def _bbox(image, bbox):
    from .satellite import subset_bbox
    return subset_bbox(image, tuple(bbox))


@step('reflectance_scale_check')
def _rsc(image):
    from .satellite import reflectance_scale_check
    return reflectance_scale_check(image)


# ------------------------------------------------------------------ runner
class Pipeline:
    """An ordered list of ``(step_name, params)``."""

    def __init__(self, steps: Optional[List[Tuple[str, dict]]] = None, verbose: bool = True):
        self.steps: List[Tuple[str, dict]] = []
        self.verbose = verbose
        for s in steps or []:
            self.add(*s)

    def add(self, name: str, params: Optional[dict] = None, **kw) -> "Pipeline":
        if name not in STEPS:
            raise KeyError("Unknown step %r. Available: %s" % (name, ', '.join(sorted(STEPS))))
        self.steps.append((name, {**(params or {}), **kw}))
        return self

    def run(self, image: Optional[HSImage] = None) -> HSImage:
        for i, (name, params) in enumerate(self.steps):
            t0 = time.time()
            image = STEPS[name](image, **params)
            dt = time.time() - t0
            if image is not None:
                image.meta.setdefault('history', [])
                if isinstance(image.meta['history'], str):
                    image.meta['history'] = [image.meta['history']]
                image.meta['history'].append(name)
            if self.verbose:
                shp = 'x'.join(map(str, image.shape)) if image is not None else '-'
                print("[%2d/%d] %-24s %6.1fs  -> %s" % (i + 1, len(self.steps), name, dt, shp))
        return image

    @classmethod
    def from_config(cls, cfg: dict, verbose: bool = True) -> "Pipeline":
        steps = []
        for entry in cfg.get('steps', []):
            if isinstance(entry, str):
                steps.append((entry, {}))
            else:
                (name, params), = entry.items()
                steps.append((name, params or {}))
        if cfg.get('input') and (not steps or steps[0][0] not in ('read', 'sensor_correction')):
            steps.insert(0, ('read', {'path': cfg['input']}))
        elif cfg.get('input') and steps[0][0] == 'read' and 'path' not in steps[0][1]:
            steps[0][1]['path'] = cfg['input']
        if cfg.get('output') and (not steps or steps[-1][0] != 'write'):
            steps.append(('write', {'path': cfg['output']}))
        return cls(steps, verbose)


def load_config(path: str) -> dict:
    with open(path) as f:
        if path.lower().endswith(('.yaml', '.yml')):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def run_config(path_or_cfg, verbose: bool = True, **overrides) -> HSImage:
    """Load a YAML/JSON config (or dict) and run it. ``overrides`` replace top-level keys (e.g. input/output)."""
    cfg = load_config(path_or_cfg) if isinstance(path_or_cfg, str) else dict(path_or_cfg)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return Pipeline.from_config(cfg, verbose).run()


def list_steps() -> Dict[str, str]:
    """Registered steps and the first line of their docstrings."""
    return {k: ((v.__doc__ or '').strip().splitlines() or [''])[0] for k, v in sorted(STEPS.items())}
