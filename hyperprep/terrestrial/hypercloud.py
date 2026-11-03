"""
Build radiometrically corrected hyperclouds from multiple terrestrial scenes.

Refactored from ``1b_Build_and_correct_hypercloud_ambient.ipynb`` and updated to
the current hylite API (``HyScene(name, root).construct``, ``get_blend_weights``
/ ``blend_scenes``; the notebook used the removed ``HyScene(image, cloud, cam)``,
``scene.correct`` and ``HyScene.build_hypercloud``). Fixes from the notebook:

* the bad-band list was deleted from the wrong image (``image`` rather than the scene);
* failure messages printed the wrong path variable;
* the reference scene for colour matching was also matched to itself when given as -1.

Steps: collect images -> common wavelengths (bad bands removed) -> per-image
projection -> atmospheric (panel ELC from header) and topographic correction ->
brightness filtering -> colour matching -> blending -> export.

Requires ``hylite``.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..calibration.elc import ELC, Panel
from ..image import HSImage
from ..noise import percentile_brightness_filter
from ..solar import sph2cart
from ..spectral import HYPERCLOUD_BAD_WINDOWS, bad_band_mask, clip_range
from ..topo import incidence, skyview_from_normals, topographic_correction
from .projection import project_frame_scene

TOPO_ALIASES = {'cfac': 'c', 'c': 'c', 'ambient': 'ambient', 'cosine': 'cosine', 'minnaert': 'minnaert', None: None}


@dataclass
class HypercloudSettings:
    """Settings mirroring the notebook's configuration cell."""

    occ_tol: float = 0.0                       # occlusion tolerance (cloud units); 0 disables occlusion
    s: int = 1                                 # point size (dilation) when projecting
    topo_correct: Union[str, Sequence[Optional[str]], None] = 'ambient'  # per scene or one for all
    low_thresh: float = 0.0                    # percentile; darker pixels removed after topo correction
    high_thresh: float = 100.0                 # percentile; brighter pixels removed
    atmos_correct: bool = True                 # ELC from panels stored in the image header
    valid_range: tuple = (0.0, 2.5)            # reflectance outside this range -> NaN pixel
    colour_correct: bool = True
    colour_method: str = 'norm'                # 'norm' or 'hist'
    colour_uniform: bool = False
    reference_index: int = -1
    blend_mode: str = 'average'                # 'average', 'gsd', 'distance', 'obliquity'
    export_bands: tuple = (0, -1)
    bad_windows: Sequence = field(default_factory=lambda: list(HYPERCLOUD_BAD_WINDOWS))
    vis: Optional[Dict[str, Sequence[float]]] = None  # name -> 3 bands for colourised exports


def collect_images(paths: Union[str, Sequence[str]], pattern: str = '*.hdr') -> List[str]:
    """Expand directories (non-recursive glob ``pattern``) and files into a sorted list of image paths."""
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, pattern))) if os.path.isdir(p) else [p]
    return out


def common_wavelengths(image_path: str, bad_windows=HYPERCLOUD_BAD_WINDOWS) -> np.ndarray:
    """Wavelengths of ``image_path`` with bad (water) windows removed - the target band set for all scenes."""
    from hylite import io as hio

    wav = np.asarray(hio.load(image_path).get_wavelengths(), float)
    return wav[bad_band_mask(wav, bad_windows)]


def header_panels(image) -> List[Panel]:
    """Calibration panels stored in a hylite image header as hyperprep :class:`Panel` objects."""
    panels = []
    for name in image.header.get_panel_names():
        p = image.header.get_panel(name)
        if p is None:
            continue
        rad = np.asarray(p.get_mean_radiance(), float).ravel()
        refl = np.asarray(p.get_reflectance(), float).ravel()
        panels.append(Panel(name, rad, refl))
    return panels


def header_sun_vector(image) -> Optional[np.ndarray]:
    """Illumination vector from ``sun azimuth`` / ``sun elevation`` header fields (None if missing)."""
    h = image.header
    if 'sun azimuth' not in h or 'sun elevation' not in h:
        return None
    return -sph2cart(float(h['sun azimuth']), float(h['sun elevation']))


def correct_scene(scene, settings: HypercloudSettings, topo_method: Optional[str] = None, vb: bool = True) -> dict:
    """Atmospheric + topographic correction of one hylite scene, in place.

    Returns a dict ``{'success': bool, 'atmos': bool, 'topo': str|None, 'reason': str}``.
    """
    img = scene.image
    wav = np.asarray(img.get_wavelengths(), float)
    hs = HSImage(np.asarray(img.data, np.float32), wav)
    status = {'success': True, 'atmos': False, 'topo': None, 'reason': ''}

    if settings.atmos_correct:
        panels = header_panels(img)
        if not panels:
            status.update(success=False, reason='missing calibration panel')
        else:
            hs = ELC(panels).apply(hs)
            status['atmos'] = True

    method = TOPO_ALIASES.get(topo_method, topo_method)
    if method is not None:
        sunvec = header_sun_vector(img)
        if sunvec is None:
            status.update(success=False, reason=(status['reason'] + '; ' if status['reason'] else '') + 'missing sun vector')
        else:
            cos_i = incidence(scene.normals, sunvec)
            hs = topographic_correction(hs, cos_i, method=method, cos_ref=1.0, skyview=skyview_from_normals(scene.normals))
            status['topo'] = method

    if settings.low_thresh > 0 or settings.high_thresh < 100:
        hs = percentile_brightness_filter(hs, settings.low_thresh, settings.high_thresh)
    if status['atmos'] and settings.valid_range is not None:
        hs = clip_range(hs, *settings.valid_range, action='nan_pixel')

    img.data = hs.data
    if 'bbl' in img.header:  # notebook deleted this from the wrong image
        del img.header['bbl']
    if vb:
        print("  atmos=%s topo=%s %s" % (status['atmos'], status['topo'], status['reason']))
    return status


def colour_match(scenes, reference_index: int = -1, method: str = 'norm', uniform: bool = False, vb: bool = True):
    """Match brightness/contrast of every scene to a reference scene (``HyScene.match_colour_to``).

    Returns backups of the original images so the step can be undone.
    """
    ref_i = reference_index % len(scenes)
    backups = {}
    for i, s in enumerate(scenes):
        if i == ref_i:
            continue
        backups[i] = s.image.copy()
        try:
            s.match_colour_to(scenes[ref_i], method=method, uniform=uniform)
        except Exception as e:  # no overlap etc.
            if vb:
                print("Colour matching of scene %d failed: %s" % (i, e))
    return backups


def blend(scenes, mode: str = 'average', bands=(0, -1), trim: bool = False, vb: bool = True):
    """Fuse scenes that share a cloud into one hypercloud (``average`` | ``gsd`` | ``distance`` | ``obliquity``)."""
    from hylite.project import blend_scenes, get_blend_weights

    method = 'equal' if mode == 'average' else mode
    weights = get_blend_weights(scenes, method, ascloud=False)
    return blend_scenes(scenes, weights, bands=bands, trim=trim, ooc=False, vb=vb)


def export_hypercloud(hypercloud, outdir: str, name: str = 'hypercloud', composites: Optional[Dict[str, Sequence]] = None,
                      compress: bool = True) -> Dict[str, str]:
    """Write the hypercloud and colourised false-colour clouds (PLY) to ``outdir``."""
    import hylite
    from hylite import io as hio

    os.makedirs(outdir, exist_ok=True)
    composites = composites or {'RGB': hylite.RGB, 'SWIR': hylite.SWIR}
    written = {}
    wav = np.asarray(hypercloud.get_wavelengths(), float)
    for label, bands in composites.items():
        if min(bands) < wav.min() - 10 or max(bands) > wav.max() + 10:
            continue  # composite outside sensor range
        c = hypercloud.copy()
        c.colourise(bands)
        c.data = None
        p = os.path.join(outdir, '%s_%s.ply' % (name, label))
        hio.saveCloudPLY(p, c)
        written[label] = p
    if compress:
        hypercloud.compress()
    p = os.path.join(outdir, '%s.ply' % name)
    hio.saveCloudPLY(p, hypercloud)
    written['hypercloud'] = p
    return written


def build_hypercloud(image_paths: Union[str, Sequence[str]], cloud_path: str, outdir: str,
                     settings: Optional[HypercloudSettings] = None, name: str = 'hypercloud', vb: bool = True):
    """Full hypercloud workflow. Returns ``(hypercloud, scenes, report)``."""
    from hylite import io as hio

    settings = settings or HypercloudSettings()
    paths = collect_images(image_paths)
    if not paths:
        raise FileNotFoundError("No images found in %s" % image_paths)
    if vb:
        print("Found %d images" % len(paths))
    wav = common_wavelengths(paths[-1], settings.bad_windows)
    cloud = hio.load(cloud_path)
    topo = settings.topo_correct
    topo = list(topo) if isinstance(topo, (list, tuple)) else [topo] * len(paths)
    if len(topo) < len(paths):
        raise ValueError("topo_correct has %d entries for %d images" % (len(topo), len(paths)))

    scenes, report = [], []
    for i, p in enumerate(paths):
        if vb:
            print("Scene %d: %s" % (i, os.path.basename(p)))
        sc = project_frame_scene(p, cloud, 'scene_%02d' % i, os.path.join(outdir, 'scenes'),
                                 occ_tol=settings.occ_tol, s=settings.s, resample_to=wav, vb=vb)
        if sc is None:
            report.append({'path': p, 'success': False, 'reason': 'no camera pose'})
            continue
        st = correct_scene(sc, settings, topo[i], vb=vb)
        st['path'] = p
        report.append(st)
        scenes.append(sc)

    if settings.colour_correct and len(scenes) > 1:
        colour_match(scenes, settings.reference_index, settings.colour_method, settings.colour_uniform, vb)
    hc = blend(scenes, settings.blend_mode, settings.export_bands, vb=vb)
    export_hypercloud(hc, outdir, name, settings.vis)
    return hc, scenes, report
