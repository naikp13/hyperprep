"""
Projection mapping: link hyperspectral pixels to a 3-D point cloud (hylite ``HyScene``).

Refactored from ``2_Projection_mapping.ipynb``. Works for

* frame / panoramic cameras (terrestrial scanners on a rotation stage, e.g. FENIX
  on a tripod), with the camera pose stored in the image header; and
* pushbroom cameras with a GNSS/IMU track (drone or airborne, e.g. HySpex
  Mjolnir with HySpexNav/PosPac output), including boresight optimisation.

Requires ``hylite``.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

from ..drone.navigation import NavTrack


def build_pushbroom(track: NavTrack, image, xfov_deg: float, lfov_deg: Optional[float] = None,
                    drop_extra_lines: bool = True):
    """Create a ``hylite.project.Pushbroom`` from a navigation track and a hylite image.

    Args:
        track: :class:`~hyperprep.drone.navigation.NavTrack` with one record per scan line
            (positions in a projected CRS, e.g. UTM metres).
        image: ``hylite.HyImage`` (x = across-track pixels, y = scan lines).
        xfov_deg: across-track IFOV of one pixel in degrees. For HySpex Mjolnir S-620 this is
            0.54 mrad -> ``np.rad2deg(0.54e-3)``.
        lfov_deg: along-track IFOV (defaults to ``xfov_deg``).
        drop_extra_lines: the image sometimes contains one more line than there are nav records;
            if so the trailing line is dropped (as in the notebook). Other mismatches raise.
    """
    from hylite.project.pushbroom import Pushbroom

    if drop_extra_lines and image.ydim() == len(track) + 1:
        image.data = image.data[:, :-1, :]
    if len(track) != image.ydim():
        raise ValueError("IMU data has %d frames, image has %d." % (len(track), image.ydim()))
    lfov_deg = xfov_deg if lfov_deg is None else lfov_deg
    return Pushbroom(track.xyz, track.rpy, xfov_deg, lfov_deg, dims=(image.xdim(), image.ydim()))


def crop_track(pb, start: int, end: int, image=None, preview=None):
    """Crop a pushbroom track (and matching image / preview) to scan lines ``start:end``."""
    pb.crop(start, end, image)
    if preview is not None:
        preview.data = preview.data[:, start:end, :]
    return pb


def solve_boresight(pb, cloud, rgb_image, initial=(0.0, 0.0, 0.0), coarse_eps: float = 0.1, **kwds):
    """Estimate boresight angles (roll, pitch, yaw offsets in degrees) by maximising the correlation
    between the projected image RGB and the point cloud RGB. Returns ``(track_opt, boresight)``.

    Use a short, high-quality subset of the swath (see :func:`crop_track`) and then apply the
    result to the full track with ``pb.apply_boresight(*boresight)``.
    """
    from hylite.project.pushbroom import optimize_boresight

    track_opt, bs, _ = optimize_boresight(pb, cloud, rgb_image, iv=np.asarray(initial, float),
                                          coarse_eps=coarse_eps, **kwds)
    return track_opt, np.asarray(bs)


def build_scene(name: str, root: str, image, cloud, camera, occ_tol: float = 2.0, s=1,
                vb: bool = True, **kwds):
    """Construct (and return) a ``hylite.HyScene`` linking ``image`` and ``cloud``.

    ``kwds`` are forwarded to ``HyScene.construct`` (e.g. ``chunk=200, step=50`` for pushbroom).
    """
    from hylite import HyScene

    os.makedirs(root, exist_ok=True)
    scene = HyScene(name, root)
    scene.construct(image, cloud, camera, occ_tol=occ_tol, s=s, vb=vb, **kwds)
    return scene


def false_colour_cloud(scene, bands, image=None, method: str = 'best', percentile: Optional[Tuple[float, float]] = None):
    """Push three bands to the cloud and store them as 8-bit RGB (replaces three duplicated notebook cells).

    Args:
        bands: three bands (indices or wavelengths), e.g. ``hylite.SWIR`` or ``(0, 1, 2)``.
        image: image to project (defaults to ``scene.image``; pass a preview PNG for true colour).
        percentile: ``(low, high)`` stretch; default normalises by the maximum like the notebook.
    """
    from hylite.project import push_to_cloud

    img = scene.image if image is None else image
    cld = push_to_cloud(scene.pmap, bands, method=method, image=img, cloud=scene.cloud)
    d = cld.data[:, :3].astype(float)
    if percentile is None:
        d = d / np.nanmax(d)
    else:
        lo, hi = np.nanpercentile(d, percentile)
        d = (d - lo) / (hi - lo)
    cld.rgb = np.clip(np.nan_to_num(d) * 255, 0, 255).astype(np.uint8)
    cld.data = None
    cld.normals = None
    return cld


def scene_geometry(scene):
    """Return per-pixel ``(xyz, normals, view, depth)`` arrays from a hylite scene in
    hyperprep's ``(rows, cols, ...)`` orientation."""
    t = lambda a: None if a is None else np.transpose(np.asarray(a), (1, 0) + tuple(range(2, np.ndim(a))))
    return t(scene.xyz), t(scene.normals), t(scene.view), t(scene.depth)


def project_frame_scene(image_path: str, cloud, name: str, root: str, occ_tol: float = 0.0, s: int = 1,
                        mask_zero: bool = True, resample_to: Optional[Sequence[float]] = None, vb: bool = False):
    """Load an image whose camera pose is stored in its header and project it onto ``cloud``.

    This is the per-image step of the hypercloud workflow (FENIX / panoramic scans).
    Returns the scene, or None if the image has no camera pose.
    """
    from hylite import io as hio

    image = hio.load(image_path)
    if resample_to is not None:
        image = image.resample(np.asarray(resample_to))
    if mask_zero:
        image.data = image.data.astype(np.float32)
        image.data[image.data == 0] = np.nan
    cam = image.header.get_camera()
    if cam is None:
        if vb:
            print("Skipping %s: image has no camera pose" % image_path)
        return None
    return build_scene(name, root, image, cloud, cam, occ_tol=occ_tol, s=s, vb=vb)
