"""
02 - Projection mapping of a pushbroom (drone / airborne) HySpex image onto a photogrammetric
point cloud, with boresight calibration (replaces 2_Projection_mapping.ipynb).

Changes from the notebook: UTM zone is detected automatically (was hard-coded 33S), the
IMU/image line check is a function, and the three copy-pasted false-colour cells are one call.
"""
import numpy as np
import hylite
from hylite import io as hio

from hyperprep.calibration import hyspex_scale_factor
from hyperprep.drone import read_hyspexnav
from hyperprep.terrestrial.projection import (build_pushbroom, build_scene, crop_track, false_colour_cloud,
                                              solve_boresight)

IMAGE = '/path/to/20211204_UIS-01_02_Mjolnir_S620_SN7051_raw_rad_keystone_smile.hdr'
PREVIEW = '/path/to/20211204_UIS-01_02_Mjolnir_S620_SN7051_raw_rad_keystone_smile.PNG'
CLOUD = '/path/to/Uis_MainPit_Denscloud_Day2.ply'
NAV = '/path/to/HyspexNav_Results/20211204_UIS-01_02_Mjolnir_S620_SN7051_raw.txt'
OUT = '/path/to/projections/'
IFOV = np.rad2deg(0.54e-3)    # Mjolnir S-620 IFOV: 0.54 mrad per pixel
CROP = (500, 1955)            # scan lines to keep
SOLVE_BORESIGHT = True
BORESIGHT = None              # or a known value, e.g. (0.256, 0.243, -0.020)

if __name__ == '__main__':
    preview = hio.load(PREVIEW)
    preview.rot90()
    preview.flip('x')

    image = hio.load(IMAGE)
    image.data = image.data / hyspex_scale_factor(image)          # decompress HySpex RAD integers
    image.data[(image.data == 0).all(axis=-1), :] = np.nan        # under-exposed pixels

    cloud = hio.load(CLOUD)
    track = read_hyspexnav(NAV)                                   # lon/lat -> UTM (zone auto-detected)
    print("Navigation in %s (EPSG:%d), %d lines" % (track.meta['utm_zone'], track.crs, len(track)))

    pb = build_pushbroom(track, image, IFOV)                      # drops one trailing line if needed
    crop_track(pb, *CROP, image=image, preview=preview)

    if SOLVE_BORESIGHT:
        pb, bs = solve_boresight(pb, cloud, preview, initial=(0, 0, 0), coarse_eps=0.1)
        print("Boresight (roll, pitch, yaw):", bs)
    elif BORESIGHT is not None:
        pb = pb.apply_boresight(*BORESIGHT)

    scene = build_scene('UisMain_Day2', OUT, image, cloud, pb, occ_tol=2.0, chunk=200, step=50)
    scene.preview = preview
    scene.swircloud = false_colour_cloud(scene, hylite.SWIR)
    scene.rgbcloud = false_colour_cloud(scene, (0, 1, 2), image=preview)
    scene.save()
    hio.save(OUT, scene)
