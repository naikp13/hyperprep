"""
03 - Illumination correction of a terrestrial FENIX scene (replaces 3_Illumination_corrections_FENIX.ipynb).

Radiance -> reflectance using a sunlit calibration panel, surface normals from the point cloud and
an Oren-Nayar + sky-view illumination model (Thiele et al., 2021). Then push reflectance to the
cloud, smooth, and compute hull-corrected previews.
"""
import numpy as np
from hylite import io as hio

from hyperprep import HSImage
from hyperprep.calibration import Panel
from hyperprep.spectral import hull_correct
from hyperprep.terrestrial.illumination import correct_scene

SCENE = '/path/to/outputs/myscene.hdr'
WHITE_PANEL_RAD = '/path/to/Panels/White_Panel_RadAverage.txt'
OUT = '/path/to/Fenix_Corrections/'
LAT, LON = -21.2, 14.8
TIME = '2021-12-03T08:47:29+00:00'      # UTC; or None to read from the image header
PANEL_NORMAL = [0.254727, -0.965015, -0.0621294]   # outward normal of the sunlit (white) panel
PANEL_SKYVIEW = 0.53
PANEL_REFLECTANCE = 0.9                  # or a (wavelengths, reflectance) tuple for a calibrated panel
CLIP = (0, 1250, 500, 900)               # region used to estimate sun/sky spectra (hylite x0, x1, y0, y1)

if __name__ == '__main__':
    S = hio.load(SCENE)
    wav = S.image.get_wavelengths()
    white = Panel.from_file('white', WHITE_PANEL_RAD, wav, PANEL_REFLECTANCE)

    res = correct_scene(S, white.radiance, white.reflectance, PANEL_NORMAL, PANEL_SKYVIEW, LAT, LON, TIME,
                        skyview=0,          # band 0 of the scene cloud holds sky-view factors; None = from normals
                        roughness=0.3, clip=CLIP)
    print("Sun azimuth %.1f, elevation %.1f" % (res['sun_azimuth'], res['sun_elevation']))
    S.refl = res['reflectance']
    hio.save(OUT + 'FENIX_refl.hdr', S.refl)

    # project reflectance subsets to hyperclouds, add saturation-boosted RGB and smooth
    from hylite.filter import boost_saturation
    for rng, rgb in (((1400., 1800.), (1593., 1538., 1766.)), ((2000., 2450.), (2200., 2340., 2380.))):
        h = S.push_to_cloud(rng, image=S.refl)
        h.rgb = (boost_saturation(h, rgb).data * 255).astype(np.uint8)
        h.smooth_savgol(window=7)
        hio.save(OUT + 'h%d_%d.hdr' % rng, h)

    # hull-corrected preview of the 1500-1750 nm range (pure numpy)
    hc = hull_correct(HSImage.from_hylite(S.refl), (1500., 1750.))
    hio.save(OUT + 'hull_1500_1750.hdr', hc.to_hylite())
