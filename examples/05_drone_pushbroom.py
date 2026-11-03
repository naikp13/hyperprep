"""
05 - Drone pushbroom workflow, end to end, on a synthetic flight line (runs without any data).

Raw DN -> radiance -> bad pixel repair / destriping -> reflectance (panel ELC) ->
cross-track BRDF normalisation -> direct georeferencing on a DEM -> map-geometry mosaic.
Replace the synthetic arrays with your sensor's files (see configs/drone_hyspex.yaml).
"""
import numpy as np

from hyperprep import HSImage, save
from hyperprep.calibration import Panel, dn_to_radiance, empirical_line
from hyperprep.drone import (DEM, NavTrack, cross_track_correction, orthorectify, pixel_look_angles)
from hyperprep.noise import destripe, repair_detector_pixels
from hyperprep.ortho import mosaic
from hyperprep.spectral import bad_band_mask, remove_bad_bands, savgol

rng = np.random.default_rng(42)
lines, pixels = 300, 120
wav = np.linspace(400, 2500, 150)
ifov = 0.1  # degrees

# --- synthetic scene: two materials, a calibration panel, detector stripes, a dead pixel, BRDF
blocks = np.kron(rng.integers(0, 2, (lines // 10, pixels // 10)), np.ones((10, 10)))  # random 10x10 patches
veg = 0.05 + 0.4 * (wav > 720)                                                           # vegetation-like
soil = 0.2 + 0.1 * (wav - 400) / 2100                                                    # soil-like
refl = np.where(blocks[..., None] == 1, veg, soil)
refl[10:20, 50:70] = 0.9                                     # white panel
view = (np.arange(pixels) - pixels / 2) * ifov
brdf = 1 + 0.004 * view + 0.0004 * view ** 2
gain = 0.05 * (1 + 0.03 * rng.normal(size=(pixels, wav.size)))   # per-detector gain -> stripes
dn = refl * brdf[None, :, None] * 1000 / gain[None]
dn[:, 33, 70] = 0                                            # dead detector element
raw = HSImage(dn.astype(np.float32), wav)

# --- navigation: heading north at 80 m over a gently sloping DEM
t = np.arange(lines) * 0.02
track = NavTrack(t, 350000 + 0 * t, 6000000 + t * 5.0, np.full(lines, 180.0), rng.normal(0, 0.3, lines),
                 rng.normal(0, 0.3, lines), np.full(lines, 0.0) + rng.normal(0, 0.2, lines), crs=32635)
xx = np.arange(400)
dem = DEM(100 + 0.05 * np.tile(xx, (400, 1)), (349800.0, 1.0, 0.0, 6000200.0, 0.0, -1.0))

if __name__ == '__main__':
    rad = dn_to_radiance(raw, gain=gain)                                   # calibrated radiance
    rad = repair_detector_pixels(rad)                                      # dead / hot detector elements
    rad = destripe(rad, method='ratio', window=15)                         # residual non-uniformity
    panel = Panel.from_region('white', rad, (10, 20, 52, 68), 0.9)
    R = empirical_line(rad, [panel])                                       # reflectance
    R = cross_track_correction(R, view, degree=2)                          # BRDF / vignetting
    R = remove_bad_bands(R)
    R = savgol(R, window=7)
    ortho, glt, geo = orthorectify(R, track, pixel_look_angles(pixels, ifov_deg=ifov), res=0.5,
                                   ground=dem, return_glt=True)
    print(ortho, 'valid %.0f %%' % (100 * np.isfinite(ortho.data[..., 0]).mean()))
    truth = refl[..., bad_band_mask(wav)]
    print('median absolute reflectance error: %.4f' % np.nanmedian(np.abs(R.data - truth)))
    mosaic([ortho])                                                        # add more flight lines here
    save('drone_line_refl_ortho.hdr', ortho)
