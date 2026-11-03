"""
04 - Build a corrected multi-scene hypercloud (replaces 1b_Build_and_correct_hypercloud_ambient.ipynb).

Each image must carry its camera pose in the header (hylite) and, for atmospheric correction,
calibration panel spectra and sun azimuth/elevation.
"""
import hylite

from hyperprep.terrestrial.hypercloud import HypercloudSettings, build_hypercloud

CLOUD = '/path/to/ca_fused_cloud_30cm.ply'
IMAGES = ['/path/to/input FENIX/2016_aligned', '/path/to/input FENIX/2019']   # folders or files
OUT = '/path/to/hyperclouds/FENIX'

settings = HypercloudSettings(
    occ_tol=0, s=1,
    topo_correct=['ambient', 'ambient', 'ambient', 'cfac', 'cfac', 'ambient', 'ambient', 'ambient', 'ambient', 'ambient'],
    low_thresh=0, high_thresh=100,
    atmos_correct=True,
    colour_correct=True, colour_method='norm', colour_uniform=False, reference_index=-1,
    blend_mode='average',
    vis={'RGB': hylite.RGB, 'SWIR': hylite.SWIR},
)

if __name__ == '__main__':
    hypercloud, scenes, report = build_hypercloud(IMAGES, CLOUD, OUT, settings, name='CA_hypercloud_30cm_amb')
    for r in report:
        print(r)
    render = hypercloud.render(scenes[-1].camera, bands=[(0, -1)])
    hylite.io.save(OUT + '/CA_hypercloud_render.hdr', render)
