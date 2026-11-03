"""
01 - Sensor corrections: raw scanner capture -> reflectance  (replaces 1_radiance_to_reflectance_Fenixsingle.ipynb)

Works for FENIX, FX10/17/50, OWL, Rikola and Telops captures via hylite. Set SENSOR and
WHITE_PANEL to what was used in the field / lab.
"""
from hyperprep.calibration.sensors import batch_correct, correct_sensor_folder

CAPTURE = '/path/to/MWIR_Li_minerals_1_0m00_1m00_2021-03-31_10-43-20'   # sensor capture folder
OUTDIR = '/path/to/reflectance/'
SENSOR = 'fx50'          # 'fenix' | 'fx10' | 'fx17' | 'fx50' | 'owl' | 'rikola' | 'telops'
WHITE_PANEL = 'R90'      # hylite target name, e.g. 'R90' (Spectralon 90 %) or 'R55_SiSu' (grey panel)

if __name__ == '__main__':
    image = correct_sensor_folder(
        CAPTURE, sensor=SENSOR, white_panel=WHITE_PANEL, outdir=OUTDIR,
        bpr=True,      # bad pixel replacement
        flip=False,    # True if the camera was mounted backwards
        # shift=True,  # FENIX only: VNIR/SWIR spatial shift (near-field / core scans)
    )
    fig, ax = image.quick_plot((80, 21, 51))
    fig.savefig(OUTDIR + 'preview.png')

    # many folders at once (failures are reported and skipped):
    # batch_correct('/path/to/session/', sensor='fenix', outdir=OUTDIR, flip=True, shift=True)
