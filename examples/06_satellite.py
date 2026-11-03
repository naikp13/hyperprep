"""
06 - Satellite workflow (EnMAP / PRISMA / EMIT / DESIS).

L1 (radiance) products: read -> merge VNIR/SWIR overlap -> smile check -> destripe ->
TOA reflectance -> dark object subtraction (or ELC with field spectra) -> masks -> bad bands -> smoothing.
L2 (surface reflectance) products: read -> rescale check -> masks -> bad bands -> smoothing.
"""
import numpy as np

from hyperprep import save
from hyperprep.masks import apply_mask, cloud_mask, dilate, water_mask
from hyperprep.noise import destripe
from hyperprep.satellite import (dark_object_subtraction, estimate_smile, merge_overlap, read_product,
                                 reflectance_scale_check, smile_correction, toa_reflectance)
from hyperprep.spectral import remove_bad_bands, savgol

PRODUCT = '/path/to/EMIT_L1B_RAD_001_20230801T100000_2321307_005.nc'   # or EnMAP folder / PRISMA .he5
SOLAR_SPECTRUM = None   # e.g. '/path/to/tsis1_hsrs_1nm.txt' (2 columns: nm, W m-2 nm-1) - recommended
OUT = 'satellite_refl.hdr'

if __name__ == '__main__':
    kw = {'ortho': True} if 'EMIT' in PRODUCT.upper() else {}
    img = read_product(PRODUCT, **kw)
    print(img, img.meta.get('units'))

    if img.meta.get('units') == 'radiance':
        img = merge_overlap(img)                                      # EnMAP/PRISMA VNIR-SWIR overlap
        shift = estimate_smile(img)                                   # O2-A band across-track shift (nm)
        print("smile range: %.2f nm" % np.ptp(shift))
        if np.ptp(shift) > 0.5 and img.transform is None:             # only meaningful in raw geometry
            img = smile_correction(img, img.wavelengths[None, :] + shift[:, None])
            img = destripe(img, method='ratio', window=31)
        # EMIT radiance is uW cm-2 sr-1 nm-1 -> W m-2 sr-1 nm-1 is x0.01; check your sensor's units
        scale = 0.01 if img.meta.get('sensor') == 'EMIT' else 1.0
        sz = img.meta.get('sun_zenith', 35.0)                         # supply from product metadata if missing
        img = toa_reflectance(img, sun_zenith=sz, esun_path=SOLAR_SPECTRUM, radiance_scale=scale)
        img = dark_object_subtraction(img, percentile=0.1)
    else:
        img = reflectance_scale_check(img)

    img = apply_mask(img, dilate(cloud_mask(img), 3))
    img = apply_mask(img, water_mask(img))
    img = remove_bad_bands(img)
    img = savgol(img, window=5)
    save(OUT, img)
