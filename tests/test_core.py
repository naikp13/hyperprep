import numpy as np
import pytest

from hyperprep import HSImage, io, masks, noise, spectral
from hyperprep.calibration import (ELC, Panel, apply_scale_factor, dn_to_radiance, hyspex_scale_factor,
                                   white_reference_reflectance)
from hyperprep.geo import latlon_to_utm, utm_to_latlon
from hyperprep.solar import earth_sun_distance, solar_irradiance, solar_position, sun_vector


def test_envi_roundtrip(tmp_path, refl_image):
    refl_image.transform = (500000.0, 1.0, 0.0, 7650000.0, 0.0, -1.0)
    refl_image.meta['acquisition date'] = 'DATE(dd-mm-yyyy): 03-12-2021'
    for il in ('bsq', 'bil', 'bip'):
        p = io.write_envi(str(tmp_path / ('img_' + il)), refl_image, interleave=il)
        back = io.read_envi(p)
        np.testing.assert_allclose(back.data, refl_image.data, rtol=1e-6)
        np.testing.assert_allclose(back.wavelengths, refl_image.wavelengths, rtol=1e-6)
        np.testing.assert_allclose(back.transform, refl_image.transform)
        assert back.meta['acquisition date'].startswith('DATE')


def test_hsimage_band_helpers(refl_image):
    assert refl_image.band_index(2200.) == int(np.argmin(np.abs(refl_image.wavelengths - 2200)))
    sub = refl_image.subset_bands((2000., 2400.))
    assert sub.wavelengths.min() >= 1990 and sub.wavelengths.max() <= 2410
    assert refl_image.rgb().dtype == np.uint8


def test_bad_bands(refl_image):
    out = spectral.remove_bad_bands(refl_image)
    assert out.bands < refl_image.bands
    assert not ((out.wavelengths > 1350) & (out.wavelengths < 1450)).any()
    out2 = spectral.remove_bad_bands(refl_image, drop=False)
    assert out2.bands == refl_image.bands and np.isnan(out2.data[..., ~out2.bbl]).all()


def test_resample_gaussian_constant(wav):
    X = np.ones((5, wav.size))
    dst = np.linspace(450, 2400, 30)
    out = spectral.resample_spectra(X, wav, dst, dst_fwhm=20)
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_savgol_and_hull(refl_image):
    sm = spectral.savgol(refl_image, 7)
    assert np.nanstd(sm.data[0, 0] - refl_image.data[0, 0]) < 0.003
    hc = spectral.hull_correct(refl_image, (2000., 2400.))
    px = hc.data[0, 0]
    assert np.nanmax(px) <= 1.0 + 1e-3
    assert abs(hc.wavelengths[np.nanargmin(px)] - 2200) < 30


def test_white_reference_and_scale(refl_image):
    dark = np.full((refl_image.cols, refl_image.bands), 100.0)
    white = np.full((refl_image.cols, refl_image.bands), 4100.0)
    raw = refl_image.copy(refl_image.data * 4000 + 100)
    R = white_reference_reflectance(raw, white, dark, panel_reflectance=1.0)
    np.testing.assert_allclose(R.data, refl_image.data, atol=1e-5)

    img = refl_image.copy((refl_image.data * 1000).round())
    img.meta['description'] = 'HySpex RAD radiance file. Scaling = 1000'
    assert hyspex_scale_factor(img) == 1000
    out = apply_scale_factor(img)
    np.testing.assert_allclose(out.data, refl_image.data, atol=1e-3)


def test_elc_two_panels(refl_image):
    gain, offset = 120.0, 3.0
    rad = dn_to_radiance(refl_image, gain, offset)
    p1 = Panel('white', np.full(refl_image.bands, 0.95 * gain + offset), 0.95)
    p2 = Panel('black', np.full(refl_image.bands, 0.05 * gain + offset), 0.05)
    elc = ELC([p1, p2])
    np.testing.assert_allclose(elc.gain, gain)
    np.testing.assert_allclose(elc.offset, offset, atol=1e-9)
    np.testing.assert_allclose(elc.apply(rad).data, refl_image.data, atol=1e-5)
    # panel from an image region
    img = rad.copy()
    img.data[:5, :5] = 0.95 * gain + offset
    p = Panel.from_region('w', img, (0, 5, 0, 5), (np.array([400, 2500.]), np.array([0.95, 0.95])))
    np.testing.assert_allclose(p.radiance, 0.95 * gain + offset, rtol=1e-5)


def test_destripe_and_bad_pixels(refl_image):
    img = refl_image.subset_bands(slice(0, 120))
    img.data = img.data[:, :20]  # homogeneous material
    stripes = 1 + 0.05 * np.random.default_rng(0).normal(size=(1, img.cols, 1))
    striped = img.copy(img.data * stripes)
    rel = lambda im: np.std(np.mean(im.data, 0) / np.mean(img.data, 0), axis=0).mean()
    assert rel(striped) > 0.02
    for method in ('ratio', 'moment'):
        assert rel(noise.destripe(striped, method=method)) < 0.005
    # a smooth cross-track gradient survives windowed destriping
    grad = img.copy(img.data * np.linspace(0.8, 1.2, img.cols)[None, :, None])
    kept = noise.destripe(grad.copy(grad.data * stripes), method='ratio', window=7)
    assert np.mean(kept.data[:, -3:]) > 1.2 * np.mean(kept.data[:, :3])

    bad = refl_image.copy()
    bad.data[:, 7, 30] = 50.0
    mask = noise.detect_bad_detector_pixels(bad)
    assert mask[7, 30]
    fixed = noise.repair_detector_pixels(bad, mask)
    assert abs(fixed.data[3, 7, 30] - refl_image.data[3, 7, 30]) < 0.05


def test_mnf_denoise(refl_image):
    noisy = refl_image.copy(refl_image.data + np.random.default_rng(3).normal(0, 0.02, refl_image.shape).astype(np.float32))
    den = noise.mnf_denoise(noisy, 5)
    assert np.nanmean((den.data - refl_image.data) ** 2) < np.nanmean((noisy.data - refl_image.data) ** 2)


def test_masks(refl_image):
    img = refl_image.copy()
    img.data[0, 0] = 0
    assert masks.nodata_mask(img)[0, 0]
    out = masks.mask_nodata(img)
    assert np.isnan(out.data[0, 0]).all()
    cloud = refl_image.copy(np.full(refl_image.shape, 0.8, np.float32))
    assert masks.cloud_mask(cloud).all()


def test_utm_roundtrip():
    lat, lon = np.array([-21.2, -21.25]), np.array([14.8, 14.85])
    e, n, z, h = latlon_to_utm(lat, lon)
    assert z == 33 and h == 'S'
    # known value (from the utm package) for (-21.2, 14.8)
    assert abs(e[0] - 479242.02) < 0.01 and abs(n[0] - 7655704.45) < 0.01
    la, lo = utm_to_latlon(e, n, z, h)
    np.testing.assert_allclose(la, lat, atol=1e-7)
    np.testing.assert_allclose(lo, lon, atol=1e-7)


def test_solar():
    az, el = solar_position(0.0, 0.0, '2021-03-20T12:07:00')
    assert el > 85
    sv, az, el = sun_vector(-21.2, 14.8, ('2021-12-03 08:47:29', '%Y-%m-%d %H:%M:%S', 'UTC'))
    assert sv[2] < 0 and 40 < el < 80 and 60 < az < 130
    assert 0.98 < earth_sun_distance('2021-01-04') < 0.99
    E = solar_irradiance(np.array([500.0, 1000.0, 2200.0]))
    assert 1.6 < E[0] < 2.2 and E[0] > E[1] > E[2]
