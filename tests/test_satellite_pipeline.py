import json
import os

import numpy as np
import pytest

from hyperprep import HSImage, io
from hyperprep.pipeline import Pipeline, list_steps, run_config
from hyperprep.satellite import (dark_object_subtraction, estimate_smile, merge_overlap, read_product,
                                 smile_correction, subset_bbox, toa_reflectance)
from hyperprep.solar import solar_irradiance

h5py = pytest.importorskip('h5py')


def test_toa_reflectance_roundtrip(refl_image):
    E = solar_irradiance(refl_image.wavelengths)
    sz = 30.0
    L = refl_image.copy(refl_image.data * E * np.cos(np.deg2rad(sz)) / np.pi)
    rho = toa_reflectance(L, sun_zenith=sz)
    np.testing.assert_allclose(rho.data, refl_image.data, rtol=1e-5)
    rho2 = toa_reflectance(L, sun_zenith=sz, time='2021-07-04')  # aphelion: d > 1
    assert np.nanmean(rho2.data) > np.nanmean(rho.data)


def test_dos(refl_image):
    hazy = refl_image.copy(refl_image.data + 0.05)
    out = dark_object_subtraction(hazy, percentile=0.0)
    assert abs(np.nanmin(out.data)) < 1e-6


def test_merge_overlap():
    wv = np.linspace(420, 1000, 30)
    ws = np.linspace(900, 2450, 60)
    wav = np.concatenate([wv, ws])
    img = HSImage(np.ones((4, 5, wav.size), np.float32), wav, meta={'n_vnir': 30})
    out = merge_overlap(img)
    assert np.all(np.diff(out.wavelengths) > 0)
    assert out.bands < img.bands
    out2 = merge_overlap(HSImage(img.data, wav), method='mean')
    assert np.all(np.diff(out2.wavelengths) > 0)


def test_smile(wav):
    w = np.linspace(700, 820, 121)
    o2 = 1 - 0.6 * np.exp(-0.5 * ((w - 761) / 2.0) ** 2)
    cols = 30
    shift = np.linspace(-1.5, 1.5, cols)  # true band-centre error per column
    data = np.empty((5, cols, w.size), np.float32)
    for c in range(cols):
        data[:, c] = np.interp(w + shift[c], w, o2)  # observed at shifted centres
    img = HSImage(data, w)
    est = estimate_smile(img, (745., 780.), max_shift=3, step=0.05)
    np.testing.assert_allclose(est - est.mean(), shift - shift.mean(), atol=0.15)
    fixed = smile_correction(img, w[None, :] + est[:, None])
    assert np.nanstd(fixed.data[0, :, 61]) < np.nanstd(img.data[0, :, 61])


def test_subset_bbox(refl_image):
    refl_image.transform = (100.0, 2.0, 0.0, 200.0, 0.0, -2.0)
    out = subset_bbox(refl_image, (110, 150, 130, 190))
    assert out.shape[:2] == (20, 10) and out.transform[0] == 110 and out.transform[3] == 190


def test_emit_reader(tmp_path):
    p = str(tmp_path / 'EMIT_L2A_RFL_001_20230101T000000_2300100_001.nc')
    rng = np.random.default_rng(0)
    rfl = rng.uniform(0, 0.5, (6, 5, 10)).astype(np.float32)
    rfl[0, 0] = -9999
    with h5py.File(p, 'w') as f:
        d = f.create_dataset('reflectance', data=rfl)
        d.attrs['_FillValue'] = np.float32(-9999)
        g = f.create_group('sensor_band_parameters')
        g['wavelengths'] = np.linspace(400, 2500, 10)
        g['fwhm'] = np.full(10, 8.0)
        g['good_wavelengths'] = np.array([1] * 8 + [0] * 2)
        loc = f.create_group('location')
        gx = np.zeros((4, 4), np.int32)
        gy = np.zeros((4, 4), np.int32)
        gx[1:3, 1:3] = [[1, 2], [3, 4]]
        gy[1:3, 1:3] = [[1, 2], [3, 4]]
        loc['glt_x'] = gx
        loc['glt_y'] = gy
        f.attrs['geotransform'] = np.array([10.0, 0.0006, 0, 50.0, 0, -0.0006])
        f.attrs['spatial_ref'] = 'GEOGCS["WGS 84"]'
    raw = read_product(p)
    assert raw.shape == (6, 5, 10) and np.isnan(raw.data[0, 0]).all() and raw.bbl.sum() == 8
    ortho = read_product(p, ortho=True)
    assert ortho.shape == (4, 4, 10) and ortho.transform[0] == 10.0
    np.testing.assert_allclose(ortho.data[2, 1], rfl[2, 2])  # glt (x=3, y=3) -> 0-based (2, 2)
    assert np.isnan(ortho.data[0, 0]).all()


def test_prisma_reader(tmp_path):
    p = str(tmp_path / 'PRS_L1_STD_OFFL_20200101.he5')
    lines, samples = 4, 3
    with h5py.File(p, 'w') as f:
        g = f.create_group('HDFEOS/SWATHS/PRS_L1_HCO/Data Fields')
        g['VNIR_Cube'] = np.full((lines, 3, samples), 200, np.uint16)
        g['SWIR_Cube'] = np.full((lines, 2, samples), 100, np.uint16)
        geo = f.create_group('HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields')
        geo['Latitude_VNIR'] = np.zeros((lines, samples))
        geo['Longitude_VNIR'] = np.zeros((lines, samples))
        f.attrs['List_Cw_Vnir'] = np.array([600.0, 500.0, 0.0])
        f.attrs['List_Fwhm_Vnir'] = np.array([10.0, 10.0, 0.0])
        f.attrs['List_Cw_Swir'] = np.array([2000.0, 1500.0])
        f.attrs['List_Fwhm_Swir'] = np.array([12.0, 12.0])
        f.attrs['ScaleFactor_Vnir'] = 100.0
        f.attrs['Offset_Vnir'] = 0.0
        f.attrs['ScaleFactor_Swir'] = 50.0
        f.attrs['Offset_Swir'] = 0.0
        f.attrs['Sun_zenith_angle'] = 35.0
    img = read_product(p)
    np.testing.assert_allclose(img.wavelengths, [500, 600, 1500, 2000])
    assert img.shape == (lines, samples, 4)
    np.testing.assert_allclose(img.data[0, 0], [2, 2, 2, 2])
    assert img.meta['sun_zenith'] == 35.0 and img.meta['units'] == 'radiance'


def test_enmap_reader(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    d = tmp_path / 'ENMAP01-____L2A-DT0000001_20220101T000000Z_001_V010110_20220102T000000Z'
    d.mkdir()
    arr = (np.random.default_rng(0).uniform(0, 0.4, (4, 6, 5)) * 10000).astype(np.int16)
    with rasterio.open(d / (d.name + '-SPECTRAL_IMAGE.TIF'), 'w', driver='GTiff', height=6, width=5, count=4,
                       dtype='int16', transform=from_origin(300000, 5000000, 30, 30), crs='EPSG:32633') as dst:
        dst.write(arr)
    bands = ''.join('<bandID number="%d"><wavelengthCenterOfBand>%.1f</wavelengthCenterOfBand>'
                    '<FWHMOfBand>8</FWHMOfBand><GainOfBand>1</GainOfBand><OffsetOfBand>0</OffsetOfBand></bandID>'
                    % (i + 1, w) for i, w in enumerate([450, 550, 650, 850]))
    xml = ('<level_X><base><temporalCoverage><startTime>2022-01-01T10:00:00Z</startTime></temporalCoverage></base>'
           '<specific><sunElevationAngle><center>40.0</center></sunElevationAngle>'
           '<sunAzimuthAngle><center>150.0</center></sunAzimuthAngle>'
           '<bandCharacterisation>%s</bandCharacterisation></specific></level_X>' % bands)
    (d / (d.name + '-METADATA.XML')).write_text(xml)
    img = read_product(str(d))
    assert img.shape == (6, 5, 4) and img.meta['units'] == 'reflectance'
    np.testing.assert_allclose(img.data, np.transpose(arr, (1, 2, 0)) / 10000.0, rtol=1e-6)
    assert img.meta['sun_zenith'] == 50.0 and img.transform[0] == 300000


def test_pipeline_yaml(tmp_path, refl_image):
    src = io.write_envi(str(tmp_path / 'in'), refl_image)
    cfg = tmp_path / 'cfg.yaml'
    out = str(tmp_path / 'out.hdr')
    cfg.write_text(
        'input: %s\noutput: %s\nsteps:\n  - read: {}\n  - remove_bad_bands: {}\n  - savgol: {window: 7}\n'
        '  - remove_spikes: {}\n  - clip_range: {vmin: 0, vmax: 1}\n' % (src, out))
    img = run_config(str(cfg), verbose=False)
    back = io.read_envi(out)
    assert back.bands == img.bands < refl_image.bands
    assert img.meta['history'][-1] == 'write'
    assert 'orthorectify' in list_steps() and 'toa_reflectance' in list_steps()


def test_pipeline_unknown_step():
    with pytest.raises(KeyError):
        Pipeline([('nope', {})])


def test_cli(tmp_path, refl_image, capsys):
    from hyperprep.cli import main
    src = io.write_envi(str(tmp_path / 'in'), refl_image)
    assert main(['info', src]) == 0
    assert '120 bands' in capsys.readouterr().out
    cfg = tmp_path / 'c.json'
    cfg.write_text(json.dumps({'input': src, 'output': str(tmp_path / 'o.hdr'), 'steps': ['iarr']}))
    assert main(['run', str(cfg), '-q']) == 0
    assert os.path.exists(tmp_path / 'o.hdr')


def test_pipeline_drone_steps(tmp_path, wav):
    lines, cols = 40, 21
    data = np.full((lines, cols, wav.size), 30.0, np.float32)
    data[:5, :5] = 90.0  # white panel radiance
    src = io.write_envi(str(tmp_path / 'line'), HSImage(data, wav))
    nav = tmp_path / 'nav.txt'
    np.savetxt(nav, [[i, 24.9, 60.2 + i * 1e-5, 150.0, 0, 0, 0, i * 0.01] for i in range(lines)])
    cfg = {'input': src, 'steps': [
        {'read': {}},
        {'elc': {'panels': [{'name': 'w', 'region': [0, 5, 0, 5], 'reflectance': 0.9}]}},
        {'cross_track_correction': {'degree': 2}},
        {'orthorectify': {'nav': str(nav), 'ifov_deg': 0.5, 'res': 1.0, 'ground': 50.0}},
    ]}
    out = run_config(cfg, verbose=False)
    assert out.crs == 32635 and out.transform[1] == 1.0
    # the bright panel slightly biases the cross-track model, hence the loose tolerance
    np.testing.assert_allclose(np.nanmedian(out.data), 0.3, rtol=0.03)
