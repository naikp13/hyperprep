import numpy as np

from hyperprep import HSImage
from hyperprep.drone import (DEM, NavTrack, cross_track_correction, georeference, irradiance_normalisation,
                             match_flightlines, orthorectify, pixel_look_angles, read_hyspexnav)
from hyperprep.ortho import mosaic
from hyperprep.terrestrial import estimate_sky_sun, illumination_reflectance, panel_illumination_factor, parse_header_time
from hyperprep.topo import incidence, normals_from_dem, oren_nayar, slope_aspect, topographic_correction


def _track(n=50, z=100.0, yaw=0.0):
    t = np.arange(n, dtype=float)
    return NavTrack(time=t, x=500000 + 0 * t, y=7650000 + t * 1.0, z=np.full(n, z), roll=np.zeros(n),
                    pitch=np.zeros(n), yaw=np.full(n, yaw))


def test_normals_from_dem():
    x = np.arange(50) * 2.0
    dem = np.tile(x * 0.5, (40, 1))  # rises to the east with slope 0.5
    n = normals_from_dem(dem, 2.0)
    slope, aspect = slope_aspect(n)
    np.testing.assert_allclose(slope[5:-5, 5:-5], np.rad2deg(np.arctan(0.5)), atol=1e-6)
    np.testing.assert_allclose(aspect[5:-5, 5:-5], 270, atol=1e-6)  # faces west
    north = np.tile((np.arange(40) * -0.5)[:, None], (1, 50))  # height decreases with row -> rises north
    _, asp = slope_aspect(normals_from_dem(north, 1.0))
    np.testing.assert_allclose(asp[5:-5, 5:-5], 180, atol=1e-6)  # faces south


def test_topographic_correction_recovers_flat(refl_image):
    rng = np.random.default_rng(0)
    cos_i = rng.uniform(0.3, 1.0, refl_image.shape[:2])
    sky = 0.15
    lit = refl_image.copy(refl_image.data * (cos_i[..., None] + sky))
    for m in ('c', 'ambient'):
        out = topographic_correction(lit, cos_i, method=m, cos_ref=1.0)
        ratio = out.data / refl_image.data
        assert np.nanstd(ratio) / np.nanmean(ratio) < 0.02, m
    cos = topographic_correction(lit, cos_i, method='cosine')
    assert np.nanstd(cos.data / refl_image.data) > np.nanstd(topographic_correction(lit, cos_i, 'c').data / refl_image.data)


def test_oren_nayar_reduces_to_lambert():
    n = np.array([[[0, 0, 1.0]]])
    sun = np.array([0, -0.5, -np.sqrt(0.75)])
    lam = incidence(n, sun)
    on = oren_nayar(n, np.array([0, 0, 1.0]), sun, roughness=0.0)
    np.testing.assert_allclose(on, lam)


def test_illumination_model_closed_form(wav):
    rng = np.random.default_rng(2)
    shape = (60, 60)
    R = np.clip(0.3 + 0.05 * rng.normal(size=shape + (1,)), 0.05, 1) * np.ones(wav.size)
    S = 0.02 + 0.01 * (wav < 800)  # skylight
    I = np.full(wav.size, 0.2)     # sunlight
    ilf = rng.uniform(0, 1, shape)
    ilf[:20] = 0.0                 # shaded block
    skv = np.full(shape, 0.6)
    L = R * (skv[..., None] * S + ilf[..., None] * I)
    pilf, pskv, Rp = 0.8, 0.5, 0.9
    Lp = Rp * (pskv * S + pilf * I)
    S_e, I_e = estimate_sky_sun(L, ilf, skv, Lp, Rp, pilf, pskv, sun_thresh=0.5)
    # The estimate assumes equal median reflectance of sun and shade pixels, so allow a few %.
    np.testing.assert_allclose(S_e, S, rtol=0.1)
    np.testing.assert_allclose(I_e, I, rtol=0.1)
    Rhat = illumination_reflectance(L, I, S, ilf, skv)
    np.testing.assert_allclose(Rhat, R, rtol=1e-6)


def test_panel_factor_and_header_time():
    assert abs(panel_illumination_factor([0, 0, 1], [0, 0, -1]) - 1) < 1e-12
    assert panel_illumination_factor([0, 0, -1], [0, 0, -1]) == 0
    t = parse_header_time({'acquisition date': 'DATE(yyyy-mm-dd): 2021-12-03', 'start time': 'UTC TIME: 08:47:29'})
    assert (t.hour, t.minute, t.second) == (8, 47, 29)


def test_hyspexnav_reader(tmp_path):
    p = tmp_path / 'nav.txt'
    rows = [[i, 14.8, -21.2 + i * 1e-5, 1100.0, 0.1, -0.2, 45.0, 1000 + i * 0.01] for i in range(10)]
    np.savetxt(p, rows)
    tr = read_hyspexnav(str(p))
    assert len(tr) == 10 and tr.crs == 32733
    assert tr.check_lines(11) == 1 and tr.check_lines(10) == 0
    tr2 = tr.interpolate(np.linspace(1000, 1000.09, 19))
    assert len(tr2) == 19 and np.allclose(tr2.yaw, 45)


def test_georeference_nadir_and_roll():
    tr = _track()
    ang = pixel_look_angles(21, ifov_deg=1.0)
    X, Y, Z, v = georeference(tr, ang, ground=0.0)
    np.testing.assert_allclose(X[:, 10], 500000, atol=1e-6)               # nadir pixel below sensor
    np.testing.assert_allclose(X[0, 20] - 500000, 100 * np.tan(np.deg2rad(10)), rtol=1e-6)  # right of track = east (heading north)
    tr.roll[:] = 5.0  # right wing down -> the belly (and a nadir sensor) swings to the left (west)
    X2, _, _, _ = georeference(tr, ang)
    np.testing.assert_allclose(X2[0, 10] - 500000, -100 * np.tan(np.deg2rad(5)), rtol=1e-6)
    # heading east: right of track is south
    X3, Y3, _, _ = georeference(_track(yaw=90.0), ang)
    assert Y3[0, 20] < Y3[0, 0]


def test_georeference_dem():
    tr = _track(z=150.0)
    ang = pixel_look_angles(21, ifov_deg=1.0)
    dem = DEM(np.full((400, 400), 50.0), (499800.0, 1.0, 0.0, 7650200.0, 0.0, -1.0))
    X, Y, Z, _ = georeference(tr, ang, dem)
    np.testing.assert_allclose(Z, 50.0)
    np.testing.assert_allclose(X[0, 20] - 500000, 100 * np.tan(np.deg2rad(10)), rtol=1e-4)


def test_orthorectify_and_mosaic(wav):
    tr = _track(n=60)
    img = HSImage(np.random.default_rng(0).uniform(size=(60, 21, 5)).astype(np.float32), wav[:5])
    ang = pixel_look_angles(21, ifov_deg=0.5)
    out, glt, geo = orthorectify(img, tr, ang, res=1.0, return_glt=True)
    assert out.transform[1] == 1.0 and out.crs is None
    filled = np.isfinite(out.data[..., 0]).mean()
    assert filled > 0.8
    # each filled pixel value comes from the source image
    ok = glt.rows >= 0
    np.testing.assert_array_equal(out.data[ok], img.data[glt.rows[ok], glt.cols[ok]])
    assert geo['view_zenith'].max() < 6
    m = mosaic([out, out.copy()], method='mean')
    assert m.shape == out.shape


def test_cross_track_and_matching(refl_image):
    va = np.linspace(-15, 15, refl_image.cols)
    brdf = 1 + 0.002 * va + 0.0005 * va ** 2
    homog = refl_image.copy(np.broadcast_to(refl_image.data[:, :1], refl_image.shape).copy())
    obs = homog.copy(homog.data * brdf[None, :, None])
    corr = cross_track_correction(obs, va, degree=2)
    np.testing.assert_allclose(corr.data, homog.data, rtol=0.01)

    a = refl_image.copy()
    a.transform = (0.0, 1.0, 0.0, 30.0, 0.0, -1.0)
    b = a.copy(a.data * 1.2 + 0.01)
    matched = match_flightlines(b, a)
    np.testing.assert_allclose(matched.data, a.data, atol=1e-4)

    E = np.full(refl_image.bands, np.pi)
    np.testing.assert_allclose(irradiance_normalisation(refl_image, E).data, refl_image.data, rtol=1e-6)
