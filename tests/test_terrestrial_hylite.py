"""Integration tests for the hylite-based terrestrial workflow (skipped when hylite is not installed)."""
import numpy as np
import pytest

hylite = pytest.importorskip('hylite')
from hylite.project import Camera  # noqa: E402

from hyperprep import HSImage  # noqa: E402
from hyperprep.drone import NavTrack  # noqa: E402
from hyperprep.solar import sun_vector  # noqa: E402
from hyperprep.terrestrial import hypercloud as H  # noqa: E402
from hyperprep.terrestrial.illumination import correct_scene  # noqa: E402
from hyperprep.terrestrial.projection import build_pushbroom, build_scene, false_colour_cloud, scene_geometry  # noqa: E402

TIME = '2021-12-03T10:00:00+00:00'
LAT, LON = -21.2, 14.8


def _floor_scene(tmp_path, data, normals_left, normals_right, name='s'):
    xx, yy = np.meshgrid(np.linspace(-5, 5, 120), np.linspace(-3, 3, 80))
    xyz = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    n = np.where((xyz[:, 0] < 0)[:, None], normals_left, normals_right)
    cloud = hylite.HyCloud(xyz, normals=n / np.linalg.norm(n, axis=1, keepdims=True))
    cam = Camera(np.array([0, 0, 20.0]), np.array([0, 0, 0.0]), 'persp', fov=30, dims=data.shape[:2])
    img = hylite.HyImage(data.astype(np.float32))
    img.set_wavelengths(np.linspace(450, 2400, data.shape[-1]))
    return build_scene(name, str(tmp_path / name), img, cloud, cam, occ_tol=0, vb=False)


def test_hsimage_hylite_roundtrip():
    img = HSImage(np.random.rand(4, 6, 3).astype(np.float32), [500., 600., 700.])
    hy = img.to_hylite()
    assert hy.data.shape == (6, 4, 3)
    back = HSImage.from_hylite(hy)
    np.testing.assert_allclose(back.data, img.data)


def test_scene_projection_and_false_colour(tmp_path):
    s = _floor_scene(tmp_path, np.random.rand(60, 40, 5) + 0.5, [0, 0, 1.], [0, 0, 1.])
    xyz, n, v, d = scene_geometry(s)
    assert n.shape == (40, 60, 3)
    np.testing.assert_allclose(np.nanmean(n.reshape(-1, 3), 0), [0, 0, 1], atol=1e-6)
    fc = false_colour_cloud(s, (0, 1, 2))
    assert fc.rgb.dtype == np.uint8 and fc.data is None


def test_hypercloud_correct_blend_export(tmp_path):
    rng = np.random.default_rng(0)
    data = 0.3 * 100 + rng.normal(0, 0.1, (60, 40, 5))  # radiance = 100 * reflectance(0.3)
    s1 = _floor_scene(tmp_path, data, [0, 0, 1.], [0, 0, 1.], 's1')
    s2 = _floor_scene(tmp_path, data * 1.1, [0, 0, 1.], [0, 0, 1.], 's2')
    for s in (s1, s2):
        wav = s.image.get_wavelengths()
        from hylite.reference.spectra import Target
        target = Target(wav, np.full(wav.size, 0.9), name='white')
        panel = hylite.correct.Panel(target, np.full((1, wav.size), 90.0), wavelengths=wav)
        s.image.header.add_panel(panel, name='white')
    st = H.correct_scene(s1, H.HypercloudSettings(), topo_method=None, vb=False)
    assert st['success'] and st['atmos']
    np.testing.assert_allclose(np.nanmean(s1.image.data), 0.3, atol=0.01)
    st2 = H.correct_scene(s2, H.HypercloudSettings(), topo_method='cfac', vb=False)
    assert st2['atmos'] and not st2['success'] and 'sun' in st2['reason']  # no sun vector in header
    H.colour_match([s1, s2], reference_index=0, vb=False)
    hc = H.blend([s1, s2], 'average', vb=False)
    assert hc.data.shape[1] == 5
    out = H.export_hypercloud(hc, str(tmp_path / 'out'), 'hc', composites={'fc': (450., 937.5, 1425.)})
    assert set(out) == {'fc', 'hypercloud'}


@pytest.mark.parametrize('method', ['numpy', 'hylite'])
def test_illumination_correct_scene(tmp_path, method):
    sunvec, az, el = sun_vector(LAT, LON, TIME)
    toward_sun = -sunvec
    h = -sunvec[:2] / np.linalg.norm(sunvec[:2])  # horizontal direction towards the sun
    a = np.deg2rad(90 - el - 5)  # tilted away from the sun by more than the sun elevation -> shaded
    away = np.array([-h[0] * np.cos(a), -h[1] * np.cos(a), np.sin(a)])
    wav = np.linspace(450, 2400, 5)
    S, I = np.full(5, 5.0), np.full(5, 60.0)
    R = 0.4
    # build once to get geometry, then synthesise radiance from the model
    s = _floor_scene(tmp_path, np.ones((60, 40, 5)), toward_sun, away)
    from hyperprep.topo import incidence, skyview_from_normals
    ilf = incidence(s.normals, sunvec)
    skv = skyview_from_normals(s.normals)
    s.image.data = (R * (skv[..., None] * S + ilf[..., None] * I)).astype(np.float32)
    pilf, pskv = float(np.dot([0, 0, 1], -sunvec)), 1.0
    Lp = 0.9 * (pskv * S + pilf * I)
    res = correct_scene(s, Lp, 0.9, [0, 0, 1], pskv, LAT, LON, TIME, roughness=0.0, method=method)
    np.testing.assert_allclose(res['sun'], I, rtol=0.05)
    np.testing.assert_allclose(res['sky'], S, rtol=0.05)
    np.testing.assert_allclose(np.nanmedian(res['reflectance'].data), R, rtol=0.05)


def test_build_pushbroom():
    n = 20
    tr = NavTrack(np.arange(n, dtype=float), np.zeros(n), np.arange(n, dtype=float), np.full(n, 100.0),
                  np.zeros(n), np.zeros(n), np.zeros(n))
    img = hylite.HyImage(np.random.rand(30, n + 1, 4).astype(np.float32))
    pb = build_pushbroom(tr, img, np.rad2deg(0.54e-3))
    assert img.ydim() == n and pb.dims == (30, n)
    with pytest.raises(ValueError):
        build_pushbroom(tr, hylite.HyImage(np.random.rand(30, n + 5, 4).astype(np.float32)), 0.03)
