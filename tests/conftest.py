import numpy as np
import pytest

from hyperprep import HSImage


@pytest.fixture
def wav():
    return np.linspace(400, 2500, 120)


@pytest.fixture
def refl_image(wav):
    """Synthetic reflectance cube with two materials and a Gaussian absorption feature."""
    rng = np.random.default_rng(1)
    rows, cols = 30, 40
    base = 0.2 + 0.1 * (wav - 400) / 2100
    feat = 1 - 0.3 * np.exp(-0.5 * ((wav - 2200) / 20) ** 2)
    a = base * feat
    b = 0.5 * np.ones_like(wav)
    data = np.empty((rows, cols, wav.size))
    data[:, :20] = a
    data[:, 20:] = b
    data += rng.normal(0, 0.002, data.shape)
    return HSImage(data.astype(np.float32), wav, fwhm=np.full(wav.size, 10.0))
