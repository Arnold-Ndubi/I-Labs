import numpy as np
import pytest
from synthetic import angle_diff_mod_pi, stripes

from alama.enhance import (
    EnhanceConfig,
    enhance,
    normalise,
    normalise_scale,
    orientation_field,
    ridge_wavelength,
)


def test_normalise_zero_mean_unit_std():
    img = stripes()
    out = normalise(img)
    assert abs(out.mean()) < 1e-3
    assert abs(out.std() - 1) < 1e-3


@pytest.mark.parametrize("theta", [0.0, np.pi / 6, np.pi / 2, 2 * np.pi / 3])
def test_orientation_field_recovers_ridge_direction(theta):
    orient, coherence = orientation_field(normalise(stripes(theta=theta)))
    centre = orient[96:160, 96:160]
    # Circular median via doubled angles.
    est = (np.angle(np.mean(np.exp(2j * centre))) / 2) % np.pi
    assert angle_diff_mod_pi(est, theta) < np.radians(3)
    assert coherence[96:160, 96:160].mean() > 0.9


@pytest.mark.parametrize("wavelength", [7.0, 9.0, 12.0])
def test_ridge_wavelength(wavelength):
    norm = normalise(stripes(wavelength=wavelength, theta=0.4))
    mask = np.ones(norm.shape, bool)
    assert ridge_wavelength(norm, mask) == pytest.approx(wavelength, rel=0.1)


def test_normalise_scale_brings_period_to_target():
    img = stripes(shape=(400, 400), wavelength=18.0, theta=0.3)
    mask = np.ones(img.shape, bool)
    out, out_mask, scale = normalise_scale(img, mask, target_wavelength=9.0)
    assert scale == pytest.approx(0.5, rel=0.1)
    assert out.shape == out_mask.shape


def test_enhance_produces_thin_ridges_on_noisy_stripes():
    img = stripes(theta=0.5, noise=0.8)
    result = enhance(img)
    inner = result.mask.copy()
    assert inner.mean() > 0.5
    ridge_share = result.ridges[inner].mean()
    assert 0.3 < ridge_share < 0.7  # about half the area is ridge
    assert result.skeleton.sum() < result.ridges.sum() / 2
    assert result.wavelength == pytest.approx(9.0, rel=0.1)


def test_bright_polarity_inverts_ridges():
    img = stripes(theta=0.2)
    dark = enhance(img, cfg=EnhanceConfig(ridge_polarity="dark")).ridges
    bright = enhance(img, cfg=EnhanceConfig(ridge_polarity="bright")).ridges
    both = dark & bright
    assert both.sum() < 0.1 * dark.sum()
