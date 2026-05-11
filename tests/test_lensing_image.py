"""Tests for strong-lensing image simulation."""

import numpy as np
import pytest

from systrophe.lensing_image import (
    LensingImage,
    chronology_horizon_caustics,
    compare_cylinder_vs_hybrid,
    cylinder_image_grid,
    hybrid_image_grid,
    shadow_metrics,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_cylinder_image_returns_correct_shape(vs):
    img = cylinder_image_grid(vs, n_pixels=11)
    assert isinstance(img, LensingImage)
    assert img.image.shape == (11, 11)


def test_cylinder_image_grids_correct_range(vs):
    img = cylinder_image_grid(vs, alpha_max=2.0, n_pixels=5)
    assert img.alpha_grid[0] == -2.0
    assert img.alpha_grid[-1] == 2.0


def test_cylinder_image_is_finite(vs):
    img = cylinder_image_grid(vs, n_pixels=7)
    assert np.all(np.isfinite(img.image))


def test_hybrid_image_returns_LensingImage(vs):
    img = hybrid_image_grid(vs, M=0.5, n_pixels=11)
    assert isinstance(img, LensingImage)
    assert img.schwarzschild_mass == 0.5


def test_hybrid_image_has_shadow(vs):
    """Schwarzschild mass should produce a shadow region."""
    img = hybrid_image_grid(vs, M=1.0, n_pixels=21, alpha_max=8.0)
    # Shadow pixels = 1.0
    n_shadow = int(np.sum(img.image >= 0.99))
    assert n_shadow > 0


def test_chronology_horizon_caustics_returns_list(vs):
    caustics = chronology_horizon_caustics(vs)
    assert isinstance(caustics, list)


def test_caustics_for_supercritical_nonempty(vs):
    """Supercritical LP should have at least one caustic."""
    caustics = chronology_horizon_caustics(vs, r_max=10.0)
    assert len(caustics) >= 1


def test_shadow_metrics_for_cylinder_no_shadow(vs):
    img = cylinder_image_grid(vs, n_pixels=11)
    m = shadow_metrics(img)
    assert m["shadow_fraction"] == 0.0


def test_shadow_metrics_for_hybrid_has_shadow(vs):
    img = hybrid_image_grid(vs, M=1.0, n_pixels=21, alpha_max=8.0)
    m = shadow_metrics(img)
    assert m["shadow_fraction"] > 0


def test_compare_cylinder_vs_hybrid_returns_dict(vs):
    cmp = compare_cylinder_vs_hybrid(vs, M=0.5, n_pixels=11)
    assert "cylinder_image" in cmp
    assert "hybrid_image" in cmp
    assert "cylinder_metrics" in cmp
    assert "hybrid_metrics" in cmp


def test_compare_shows_shadow_only_in_hybrid(vs):
    cmp = compare_cylinder_vs_hybrid(vs, M=1.0, n_pixels=21, alpha_max=8.0)
    assert cmp["cylinder_metrics"]["shadow_fraction"] == 0.0
    assert cmp["hybrid_metrics"]["shadow_fraction"] > 0
