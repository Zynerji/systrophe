"""Tests for the Bobrick-Martire generalised warp-bubble class."""

import numpy as np

from systrophe.bobrick_martire import (
    bm_NEC_radial,
    bm_energy_density,
    bm_metric_components,
    bm_shape,
    novelty_scan,
)


def test_shape_is_unity_at_centre():
    val = bm_shape(0.0, 1.0, R=1.0, sigma=4.0)
    assert val > 0.99


def test_shape_decays_far():
    assert bm_shape(5.0, 5.0, R=1.0, sigma=4.0) < 0.01


def test_metric_minkowski_far():
    g = bm_metric_components(5.0, 5.0, m_ADM=1.0)
    assert abs(g["g_tt"] - (-1.0)) < 0.01


def test_energy_density_sign_follows_m_ADM():
    """E_tt has the same sign as m_ADM."""
    pos = bm_energy_density(0.0, 1.0, m_ADM=1.0)
    neg = bm_energy_density(0.0, 1.0, m_ADM=-1.0)
    assert pos > 0
    assert neg < 0


def test_NEC_sign_follows_m_ADM():
    pos = bm_NEC_radial(0.0, 1.0, m_ADM=1.0)
    neg = bm_NEC_radial(0.0, 1.0, m_ADM=-1.0)
    assert pos > 0
    assert neg < 0


def test_novelty_scan_returns_verdict():
    res = novelty_scan(m_ADM_range=(-2.0, 2.0), n_m=11,
                        n_x=11, n_rho=11)
    assert "novelty_verdict" in res
    assert res["novelty_verdict"] in ("smooth", "uniform", "novel_structure")
