"""Tests for the Krasnikov tube module."""

import math

from systrophe.krasnikov_tube import (
    krasnikov_NEC_radial,
    krasnikov_energy_density,
    krasnikov_kernel,
    krasnikov_metric_components,
    novelty_scan,
)


def test_kernel_smooth_at_origin():
    """kernel is well-defined and finite at (x_0, t_0)."""
    k = krasnikov_kernel(0.0, 0.0, x_0=0.0, t_0=0.0, alpha=4.0)
    assert 0 <= k <= 1


def test_metric_components_keys():
    g = krasnikov_metric_components(0.5, 0.0)
    assert "g_tt" in g and "g_tx" in g and "g_xx" in g and "k" in g


def test_energy_density_negative_in_wall():
    """T_tt is negative near the wall x = x_0."""
    val = krasnikov_energy_density(0.0, t=1.0, x_0=0.0, alpha=4.0)
    assert val < 0


def test_energy_density_vanishes_far():
    """Far from the wall, T_tt ~ 0."""
    val = krasnikov_energy_density(5.0, t=1.0, x_0=0.0, alpha=4.0)
    assert abs(val) < 1e-3


def test_NEC_negative_in_wall():
    val = krasnikov_NEC_radial(0.0, t=1.0, x_0=0.0, alpha=4.0)
    assert val < 0


def test_NEC_scales_with_alpha_squared():
    """|T_kk| ~ alpha^2 at the wall."""
    v1 = krasnikov_NEC_radial(0.0, t=1.0, alpha=2.0)
    v2 = krasnikov_NEC_radial(0.0, t=1.0, alpha=4.0)
    ratio = v2 / v1
    assert 3.5 < ratio < 4.5


def test_novelty_scan_returns_verdict():
    res = novelty_scan(alpha_range=(1.0, 10.0), n_alpha=8, n_x=21)
    assert "novelty_verdict" in res
    assert res["novelty_verdict"] in ("smooth", "uniform", "novel_structure")
