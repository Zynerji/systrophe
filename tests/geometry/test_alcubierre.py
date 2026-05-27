"""Tests for the Alcubierre warp-drive module."""

import math

import numpy as np

from systrophe.geometry.alcubierre import (
    alcubierre_NEC_radial,
    alcubierre_energy_density,
    alcubierre_metric_components,
    alcubierre_shape,
    alcubierre_shape_derivative,
    alcubierre_total_negative_energy,
    novelty_scan,
    pfenning_ford_quantum_bound,
)


def test_shape_is_unity_inside_bubble():
    """Top-hat = 1 at r_s = 0."""
    assert alcubierre_shape(0.0, R=1.0, sigma=8.0) > 0.99


def test_shape_decays_far_outside():
    assert alcubierre_shape(5.0, R=1.0, sigma=8.0) < 0.01


def test_shape_derivative_zero_at_origin():
    """f is even in r_s about origin so df/dr_s|_0 = 0."""
    assert abs(alcubierre_shape_derivative(0.0, R=1.0, sigma=8.0)) < 1e-8


def test_shape_derivative_peaks_at_wall():
    """The derivative peaks near r_s = R."""
    rs = np.linspace(0.1, 2.0, 41)
    dfs = np.array([alcubierre_shape_derivative(float(r), R=1.0, sigma=8.0)
                    for r in rs])
    i = int(np.argmin(dfs))  # most-negative; df is negative at outer wall
    assert 0.8 < rs[i] < 1.2


def test_metric_components_minkowski_outside():
    g = alcubierre_metric_components(5.0, v_s=1.0, R=1.0, sigma=8.0)
    assert abs(g["g_tt"] - (-1.0)) < 0.01
    assert abs(g["g_tx"]) < 0.01


def test_energy_density_nonpositive():
    """T_tt is everywhere <= 0 (exotic matter)."""
    rs = np.linspace(0.01, 3.0, 31)
    vals = [alcubierre_energy_density(float(r), v_s=1.0)
            for r in rs]
    assert all(v <= 0 for v in vals)


def test_NEC_radial_nonpositive():
    rs = np.linspace(0.01, 3.0, 31)
    vals = [alcubierre_NEC_radial(float(r), v_s=1.0)
            for r in rs]
    assert all(v <= 0 for v in vals)


def test_total_negative_energy_scales_with_v_squared():
    """E_neg ~ -v_s^2."""
    e1 = alcubierre_total_negative_energy(v_s=1.0, R=1.0, sigma=8.0)
    e2 = alcubierre_total_negative_energy(v_s=2.0, R=1.0, sigma=8.0)
    ratio = e2 / e1
    assert 3.5 < ratio < 4.5


def test_pfenning_ford_bound_positive():
    """|E_neg| tau lower bound is strictly positive."""
    b = pfenning_ford_quantum_bound(v_s=1.0, R=1.0, sigma=8.0)
    assert b > 0


def test_pfenning_ford_inverse_sigma_squared():
    b1 = pfenning_ford_quantum_bound(sigma=2.0)
    b2 = pfenning_ford_quantum_bound(sigma=4.0)
    ratio = b1 / b2
    assert 3.5 < ratio < 4.5  # b ~ 1/sigma^2


def test_novelty_scan_returns_verdict():
    res = novelty_scan(v_s_range=(0.1, 3.0), sigma_range=(2.0, 10.0),
                        n_vs=8, n_sigma=8, R=1.0)
    assert "novelty_verdict" in res
    assert res["novelty_verdict"] in ("smooth", "uniform", "novel_structure")
