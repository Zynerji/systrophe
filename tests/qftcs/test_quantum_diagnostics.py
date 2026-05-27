"""Quantum-diagnostic (classical pre-quantum) tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.qftcs.quantum_diagnostics import (
    cauchy_horizon_estimate,
    chronology_protection_indicator,
    tolman_blueshift_factor,
)


def test_tolman_blueshift_unity_at_F_one():
    """1 / sqrt(1) = 1."""
    assert float(tolman_blueshift_factor(1.0)) == pytest.approx(1.0)


def test_tolman_blueshift_diverges_at_F_zero():
    """1 / sqrt(F) -> infinity as F -> 0+."""
    assert float(tolman_blueshift_factor(0.0)) == np.inf


def test_chronology_indicator_diverges_near_F_zero():
    """1 / F^2 grows unboundedly as F -> 0."""
    F_vals = np.array([1.0, 0.1, 0.01, 0.001])
    indicator = chronology_protection_indicator(F_vals)
    assert indicator[0] == pytest.approx(1.0)
    assert indicator[1] == pytest.approx(100.0)
    assert indicator[3] > indicator[2] > indicator[1] > indicator[0]


def test_cauchy_horizon_estimate_supercritical():
    """For supercritical Tipler, returns the closed-form F-zero radii."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    horizons = cauchy_horizon_estimate(vs)
    assert horizons.size >= 1
    # First horizon is at r = exp((pi - gamma)/alpha)
    alpha = vs.alpha
    gamma = float(np.pi - np.arctan(alpha))
    expected_first = float(np.exp((np.pi - gamma) / alpha))
    assert horizons[0] == pytest.approx(expected_first, rel=1e-12)


def test_cauchy_horizon_subcritical_rejected():
    vs = VanStockumInterior(omega=0.3, R=1.0)
    with pytest.raises(ValueError):
        cauchy_horizon_estimate(vs)


def test_ricci_residual_small_in_vacuum_exterior():
    """The vacuum Ernst residual F(F''+F'/r) - (F'^2 - c^2) is zero analytically;
    numerical residual is finite-difference error only."""
    from systrophe.qftcs.quantum_diagnostics import ricci_scalar

    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = np.array([1.5, 2.0, 3.0])
    residual = ricci_scalar(vs, r)
    assert np.all(np.abs(residual) < 1e-3)


def test_conformal_anomaly_proxy_finite_in_well_conditioned_region():
    from systrophe.qftcs.quantum_diagnostics import conformal_anomaly_2d_proxy

    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = np.array([1.2, 1.5, 1.7])  # before first F-zero
    proxy = conformal_anomaly_2d_proxy(vs, r)
    assert np.all(np.isfinite(proxy))


def test_surface_gravity_at_horizon():
    """Surface gravity kappa = |F'|/2 at a Cauchy horizon."""
    from systrophe.qftcs.quantum_diagnostics import (
        cauchy_horizon_estimate, hawking_temperature_at_horizon, surface_gravity_at_horizon
    )

    vs = VanStockumInterior(omega=1.0, R=1.0)
    horizons = cauchy_horizon_estimate(vs)
    assert horizons.size >= 1
    kappa = surface_gravity_at_horizon(vs, float(horizons[0]))
    assert kappa > 0
    T_H = hawking_temperature_at_horizon(vs, float(horizons[0]))
    assert T_H == pytest.approx(kappa / (2 * np.pi))


def test_cauchy_horizon_log_uniform_spacing():
    """Adjacent Cauchy horizon radii are separated by exp(pi / alpha) (half a Tipler period)."""
    vs = VanStockumInterior(omega=1.5, R=1.0)
    horizons = cauchy_horizon_estimate(vs)
    if horizons.size >= 2:
        ratios = horizons[1:] / horizons[:-1]
        expected_ratio = float(np.exp(np.pi / vs.alpha))
        np.testing.assert_allclose(ratios, expected_ratio, rtol=1e-12)
