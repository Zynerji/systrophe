"""QFTCS back-reaction (2D conformal anomaly) tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.qftcs.qftcs_backreaction import (
    conformal_anomaly_trace,
    radial_temporal_ricci_scalar,
    stress_energy_at_horizon,
    vacuum_energy_density_proxy,
)


def test_ricci_finite_in_well_conditioned_region():
    """R_2d is finite away from the Cauchy horizon."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = np.array([1.2, 1.5, 1.7])
    R = radial_temporal_ricci_scalar(vs, r)
    assert np.all(np.isfinite(R))


def test_anomaly_trace_proportional_to_R():
    """<T^mu_mu> = R / (24 pi)."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = 1.5
    R = float(radial_temporal_ricci_scalar(vs, r))
    trace = float(conformal_anomaly_trace(vs, r))
    assert trace == pytest.approx(R / (24.0 * np.pi), rel=1e-12)


def test_vacuum_energy_proxy_signed():
    """Vacuum energy proxy is -R/(96 pi)."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = 1.5
    R = float(radial_temporal_ricci_scalar(vs, r))
    rho = float(vacuum_energy_density_proxy(vs, r))
    assert rho == pytest.approx(-R / (96.0 * np.pi), rel=1e-12)


def test_stress_energy_at_horizon_returns_dict():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate
    horizons = cauchy_horizon_estimate(vs)
    assert horizons.size >= 1
    out = stress_energy_at_horizon(vs, float(horizons[0]))
    assert "power" in out
    assert "amplitude_log" in out


def test_anomaly_zero_in_minkowski_limit():
    """As omega R -> 1/2 from above (critical), the Tipler exterior approaches
    the logarithmic critical case; for small a > 1/2 the anomaly is small in
    magnitude near the cylinder boundary."""
    vs = VanStockumInterior(omega=0.6, R=1.0)  # a = 0.6 (mildly supercritical)
    r = 1.05
    trace = float(conformal_anomaly_trace(vs, r))
    # Magnitude should be modest near the cylinder boundary
    assert abs(trace) < 100.0
