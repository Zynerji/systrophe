"""Dirac-sea structure tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.qftcs.dirac_sea import (
    chronology_horizon_pressure_divergence_rate,
    density_of_states_radial,
    dirac_sea_pressure_proxy,
    local_energy,
)


def test_local_energy_unity_at_F_one():
    """Tolman energy E_local = E_infty / sqrt(1) = E_infty."""
    assert float(local_energy(1.0, E_infty=2.0)) == pytest.approx(2.0)


def test_local_energy_diverges_at_F_zero():
    assert float(local_energy(0.0, E_infty=1.0)) == np.inf


def test_density_of_states_zero_at_threshold():
    """At omega_local = mass, the density of states vanishes."""
    # Choose F so that omega_local = E_infty/sqrt(F) = mass exactly
    mass = 1.0
    E_infty = 1.0
    F_threshold = (E_infty / mass) ** 2
    rho = float(density_of_states_radial(F_threshold, m=0, k=0, mass=mass, E_infty=E_infty))
    assert rho == pytest.approx(0.0, abs=1e-9)


def test_density_of_states_positive_above_threshold():
    """Above threshold, rho > 0."""
    mass = 0.5
    E_infty = 1.0
    rho = float(density_of_states_radial(0.5, m=0, k=0, mass=mass, E_infty=E_infty))
    assert rho > 0


def test_dirac_sea_pressure_proxy_diverges_at_F_zero():
    """The 1/F^2 proxy diverges as F -> 0."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    # Far from F=0, finite
    p_far = float(dirac_sea_pressure_proxy(vs, 1.5))
    assert np.isfinite(p_far)
    # Get F-zero radius
    from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate
    horizons = cauchy_horizon_estimate(vs)
    assert horizons.size >= 1
    # Just before the horizon, pressure should be huge
    p_close = float(dirac_sea_pressure_proxy(vs, horizons[0] - 1e-3))
    assert p_close > 1e3


def test_pressure_divergence_rate_close_to_2():
    """Power-law divergence rate should be ~2 near the Cauchy horizon."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate
    horizons = cauchy_horizon_estimate(vs)
    # Heuristic: rate = log(P(eps)/P(2*eps)) / log(2) ~ 2 for 1/eps^2 scaling
    rate = chronology_horizon_pressure_divergence_rate(vs, float(horizons[0]), eps=1e-4)
    assert rate == pytest.approx(2.0, abs=0.5)
