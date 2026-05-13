"""Tests for hadamard_modesum (Phase 2b)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from systrophe import (
    cauchy_horizon_estimate,
    fit_4d_quantity_at_horizon,
    hadamard_chronology_report,
    hadamard_modesum_novelty_scan,
    hadamard_subtraction_residual,
    hadamard_V_0,
    hadamard_V_1,
    kretschmann_scalar,
    phi_squared_wkb_partial_sum,
    squared_geodesic_distance_radial,
    trace_anomaly_4d_exact,
    trace_anomaly_via_V_1,
    wkb_mode_density,
    wkb_radial_mode,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture(scope="module")
def vs_super() -> VanStockumInterior:
    return VanStockumInterior(omega=2.0, R=1.0)


# ---------------------------------------------------------------------------
# Hadamard biparametrix coefficients
# ---------------------------------------------------------------------------


def test_V_0_vanishes_in_vacuum_conformal(vs_super):
    """Massless conformally-coupled scalar on vacuum LP gives V_0 = 0 exactly."""
    for r in (1.10, 1.20, 1.30):
        V_0 = hadamard_V_0(vs_super, r, mass=0.0, xi=1.0 / 6.0)
        # Should be tiny: numerical Ricci scalar at LP vacuum is FD-noise.
        assert abs(V_0) < 1e-2, f"r={r}: V_0 = {V_0} (expected ~0 in vacuum)"


def test_V_1_proportional_to_Kretschmann(vs_super):
    """V_1 = K/720 in vacuum conformal case."""
    for r in (1.10, 1.20, 1.30):
        V_1 = hadamard_V_1(vs_super, r)
        K = kretschmann_scalar(vs_super, r)
        assert math.isclose(V_1, K / 720.0, rel_tol=1e-10)


def test_trace_anomaly_via_V_1_agrees_with_point_splitting(vs_super):
    """2 V_1 / (8 pi^2) recovers the point_splitting trace anomaly K/(2880 pi^2)."""
    for r in (1.10, 1.20, 1.30):
        anom_new = trace_anomaly_via_V_1(vs_super, r)
        anom_old = trace_anomaly_4d_exact(vs_super, r)
        assert math.isclose(anom_new, anom_old, rel_tol=1e-10), (
            f"r={r}: V_1 path {anom_new} vs point_splitting {anom_old}"
        )


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------


def test_squared_geodesic_distance_basic(vs_super):
    sigma = squared_geodesic_distance_radial(vs_super, 1.20, 1.30)
    assert math.isclose(sigma, 0.5 * 0.01, rel_tol=1e-12)
    # Symmetric
    sigma_rev = squared_geodesic_distance_radial(vs_super, 1.30, 1.20)
    assert math.isclose(sigma, sigma_rev, rel_tol=1e-12)


def test_squared_geodesic_distance_zero_at_coincidence(vs_super):
    sigma = squared_geodesic_distance_radial(vs_super, 1.25, 1.25)
    assert sigma == 0.0


# ---------------------------------------------------------------------------
# WKB modes
# ---------------------------------------------------------------------------


def test_wkb_radial_mode_returns_complex(vs_super):
    """At a generic radius and frequency, mode amplitude is complex non-zero."""
    chi = wkb_radial_mode(vs_super, 1.20, omega=2.0, n_phi=0)
    assert isinstance(chi, complex)
    assert abs(chi) > 0


def test_wkb_mode_density_scales_with_inverse_sqrt_V(vs_super):
    """In the allowed region (V > 0), density ~ V^{-1/2}."""
    # high omega -> V_eff very positive
    d_high = wkb_mode_density(vs_super, 1.20, omega=10.0)
    d_low = wkb_mode_density(vs_super, 1.20, omega=2.0)
    # |chi|^2 = V^{-1/2}; higher V (higher omega^2) -> lower density
    assert d_high < d_low


# ---------------------------------------------------------------------------
# Mode-sum
# ---------------------------------------------------------------------------


def test_phi_squared_wkb_partial_sum_finite(vs_super):
    result = phi_squared_wkb_partial_sum(
        vs_super, 1.20, omega_min=0.1, omega_max=10.0, n_omega=40, n_phi_max=2,
    )
    assert np.isfinite(result["phi_squared_partial"])
    assert result["leading_UV_scaling"] > 0


def test_hadamard_subtraction_finite(vs_super):
    """Hadamard subtraction at a regular radius gives a finite real number."""
    val = hadamard_subtraction_residual(
        vs_super, 1.20, omega_max=8.0, n_omega=32, n_phi_max=2,
    )
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# 4D chronology-protection scan
# ---------------------------------------------------------------------------


def test_V_1_bounded_at_first_horizon(vs_super):
    """4D Hadamard V_1 ~ K/720 stays BOUNDED at the first Cauchy horizon.

    Confirms v0.10.0 result: the *local* 4D Hadamard biparametrix is
    finite at simple F-zeros (those are coordinate singularities).
    """
    horizons = cauchy_horizon_estimate(vs_super)
    r_H = float(horizons[0])
    fit = fit_4d_quantity_at_horizon(
        vs_super, r_H, quantity="V_1",
        n_samples=12, eps_min=1e-3, eps_max=2e-2,
    )
    # Bounded: |power| < 0.5
    assert abs(fit.power) < 0.5, f"V_1 power = {fit.power} (expected ~0)"
    assert not fit.diverges


def test_kretschmann_bounded_at_first_horizon(vs_super):
    horizons = cauchy_horizon_estimate(vs_super)
    r_H = float(horizons[0])
    fit = fit_4d_quantity_at_horizon(
        vs_super, r_H, quantity="kretschmann",
        n_samples=12, eps_min=1e-3, eps_max=2e-2,
    )
    assert abs(fit.power) < 0.5
    assert not fit.diverges


def test_hadamard_chronology_report(vs_super):
    rep = hadamard_chronology_report(
        vs_super, n_horizons=2, n_samples=10, eps_min=2e-3, eps_max=2e-2,
    )
    assert rep.n_horizons_scanned == 2
    # All V_1 fits should be bounded (powers near 0)
    powers = np.array([f.power for f in rep.V_1_fits])
    assert np.all(np.abs(powers) < 0.5), f"V_1 powers = {powers}"
    assert "BOUNDED" in rep.summary or "bounded" in rep.summary


def test_chronology_report_requires_supercritical():
    vs_sub = VanStockumInterior(omega=0.3, R=1.0)
    with pytest.raises(ValueError):
        hadamard_chronology_report(vs_sub)


# ---------------------------------------------------------------------------
# Catcher (always-on)
# ---------------------------------------------------------------------------


def test_hadamard_novelty_scan_runs(vs_super):
    result = hadamard_modesum_novelty_scan(vs_super, n_radii=20)
    assert result["verdict"] in {"smooth", "novel_structure", "uniform"}
    assert "lambda_2_at_radius" in result
