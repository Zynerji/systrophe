"""Tests for the LQG + holographic-complexity probes of the Toroidal Knopp Drive."""

import math

import pytest

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_quantum import (
    ToroidalBandComplexity,
    ToroidalBandLQG,
    ToroidalQuantumDiagnostics,
    action_complexity_proxy,
    lloyd_growth_rate,
    lqg_area_spectrum_at_j,
    planck_area_unit,
    summarise_toroidal_quantum,
    toroidal_band_boundary_area,
    toroidal_band_complexity,
    toroidal_band_lqg_discretization,
    toroidal_band_volume,
    toroidal_quantum_diagnostics,
    volume_complexity_proxy,
)


# ----- LQG constants ------------------------------------------------------


def test_planck_area_unit_value():
    expected = 8.0 * math.pi * 0.2375 * 1.0 ** 2
    assert planck_area_unit() == pytest.approx(expected, rel=1e-12)


def test_lqg_area_spectrum_zero_at_j_zero():
    assert lqg_area_spectrum_at_j(0.0) == 0.0


def test_lqg_area_spectrum_monotone_in_j():
    A_vals = [lqg_area_spectrum_at_j(float(j)) for j in [0, 0.5, 1, 2, 5, 10]]
    diffs = [A_vals[i + 1] - A_vals[i] for i in range(len(A_vals) - 1)]
    assert all(d > 0.0 for d in diffs)


def test_lqg_area_spectrum_rejects_negative_j():
    with pytest.raises(ValueError):
        lqg_area_spectrum_at_j(-0.5)


# ----- toroidal band geometry --------------------------------------------


@pytest.fixture
def binary_in_band():
    return EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)


@pytest.fixture
def binary_subcritical():
    return EffectiveToroidalKerrBinary(M=1.0, d=10.0, chi=1.0)


def test_boundary_area_positive(binary_in_band):
    A = toroidal_band_boundary_area(binary_in_band, rho=1.0)
    assert A > 0.0
    assert math.isfinite(A)


def test_boundary_area_increases_with_rho(binary_in_band):
    # In the LT-weak-field regime, A_proper ~ 2 pi rho L_z grows with rho.
    A_lo = toroidal_band_boundary_area(binary_in_band, rho=1.0)
    A_hi = toroidal_band_boundary_area(binary_in_band, rho=3.0)
    assert A_hi > A_lo


def test_boundary_area_rejects_bad_inputs(binary_in_band):
    with pytest.raises(ValueError):
        toroidal_band_boundary_area(binary_in_band, rho=-0.5)
    with pytest.raises(ValueError):
        toroidal_band_boundary_area(binary_in_band, rho=1.0, L_z=-1.0)


def test_band_volume_positive_when_band_exists(binary_in_band):
    V = toroidal_band_volume(binary_in_band)
    assert V > 0.0
    assert math.isfinite(V)


def test_band_volume_zero_when_no_band(binary_subcritical):
    V = toroidal_band_volume(binary_subcritical)
    assert V == 0.0


# ----- LQG discretization -------------------------------------------------


def test_lqg_discretization_returns_dataclass(binary_in_band):
    out = toroidal_band_lqg_discretization(binary_in_band)
    assert isinstance(out, ToroidalBandLQG)


def test_lqg_spins_positive_when_band_exists(binary_in_band):
    out = toroidal_band_lqg_discretization(binary_in_band)
    assert out.j_inner > 0.0
    assert out.j_outer > 0.0
    # Outer boundary has larger area -> larger j
    assert out.j_outer > out.j_inner


def test_lqg_relative_error_small(binary_in_band):
    out = toroidal_band_lqg_discretization(binary_in_band)
    # Rounded half-integer j should match within a few percent for
    # band radii of a few Planck units.
    assert out.rel_error_inner < 0.1
    assert out.rel_error_outer < 0.1


def test_lqg_no_band_returns_trivial_protection(binary_subcritical):
    out = toroidal_band_lqg_discretization(binary_subcritical)
    assert out.rho_inner is None
    assert out.rho_outer is None
    assert out.chronology_protected_by_discreteness is True


def test_lqg_discreteness_threshold_below_band_width(binary_in_band):
    # Default threshold = 1 Planck unit; band width ~ 2.9 here, so NOT
    # protected by discreteness alone.
    out = toroidal_band_lqg_discretization(
        binary_in_band, discreteness_threshold_planck=1.0,
    )
    assert out.band_width_in_planck_units > 1.0
    assert out.chronology_protected_by_discreteness is False


def test_lqg_discreteness_threshold_above_band_width(binary_in_band):
    # A huge threshold -> the band is judged "unresolvable."
    out = toroidal_band_lqg_discretization(
        binary_in_band, discreteness_threshold_planck=100.0,
    )
    assert out.chronology_protected_by_discreteness is True


# ----- holographic complexity --------------------------------------------


def test_cv_proxy_positive_when_band_exists(binary_in_band):
    cv = volume_complexity_proxy(binary_in_band)
    assert cv > 0.0


def test_cv_proxy_zero_when_no_band(binary_subcritical):
    cv = volume_complexity_proxy(binary_subcritical)
    assert cv == 0.0


def test_ca_proxy_positive_when_band_exists(binary_in_band):
    ca = action_complexity_proxy(binary_in_band)
    assert ca > 0.0


def test_ca_proxy_zero_when_no_band(binary_subcritical):
    ca = action_complexity_proxy(binary_subcritical)
    assert ca == 0.0


def test_lloyd_growth_rate_scales_with_M():
    b1 = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    b2 = EffectiveToroidalKerrBinary(M=2.0, d=4.0)
    r1 = lloyd_growth_rate(b1)
    r2 = lloyd_growth_rate(b2)
    # dC/dt_max = 2 E / pi = 4 M / pi -> linear in M
    assert r2 == pytest.approx(2.0 * r1, rel=1e-12)


def test_lloyd_growth_rate_formula():
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    assert lloyd_growth_rate(b, hbar=1.0) == pytest.approx(
        2.0 * (2.0 * 1.0) / math.pi, rel=1e-12,
    )


def test_complexity_returns_dataclass(binary_in_band):
    out = toroidal_band_complexity(binary_in_band)
    assert isinstance(out, ToroidalBandComplexity)
    assert out.band_volume > 0.0
    assert out.cv_proxy > 0.0
    assert out.lloyd_growth_rate_max > 0.0
    assert out.cv_traversal_time is not None and out.cv_traversal_time > 0.0


def test_complexity_no_band(binary_subcritical):
    out = toroidal_band_complexity(binary_subcritical)
    assert out.band_volume == 0.0
    assert out.cv_proxy == 0.0
    assert out.ca_proxy == 0.0
    assert out.cv_traversal_time is None


def test_volume_complexity_rejects_bad_ell_AdS(binary_in_band):
    with pytest.raises(ValueError):
        volume_complexity_proxy(binary_in_band, ell_AdS=-1.0)


# ----- combined diagnostics ----------------------------------------------


def test_diagnostics_in_band(binary_in_band):
    d = toroidal_quantum_diagnostics(binary_in_band)
    assert isinstance(d, ToroidalQuantumDiagnostics)
    assert d.has_band is True
    assert d.lqg.rho_inner is not None
    assert d.complexity.band_volume > 0.0


def test_diagnostics_no_band(binary_subcritical):
    d = toroidal_quantum_diagnostics(binary_subcritical)
    assert d.has_band is False
    assert d.lqg.chronology_protected_by_discreteness is True
    assert d.complexity.band_volume == 0.0


def test_summary_string(binary_in_band, binary_subcritical):
    d_in = toroidal_quantum_diagnostics(binary_in_band)
    d_out = toroidal_quantum_diagnostics(binary_subcritical)
    s_in = summarise_toroidal_quantum(d_in)
    s_out = summarise_toroidal_quantum(d_out)
    assert "Toroidal CTC band" in s_in
    assert "LQG boundary" in s_in
    assert "Holographic complexity" in s_in
    assert "NONE" in s_out
