"""Tests for acoustic-hawking."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from acoustic_hawking import (
    AnalogHorizonAnalyser,
    AnalogHorizonReport,
    SpectrumReport,
    SteinhauerBenchmark,
    benchmark_against_steinhauer_2019,
    compute_phonon_spectrum,
)


# ---------------------------------------------------------------------------
# AnalogHorizonAnalyser
# ---------------------------------------------------------------------------


def test_analyser_construction_supercritical():
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    assert ah.omega == 2.0
    assert ah.R == 1.0


def test_supercritical_has_acoustic_horizon():
    """A supercritical Tipler exterior (omega*R > 1/2) has F=0 surfaces
    in the search range — those are the acoustic horizons."""
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    r_h = ah.horizon()
    assert r_h is not None
    assert r_h > 1.0


def test_narrow_search_range_finds_no_horizon():
    """If the search range is tight enough to exclude all F=0 zeros,
    horizon() returns None."""
    # For (omega=2, R=1) the first horizon is at ~1.4; search [1.05,
    # 1.2] excludes it.
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0,
                                r_search_min=1.05, r_search_max=1.2)
    assert ah.horizon() is None


def test_horizon_F_is_zero():
    """At the reported horizon, F should be very small."""
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    r_h = ah.horizon()
    assert r_h is not None
    F_at_h = float(ah.vs.analytic_exterior_F(r_h))
    assert abs(F_at_h) < 1e-3


def test_surface_gravity_positive():
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    kappa = ah.surface_gravity()
    assert kappa > 0


def test_acoustic_equals_gravitational_T_H():
    """The Unruh identification c² - v² = F makes T_H_acoustic = T_H_gravitational."""
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    cmp_ = ah.compare_acoustic_vs_gravitational()
    assert cmp_["rel_diff"] < 1e-10


def test_line_element_at_supercritical_radius():
    """At a radius inside a CTC band (F<0), regime should be 'supersonic'."""
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    r_h = ah.horizon()
    info_inside = ah.line_element_at(r_h * 1.5)
    assert info_inside["regime"] in {"subsonic", "supersonic", "sonic"}


def test_ctc_supersonic_consistency():
    """In every F<0 region the analyser must report supersonic flow."""
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    audit = ah.audit_ctc_supersonic(n_samples=50)
    assert audit["ctc_supersonic_consistency"]


def test_report_returns_dataclass():
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    rep = ah.report(n_audit_samples=30)
    assert isinstance(rep, AnalogHorizonReport)
    assert rep.horizon_r is not None
    assert rep.T_hawking_acoustic is not None
    assert rep.ctc_supersonic_consistent


def test_report_narrow_range_no_T():
    """Narrow search range excluding all horizons: report has horizon_r=None."""
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0,
                                r_search_min=1.05, r_search_max=1.2)
    rep = ah.report(n_audit_samples=10)
    assert rep.horizon_r is None
    assert rep.T_hawking_acoustic is None


def test_surface_gravity_raises_when_no_horizon():
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0,
                                r_search_min=1.05, r_search_max=1.2)
    with pytest.raises(ValueError):
        ah.surface_gravity()


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


def test_compute_phonon_spectrum():
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    r_h = ah.horizon()
    spec = compute_phonon_spectrum(
        omega=2.0, R=1.0, r_horizon=r_h,
        omega_range=(0.01, 3.0), n_omega=50,
    )
    assert isinstance(spec, SpectrumReport)
    assert spec.horizon_r == pytest.approx(r_h)
    assert spec.T_hawking > 0
    assert spec.omegas.shape == (50,)
    assert spec.mean_phonon_number.shape == (50,)
    # n(ω) should be monotonically decreasing
    assert spec.mean_phonon_number[0] > spec.mean_phonon_number[-1]


def test_total_emission_power_positive():
    ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
    r_h = ah.horizon()
    spec = compute_phonon_spectrum(
        omega=2.0, R=1.0, r_horizon=r_h, n_omega=20, n_omega_power=100,
    )
    assert spec.total_emission_power > 0


# ---------------------------------------------------------------------------
# Steinhauer 2019 benchmark
# ---------------------------------------------------------------------------


def test_benchmark_returns_dataclass():
    """BEC params chosen to roughly match Steinhauer 2019 setup."""
    # Rb-87, typical Steinhauer values
    bench = benchmark_against_steinhauer_2019(
        omega=1000.0, R=1e-6, n_density=1e18,
        atom_mass=1.443e-25, a_scattering=5.3e-9,
    )
    assert isinstance(bench, SteinhauerBenchmark)
    assert bench.T_steinhauer_nK == pytest.approx(0.124)
    assert bench.T_steinhauer_uncertainty_nK == pytest.approx(0.012)
    assert "T_H_acoustic_nK" in bench.bec_predictions


def test_benchmark_sigma_deviation_finite():
    bench = benchmark_against_steinhauer_2019(
        omega=1000.0, R=1e-6, n_density=1e18,
        atom_mass=1.443e-25,
    )
    assert np.isfinite(bench.sigma_deviation)
    assert bench.sigma_deviation >= 0
