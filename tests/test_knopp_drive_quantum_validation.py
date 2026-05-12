"""Tests for the Knopp Drive quantum back-reaction validation."""

import math

import numpy as np

from systrophe.knopp_drive_quantum_validation import (
    KnoppQuantumPoint,
    KnoppQuantumValidationReport,
    knopp_quantum_point,
    pfenning_ford_bound_local,
    summarise_quantum_validation,
    validate_knopp_drive_quantum,
)
from systrophe.vanstockum import VanStockumInterior


def test_pfenning_ford_bound_positive():
    """The Pfenning-Ford lower bound is strictly positive."""
    b = pfenning_ford_bound_local(sigma=4.0)
    assert b > 0


def test_pfenning_ford_inverse_sigma_squared():
    """The bound scales as 1/sigma^2."""
    b1 = pfenning_ford_bound_local(sigma=2.0)
    b2 = pfenning_ford_bound_local(sigma=4.0)
    ratio = b1 / b2
    assert 3.5 < ratio < 4.5


def test_quantum_point_returns_dataclass():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    p = knopp_quantum_point(vs, r=1.5)
    assert isinstance(p, KnoppQuantumPoint)


def test_quantum_point_inside_band_has_zero_gate():
    """At r=1.5 we are inside the first CTC band (tilt > 1)."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    p = knopp_quantum_point(vs, r=1.5)
    assert p.tipler_gate_factor == 0.0


def test_quantum_point_T_tt_finite_inside_band():
    """T_tt is finite inside the CTC band."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    p = knopp_quantum_point(vs, r=1.5)
    assert math.isfinite(p.T_tt)
    assert math.isfinite(p.T_phi_phi)
    assert math.isfinite(p.T_zz)
    assert math.isfinite(p.T_rr)


def test_quantum_point_kretschmann_positive():
    """The Kretschmann scalar is non-negative in the supercritical
    exterior."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    p = knopp_quantum_point(vs, r=1.5)
    assert p.kretschmann >= 0


def test_quantum_point_bounded_inside_band():
    """The is_bounded flag is True inside the CTC band."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    p = knopp_quantum_point(vs, r=1.5)
    assert p.is_bounded is True


def test_validate_returns_report():
    """End-to-end validation returns a complete report."""
    report = validate_knopp_drive_quantum(
        r_range=(1.05, 5.0), n_r=10,
    )
    assert isinstance(report, KnoppQuantumValidationReport)


def test_validate_contains_finite_points():
    """Some points should be finite (those inside the CTC band)."""
    report = validate_knopp_drive_quantum(
        r_range=(1.05, 12.0), n_r=20,
    )
    finite_points = [p for p in report.points if p.is_finite]
    assert len(finite_points) > 0


def test_validate_contrast_ratio_is_finite():
    """The renormalised vs nonband contrast ratio is a finite number."""
    report = validate_knopp_drive_quantum(
        r_range=(1.05, 12.0), n_r=20,
    )
    assert math.isfinite(report.renormalised_vs_classical_ratio)


def test_validate_pfenning_ford_consistent():
    """The Pfenning-Ford consistency flag is True for a normal sweep."""
    report = validate_knopp_drive_quantum(
        r_range=(1.05, 12.0), n_r=20,
    )
    assert report.pfenning_ford_consistent is True


def test_validate_catcher_flags_chronology_horizon():
    """The catcher should flag a sharp transition at the chronology
    horizon (where F crosses zero), which separates the inside-band
    region from the rest."""
    report = validate_knopp_drive_quantum(
        r_range=(1.05, 12.0), n_r=30,
    )
    # The sweep should produce at least one sharp transition
    assert report.novelty_n_sharp >= 1


def test_summarise_returns_string():
    report = validate_knopp_drive_quantum(r_range=(1.05, 5.0), n_r=10)
    s = summarise_quantum_validation(report)
    assert isinstance(s, str)
    assert "renormalised" in s.lower()
