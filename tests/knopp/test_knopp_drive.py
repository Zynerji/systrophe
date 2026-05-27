"""Tests for the Knopp Drive composite warp engineering bound."""

import math

import pytest

from systrophe.knopp.knopp_drive import (
    KnoppDriveBudget,
    KnoppDriveConfig,
    knopp_budget,
    knopp_drive_inside_band,
    novelty_scan,
    summarise_knopp_budget,
)


def test_default_budget_returns_KnoppDriveBudget():
    b = knopp_budget()
    assert isinstance(b, KnoppDriveBudget)


def test_budget_has_all_factors():
    b = knopp_budget()
    assert hasattr(b, "tipler_gate_factor")
    assert hasattr(b, "feedback_factor")
    assert hasattr(b, "horn_amplification")
    assert hasattr(b, "composite_E_neg")
    assert hasattr(b, "sustained_drive_power")
    assert hasattr(b, "pfenning_ford_compatible")
    assert hasattr(b, "steering_magnitude")


def test_feedback_factor_inverse_square_Q():
    """feedback_factor = 1/Q^2."""
    b1 = knopp_budget(Q=10.0)
    b2 = knopp_budget(Q=100.0)
    ratio = b1.feedback_factor / b2.feedback_factor
    assert abs(ratio - 100.0) < 0.5


def test_tipler_gate_zero_inside_band():
    """At r=1.5 (within first CTC band of omega=1, R=1), tipler tilt > 1
    so the gate factor is exactly zero."""
    b = knopp_budget(r_orbit=1.5)
    assert b.tipler_gate_factor == 0.0


def test_tipler_gate_outer_band():
    """At a radius beyond the CTC band the gate factor is positive."""
    b = knopp_budget(r_orbit=10.0)
    assert 0.0 <= b.tipler_gate_factor <= 1.0


def test_composite_E_neg_zero_inside_band():
    """Composite E_neg vanishes when tipler gate is zero (band)."""
    b = knopp_budget(r_orbit=1.5)
    assert b.composite_E_neg == 0.0


def test_steering_zero_at_epsilon_zero():
    b = knopp_budget(epsilon_horn=0.0)
    assert b.steering_magnitude < 1e-6


def test_steering_grows_with_epsilon():
    b1 = knopp_budget(epsilon_horn=0.0)
    b2 = knopp_budget(epsilon_horn=0.5)
    assert b2.steering_magnitude > b1.steering_magnitude


def test_horn_amplification_at_horn_side():
    """horn_amp = 1 + epsilon."""
    b = knopp_budget(epsilon_horn=0.3)
    assert abs(b.horn_amplification - 1.3) < 1e-12


def test_inside_band_helper():
    """At r=1.5 we're in the first CTC band; at r=3.0 we're between
    bands (tilt < 1)."""
    assert knopp_drive_inside_band(r_orbit=1.5) is True
    assert knopp_drive_inside_band(r_orbit=3.0) is False


def test_summary_string_contains_key_fields():
    b = knopp_budget()
    s = summarise_knopp_budget(b)
    assert "Knopp Drive" in s
    assert "E_neg" in s
    assert "P_drive" in s
    assert "tipler_gate" in s
    assert "steering" in s


def test_novelty_scan_returns_verdict():
    res = novelty_scan(r_orbit_range=(1.05, 8.0), n_r=10,
                        Q_range=(2.0, 20.0), n_Q=5,
                        epsilon_range=(0.0, 0.5), n_eps=5)
    assert "novelty_verdict" in res
    assert res["novelty_verdict"] in ("smooth", "uniform", "novel_structure")


def test_dataclass_config_roundtrip():
    cfg = KnoppDriveConfig(Q=50.0, epsilon_horn=0.4)
    b = knopp_budget(cfg)
    assert b.config.Q == 50.0
    assert b.config.epsilon_horn == 0.4
    assert abs(b.feedback_factor - 1.0 / 50.0 ** 2) < 1e-12
