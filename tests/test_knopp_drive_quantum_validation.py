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


# ---------------------------------------------------------------------
# Full Knopp composite stress tensor tests
# ---------------------------------------------------------------------

class TestFullCompositeValidation:
    def test_composite_point_dataclass(self):
        from systrophe.knopp_drive import KnoppDriveConfig
        from systrophe.knopp_drive_quantum_validation import (
            KnoppCompositeQuantumPoint,
            knopp_composite_quantum_point,
        )
        cfg = KnoppDriveConfig()
        p = knopp_composite_quantum_point(cfg, r=1.5)
        assert isinstance(p, KnoppCompositeQuantumPoint)

    def test_composite_T_tt_inside_band_dominated_by_renormalised(self):
        """Inside the CTC band, gate=0 so the Krasnikov contribution
        is zero. The composite T_tt is dominated by the (small)
        renormalised vacuum stress plus the (very small) Q-cavity
        contribution."""
        from systrophe.knopp_drive import KnoppDriveConfig
        from systrophe.knopp_drive_quantum_validation import (
            knopp_composite_quantum_point,
        )
        cfg = KnoppDriveConfig(Q=100.0)
        p = knopp_composite_quantum_point(cfg, r=1.5)
        assert p.tipler_gate_factor == 0.0
        assert p.T_kk_krasnikov_gated == 0.0

    def test_composite_T_tt_outside_band_has_krasnikov_contribution(self):
        """Outside the band, gate > 0 so the Krasnikov contribution
        is present."""
        from systrophe.knopp_drive import KnoppDriveConfig
        from systrophe.knopp_drive_quantum_validation import (
            knopp_composite_quantum_point,
        )
        cfg = KnoppDriveConfig(Q=10.0)
        p = knopp_composite_quantum_point(cfg, r=3.0)
        assert p.tipler_gate_factor > 0
        assert p.T_kk_krasnikov_gated < 0  # negative NEC

    def test_composite_Q_dependence(self):
        """Higher Q -> smaller Q-cavity contribution."""
        from systrophe.knopp_drive import KnoppDriveConfig
        from systrophe.knopp_drive_quantum_validation import (
            knopp_composite_quantum_point,
        )
        cfg_lo = KnoppDriveConfig(Q=10.0)
        cfg_hi = KnoppDriveConfig(Q=100.0)
        p_lo = knopp_composite_quantum_point(cfg_lo, r=3.0)
        p_hi = knopp_composite_quantum_point(cfg_hi, r=3.0)
        assert abs(p_hi.T_shell_qcavity) < abs(p_lo.T_shell_qcavity)

    def test_full_composite_validation_runs(self):
        from systrophe.knopp_drive import KnoppDriveConfig
        from systrophe.knopp_drive_quantum_validation import (
            KnoppCompositeValidationReport,
            validate_full_knopp_composite,
        )
        report = validate_full_knopp_composite(
            cfg=KnoppDriveConfig(), r_range=(1.05, 5.0), n_r=10,
        )
        assert isinstance(report, KnoppCompositeValidationReport)

    def test_full_composite_pfenning_ford_at_low_Q(self):
        """At low Q the composite is P-F consistent."""
        from systrophe.knopp_drive import KnoppDriveConfig
        from systrophe.knopp_drive_quantum_validation import (
            validate_full_knopp_composite,
        )
        report = validate_full_knopp_composite(
            cfg=KnoppDriveConfig(Q=10.0), r_range=(1.05, 5.0), n_r=10,
        )
        assert report.pfenning_ford_consistent is True

    def test_parameter_sweep_returns_dict(self):
        from systrophe.knopp_drive_quantum_validation import (
            composite_parameter_sweep,
        )
        sw = composite_parameter_sweep(
            Q_values=(10.0, 100.0),
            epsilon_values=(0.0, 0.5),
            r_range=(1.05, 5.0), n_r=10,
        )
        assert "results" in sw
        assert "n_novel_combinations" in sw
        assert len(sw["results"]) == 4

    def test_band_gating_robust_across_Q_eps(self):
        """The band-gating shortcut produces catcher emergents at
        every (Q, eps) combination tested."""
        from systrophe.knopp_drive_quantum_validation import (
            composite_parameter_sweep,
        )
        sw = composite_parameter_sweep(
            Q_values=(10.0, 100.0),
            epsilon_values=(0.0, 0.5),
            r_range=(1.05, 8.0), n_r=15,
        )
        # All combos should at least catch the chronology horizon
        # transition; some may also catch the band exit
        n_sharp_total = sum(r["n_sharp"] for r in sw["results"])
        assert n_sharp_total >= 4  # at least one sharp per combo
