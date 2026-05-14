"""Tests for guderley-shock."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from guderley_shock import (
    GuderleyExponent,
    GuderleyProfile,
    ShockHorizonComparison,
    compare_to_cauchy_horizon,
    compute_guderley_exponent,
    density_power_at_focus,
    integrate_post_shock_profile,
)
from systrophe.vanstockum import VanStockumInterior


# ---------------------------------------------------------------------------
# Guderley exponent: literature reference
# ---------------------------------------------------------------------------


def test_guderley_exponent_spherical_53():
    """β ≈ 0.6884 for γ=5/3, n=3 (Guderley 1942 / Lazarus 1981)."""
    e = compute_guderley_exponent(gamma=5.0 / 3.0, n=3)
    assert isinstance(e, GuderleyExponent)
    assert e.method == "literature"
    assert e.is_literature_match
    assert e.beta == pytest.approx(0.688376, abs=1e-6)


def test_guderley_exponent_spherical_75():
    """β ≈ 0.7172 for γ=7/5, n=3."""
    e = compute_guderley_exponent(gamma=7.0 / 5.0, n=3)
    assert e.beta == pytest.approx(0.717174, abs=1e-6)


def test_guderley_exponent_cylindrical_53():
    """β ≈ 0.8156 for γ=5/3, n=2."""
    e = compute_guderley_exponent(gamma=5.0 / 3.0, n=2)
    assert e.beta == pytest.approx(0.815625, abs=1e-6)


def test_guderley_exponent_planar():
    """β = 1 for n=1 (planar) — shock advances linearly in time."""
    e = compute_guderley_exponent(gamma=5.0 / 3.0, n=1)
    assert e.beta == pytest.approx(1.0)


def test_guderley_exponent_invalid_gamma_raises():
    with pytest.raises(ValueError):
        compute_guderley_exponent(gamma=1.0, n=3)


def test_guderley_exponent_invalid_n_raises():
    with pytest.raises(ValueError):
        compute_guderley_exponent(gamma=5.0 / 3.0, n=4)


# ---------------------------------------------------------------------------
# Density-power asymptotic
# ---------------------------------------------------------------------------


def test_density_power_53_spherical():
    """ρ-divergence power for γ=5/3, n=3 ≈ -0.9054."""
    p = density_power_at_focus(gamma=5.0 / 3.0, n=3)
    expected = -2.0 * (1.0 - 0.688376) / 0.688376
    assert p == pytest.approx(expected, abs=1e-6)
    assert -0.92 < p < -0.89


def test_density_power_75_spherical():
    """ρ-divergence power for γ=7/5, n=3 ≈ -0.7889."""
    p = density_power_at_focus(gamma=7.0 / 5.0, n=3)
    assert -0.80 < p < -0.78


def test_density_power_more_negative_for_softer_eos():
    """Smaller β → more negative power."""
    p_53 = density_power_at_focus(gamma=5.0 / 3.0, n=3)
    p_75 = density_power_at_focus(gamma=7.0 / 5.0, n=3)
    # γ=5/3 has smaller β (~0.69 vs ~0.72) → more negative power.
    assert p_53 < p_75


# ---------------------------------------------------------------------------
# Post-shock profile integration (deliberately NotImplemented; document)
# ---------------------------------------------------------------------------


def test_profile_integration_raises_not_implemented():
    """The full Guderley profile integrator is intentionally not implemented.

    Naive forwards-from-shock integration blows up at the singular
    saddle. A correct integrator uses Lazarus 1981's
    backwards-from-sonic procedure, which is out of scope for this
    tool. The asymptotic-power diagnostic (density_power_at_focus) is
    a sufficient surrogate.
    """
    with pytest.raises(NotImplementedError):
        integrate_post_shock_profile(gamma=5.0 / 3.0, n=3,
                                      xi_min=0.5, n_points=20)


def test_post_shock_jump_values():
    """Strong-shock Rankine-Hugoniot jump values are returned."""
    from guderley_shock.shock import post_shock_jump
    V_1, G_1, C2_1 = post_shock_jump(5.0 / 3.0)
    assert V_1 == pytest.approx(0.75)
    assert G_1 == pytest.approx(4.0)
    assert C2_1 == pytest.approx(2.0 * (5.0/3.0) * (2.0/3.0) / (8.0/3.0)**2)


def test_compute_exponent_non_literature_raises():
    """No literature value for (n=3, γ=2.0) → tool refuses to guess."""
    with pytest.raises(NotImplementedError):
        compute_guderley_exponent(gamma=2.0, n=3)


# ---------------------------------------------------------------------------
# Compare to Cauchy horizon (QFTCS)
# ---------------------------------------------------------------------------


def test_compare_to_cauchy_horizon_returns_comparison():
    vs = VanStockumInterior(omega=2.0, R=1.0)  # canonical supercritical
    cmp_ = compare_to_cauchy_horizon(vs, gamma=5.0 / 3.0, n=3)
    assert isinstance(cmp_, ShockHorizonComparison)
    assert cmp_.gamma == pytest.approx(5.0 / 3.0)
    assert cmp_.n == 3
    assert cmp_.beta == pytest.approx(0.688376, abs=1e-6)


def test_qftcs_power_is_minus_one_universally():
    """Phase 2a's headline result: <T_tt>_Boulware ~ -1 power at any Cauchy horizon."""
    vs = VanStockumInterior(omega=2.0, R=1.0)
    cmp_ = compare_to_cauchy_horizon(vs, gamma=5.0 / 3.0, n=3)
    assert cmp_.qftcs_T_tt_power == pytest.approx(-1.0, abs=0.05)


def test_residual_is_about_0_1_for_gamma_53():
    """Empirical fact: |p_guderley - p_qft| ≈ 0.1 for γ=5/3, n=3."""
    vs = VanStockumInterior(omega=2.0, R=1.0)
    cmp_ = compare_to_cauchy_horizon(vs, gamma=5.0 / 3.0, n=3)
    # power_guderley ≈ -0.905, power_qftcs ≈ -1.000 → residual ≈ 0.095
    assert 0.05 < cmp_.absolute_residual < 0.20


def test_compare_raises_on_subcritical():
    """Subcritical vs has no Cauchy horizons; comparison should refuse."""
    vs = VanStockumInterior(omega=0.4, R=1.0)  # a=0.4 < 1/2
    with pytest.raises(ValueError):
        compare_to_cauchy_horizon(vs, gamma=5.0 / 3.0, n=3)


def test_compare_horizon_index_validation():
    """Asking for the 100th horizon should raise."""
    vs = VanStockumInterior(omega=2.0, R=1.0)
    with pytest.raises(ValueError):
        compare_to_cauchy_horizon(vs, gamma=5.0 / 3.0, n=3, n_horizon=100)
