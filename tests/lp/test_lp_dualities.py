"""Tests for LP dualities module."""

import math

import pytest

from systrophe.lp.lp_dualities import (
    duality_invariant_observables,
    novelty_scan,
    s_dual_a,
    t_dual_L_xi,
    t_dual_kk_to_winding,
    t_dual_orbit,
)
from systrophe.geometry.vanstockum import VanStockumInterior


def test_t_dual_inverse():
    L = 2.0
    L_dual = t_dual_L_xi(L)
    assert L_dual == pytest.approx(0.5, rel=1e-12)


def test_t_dual_idempotent_after_two():
    L = 3.0
    L1 = t_dual_L_xi(L)
    L2 = t_dual_L_xi(L1)
    assert L2 == pytest.approx(L, rel=1e-12)


def test_t_dual_invalid_L_raises():
    with pytest.raises(ValueError):
        t_dual_L_xi(0.0)


def test_t_dual_kk_to_winding_returns_dict():
    res = t_dual_kk_to_winding(n=1, L_xi=1.0)
    assert "kk_mass_original" in res
    assert "winding_mass_dual" in res


def test_s_dual_at_critical_fixed():
    """a = 1/2 is fixed under S-duality."""
    assert s_dual_a(0.5) == pytest.approx(1.0, rel=1e-12)
    # Wait — 1/(2*0.5) = 1, not 0.5. So fixed point of a -> 1/(2a) is a=1/sqrt(2).
    # Adjust: fixed point of a -> 1/(2a) satisfies a = 1/(2a) -> a^2 = 1/2 -> a = 1/sqrt(2).
    a_fixed = 1.0 / math.sqrt(2)
    assert s_dual_a(a_fixed) == pytest.approx(a_fixed, rel=1e-9)


def test_duality_invariant_at_critical():
    vs = VanStockumInterior(omega=1.0 / math.sqrt(2), R=1.0)
    res = duality_invariant_observables(vs)
    # a_S should be approximately equal to a at the fixed point
    assert res["a_S"] == pytest.approx(res["a"], rel=1e-9)


def test_duality_invariant_general_finite():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    res = duality_invariant_observables(vs)
    assert math.isfinite(res["a"])
    assert math.isfinite(res["a_S"])


def test_t_dual_orbit_returns_initial():
    L = 5.0
    orbit = t_dual_orbit(L, n_iter=2)
    assert orbit[2] == pytest.approx(L, rel=1e-12)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_a_values=10)
    assert "verdict" in res
