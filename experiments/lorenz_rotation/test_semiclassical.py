"""Tests for the self-consistent semiclassical backreaction at the chronology horizon."""

import numpy as np

from semiclassical_backreaction import (
    breakdown_shell, breakdown_scaling, self_consistent_effective_horizon,
    chaotic_breakdown, A_CRIT,
)


def test_horizon_shrouded_for_any_hbar():
    """For any ell^2 > 0 the breakdown shell has positive width: the classical
    chronology horizon is shrouded by quantum backreaction."""
    for ell2 in (1e-5, 1e-3, 1e-1):
        s = breakdown_shell(1.0, ell2)
        assert s["eps_breakdown"] > 0
        assert s["protected"]
        assert s["r_effective_horizon"] < s["r_horizon"]


def test_breakdown_scales_linearly_with_planck_area():
    """eps_bd ~ ell^2 (the 1/(r-r_H) divergence gives power 1 in ell^2)."""
    sc = breakdown_scaling(1.0, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    assert 0.9 < sc["power_vs_ell2"] < 1.1


def test_shell_closes_in_classical_limit():
    """eps_bd -> 0 as ell^2 -> 0 (CTCs only in the strict classical limit)."""
    big = breakdown_shell(1.0, 1e-2)["eps_breakdown"]
    small = breakdown_shell(1.0, 1e-6)["eps_breakdown"]
    assert small < big
    assert small < 1e-3


def test_self_consistent_horizon_is_shrouded():
    it = self_consistent_effective_horizon(1.0, 1e-3)
    assert it["horizon_shrouded"]
    assert it["r_effective_horizon"] < it["r_horizon"]
    assert it["eps_breakdown"] > 0


def test_subcritical_has_no_horizon():
    """Below threshold there is no chronology horizon to shroud."""
    import pytest
    from systrophe.geometry.vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=0.4, R=1.0)
    assert not vs.is_supercritical()


def test_chaotic_breakdown_flickers():
    cb = chaotic_breakdown(ell2=1e-3, a_center=0.7, amp=0.18)
    assert 0.1 < cb["fraction_protected"] < 0.99   # shell toggles with a(t)
    assert cb["max_shell"] > 0
