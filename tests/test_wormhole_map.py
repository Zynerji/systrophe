"""Tests for the full traversable-wormhole map (classical / Casimir / ER=EPR)."""

import math

import pytest

from systrophe.wormhole_map import (
    WormholeMapReport,
    dual_black_hole_mass_energy_J,
    erepr_entanglement_ebits,
    map_wormhole,
    morris_thorne_exotic_energy_J,
    summarise_wormhole_map,
)


def test_classical_exotic_energy_scales_linearly():
    """Morris-Thorne exotic energy ~ (c^4/G) r_t -> linear in throat radius."""
    e1 = morris_thorne_exotic_energy_J(1.0)
    e10 = morris_thorne_exotic_energy_J(10.0)
    assert e10 == pytest.approx(10.0 * e1, rel=1e-6)


def test_entanglement_scales_as_area():
    """ER=EPR entanglement (RT: S = A/4G_N) ~ r_t^2."""
    s1 = erepr_entanglement_ebits(1.0)
    s10 = erepr_entanglement_ebits(10.0)
    assert s10 == pytest.approx(100.0 * s1, rel=1e-6)


def test_metre_throat_needs_bekenstein_scale_entanglement():
    """A metre throat requires ~1e70 ebits -- a black-hole's worth."""
    s = erepr_entanglement_ebits(1.0)
    assert 1e69 < s < 1e71


def test_energy_entanglement_duality_is_order_unity():
    """Classical exotic energy and the ER=EPR dual black-hole mass-energy are
    the SAME wall (ratio ~ O(1)), at every scale."""
    for rt in (1e-6, 1.0, 100.0):
        r = map_wormhole(r_throat_m=rt)
        assert 0.1 < r.energy_entanglement_duality_ratio < 10.0


def test_dual_black_hole_mass_is_schwarzschild():
    # M c^2 = c^4 r_t / (2 G)
    from systrophe.knopp_ratchet import _C_SI, _G_SI
    assert dual_black_hole_mass_energy_J(1.0) == pytest.approx(
        _C_SI ** 4 * 1.0 / (2.0 * _G_SI), rel=1e-9)


def test_wormhole_does_not_beat_the_wall():
    """Headline: no route transports macroscopic matter below the wall, though
    the quantum route does transport information."""
    r = map_wormhole(r_throat_m=1.0)
    assert isinstance(r, WormholeMapReport)
    assert r.beats_the_wall is False
    assert r.transports_macroscopic_matter is False
    assert r.transports_information is True
    assert r.residual_oom > 30.0


def test_summary():
    s = summarise_wormhole_map(map_wormhole())
    assert "BEATS_WALL=False" in s and "ER=EPR" in s
