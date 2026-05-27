"""Tests for CTC tunneling module."""

import math

import pytest

from systrophe.ctc.ctc_tunneling import (
    CTCTunnelingRate,
    attempt_frequency,
    effective_radial_potential,
    escape_to_chronology_safe_region,
    multi_band_tunneling_diagram,
    tunneling_action,
    tunneling_rate,
    tunneling_resonance_locus,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_effective_radial_potential_finite(vs_super):
    V = effective_radial_potential(vs_super, r=1.5)
    assert math.isfinite(V)


def test_effective_radial_potential_grows_near_CH(vs_super):
    # First CH ~ 1.83. V grows as 1/|F| diverges
    V_far = effective_radial_potential(vs_super, r=1.5)
    V_near = effective_radial_potential(vs_super, r=1.82)
    assert V_near > V_far


def test_tunneling_action_non_negative(vs_super):
    S = tunneling_action(vs_super, r_inner=1.5, r_outer=2.0)
    assert S >= 0


def test_attempt_frequency_positive(vs_super):
    nu = attempt_frequency(vs_super, r_inner=1.83, r_outer=11.23)
    assert nu > 0


def test_tunneling_rate_returns_CTCTunnelingRate(vs_super):
    rate = tunneling_rate(vs_super, r_inner=1.83, r_outer=11.23)
    assert isinstance(rate, CTCTunnelingRate)


def test_tunneling_rate_non_negative(vs_super):
    rate = tunneling_rate(vs_super, r_inner=1.83, r_outer=11.23)
    assert rate.escape_rate >= 0
    assert rate.half_life >= 0


def test_tunneling_rate_decreases_with_wider_barrier(vs_super):
    """Wider barrier -> smaller escape rate."""
    rate1 = tunneling_rate(vs_super, r_inner=2.0, r_outer=3.0)
    rate2 = tunneling_rate(vs_super, r_inner=2.0, r_outer=8.0)
    assert rate2.escape_rate <= rate1.escape_rate + 1e-30


def test_escape_to_safe_region_returns_dict(vs_super):
    res = escape_to_chronology_safe_region(vs_super, r_inner_CH=1.83, r_safe=1.5)
    assert "escape_rate" in res
    assert "half_life" in res


def test_escape_invalid_safe_raises(vs_super):
    """r_safe must be < r_inner_CH AND F(r_safe) > 0."""
    with pytest.raises(ValueError):
        escape_to_chronology_safe_region(vs_super, r_inner_CH=1.5,
                                           r_safe=3.35)


def test_escape_safe_not_outside_CH_raises(vs_super):
    """r_safe must be in chronology-safe region."""
    # r=3.35 has F<0; should fail.
    with pytest.raises(ValueError):
        escape_to_chronology_safe_region(vs_super, r_inner_CH=5.0,
                                           r_safe=3.35)


def test_multi_band_returns_list(vs_super):
    bands = multi_band_tunneling_diagram(vs_super, n_bands=2)
    assert isinstance(bands, list)


def test_multi_band_subcritical_empty(vs_sub):
    bands = multi_band_tunneling_diagram(vs_sub)
    assert bands == []


def test_multi_band_entries_have_required_keys(vs_super):
    bands = multi_band_tunneling_diagram(vs_super, n_bands=2)
    for b in bands:
        assert "band_index" in b
        assert "barrier_action" in b
        assert "escape_rate" in b
        assert "half_life" in b


def test_tunneling_resonance_locus_returns_dict(vs_super):
    res = tunneling_resonance_locus(vs_super, n_energies=10)
    assert "energies" in res
    assert "rates" in res


def test_tunneling_resonance_rates_length_matches(vs_super):
    res = tunneling_resonance_locus(vs_super, n_energies=15)
    assert len(res["energies"]) == len(res["rates"]) == 15
