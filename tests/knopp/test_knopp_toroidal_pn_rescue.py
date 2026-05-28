"""Tests for the PN rescue path of the Toroidal Knopp binary."""

import math

import pytest

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_pn_rescue import (
    PNRescueReport,
    PNRescueVerdict,
    pn_luminosity_factor,
    pn_merger_time,
    pn_rescue_report,
    summarise_pn_rescue,
)


@pytest.fixture
def binary_tight():
    return EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)


@pytest.fixture
def binary_wide():
    return EffectiveToroidalKerrBinary(M=1.0, d=50.0, chi=1.0)


# ----- pn_luminosity_factor ----------------------------------------------


def test_pn_factor_at_order_zero_is_unity(binary_tight):
    assert pn_luminosity_factor(binary_tight, pn_order=0.0) == 1.0


def test_pn_factor_rejects_unknown_order(binary_tight):
    with pytest.raises(ValueError):
        pn_luminosity_factor(binary_tight, pn_order=3.0)


def test_pn_factor_breaks_down_at_strong_field(binary_tight):
    """At d = 2M (M/r = 0.5), the 1PN coefficient -3.895 makes the
    factor negative -- PN series has lost convergence."""
    fac = pn_luminosity_factor(binary_tight, pn_order=1.0)
    assert fac < 0.0


def test_pn_factor_positive_at_weak_field(binary_wide):
    """At d = 50M (M/r = 0.02), the PN series remains positive."""
    for p in (0.0, 1.0, 1.5, 2.0):
        assert pn_luminosity_factor(binary_wide, pn_order=p) > 0.0


def test_pn_factor_spin_orbit_zero_for_antiparallel(binary_wide):
    """Antiparallel maximal spins -> S_1 + S_2 = 0 -> SO term vanishes.
    The 1PN and 1.5PN factors should therefore be equal."""
    f_1 = pn_luminosity_factor(binary_wide, pn_order=1.0)
    f_1p5 = pn_luminosity_factor(binary_wide, pn_order=1.5)
    assert f_1 == pytest.approx(f_1p5, rel=1e-12)


# ----- pn_merger_time ----------------------------------------------------


def test_pn_merger_time_order_zero_matches_peters(binary_wide):
    from systrophe.knopp.knopp_toroidal_stability import time_to_merger
    assert pn_merger_time(binary_wide, pn_order=0.0) == pytest.approx(
        time_to_merger(binary_wide), rel=1e-12,
    )


def test_pn_merger_time_inf_when_pn_series_breaks(binary_tight):
    """Negative PN factor -> sentinel infinity."""
    t = pn_merger_time(binary_tight, pn_order=1.0)
    assert math.isinf(t)


def test_pn_merger_time_increases_with_PN_at_weak_field(binary_wide):
    """At weak field, PN corrections reduce dE/dt slightly -> longer t_merge."""
    t0 = pn_merger_time(binary_wide, pn_order=0.0)
    t2 = pn_merger_time(binary_wide, pn_order=2.0)
    assert t2 > t0


# ----- pn_rescue_report --------------------------------------------------


def test_report_returns_dataclass(binary_tight):
    r = pn_rescue_report(binary_tight)
    assert isinstance(r, PNRescueReport)
    assert len(r.verdicts) == 4
    for v in r.verdicts:
        assert isinstance(v, PNRescueVerdict)


def test_report_strong_field_pn_breakdown(binary_tight):
    """At d=2M, all PN orders >= 1 should be unreliable."""
    r = pn_rescue_report(binary_tight)
    assert r.verdicts[0].pn_series_reliable is True
    assert r.verdicts[1].pn_series_reliable is False
    assert r.verdicts[2].pn_series_reliable is False
    assert r.verdicts[3].pn_series_reliable is False
    # And rescue does NOT succeed (since PN breakdown disqualifies)
    assert r.rescue_succeeds is False


def test_report_weak_field_pn_reliable(binary_wide):
    """At d=50M all PN orders are reliable."""
    r = pn_rescue_report(binary_wide)
    for v in r.verdicts:
        assert v.pn_series_reliable is True


def test_report_n_orbits_leading_matches_stability(binary_tight):
    """The leading-order n_orbits matches the stability module."""
    from systrophe.knopp.knopp_toroidal_stability import stability_report
    s = stability_report(binary_tight)
    r = pn_rescue_report(binary_tight)
    # leading-order PN factor = 1, so n_orbits at PN=0 = n_orbits leading.
    # Stability module uses corrected_merger_time (with SS correction),
    # so its n_orbits is SLIGHTLY DIFFERENT. But the same order of mag.
    assert abs(r.n_orbits_leading - s.band_lifetime_vs_ctc_window) < 0.05


def test_summary_string_tight_falsified(binary_tight):
    r = pn_rescue_report(binary_tight)
    s = summarise_pn_rescue(r)
    assert "PN SERIES BREAKS DOWN" in s
    assert "FALSIFIED" in s


def test_summary_string_wide_open(binary_wide):
    r = pn_rescue_report(binary_wide)
    s = summarise_pn_rescue(r)
    assert "OPEN at some reliable PN order" in s
