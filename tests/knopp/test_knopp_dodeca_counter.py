"""Tests for the counter-rotating pair (knopp_dodeca_counter)."""

import math

import pytest

from systrophe.knopp.knopp_dodeca_counter import (
    counter_report,
    mathieu_growth,
    net_drag_parameter,
    parametric_band,
    ring_m5,
    standing_flash_check,
    summarise_counter,
    trap_frequency,
    window_closed_for_counter_pair,
)


def test_counter_pair_cancels_drag():
    assert net_drag_parameter(1.2, -1.2) == 0.0
    assert net_drag_parameter(1.0, 1.0) == 1.0       # co-rotation keeps drag
    assert window_closed_for_counter_pair(2.0)
    with pytest.raises(ValueError):
        window_closed_for_counter_pair(-1.0)


def test_standing_flash():
    flash = standing_flash_check()
    assert flash["standing"]
    assert flash["single_phase_drift"] > 0.2          # rotor phase advances
    assert flash["pair_phase_drift_mod_pi"] < 0.02    # pair phase locked


def test_pair_amplitude_pulses():
    # the standing wave passes through (near) zero each half-beat
    amps = [ring_m5(p, counter=True)[0] for p in (0.0, 10.0, 20.0, 30.0)]
    assert min(amps) < 0.3 * max(amps)


def test_parametric_bands_above_grip():
    wt = trap_frequency()
    assert 100.0 < wt < 140.0
    assert math.isclose(parametric_band(1), 2 * wt / 5, rel_tol=1e-12)
    assert parametric_band(1) > 40.0                  # far above grip ~12
    with pytest.raises(ValueError):
        parametric_band(0)


def test_mathieu_stable_in_reach_unstable_at_band():
    assert mathieu_growth(5.0, T=4.0) < 0.3           # reachable: stable
    assert mathieu_growth(parametric_band(1), T=4.0) > 3.0   # band: explosive
    with pytest.raises(ValueError):
        mathieu_growth(0.0)


def test_counter_report():
    r = counter_report()
    assert r.drag_cancels and r.comms_exclusion_lifted
    assert r.standing_flash
    assert r.cargo_stable_in_reach
    assert r.growth_at_principal > 3.0
    assert r.catcher_verdict in ("novel_structure", "smooth", "uniform")
    text = summarise_counter(r)
    for tag in ("K1", "K2", "K3", "catcher"):
        assert tag in text
