"""Tests for through-the-wall comb communication (knopp_dodeca_comb_channel)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_dodeca_comb_channel import (
    WALL_45,
    WALL_EQUATOR,
    channel_report,
    clean_channel,
    comb_bit_rate,
    lines_are_5omega_selected,
    max_clean_bit_rate,
    summarise_channel,
    wall_comb_spectrum,
)


def test_wall_sees_the_comb():
    s = wall_comb_spectrum(WALL_45, 1.0, 0.0)
    assert s["modulation_depth"] > 0.15          # readable imprint
    assert lines_are_5omega_selected(s["lines_per_omega"])


def test_tilt_keying_readable_at_wall():
    s = wall_comb_spectrum(WALL_45, 1.0, 4.0)
    assert not lines_are_5omega_selected(s["lines_per_omega"])
    with pytest.raises(ValueError):
        wall_comb_spectrum(WALL_45, 0.0)


def test_equator_mirror_doubling():
    s = wall_comb_spectrum(WALL_EQUATOR, 1.0, 0.0)
    lines = s["lines_per_omega"]
    # y-mirror symmetry: only even multiples of 5 Omega survive
    assert np.all(np.abs(lines / 10.0 - np.round(lines / 10.0)) < 0.05)


def test_bit_rate_formula():
    assert math.isclose(comb_bit_rate(2.0 * math.pi), 1.0, rel_tol=1e-12)
    assert math.isclose(comb_bit_rate(2.0 * math.pi, levels=4), 2.0,
                        rel_tol=1e-12)
    with pytest.raises(ValueError):
        comb_bit_rate(1.0, levels=1)


def test_comms_chronology_exclusion():
    assert clean_channel(0.5)        # below the window
    assert not clean_channel(1.2)    # inside: CTC band, no clean signalling
    assert clean_channel(2.0)        # above
    assert max_clean_bit_rate() > 1.0


def test_channel_report():
    r = channel_report()
    assert r.wall_modulation_depth_45 > 0.15
    assert r.wall_5omega_selected
    assert r.wall_demultiplied_on_tilt
    assert math.isclose(r.equator_spacing_per_omega, 10.0, rel_tol=0.05)
    assert r.comms_chronology_disjoint
    assert r.bit_rate_at_grip_bound > r.bit_rate_above_window \
        > r.bit_rate_below_window
    assert r.catcher_verdict in ("novel_structure", "smooth", "uniform")
    text = summarise_channel(r)
    for tag in ("C1", "C2", "C3", "catcher"):
        assert tag in text
