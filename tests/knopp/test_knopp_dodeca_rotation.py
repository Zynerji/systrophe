"""Tests for the rotation derivations (knopp_dodeca_rotation)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_dodeca_alignment import DEFAULT_R
from systrophe.knopp.knopp_dodeca_rotation import (
    conveyor_rigidity_residual,
    cornered_coupling,
    cornered_spiral_source,
    high_band_gain,
    horn_spiral_handedness,
    inversion_symmetry_residual,
    max_conveyor_rate,
    mean_interior_pressure,
    ring_coherence_under_spin,
    ring_lobe_phase,
    rotation_report,
    spun_axes,
    summarise_rotation,
    tipler_assessment,
    van_stockum_window,
)


# ----- R1: conveyor -----------------------------------------------------------


def test_spun_axes_stay_unit():
    ax = spun_axes(33.0, 5.0)
    assert ax.shape == (6, 3)
    assert np.allclose(np.linalg.norm(ax, axis=1), 1.0)


def test_lattice_corotates_exactly():
    assert conveyor_rigidity_residual(25.0) < 1e-10
    assert conveyor_rigidity_residual(140.0) < 1e-10


def test_lock_survives_spin():
    for psi in (0.0, 17.0, 45.0, 90.0):
        assert ring_coherence_under_spin(psi) > 0.99


def test_max_conveyor_rate():
    om = max_conveyor_rate(DEFAULT_R)
    assert 8.0 < om < 20.0
    with pytest.raises(ValueError):
        max_conveyor_rate(0.0)


# ----- R2 + R5: Tipler window ----------------------------------------------------


def test_van_stockum_window_closed_form():
    u = mean_interior_pressure()
    lo, hi = van_stockum_window(u)
    assert math.isclose(lo, 1.0 / (2.0 * DEFAULT_R), rel_tol=1e-12)
    assert math.isclose(hi, math.sqrt(2.0 * math.pi * u / math.e),
                        rel_tol=1e-12)
    assert lo < hi          # the window exists at lock
    with pytest.raises(ValueError):
        van_stockum_window(u=-1.0)


def test_tipler_assessment_window_edges():
    inside = tipler_assessment(1.0)
    assert inside["supercritical"]
    assert inside["ctc_radius_in_tube"]
    assert inside["conveyor_can_reach"]
    slow = tipler_assessment(0.5)
    assert not slow["supercritical"]          # CTC radius outside the tube
    assert not slow["ctc_radius_in_tube"]
    fast = tipler_assessment(2.5)
    assert not fast["supercritical"]          # field density insufficient
    assert fast["mu_required_at_band"] > fast["mu_available"]
    assert "chronology" in inside["caveats"]
    with pytest.raises(ValueError):
        tipler_assessment(0.0)


# ----- R3: counter-helicity --------------------------------------------------------


def test_horns_are_parity_twins():
    assert inversion_symmetry_residual() < 1e-12


def test_handedness_opposite():
    assert horn_spiral_handedness(1) == 1
    assert horn_spiral_handedness(-1) == -1
    with pytest.raises(ValueError):
        horn_spiral_handedness(0)


def test_rings_corotate_same_sense():
    # +psi rotation advances the m=5 phase by -5 psi on BOTH rings
    psi, tilt = 20.0, 4.0
    expect = (-5.0 * math.radians(psi)) % (2.0 * math.pi)
    for ys in (1, -1):
        p0 = ring_lobe_phase(ys, 0.0, tilt)
        p1 = ring_lobe_phase(ys, psi, tilt)
        d = (p1["phase_m5"] - p0["phase_m5"]) % (2.0 * math.pi)
        assert min(abs(d - expect), abs(d - expect + 2 * math.pi)) < 1e-9


def test_tilt_creates_m1_sideband():
    flat = ring_lobe_phase(1, 20.0, 0.0)["amp_m1"]
    tilted = ring_lobe_phase(1, 20.0, 4.0)["amp_m1"]
    assert tilted > 10.0 * max(flat, 1e-15)
    with pytest.raises(ValueError):
        ring_lobe_phase(2, 0.0)


# ----- R4: cornered spiral ----------------------------------------------------------


def test_cornered_source_reduces_to_pure_spiral():
    rho = np.linspace(0.0, 1.0, 500)
    pure = cornered_spiral_source(rho, 0.0)
    chi = 60.0 * np.sqrt(rho)
    assert np.allclose(pure, (rho <= 0.8) * (1.0 + np.cos(chi)))
    with pytest.raises(ValueError):
        cornered_spiral_source(rho, 1.5)


def test_corners_extend_the_comb_upward():
    assert math.isclose(high_band_gain(0.0), 1.0, rel_tol=1e-9)
    assert high_band_gain(0.8) > 1.5
    c = cornered_coupling(0.8)
    assert len(c) == 48


# ----- report -------------------------------------------------------------------------


def test_rotation_report():
    r = rotation_report()
    assert r.conveyor_rigidity_residual < 1e-10
    assert r.lock_coherence_under_spin > 0.99
    assert r.conveyor_covers_window
    assert r.tipler_window[0] < 1.0 < r.tipler_window[1]
    assert r.inversion_residual < 1e-12
    assert r.handedness_upper == -r.handedness_lower
    assert r.corotation_phase_slip < 1e-9
    assert r.tilt_sideband_ratio > 10.0
    assert r.corner_gain_a5_08 > 1.5
    assert r.catcher_verdict in ("novel_structure", "smooth", "uniform")
    text = summarise_rotation(r)
    for tag in ("R1", "R2/R5", "R3", "R4", "catcher"):
        assert tag in text
