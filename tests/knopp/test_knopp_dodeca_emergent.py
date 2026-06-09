"""Tests for scale materiality + emergent phenomena (knopp_dodeca_emergent)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_dodeca_alignment import FACE_LOCK_DEG
from systrophe.knopp.knopp_dodeca_emergent import (
    comb_spacing,
    ctc_cargo_fraction,
    orientation_hysteresis,
    registry_torque_landscape,
    rotational_comb,
    scale_materiality_audit,
    _trap_points,
)


# ----- S1: scale materiality ------------------------------------------------------


def test_mechanism_is_scale_free_registry_is_not():
    audit = scale_materiality_audit(np.linspace(0.08, 0.70, 7))
    assert audit["mechanism_scale_free"]
    assert audit["cv_saturation"] < 0.01
    assert audit["registry_scale_sensitive"]
    assert audit["cv_crystal_X"] > 0.3
    assert np.all(audit["saturation"] > 0.99)


# ----- S2: rotational comb ----------------------------------------------------------


def test_aligned_spin_comb_is_5omega_spaced():
    lines = rotational_comb(1.0, 0.0)
    assert len(lines) >= 3
    # every significant line a multiple of 5*Omega; none in (0, 5)
    for f in lines[:5]:
        assert abs(f / 5.0 - round(f / 5.0)) < 0.05
    assert lines[0] > 4.5


def test_tilt_demultiplies_the_comb():
    assert math.isclose(comb_spacing(1.0, 0.0), 5.0, rel_tol=0.05)
    assert math.isclose(comb_spacing(1.0, 4.0), 1.0, rel_tol=0.10)
    with pytest.raises(ValueError):
        rotational_comb(0.0)


def test_comb_scales_with_omega():
    assert math.isclose(comb_spacing(2.0, 0.0), 10.0, rel_tol=0.05)


# ----- S3: chronology cargo ----------------------------------------------------------


def test_ctc_cargo_fraction_monotone_in_omega():
    pts = _trap_points(n=110)
    assert len(pts) > 500
    f_low = ctc_cargo_fraction(0.5, pts)     # r_ctc = 2 > tube: no cargo
    f_mid = ctc_cargo_fraction(1.2, pts)
    f_hi = ctc_cargo_fraction(2.0, pts)
    assert f_low == 0.0
    assert 0.2 < f_mid < 0.9
    assert f_hi > f_mid
    with pytest.raises(ValueError):
        ctc_cargo_fraction(0.0, pts)


# ----- S4: self-holding lock -----------------------------------------------------------


def test_registry_landscape_is_multiwell_with_lock_as_global_peak():
    land = registry_torque_landscape(np.linspace(20.0, 70.0, 51))
    assert len(land["peaks_deg"]) >= 2
    best = land["peaks_deg"][int(np.argmax(
        [land["X"][list(land["betas"]).index(p)] for p in land["peaks_deg"]]))]
    assert abs(best - FACE_LOCK_DEG) < 1.5
    assert land["max_slope_per_rad"] > 5.0


def test_lock_holds_and_torque_sweep_shows_hysteresis():
    land = registry_torque_landscape(np.linspace(15.0, 75.0, 61))
    hyst = orientation_hysteresis(land)
    assert hyst["holding_tau"] > 0.5          # the lock holds itself
    assert hyst["max_separation_deg"] > 10.0  # forward != backward path
    assert hyst["max_jump_deg"] > 5.0         # stick-slip between wells
