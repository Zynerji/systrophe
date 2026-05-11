"""Tests for horned-torus topology with regular and inverted variants."""

import numpy as np
import pytest

from systrophe.horned_torus import (
    HornedTorus,
    compare_horn_modes,
    horn_circumference_at_z,
    inverted_horn_profile,
    regular_horn_profile,
)
from systrophe.sinusoid import TiplerSinusoid


# ----- horn profiles -----------------------------------------------------

def test_regular_profile_at_zero():
    """Pinch: h(0) = h_min."""
    assert regular_horn_profile(0.0, h_min=0.1, sigma=1.0) == pytest.approx(0.1, abs=1e-12)


def test_regular_profile_at_infinity():
    """Far from horn: h -> 1."""
    h = regular_horn_profile(100.0, h_min=0.1, sigma=1.0)
    assert h == pytest.approx(1.0, abs=1e-6)


def test_inverted_profile_at_zero():
    """Bulge: h(0) = h_max."""
    assert inverted_horn_profile(0.0, h_max=3.5, sigma=1.0) == pytest.approx(3.5, abs=1e-12)


def test_inverted_profile_at_infinity():
    """Far from bulge: h -> 1."""
    h = inverted_horn_profile(100.0, h_max=3.5, sigma=1.0)
    assert h == pytest.approx(1.0, abs=1e-30)


def test_regular_profile_rejects_h_min_out_of_range():
    with pytest.raises(ValueError):
        regular_horn_profile(0.0, h_min=1.5)
    with pytest.raises(ValueError):
        regular_horn_profile(0.0, h_min=-0.1)


def test_inverted_profile_rejects_h_max_lt_1():
    with pytest.raises(ValueError):
        inverted_horn_profile(0.0, h_max=0.5)


# ----- HornedTorus class -------------------------------------------------

def _toy_L(r):
    """Toy L(r) with a single CTC band r in (2, 4)."""
    r = np.asarray(r, dtype=float)
    return (r - 2) * (r - 4) * (-1)  # negative inside (2, 4)


def test_horned_torus_construction_regular():
    h = HornedTorus(L_func=_toy_L, mode="regular", h_param=0.2, sigma=1.0)
    assert h.mode == "regular"
    assert h.h(0.0) == pytest.approx(0.2, abs=1e-12)


def test_horned_torus_construction_inverted():
    h = HornedTorus(L_func=_toy_L, mode="inverted", h_param=2.5, sigma=1.0)
    assert h.mode == "inverted"
    assert h.h(0.0) == pytest.approx(2.5, abs=1e-12)


def test_horned_torus_rejects_bad_mode():
    with pytest.raises(ValueError):
        HornedTorus(L_func=_toy_L, mode="bogus")


def test_horned_torus_validates_h_param():
    with pytest.raises(ValueError):
        HornedTorus(L_func=_toy_L, mode="regular", h_param=1.5)
    with pytest.raises(ValueError):
        HornedTorus(L_func=_toy_L, mode="inverted", h_param=0.5)


def test_L_horned_preserves_sign():
    """Since h(z) > 0, L_horned has same sign as L(r) at each r."""
    horn = HornedTorus(L_func=_toy_L, mode="regular", h_param=0.1, sigma=1.0)
    r = np.array([1.5, 3.0, 5.0])
    z = np.array([-2.0, 0.0, 2.0])
    for ri in r:
        for zi in z:
            L_h = float(horn.L_horned(ri, zi))
            L_base = float(_toy_L(ri))
            assert np.sign(L_h) == np.sign(L_base)


def test_ctc_indicator_constant_in_z():
    """CTC region {L<0} is z-independent because h > 0 everywhere."""
    horn = HornedTorus(L_func=_toy_L, mode="regular", h_param=0.1, sigma=1.0)
    r = np.linspace(1.0, 5.0, 41)
    z = np.linspace(-3.0, 3.0, 31)
    ind = horn.ctc_indicator(r, z)
    # Each row (same r, varying z) must be constant
    for i in range(len(r)):
        assert np.all(ind[i, :] == ind[i, 0])


def test_inverted_horn_increases_proper_area_vs_regular():
    """Bulge increases the CTC proper-area integral; pinch decreases it.

    This is the main physical distinction between the two horn modes.
    """
    cmp = compare_horn_modes(
        L_func=_toy_L,
        h_min=0.1, h_max=3.0, sigma=1.0,
        r_min=1.0, r_max=5.0, z_min=-3.0, z_max=3.0,
        n_r=201, n_z=101,
    )
    assert cmp["regular_proper_area"] < cmp["flat_proper_area"]
    assert cmp["inverted_proper_area"] > cmp["flat_proper_area"]
    assert cmp["ratio_inv_over_reg"] > 1.0


def test_compare_modes_on_real_tipler_sinusoid():
    """Use a real TiplerSinusoid as the base L."""
    s = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    cmp = compare_horn_modes(
        L_func=s.L,
        h_min=0.2, h_max=2.5, sigma=1.0,
        r_min=1.0, r_max=8.0, z_min=-3.0, z_max=3.0,
        n_r=301, n_z=151,
    )
    assert cmp["flat_proper_area"] > 0  # there are real CTC bands
    assert cmp["regular_proper_area"] < cmp["flat_proper_area"]
    assert cmp["inverted_proper_area"] > cmp["flat_proper_area"]


def test_topology_classification():
    flat = HornedTorus(L_func=_toy_L, mode="regular", h_param=1.0, sigma=1.0)
    thinned = HornedTorus(L_func=_toy_L, mode="regular", h_param=0.3, sigma=1.0)
    fattened = HornedTorus(L_func=_toy_L, mode="inverted", h_param=2.0, sigma=1.0)
    assert flat.topology_class() == "flat_T2"
    assert thinned.topology_class() == "thinned_T2"
    assert fattened.topology_class() == "fattened_T2"


def test_horn_circumference_at_pinch_vs_bulge():
    """At z = 0: regular gives smaller proper length, inverted gives larger."""
    L_band = -1.0  # CTC band
    regular = HornedTorus(L_func=_toy_L, mode="regular", h_param=0.05, sigma=1.0)
    inverted = HornedTorus(L_func=_toy_L, mode="inverted", h_param=4.0, sigma=1.0)
    c_reg = horn_circumference_at_z(L_band, 0.0, regular)
    c_inv = horn_circumference_at_z(L_band, 0.0, inverted)
    assert c_reg < c_inv
    # Far from horn (z -> infinity), both approach the flat-h value
    c_reg_far = horn_circumference_at_z(L_band, 10.0, regular)
    c_inv_far = horn_circumference_at_z(L_band, 10.0, inverted)
    assert c_reg_far == pytest.approx(c_inv_far, rel=1e-6)
