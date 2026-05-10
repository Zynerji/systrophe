"""Tests for the three singularity reinterpretations of Systrophe sources.

(1) Rotating line singularity (Lewis/Tipler) -- LineSingularity
(2) Non-rotating cosmic string (Vilenkin) -- CosmicString
(3) Kerr ring singularity (4D analog) -- KerrSpacetime
"""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.spacetimes import GottPair
from systrophe.spacetimes.line_singularity import LineSingularity
from systrophe.spacetimes.cosmic_string import CosmicString
from systrophe.spacetimes.kerr import KerrSpacetime


# ----------------------------------------------------------------------
# (1) LineSingularity tests
# ----------------------------------------------------------------------


def test_line_singularity_construction_validates():
    LineSingularity(omega=1.0, R_match=1.0)
    with pytest.raises(ValueError):
        LineSingularity(omega=0.0, R_match=1.0)
    with pytest.raises(ValueError):
        LineSingularity(omega=1.0, R_match=0.0)


def test_line_singularity_exterior_matches_van_stockum():
    """The line-singularity exterior is identical to the van Stockum exterior."""
    omega, R = 1.0, 1.0
    ls = LineSingularity(omega=omega, R_match=R)
    vs = VanStockumInterior(omega=omega, R=R)
    rs = np.array([1.5, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(ls.F(rs), vs.analytic_exterior_F(rs), rtol=1e-14)
    np.testing.assert_allclose(ls.K(rs), vs.analytic_exterior_K(rs), rtol=1e-14)
    np.testing.assert_allclose(ls.L(rs), vs.analytic_exterior_L(rs), rtol=1e-14)


def test_line_singularity_twist_constant():
    ls = LineSingularity(omega=1.5, R_match=1.0)
    assert ls.twist_constant == pytest.approx(3.0)


def test_line_singularity_regime_matches_van_stockum():
    ls = LineSingularity(omega=1.0, R_match=1.0)
    assert ls.regime == "supercritical"


def test_line_singularity_from_M_J_recovers_omega():
    """from_mass_and_angular_momentum gives a self-consistent (omega, R) pair."""
    # For omega=1, R=1, a=1: M = (1/2)(e - 1) ~ 0.859
    M_target = 0.5 * (np.exp(1.0) - 1.0)
    ls = LineSingularity.from_mass_and_angular_momentum(M=M_target, J=0.0, R_match=1.0)
    assert ls.omega == pytest.approx(1.0, rel=1e-12)
    assert ls.R_match == 1.0


def test_van_stockum_mass_per_unit_length():
    """M = (1/2)(exp(a^2) - 1)."""
    vs = VanStockumInterior(omega=1.0, R=1.0)  # a = 1
    expected = 0.5 * (np.exp(1.0) - 1.0)
    assert vs.mass_per_unit_length == pytest.approx(expected, rel=1e-12)


def test_as_line_singularity_summary_returns_dict():
    vs = VanStockumInterior(omega=1.5, R=1.0)
    summary = vs.as_line_singularity_summary()
    assert "mass_per_unit_length_M" in summary
    assert "angular_momentum_per_unit_length_J" in summary
    assert "twist_constant_c" in summary
    assert summary["twist_constant_c"] == pytest.approx(3.0)


# ----------------------------------------------------------------------
# (2) CosmicString tests
# ----------------------------------------------------------------------


def test_cosmic_string_construction_validates_mu():
    CosmicString(mu=0.05)
    with pytest.raises(ValueError):
        CosmicString(mu=0.0)
    with pytest.raises(ValueError):
        CosmicString(mu=0.3)  # >= 1/4


def test_cosmic_string_conical_factor():
    cs = CosmicString(mu=0.1)
    assert cs.conical_factor == pytest.approx(1.0 - 0.4)


def test_cosmic_string_deficit_angle():
    cs = CosmicString(mu=0.05)
    assert cs.deficit_angle == pytest.approx(8.0 * np.pi * 0.05)


def test_cosmic_string_no_local_ctc():
    """A static cosmic string never has a local CTC."""
    cs = CosmicString(mu=0.05)
    assert not cs.has_local_ctc(r=1.0)
    assert not cs.has_local_ctc(r=10.0)


def test_cosmic_string_g_phiphi_positive():
    cs = CosmicString(mu=0.1)
    rs = np.array([1.0, 2.0, 5.0])
    L = cs.line_element_factor(rs)
    assert np.all(L > 0)


def test_compose_with_gott_creates_pair():
    """Two cosmic strings -> GottPair at chosen velocity."""
    cs1 = CosmicString(mu=0.05)
    cs2 = CosmicString(mu=0.05)
    pair = cs1.compose_with_gott(cs2, v=0.8)
    assert isinstance(pair, GottPair)
    assert pair.mu == pytest.approx(0.05)
    assert pair.v == pytest.approx(0.8)


def test_compose_with_gott_rejects_mismatched_masses():
    cs1 = CosmicString(mu=0.05)
    cs2 = CosmicString(mu=0.10)
    with pytest.raises(ValueError):
        cs1.compose_with_gott(cs2, v=0.8)


def test_compose_with_gott_rejects_non_string():
    cs = CosmicString(mu=0.05)
    with pytest.raises(TypeError):
        cs.compose_with_gott("not a string", v=0.8)


# ----------------------------------------------------------------------
# (3) Kerr ring singularity verification
# ----------------------------------------------------------------------


def test_kerr_ring_singularity_at_origin_equator():
    """Sigma = r^2 + a^2 cos^2 theta vanishes at r = 0, theta = pi / 2."""
    k = KerrSpacetime(M=1.0, a=0.5)
    assert float(k.Sigma(0.0, np.pi / 2)) == pytest.approx(0.0, abs=1e-12)


def test_three_reinterpretations_share_pair_structure():
    """Conceptual: each of {LineSingularity, CosmicString, KerrSpacetime} can be
    paired up; the Systrophe construction is two LineSingularities, the Gott
    construction is two CosmicStrings, and a Kerr binary is two KerrSpacetimes.
    Verify that each pair-formation mechanism is available.
    """
    # Pair of LineSingularities (just construct two and check exterior match)
    ls1 = LineSingularity(omega=1.0, R_match=1.0)
    ls2 = LineSingularity(omega=1.0, R_match=1.0)
    assert ls1.regime == ls2.regime == "supercritical"

    # Pair of CosmicStrings -> GottPair
    cs = CosmicString(mu=0.05)
    pair = cs.compose_with_gott(cs, v=0.8)
    assert isinstance(pair, GottPair)

    # Pair of KerrSpacetimes (just instantiate two; binary dynamics is out of scope)
    k1 = KerrSpacetime(M=1.0, a=0.5)
    k2 = KerrSpacetime(M=1.0, a=0.5)
    assert k1.M == k2.M
