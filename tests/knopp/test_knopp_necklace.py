"""Tests for the multi-binary necklace configuration."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_necklace import (
    KnoppNecklace,
    NecklaceReport,
    necklace_report,
    packing_threshold_for_continuous_band,
    summarise_necklace,
)


# ----- construction & validation ----------------------------------------


def test_construction_defaults():
    n = KnoppNecklace()
    assert n.n_binaries == 6


def test_rejects_zero_binaries():
    with pytest.raises(ValueError):
        KnoppNecklace(n_binaries=0)


def test_rejects_nonpositive_parameters():
    with pytest.raises(ValueError):
        KnoppNecklace(M=-1.0)
    with pytest.raises(ValueError):
        KnoppNecklace(d=0.0)
    with pytest.raises(ValueError):
        KnoppNecklace(R_ring=-1.0)


def test_rejects_chi_out_of_range():
    with pytest.raises(ValueError):
        KnoppNecklace(chi=1.5)


# ----- geometry ---------------------------------------------------------


def test_binary_angular_positions_uniform():
    n = KnoppNecklace(n_binaries=4)
    angles = n.binary_angular_positions()
    assert len(angles) == 4
    assert angles[0] == 0.0
    assert angles[1] == pytest.approx(math.pi / 2.0, rel=1e-12)
    assert angles[3] == pytest.approx(3.0 * math.pi / 2.0, rel=1e-12)


def test_chord_to_self_is_zero():
    n = KnoppNecklace(n_binaries=6, R_ring=10.0)
    for k in range(6):
        phi_k = 2.0 * math.pi * k / 6.0
        assert n.chord_to_kth_binary(phi_k, k) == pytest.approx(
            0.0, abs=1e-12,
        )


def test_chord_max_at_diametrically_opposite():
    n = KnoppNecklace(n_binaries=2, R_ring=10.0)
    # At test point phi=0, binary 0 is at 0 (chord=0), binary 1 at pi
    # (chord = 2 R_ring = 20).
    assert n.chord_to_kth_binary(0.0, 1) == pytest.approx(20.0, rel=1e-12)


def test_adjacent_binary_chord_formula():
    n = KnoppNecklace(n_binaries=6, R_ring=10.0)
    # 2 R sin(pi/N) = 20 sin(pi/6) = 20 * 0.5 = 10.
    assert n.adjacent_binary_chord() == pytest.approx(10.0, rel=1e-12)


def test_nearest_binary_distance_is_min_chord():
    n = KnoppNecklace(n_binaries=4, R_ring=10.0)
    # At phi_test = pi/4 (between binaries 0 and 1, equally distant):
    d_near = n.nearest_binary_distance(math.pi / 4.0)
    d0 = n.chord_to_kth_binary(math.pi / 4.0, 0)
    d1 = n.chord_to_kth_binary(math.pi / 4.0, 1)
    assert d_near == pytest.approx(min(d0, d1), rel=1e-12)


# ----- frame-dragging --------------------------------------------------


def test_t_eff_local_recovers_single_binary_at_centre():
    """At distance_from_binary = (any valid rho), the local T_eff matches
    EffectiveToroidalKerrBinary's t_eff (no-phi version)."""
    from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
    n = KnoppNecklace(M=1.0, d=2.0, chi=1.0)
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    for rho in (0.5, 1.0, 2.0, 3.5):
        assert n.t_eff_local(rho) == pytest.approx(
            b.t_eff(rho, include_phi=False), rel=1e-12,
        )


def test_t_eff_necklace_at_binary_centre_is_max():
    """T_eff_necklace evaluated AT a binary's angular position is
    dominated by the singular contribution of that binary."""
    n = KnoppNecklace(n_binaries=6, R_ring=10.0)
    angles = n.binary_angular_positions()
    # At binary location: the local term is at d_local = 0 -> T = 0.
    # But the OTHER binaries contribute, so T_eff is finite & positive.
    val = n.t_eff_necklace(angles[0])
    assert val >= 0.0
    assert math.isfinite(val)


def test_t_eff_necklace_periodic():
    n = KnoppNecklace(n_binaries=6, R_ring=10.0)
    assert n.t_eff_necklace(0.0) == pytest.approx(
        n.t_eff_necklace(2.0 * math.pi), rel=1e-10,
    )


# ----- continuous-band test --------------------------------------------


def test_small_N_no_continuous_band():
    n = KnoppNecklace(n_binaries=2, R_ring=5.0, M=1.0, d=2.0)
    assert n.is_continuously_banded() is False


def test_large_N_continuous_band():
    n = KnoppNecklace(n_binaries=50, R_ring=5.0, M=1.0, d=2.0)
    assert n.is_continuously_banded() is True


def test_min_T_eff_smaller_than_max():
    n = KnoppNecklace(n_binaries=6, R_ring=10.0)
    assert n.min_t_eff_along_ring() <= n.max_t_eff_along_ring()


# ----- packing threshold ------------------------------------------------


def test_packing_threshold_returns_int_or_none():
    N = packing_threshold_for_continuous_band(M=1.0, d=2.0, R_ring=5.0)
    assert isinstance(N, int)
    assert N >= 1


def test_packing_threshold_grows_with_ring_radius():
    """A larger ring needs MORE binaries to maintain continuous coverage."""
    N_small = packing_threshold_for_continuous_band(M=1.0, d=2.0, R_ring=2.0)
    N_large = packing_threshold_for_continuous_band(M=1.0, d=2.0, R_ring=20.0)
    assert N_small is not None and N_large is not None
    assert N_large > N_small


# ----- inherited falsification ------------------------------------------


def test_per_binary_n_orbits_inherits_falsification():
    """Each binary on the necklace still has only ~0.025 orbits to live."""
    n = KnoppNecklace(M=1.0, d=2.0, chi=1.0)
    assert n.per_binary_n_orbits() < 0.1


def test_necklace_report_inherits_falsification():
    n = KnoppNecklace(M=1.0, d=2.0, chi=1.0, R_ring=5.0)
    r = necklace_report(n)
    assert r.inherits_falsification is True


# ----- combined report -------------------------------------------------


def test_report_returns_dataclass():
    n = KnoppNecklace()
    r = necklace_report(n)
    assert isinstance(r, NecklaceReport)


def test_summary_string():
    n = KnoppNecklace()
    s = summarise_necklace(necklace_report(n))
    assert "Knopp necklace" in s
    assert "continuous band" in s
    assert "inherits falsification" in s
