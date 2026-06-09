"""Tests for the bubble-interior derivation (knopp_dodeca_bubble_interior)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_dodeca_bubble_interior import (
    SQRT5,
    axis_potential,
    axis_projections,
    bubble_interior_report,
    cubic_control_census,
    localization_scan,
    participation,
    payload_grip,
    recoil_energy,
    registry_identity_residual,
    summarise_interior,
    trap_census,
)


# ----- B0: axis potential == registry function ----------------------------------


def test_axis_projections_are_one_and_inv_sqrt5():
    p = np.sort(axis_projections())
    assert math.isclose(p[-1], 1.0, abs_tol=1e-9)
    assert np.allclose(p[:-1], 1.0 / SQRT5, atol=1e-9)


def test_registry_identity_is_exact():
    assert registry_identity_residual() < 1e-12


def test_axis_potential_form():
    x = np.array([0.0])
    assert math.isclose(float(axis_potential(x, 6.0, 38.0)[0]), 6.0)
    # zero-mean over many quasiperiods
    xs = np.linspace(0.0, 50.0, 200001)
    assert abs(float(np.mean(axis_potential(xs, 1.0, 38.0)))) < 5e-3


# ----- B2: localization ------------------------------------------------------------


def test_recoil_energy():
    assert math.isclose(recoil_energy(38.0), (76.0 / SQRT5) ** 2, rel_tol=1e-12)


def test_extended_at_lock_localized_at_high_amplitude():
    lo = participation(2.5, n=3000, L=6.0)
    hi = participation(4000.0, n=3000, L=6.0)
    assert lo < 5.0          # extended
    assert hi > 30.0         # localized
    with pytest.raises(ValueError):
        participation(-1.0)


def test_localization_threshold_near_recoil_scale():
    scan = localization_scan(
        V0_values=np.array([2.5, 400.0, 1000.0, 2000.0, 4000.0]))
    assert 800.0 <= scan["V0_star"] <= 3000.0
    assert scan["V0_star"] > 100.0 * 2.54    # far above the lock amplitude
    assert math.isclose(scan["recoil"], recoil_energy(38.0), rel_tol=1e-9)


# ----- B1: trap quasilattice ---------------------------------------------------------


def test_trap_lattice_is_quasilattice_vs_cubic_control():
    c = trap_census(n=112)
    k = cubic_control_census(n=112)
    assert c.n_traps > 200
    assert c.nn_shell_count >= 2          # discrete multi-shell spacing
    assert k.nn_shell_count == 1          # periodic control: single spacing
    assert c.median_depth < -0.5          # deep traps (ceiling ~2.54)
    assert 0.05 < c.median_spacing < 0.15


# ----- B3: payload grip ----------------------------------------------------------------


def test_payload_pinned_during_collapse():
    g = payload_grip()
    assert g > 50.0
    with pytest.raises(ValueError):
        payload_grip(eps=0.0)


# ----- report ------------------------------------------------------------------------------


def test_bubble_interior_report():
    r = bubble_interior_report(census_n=112)
    assert r.registry_identity_residual < 1e-12
    assert r.axis_projection_check
    assert r.is_quasilattice
    assert r.waves_extended_at_lock
    assert r.localization_shortfall > 100.0
    assert r.payload_pinned
    assert r.catcher_verdict in ("novel_structure", "smooth", "uniform")
    text = summarise_interior(r)
    for tag in ("B0", "B1", "B2", "B3", "catcher"):
        assert tag in text
