"""Tests for Z_3 anomaly inflow."""

import numpy as np
import pytest

from systrophe.anomaly_inflow import (
    AXIAL_ANOMALY_COEFFICIENT,
    axial_anomaly_density,
    callan_harvey_bulk_inflow,
    callan_harvey_consistency,
    chern_simons_5form_coefficient,
    dirac_eta_invariant,
    index_density_2form,
    z3_anomaly_inflow_balance,
    z3_branch_etas,
    z3_branch_twists,
    z3_total_eta,
)


# ----- APS eta-invariant -------------------------------------------------

def test_eta_at_antiperiodic():
    """eta(1/2) = 0 (anti-periodic, no spectral asymmetry)."""
    assert dirac_eta_invariant(0.5) == pytest.approx(0.0, abs=1e-12)


def test_eta_at_periodic():
    """eta(0) = 0 (symmetric regularisation; zero mode contribution dropped)."""
    assert float(dirac_eta_invariant(0.0)) == 0.0
    # And by wrapping, eta(1) = eta(0) = 0
    assert float(dirac_eta_invariant(1.0)) == 0.0


def test_eta_at_third():
    """eta(1/3) = 1 - 2/3 = 1/3."""
    assert dirac_eta_invariant(1 / 3) == pytest.approx(1 / 3, rel=1e-12)


def test_eta_at_two_thirds():
    """eta(2/3) = 1 - 4/3 = -1/3."""
    assert dirac_eta_invariant(2 / 3) == pytest.approx(-1 / 3, rel=1e-12)


def test_eta_array_input():
    alphas = np.array([0.0, 1 / 4, 1 / 2, 3 / 4])
    expected = np.array([0.0, 0.5, 0.0, -0.5])
    assert np.allclose(dirac_eta_invariant(alphas), expected, atol=1e-12)


# ----- Z_3 branch twists / etas -----------------------------------------

def test_z3_branch_twists_at_zero_gamma():
    twists = z3_branch_twists(0.0)
    assert twists == pytest.approx(np.array([0.0, 1 / 3, 2 / 3]), abs=1e-12)


def test_z3_branch_twists_with_gamma():
    """gamma_eff = pi shifts each branch by 1/2 (mod 1)."""
    twists = z3_branch_twists(np.pi)
    # Branch 0: 0 + 1/2 = 1/2
    # Branch 1: 1/3 + 1/2 = 5/6
    # Branch 2: 2/3 + 1/2 = 7/6 -> 1/6
    expected = np.array([0.5, 5 / 6, 1 / 6])
    assert np.allclose(twists, expected, atol=1e-12)


def test_z3_branch_etas_at_zero():
    etas = z3_branch_etas(0.0)
    # (eta(0), eta(1/3), eta(2/3)) = (0, 1/3, -1/3)
    assert etas == pytest.approx(np.array([0.0, 1 / 3, -1 / 3]), abs=1e-12)


def test_z3_total_eta_vanishes_at_zero_gamma():
    """Z_3 sum cancels: 0 + 1/3 + (-1/3) = 0."""
    assert z3_total_eta(0.0) == pytest.approx(0.0, abs=1e-12)


def test_z3_total_eta_nonzero_for_general_gamma():
    """For general gamma_eff the cancellation is broken."""
    # Pick a gamma that doesn't make all twists rational equal-step
    total = z3_total_eta(0.7)
    assert total != 0.0


# ----- Axial anomaly density --------------------------------------------

def test_axial_anomaly_coefficient_value():
    """1 / (16 pi^2)."""
    assert AXIAL_ANOMALY_COEFFICIENT == pytest.approx(1 / (16 * np.pi ** 2), rel=1e-12)


def test_axial_anomaly_density_simple():
    """For F with F_01 = E, F_dual_23 = E, F̃F contraction = 2 * 2 * E^2 = 4 E^2.

    Actually F̃F = F̃^{mu nu} F_{mu nu}. With both F and F_dual having
    one upper non-zero pair (01) and (23) respectively, the contraction
    is zero (orthogonal pairs). Use a non-trivial example."""
    # Take F = block-diag electric component F_01 = -F_10 = E, and
    # F_dual = block-diag magnetic component F_dual_01 = -F_dual_10 = B.
    F = np.zeros((4, 4))
    F_dual = np.zeros((4, 4))
    F[0, 1] = 3.0
    F[1, 0] = -3.0
    F_dual[0, 1] = 5.0
    F_dual[1, 0] = -5.0
    # F_dual.F = sum_{ij} F_dual[i,j] F[i,j] = 2 * (3 * 5) - well, both
    # asymmetric -> 2 * 3 * 5 + 2 * 3 * 5 (no, signs match -> +) = 60? Let me think.
    # F_dual[0,1] F[0,1] = 5 * 3 = 15
    # F_dual[1,0] F[1,0] = -5 * -3 = 15
    # Sum = 30
    density = axial_anomaly_density(F, F_dual)
    expected = AXIAL_ANOMALY_COEFFICIENT * 30.0
    assert density == pytest.approx(expected, rel=1e-12)


# ----- Callan-Harvey ----------------------------------------------------

def test_callan_harvey_bulk_inflow_at_zero():
    """Bulk inflow required = -1/2 * boundary sum; zero when boundary sum is zero."""
    etas = z3_branch_etas(0.0)
    inflow = callan_harvey_bulk_inflow(etas)
    assert inflow == pytest.approx(0.0, abs=1e-12)


def test_callan_harvey_bulk_inflow_nontrivial():
    """For nonzero gamma_eff, the required bulk inflow is nonzero."""
    etas = z3_branch_etas(0.4)
    inflow = callan_harvey_bulk_inflow(etas)
    assert abs(inflow) > 0.0


def test_callan_harvey_consistency_at_zero():
    etas = z3_branch_etas(0.0)
    assert callan_harvey_consistency(etas)


# ----- Chern-Simons -----------------------------------------------------

def test_chern_simons_coefficient():
    """k = 1 / (24 pi^2)."""
    assert chern_simons_5form_coefficient() == pytest.approx(
        1 / (24 * np.pi ** 2), rel=1e-12
    )


# ----- Index density ----------------------------------------------------

def test_index_density_flux_quantum():
    """B = 2 pi, area = 1: q = 1 (one flux quantum)."""
    assert index_density_2form(B_z=2 * np.pi, area=1.0) == pytest.approx(1.0, rel=1e-12)


# ----- Full balance -----------------------------------------------------

def test_z3_anomaly_balance_closed_at_zero():
    """gamma_eff = 0, B_z = 0: cover is fully anomaly-closed."""
    result = z3_anomaly_inflow_balance(gamma_eff=0.0, B_z=0.0, area=1.0)
    assert result["boundary_anomaly_sum"] == pytest.approx(0.0, abs=1e-12)
    assert result["required_bulk_inflow"] == pytest.approx(0.0, abs=1e-12)
    assert result["residual"] == pytest.approx(0.0, abs=1e-12)


def test_z3_anomaly_balance_inflow_required():
    """gamma_eff = 0.6 produces nonzero required inflow, B field cancels it."""
    gamma = 0.6
    # First compute required inflow with no flux
    base = z3_anomaly_inflow_balance(gamma_eff=gamma, B_z=0.0, area=1.0)
    required = base["required_bulk_inflow"]
    cs_coeff = base["chern_simons_coeff"]
    # Set B and area so that cs_coeff * (B * area / 2pi) = required
    target_q = required / cs_coeff
    B_target = target_q * 2 * np.pi  # area = 1
    balanced = z3_anomaly_inflow_balance(gamma_eff=gamma, B_z=B_target, area=1.0)
    assert balanced["residual"] == pytest.approx(0.0, abs=1e-12)
