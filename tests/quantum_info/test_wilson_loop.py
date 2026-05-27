"""Tests for U(1) Wilson loop identification of Z_3 monodromy."""

import numpy as np
import pytest

from systrophe.quantum_info.wilson_loop import (
    field_strength_is_zero,
    flat_u1_connection,
    gauge_field_strength_4d,
    integrated_chern_number,
    verify_wilson_loop_matches_z3,
    wilson_loop,
    wilson_loop_sum_over_branches,
    z3_branch_holonomy,
)


# ----- flat connection ---------------------------------------------------

def test_connection_is_constant():
    """A_phi is constant in phi (flat connection)."""
    A = flat_u1_connection(gamma_eff=0.5, branch=1)
    assert A(0.0) == pytest.approx(A(1.0))
    assert A(0.0) == pytest.approx(A(np.pi))


def test_connection_branch_0_at_zero_gamma():
    """A = 0 for gamma_eff = 0, branch = 0."""
    A = flat_u1_connection(gamma_eff=0.0, branch=0)
    assert A(1.0) == pytest.approx(0.0, abs=1e-12)


def test_connection_field_strength_zero():
    """F = dA = 0 for flat connection."""
    A = flat_u1_connection(gamma_eff=0.3, branch=2)
    result = field_strength_is_zero(A)
    assert result["is_flat"]


# ----- Wilson loop calculation -----------------------------------------

def test_wilson_loop_returns_complex():
    A = flat_u1_connection(gamma_eff=0.0, branch=0)
    W = wilson_loop(A)
    assert isinstance(W, complex)


def test_wilson_loop_at_branch_0_zero_gamma():
    """W = exp(0) = 1 at branch 0, gamma_eff = 0."""
    A = flat_u1_connection(gamma_eff=0.0, branch=0)
    W = wilson_loop(A)
    assert abs(W - 1.0) < 1e-9


def test_wilson_loop_at_branch_1():
    """W = exp(2 pi i / 3) at branch 1, gamma_eff = 0."""
    A = flat_u1_connection(gamma_eff=0.0, branch=1)
    W = wilson_loop(A)
    expected = complex(np.exp(2j * np.pi / 3))
    assert abs(W - expected) < 1e-9


def test_wilson_loop_at_branch_2():
    """W = exp(4 pi i / 3) at branch 2."""
    A = flat_u1_connection(gamma_eff=0.0, branch=2)
    W = wilson_loop(A)
    expected = complex(np.exp(4j * np.pi / 3))
    assert abs(W - expected) < 1e-9


# ----- closed-form Z_3 holonomy ----------------------------------------

def test_z3_holonomy_cube_root_of_unity():
    """Branches 0, 1, 2 produce the three cube roots of 1."""
    W0 = z3_branch_holonomy(0.0, 0)
    W1 = z3_branch_holonomy(0.0, 1)
    W2 = z3_branch_holonomy(0.0, 2)
    # W^3 = 1 for each
    assert abs(W0 ** 3 - 1.0) < 1e-12
    assert abs(W1 ** 3 - 1.0) < 1e-12
    assert abs(W2 ** 3 - 1.0) < 1e-12


def test_z3_holonomy_validates_branch():
    with pytest.raises(ValueError):
        z3_branch_holonomy(branch=3)
    with pytest.raises(ValueError):
        z3_branch_holonomy(branch=-1)


def test_z3_holonomy_with_gamma_eff():
    """gamma_eff shifts holonomy by an overall phase."""
    g = 0.5
    W0 = z3_branch_holonomy(g, 0)
    expected = complex(np.exp(1j * g))
    assert abs(W0 - expected) < 1e-12


# ----- verification full battery ----------------------------------------

def test_verify_wilson_matches_z3_at_zero_gamma():
    """All three branches: numerical Wilson loop matches closed form."""
    result = verify_wilson_loop_matches_z3(gamma_eff=0.0)
    assert result["all_consistent"]


def test_verify_wilson_matches_z3_with_gamma():
    """With nonzero gamma_eff, consistency still holds."""
    result = verify_wilson_loop_matches_z3(gamma_eff=0.7)
    assert result["all_consistent"]


# ----- sum-over-branches -----------------------------------------------

def test_sum_over_branches_vanishes_at_zero_gamma():
    """W_0 + W_1 + W_2 = 1 + omega + omega^2 = 0."""
    total = wilson_loop_sum_over_branches(gamma_eff=0.0)
    assert abs(total) < 1e-12


def test_sum_over_branches_phase_with_gamma():
    """For non-zero gamma_eff, sum is exp(i gamma) * (1 + omega + omega^2) = 0
    still --- the overall phase factors out."""
    total = wilson_loop_sum_over_branches(gamma_eff=0.3)
    # Still zero! exp(i gamma) * 0 = 0
    assert abs(total) < 1e-12


# ----- field strength + Chern --------------------------------------------

def test_field_strength_4d_is_zero():
    F = gauge_field_strength_4d(gamma_eff=0.5)
    assert F.shape == (4, 4)
    assert np.all(F == 0)


def test_chern_number_vanishes():
    """Flat connection: c_1 = 0."""
    c1 = integrated_chern_number(gamma_eff=0.5)
    assert c1 == 0.0
