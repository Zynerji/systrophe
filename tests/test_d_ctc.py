"""Tests for D-CTC quantum information module."""

import numpy as np
import pytest

from systrophe.d_ctc import (
    apply_channel,
    dctc_fixed_point,
    density_matrix_diagnostics,
    maximally_mixed_state,
    verify_fixed_point,
    z3_ctc_unitary,
)


# ----- channel application ----------------------------------------------

def test_apply_channel_preserves_trace():
    """Tr(rho_ctc') = Tr(rho_ctc) for trace-preserving channel."""
    dim_cr, dim_ctc = 2, 3
    U = np.eye(dim_cr * dim_ctc, dtype=complex)
    sigma_cr = maximally_mixed_state(dim_cr)
    rho_ctc = maximally_mixed_state(dim_ctc)
    out = apply_channel(U, sigma_cr, rho_ctc, dim_cr)
    assert np.real(np.trace(out)) == pytest.approx(1.0, abs=1e-12)


def test_apply_channel_identity_returns_input_state():
    """U = identity: rho_ctc' = rho_ctc."""
    dim_cr, dim_ctc = 2, 3
    U = np.eye(dim_cr * dim_ctc, dtype=complex)
    sigma_cr = maximally_mixed_state(dim_cr)
    rho_ctc = maximally_mixed_state(dim_ctc)
    out = apply_channel(U, sigma_cr, rho_ctc, dim_cr)
    assert np.allclose(out, rho_ctc, atol=1e-12)


def test_apply_channel_shape_validation():
    with pytest.raises(ValueError):
        apply_channel(np.eye(3) + 0j, np.eye(2) + 0j, np.eye(3) + 0j, dim_cr=2)


# ----- Z_3 unitary ------------------------------------------------------

def test_z3_ctc_unitary_is_unitary():
    U = z3_ctc_unitary(dim_cr=2, phase=0.5)
    assert np.allclose(U @ U.conj().T, np.eye(6), atol=1e-12)


def test_z3_unitary_block_structure():
    """U = I_CR (x) S for default CR action."""
    U = z3_ctc_unitary(dim_cr=2, phase=0.0)
    assert U.shape == (6, 6)
    # First 3x3 block should be identity-like multiplied by S
    # The (0, 0) block corresponds to CR-state |0>; check that the
    # internal 3x3 block matches a Z_3 cyclic shift
    block_00 = U[:3, :3]
    # S |0> = |1>, S |1> = |2>, S |2> = |0>: action = [[0,0,1],[1,0,0],[0,1,0]]
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    assert np.allclose(block_00, expected, atol=1e-12)


# ----- Fixed point ------------------------------------------------------

def test_fixed_point_converges_for_identity():
    """Identity unitary: maximally mixed state is the (only) fixed point."""
    dim_cr, dim_ctc = 2, 3
    U = np.eye(dim_cr * dim_ctc, dtype=complex)
    sigma_cr = maximally_mixed_state(dim_cr)
    result = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                rho_ctc_init=maximally_mixed_state(dim_ctc))
    assert result["converged"]
    # Identity gives maximally-mixed state as fixed point
    assert np.allclose(result["rho_ctc"], maximally_mixed_state(dim_ctc),
                       atol=1e-10)


def test_fixed_point_verifies_self_consistent():
    dim_cr = 2
    U = z3_ctc_unitary(dim_cr=dim_cr, phase=0.2)
    sigma_cr = maximally_mixed_state(dim_cr)
    result = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                rho_ctc_init=maximally_mixed_state(3))
    if result["converged"]:
        ver = verify_fixed_point(U, sigma_cr, result["rho_ctc"], dim_cr=dim_cr)
        assert ver["is_fixed_point"]


def test_fixed_point_z3_cycle_invariant():
    """Z_3 cyclic shift on CTC: maximally-mixed state is fixed."""
    dim_cr = 2
    U = z3_ctc_unitary(dim_cr=dim_cr, phase=0.0)
    sigma_cr = maximally_mixed_state(dim_cr)
    result = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                rho_ctc_init=maximally_mixed_state(3))
    assert result["converged"]
    # The maximally mixed state on CTC is invariant under any unitary
    assert np.allclose(result["rho_ctc"], maximally_mixed_state(3),
                       atol=1e-10)


# ----- Diagnostics -----------------------------------------------------

def test_diagnostics_for_maximally_mixed_state():
    rho = maximally_mixed_state(3)
    d = density_matrix_diagnostics(rho)
    assert d["trace"] == pytest.approx(1.0, abs=1e-12)
    assert d["positive"]
    assert d["hermitian"] < 1e-12
    assert d["purity"] == pytest.approx(1 / 3, abs=1e-12)


def test_diagnostics_for_pure_state():
    psi = np.array([1, 0, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    d = density_matrix_diagnostics(rho)
    assert d["purity"] == pytest.approx(1.0, abs=1e-12)
    assert d["max_eig"] == pytest.approx(1.0, abs=1e-12)
