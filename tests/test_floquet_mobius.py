"""Tests for joint Floquet-Mobius spectrum on (time-circle x Z_3 branch)."""

import numpy as np
import pytest

from systrophe.floquet_mobius import (
    analyze_floquet_mobius,
    brillouin_zone_wrap,
    floquet_propagator,
    joint_floquet_spectrum,
    joint_static_hamiltonian,
    static_limit_check,
    z3_cycle_shift,
    z3_hopping_matrix,
    z3_symmetry_check,
)


# ----- Z_3 operators -----------------------------------------------------

def test_z3_hopping_is_hermitian():
    H = z3_hopping_matrix(g=1.7)
    assert np.allclose(H, H.conj().T, atol=1e-14)


def test_z3_cycle_shift_is_unitary():
    S = z3_cycle_shift()
    assert np.allclose(S @ S.conj().T, np.eye(3), atol=1e-14)


def test_z3_cycle_shift_cube_is_identity():
    S = z3_cycle_shift()
    assert np.allclose(S @ S @ S, np.eye(3), atol=1e-14)


# ----- Static joint Hamiltonian -----------------------------------------

def test_joint_static_hamiltonian_diagonal():
    """Hopping = 0 gives a diagonal matrix."""
    H = joint_static_hamiltonian(np.array([1.0, 2.0, 3.0]), hopping=0.0)
    assert np.allclose(H, np.diag([1.0, 2.0, 3.0]), atol=1e-14)


def test_joint_static_hamiltonian_with_hopping():
    H = joint_static_hamiltonian(np.array([0.0, 0.0, 0.0]), hopping=0.5)
    # Should be all 0.5 on off-diagonal in cyclic pattern
    assert H[0, 1] == 0.5
    assert H[1, 2] == 0.5
    assert H[2, 0] == 0.5
    assert H[0, 0] == 0.0


def test_joint_static_hamiltonian_validates_shape():
    with pytest.raises(ValueError):
        joint_static_hamiltonian(np.array([1.0, 2.0]), hopping=0.0)


# ----- Floquet propagator -----------------------------------------------

def test_floquet_propagator_is_unitary():
    H = joint_static_hamiltonian(np.array([0.5, 1.0, 1.5]), hopping=0.1)
    U = floquet_propagator(H, drive_amp=0.2, omega_drive=1.0, n_steps=500)
    assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-6)


def test_no_drive_propagator_is_diagonal_phase():
    """Drive amp = 0, hopping = 0: U(T) = diag(exp(-i e_b T))."""
    energies = np.array([0.5, 1.0, 1.5])
    H = joint_static_hamiltonian(energies, hopping=0.0)
    omega = 2.0
    T = 2 * np.pi / omega
    U = floquet_propagator(H, drive_amp=0.0, omega_drive=omega, n_steps=500)
    expected = np.diag(np.exp(-1j * energies * T))
    assert np.allclose(U, expected, atol=1e-6)


# ----- Spectrum extraction ----------------------------------------------

def test_static_limit_recovers_branch_energies():
    energies = np.array([0.2, 0.5, 0.8])
    omega = 4.0  # > max energy, no BZ wrap needed
    check = static_limit_check(energies, omega_drive=omega, n_steps=500)
    assert check["max_err"] < 1e-5


def test_brillouin_zone_wrap():
    eps = np.array([0.0, 1.5, -2.0, 3.5])
    omega = 2.0
    wrapped = brillouin_zone_wrap(eps, omega)
    # All in (-1, 1]
    assert np.all(wrapped > -1.0 - 1e-12)
    assert np.all(wrapped <= 1.0 + 1e-12)


def test_z3_symmetry_invariance():
    """Cyclic permutation of branch_energies leaves the spectrum invariant."""
    energies = np.array([0.2, 0.5, 0.8])
    check = z3_symmetry_check(
        energies, hopping=0.1, drive_amp=0.3, omega_drive=2.5,
        n_steps=500,
    )
    assert check["max_set_diff"] < 1e-5


# ----- Drive opens gaps -------------------------------------------------

def test_drive_modifies_spectrum():
    """A nonzero drive shifts the spectrum from the static result."""
    energies = np.array([0.2, 0.5, 0.8])
    omega = 4.0
    spec_static = joint_floquet_spectrum(
        joint_static_hamiltonian(energies, hopping=0.0),
        drive_amp=0.0, omega_drive=omega, n_steps=500,
    )
    spec_driven = joint_floquet_spectrum(
        joint_static_hamiltonian(energies, hopping=0.0),
        drive_amp=0.4, omega_drive=omega, n_steps=500,
    )
    diff = np.max(np.abs(np.sort(spec_driven) - np.sort(spec_static)))
    assert diff > 1e-3


# ----- Full analysis ----------------------------------------------------

def test_analyze_returns_result():
    energies = np.array([0.2, 0.5, 0.8])
    result = analyze_floquet_mobius(
        energies, hopping=0.1, drive_amp=0.3, omega_drive=2.5,
        n_steps=200,
    )
    assert result.quasi_energies.shape == (3,)
    assert np.all(np.isfinite(result.quasi_energies))
    assert result.propagator.shape == (3, 3)
