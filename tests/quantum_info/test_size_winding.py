"""Tests for the first-principles size-winding analysis of the GJW channel."""

import numpy as np
import pytest

from systrophe.quantum_info.size_winding import (
    SizeWindingReport,
    epr_state,
    first_principles_report,
    mean_operator_size,
    pauli_size_distribution,
    size_operator,
    summarise,
    verify_size_operator,
)


def test_coupling_is_the_operator_size_operator():
    """The central derivation (star): V P_L|EPR> = (1 - 2 size/n) P_L|EPR>,
    verified to machine precision for every Pauli on L."""
    assert verify_size_operator(2) < 1e-12
    assert verify_size_operator(3) < 1e-12
    assert verify_size_operator(4) < 1e-12


def test_size_operator_is_hermitian_with_epr_top_eigenstate():
    n = 3
    V = size_operator(n)
    assert np.allclose(V, V.conj().T)
    epr = epr_state(n)
    # EPR is the +1 eigenstate (size 0)
    assert np.vdot(epr, V @ epr).real == pytest.approx(1.0, abs=1e-9)


def test_pauli_size_distribution_normalized():
    n = 3
    X0 = np.kron(np.array([[0, 1], [1, 0]], dtype=complex),
                 np.eye(2 ** (n - 1), dtype=complex))
    dist = pauli_size_distribution(X0, n)
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-9)
    assert dist.get(1, 0.0) == pytest.approx(1.0, abs=1e-9)  # X_0 has size 1


def test_chaotic_scrambling_grows_operator_size():
    """SYK scrambling grows the operator size (precondition for size-winding)."""
    from scipy.linalg import expm
    from systrophe.quantum_info.erepr_channel import syk_hamiltonian
    n = 3
    X0 = np.kron(np.array([[0, 1], [1, 0]], dtype=complex),
                 np.eye(2 ** (n - 1), dtype=complex))
    H = syk_hamiltonian(n, seed=1)
    s0 = mean_operator_size(X0, n)
    U = expm(-1j * H * 2.0)
    s_scrambled = mean_operator_size(U.conj().T @ X0 @ U, n)
    assert s0 == pytest.approx(1.0, abs=1e-6)
    assert s_scrambled > 1.5   # operator has grown


def test_first_principles_report():
    r = first_principles_report(3)
    assert isinstance(r, SizeWindingReport)
    assert r.size_operator_residual < 1e-12          # (star) rigorous
    assert r.mechanism_active is True                 # SYK activates winding
    assert r.syk_teleport_fidelity > r.haar_teleport_fidelity
    assert r.deterministic_channel_fidelity == pytest.approx(1.0, abs=1e-9)


def test_summary():
    s = summarise(first_principles_report(3))
    assert "size-operator" in s and "deterministic F=1.000" in s
