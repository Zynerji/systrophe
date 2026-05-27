"""Tests for the Born-rule D-CTC witness framework."""

import math

import numpy as np
import pytest

from systrophe.foundations.born_rule_dctc import (
    BornWitness,
    born_rule_witness,
    brun_wilde_unitary,
    cnot_then_hadamard_unitary,
    dctc_output_state,
    hadamard_swap_unitary,
    helstrom_bound_density,
    helstrom_bound_pure,
    mobius_smoke_test,
)


# ----- Helstrom bound ----------------------------------------------------


def test_helstrom_orthogonal_states_is_one():
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    psi_1 = np.array([0.0, 1.0], dtype=complex)
    assert helstrom_bound_pure(psi_0, psi_1) == pytest.approx(1.0, abs=1e-12)


def test_helstrom_identical_states_is_half():
    psi = np.array([1.0, 0.0], dtype=complex)
    assert helstrom_bound_pure(psi, psi) == pytest.approx(0.5, abs=1e-12)


def test_helstrom_zero_and_plus():
    # P = 1/2 + (1/2) sqrt(1 - 1/2) = 1/2 + 1/(2 sqrt(2)).
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    psi_1 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    expected = 0.5 + 0.5 * math.sqrt(0.5)
    assert helstrom_bound_pure(psi_0, psi_1) == pytest.approx(
        expected, rel=1e-12,
    )


def test_helstrom_density_matches_pure_for_pure_inputs():
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    psi_1 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    rho_0 = np.outer(psi_0, psi_0.conj())
    rho_1 = np.outer(psi_1, psi_1.conj())
    p_pure = helstrom_bound_pure(psi_0, psi_1)
    p_dens = helstrom_bound_density(rho_0, rho_1)
    assert p_dens == pytest.approx(p_pure, abs=1e-12)


# ----- unitary builders --------------------------------------------------


def _is_unitary(U: np.ndarray) -> bool:
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10)


def test_hadamard_swap_is_unitary():
    assert _is_unitary(hadamard_swap_unitary())


def test_cnot_then_hadamard_is_unitary():
    assert _is_unitary(cnot_then_hadamard_unitary())


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.5, 3.0])
def test_brun_wilde_unitary_is_unitary(theta):
    assert _is_unitary(brun_wilde_unitary(theta))


# ----- D-CTC output state ------------------------------------------------


def test_dctc_output_state_converges_for_swap():
    # Simple SWAP unitary: U|x>_CR |y>_CTC = |y>_CR |x>_CTC.
    SWAP = np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=complex,
    )
    sigma = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    out = dctc_output_state(SWAP, sigma, dim_cr=2)
    assert out["converged"]
    # CR output trace = 1 and Hermitian.
    rho = out["rho_cr_out"]
    assert np.real(np.trace(rho)) == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(rho, rho.conj().T, atol=1e-10)


# ----- Born witness ------------------------------------------------------


def test_born_witness_returns_finite_numbers():
    w = born_rule_witness()
    assert isinstance(w, BornWitness)
    assert math.isfinite(w.P_helstrom)
    assert math.isfinite(w.P_dctc_max)
    assert 0.5 <= w.P_helstrom <= 1.0
    assert 0.0 <= w.P_dctc_max <= 1.0
    assert w.best_unitary != ""


def test_born_witness_baseline_zero_plus():
    # Helstrom for |0> vs |+> = 1/2 + 1/(2 sqrt(2)) ~ 0.8536.
    w = born_rule_witness()
    expected_hel = 0.5 + 0.5 * math.sqrt(0.5)
    assert w.P_helstrom == pytest.approx(expected_hel, abs=1e-10)


def test_born_witness_simple_unitaries_dont_violate():
    """The shipped 2-dim-CTC unitary family does NOT exhibit Born violation.

    This is the documented standing finding: SWAP-like unitaries preserve
    distinguishability, and the parametric Brun-Wilde-style family
    chosen here doesn't reach the cyclic-power regime that Brun-Wilde
    Section 5 requires. The witness is correct; the open knob is U.
    """
    w = born_rule_witness()
    assert w.born_violated is False
    assert w.margin <= 1e-8


def test_born_witness_custom_states():
    # Pick a different non-orthogonal pair: |0> vs (cos theta |0> + sin theta |1>)
    theta = math.pi / 3.0
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    psi_1 = np.array([math.cos(theta), math.sin(theta)], dtype=complex)
    w = born_rule_witness(psi_0=psi_0, psi_1=psi_1, n_theta=11)
    overlap = abs(np.vdot(psi_0, psi_1))
    expected_hel = 0.5 + 0.5 * math.sqrt(max(0.0, 1.0 - overlap ** 2))
    assert w.P_helstrom == pytest.approx(expected_hel, abs=1e-10)


# ----- Möbius smoke ------------------------------------------------------


def test_mobius_smoke_test_returns_dict_with_available_flag():
    out = mobius_smoke_test()
    assert isinstance(out, dict)
    assert "available" in out
    assert isinstance(out["available"], bool)
    # The vendored subset does NOT include dinos.temporal_loop, so the
    # smoke test should report unavailable on a clean install.
    assert out["available"] is False
    assert "error" in out
    assert "dinos" in out["error"].lower()
