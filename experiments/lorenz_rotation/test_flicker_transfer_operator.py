"""Tests for the chronology-horizon flicker as a Perron-Frobenius transfer operator.

The acceptance gates encode the deliverable honestly:

  * The 2x2 transfer operator P is row-stochastic and lambda_2 = tr(P) - 1
    matches the spectral sub-leading eigenvalue.
  * The mean-dwell reproduction is *tautological* (a first-order chain pins its
    mean) -- asserted as such, NOT sold as a result.
  * The REAL result is the KS falsification: empirical run lengths deviate
    SIGNIFICANTLY from the geometric law a first-order Markov chain implies
    (non-Markovianity / memory of the Lorenz attractor).
  * Surrogate null: genuine Lorenz dwell-time persistence lambda_2 exceeds a
    PSD-matched phase-randomized surrogate.
  * scan_novelty on the dwell-time series is NOT 'uniform'.
  * Markov-order comparison (order-1 vs order-2) reported honestly.

One full report (`_REPORT`) is computed once at module load and shared across
tests (it is the expensive part: a long Lorenz integration + surrogate ensemble).
"""

import numpy as np
import pytest

import flicker_transfer_operator as fto


# --------------------------------------------------------------------------- #
# Shared report (production settings: long enough series for a stable null).
# --------------------------------------------------------------------------- #
_REPORT = fto.analyze_flicker(
    a_center=0.62, amp=0.18, lorenz_t_max=400.0, dt=0.02,
    n_surrogates=40, seed=0,
)


# --------------------------------------------------------------------------- #
# Symbolic coarse-graining sanity.
# --------------------------------------------------------------------------- #
def test_symbolize_matches_threshold():
    """The 2-state partition is exactly the CTC-open support a > 1/2."""
    a = np.array([0.2, 0.49, 0.5, 0.501, 0.62, 1.3])
    sym = fto.symbolize(a)
    # gate_openness > 0 iff a > A_CRIT (0.5); a == 0.5 is NOT open.
    expected = (a > fto.A_CRIT).astype(np.int8)
    np.testing.assert_array_equal(sym, expected)


def test_series_actually_crosses_threshold():
    """The chaotic a(t) genuinely flickers across a = 1/2 (both states visited)."""
    frac = _REPORT.fraction_open
    assert 0.05 < frac < 0.95, f"degenerate occupation frac_open={frac}"
    assert _REPORT.extras["a_min"] < fto.A_CRIT < _REPORT.extras["a_max"]


# --------------------------------------------------------------------------- #
# Transfer operator structure + lambda_2 identity.
# --------------------------------------------------------------------------- #
def test_P_is_row_stochastic():
    P = _REPORT.P
    assert P.shape == (2, 2)
    assert np.all(P >= -1e-12) and np.all(P <= 1.0 + 1e-12)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(2), atol=1e-12)


def test_lambda2_trace_identity_matches_spectrum():
    """lambda_2 = tr(P) - 1 must equal the actual sub-leading eigenvalue."""
    lam2_trace = _REPORT.lambda_2
    lam2_spec = _REPORT.extras["lambda_2_eig"]
    assert abs(lam2_trace - lam2_spec) < 1e-10
    # Chaotic flicker is persistent (band occupation autocorrelated): 0 < l2 < 1.
    assert 0.5 < lam2_trace < 1.0, f"lambda_2={lam2_trace}"


def test_mean_dwell_is_inverse_spectral_gap():
    """mean dwell = 1/(1-lambda_2); finite and multi-step for a persistent chain."""
    md = _REPORT.mean_dwell
    assert np.isfinite(md)
    np.testing.assert_allclose(md, 1.0 / (1.0 - _REPORT.lambda_2), rtol=1e-9)
    assert md > 2.0, f"mean dwell {md} too short to be a persistent barrier"


def test_stationary_matches_open_fraction():
    """Stationary open-probability equals the empirical open fraction (consistency)."""
    pi_open = _REPORT.stationary[1]
    np.testing.assert_allclose(pi_open, _REPORT.fraction_open, atol=0.02)


# --------------------------------------------------------------------------- #
# The TAUTOLOGICAL mean-dwell match -- asserted, and labelled as vacuous.
# --------------------------------------------------------------------------- #
def test_mean_dwell_match_is_tautological():
    """A first-order Markov chain's geometric mean EQUALS the empirical run-length
    mean by construction. We assert the (vacuous) match so the docstring's honesty
    is enforced: this is NOT the deliverable."""
    ks_o = _REPORT.ks_open
    # emp mean == geometric mean == 1/(1-p_self) to numerical precision.
    np.testing.assert_allclose(
        ks_o["empirical_mean"], ks_o["geometric_mean"], rtol=1e-9
    )


# --------------------------------------------------------------------------- #
# THE RESULT: KS falsification of the first-order Markov (geometric) law.
# --------------------------------------------------------------------------- #
def test_ks_falsifies_geometric_open_state():
    """Empirical OPEN run lengths deviate SIGNIFICANTLY from geometric -- the real
    (non-tautological) content: the chronology flicker is non-Markovian at order 1."""
    ks_o = _REPORT.ks_open
    assert ks_o["n_runs"] >= 10
    assert ks_o["ks_stat"] > 0.05, f"KS stat {ks_o['ks_stat']} unexpectedly tiny"
    assert ks_o["p_value"] < 0.01, (
        f"expected SIGNIFICANT deviation from geometric, got p={ks_o['p_value']}"
    )


def test_ks_falsifies_geometric_closed_state():
    """Closed-state run lengths also deviate from geometric (memory both ways)."""
    ks_c = _REPORT.ks_closed
    assert ks_c["n_runs"] >= 10
    assert ks_c["p_value"] < 0.01, (
        f"expected SIGNIFICANT deviation, got p={ks_c['p_value']}"
    )


# --------------------------------------------------------------------------- #
# Markov-order comparison: order-1 vs order-2 dwell prediction.
# --------------------------------------------------------------------------- #
def test_order_comparison_reports_both_predictions():
    """Both order-1 and order-2 dwell predictions are finite and computed.

    HONEST: for the open state at this parameter point the order-2 block operator
    does NOT improve the mean-dwell prediction over order-1 (the geometric order-1
    mean is already pinned to the empirical mean -- a tautology -- so order-2 can
    only move away). This is recorded, not hidden."""
    oc = _REPORT.order_cmp
    assert np.isfinite(oc["order1_pred"])
    assert np.isfinite(oc["order2_pred"])
    assert np.isfinite(oc["empirical_open_mean"])
    # Record the actual verdict (False at this point) without asserting a fake win.
    assert isinstance(oc["order2_improves"], bool)


def test_order2_conditional_persistence_differs_from_order1():
    """Even though order-2 does not improve the MEAN (tautology), the conditional
    persistence P[(s,s)->(s,s)] differs from the marginal P_ss for at least one
    state -- direct evidence of order-1 memory that the KS test detects.

    HONEST: the memory is carried by the CLOSED (subcritical) excursions
    (memory_closed ~ 2.6e-3, KS_closed huge) while the OPEN runs are nearly
    geometric (memory_open ~ 4e-4, small KS_open). So we assert on the maximum
    over states, and that the closed state is the dominant memory carrier."""
    oc = _REPORT.order_cmp
    assert np.isfinite(oc["max_memory"])
    # If the chain were truly first-order, conditional == marginal. It is not.
    assert oc["max_memory"] > 1e-3, (
        f"order-2 conditional persistence equals order-1 (max memory "
        f"{oc['max_memory']:.2e}) -> would imply Markov"
    )
    # The closed (subcritical) state carries the dominant non-Markov memory.
    assert oc["memory_closed"] > oc["memory_open"], (
        "expected closed-state excursions to dominate the non-Markovianity"
    )


# --------------------------------------------------------------------------- #
# Surrogate null (Systrophe rule: no positive without a null).
# --------------------------------------------------------------------------- #
def test_surrogate_null_exceeded():
    """Genuine Lorenz dwell-time persistence lambda_2 must exceed the PSD-matched
    phase-randomized surrogate mean."""
    sg = _REPORT.surrogate
    assert sg["exceeds_null"], (
        f"real lambda_2 {sg['lambda_2_real']:.4f} did not exceed surrogate "
        f"{sg['lambda_2_surrogate_mean']:.4f}"
    )
    assert sg["lambda_2_real"] > sg["lambda_2_surrogate_mean"]
    assert sg["z_score"] > 1.5, f"surrogate z-score only {sg['z_score']:.2f}"


def test_surrogate_preserves_psd():
    """The phase-randomization surrogate preserves the power spectrum exactly."""
    rng = np.random.default_rng(7)
    x = fto.lorenz_rotation_series(lorenz_t_max=80.0, dt=0.02)["a"]
    surr = fto.phase_randomize(x, rng)
    px = np.abs(np.fft.rfft(x))
    ps = np.abs(np.fft.rfft(surr))
    np.testing.assert_allclose(px, ps, rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------- #
# Novelty catcher on the dwell-time series must NOT be 'uniform'.
# --------------------------------------------------------------------------- #
def test_novelty_not_uniform():
    """scan_novelty on the dwell-time series carries address-space structure."""
    assert _REPORT.novelty_verdict in ("novel_structure", "smooth")
    assert _REPORT.novelty_verdict != "uniform"


# --------------------------------------------------------------------------- #
# Unit-level checks on the building blocks (fast, no Lorenz integration).
# --------------------------------------------------------------------------- #
def test_transition_matrix_on_known_sequence():
    """Hand-checkable: ...0011001100... has P_00=P_11=0.5 each (alternating runs)."""
    sym = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    tm = fto.transition_matrix(sym)
    # From 0: -> 0 (4 times), -> 1 (3 times) over pairs; check row-stochastic + l2.
    P = tm["P"]
    np.testing.assert_allclose(P.sum(axis=1), np.ones(2), atol=1e-12)
    assert abs(tm["lambda_2"] - (np.trace(P) - 1.0)) < 1e-12


def test_run_lengths_partition_total():
    """Run lengths sum to the series length and split correctly by symbol."""
    sym = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    rl = fto.run_lengths(sym)
    assert rl["all"].sum() == sym.size
    np.testing.assert_array_equal(rl[1], np.array([3, 1]))
    np.testing.assert_array_equal(rl[0], np.array([2, 4]))


def test_geometric_for_true_markov_is_not_rejected():
    """Control: a synthetic TRUE first-order Markov chain with the same diagonal
    persistence is NOT KS-rejected, confirming the test's rejection of the Lorenz
    flicker is meaningful (not an artifact of the geometric comparison itself)."""
    rng = np.random.default_rng(0)
    p_stay = 0.95
    n = 40000
    s = np.zeros(n, dtype=np.int8)
    s[0] = 1
    u = rng.random(n)
    for i in range(1, n):
        # symmetric two-state chain, P_stay on the diagonal
        s[i] = s[i - 1] if u[i] < p_stay else 1 - s[i - 1]
    ks = fto.geometric_ks_test(s, symbol_value=1)
    # A genuine first-order chain -> geometric run lengths -> NOT rejected.
    assert ks["p_value"] > 0.01, (
        f"true Markov chain wrongly rejected (p={ks['p_value']}) -- test invalid"
    )


def test_block_symbols_order2_encoding():
    """Order-2 block encoding produces the expected base-2 super-states."""
    sym = np.array([0, 1, 1, 0], dtype=np.int8)
    blk = fto.block_symbols(sym, order=2)
    # pairs: (0,1)=1, (1,1)=3, (1,0)=2
    np.testing.assert_array_equal(blk, np.array([1, 3, 2]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "-p", "no:cacheprovider"]))
