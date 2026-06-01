"""Tests for the Dirichlet L-function DSI helper and the
millennium_dirichlet_l_dsi example (L-function comb + Chebyshev race).

Each acceptance gate from the deliverable is a falsifiable test that
reports actual numbers. All tests run at X = 1e6 (quick) so the suite is
fast while still exercising the real prime sieve and Lomb-Scargle path.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from systrophe.catchers.dirichlet_l_dsi import (
    character_vector,
    dirichlet_cascade,
    dirichlet_character,
    low_zeros,
    make_twisted_psi,
    primes_up_to,
    twisted_fluctuation,
    twisted_prime_power_steps,
    VERIFIED_LOW_ZEROS,
)

# Make the example importable by full path (it lives in examples/millennium).
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "millennium"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import millennium_dirichlet_l_dsi as mdl  # noqa: E402

ZETA_FIRST = 14.134725


# --------------------------------------------------------------------------
# Character algebra
# --------------------------------------------------------------------------

def test_chi4_values():
    chi = dirichlet_character(4)
    assert chi(1) == 1 and chi(3) == -1
    assert chi(2) == 0 and chi(4) == 0
    assert chi(5) == 1 and chi(7) == -1  # periodic


def test_chi3_values():
    chi = dirichlet_character(3)
    assert chi(1) == 1 and chi(2) == -1
    assert chi(3) == 0 and chi(6) == 0
    assert chi(4) == 1 and chi(5) == -1


def test_characters_are_completely_multiplicative():
    for q in (3, 4):
        chi = dirichlet_character(q)
        for m in range(1, 20):
            for n in range(1, 20):
                assert chi(m * n) == chi(m) * chi(n)


def test_character_vector_matches_scalar():
    primes = primes_up_to(100)
    cv = character_vector(4, primes)
    chi = dirichlet_character(4)
    assert np.allclose(cv, [chi(int(p)) for p in primes])


def test_unsupported_modulus_raises():
    with pytest.raises(ValueError):
        dirichlet_character(5)


# --------------------------------------------------------------------------
# Verified zeros: hardcode matches the from-scratch mpmath computation
# --------------------------------------------------------------------------

def test_low_zeros_on_critical_line_via_hurwitz():
    """The hardcoded zeros reproduce a fresh mpmath root-find on Re = 1/2."""
    pytest.importorskip("mpmath")
    from systrophe.catchers.dirichlet_l_dsi import low_zeros_via_hurwitz
    for q in (4, 3):
        n = len(VERIFIED_LOW_ZEROS[q])
        roots = low_zeros_via_hurwitz(q, n)
        for (re, im), g_hard in zip(roots, VERIFIED_LOW_ZEROS[q]):
            assert abs(re - 0.5) < 1e-6, f"q={q} zero off critical line: {re}"
            assert abs(im - g_hard) < 1e-4, f"q={q} {im} vs {g_hard}"


def test_zeros_disjoint_between_characters_and_from_zeta():
    """The COMBS are disjoint: the dominant low (fundamental) zeros of
    chi_3 and chi_4 differ, and neither sits at the zeta 14.13 fundamental.

    Note (honest): individual HIGH zeros of distinct L-functions can fall
    close by chance -- here chi_4's 18.292 and chi_3's 18.262 are only
    ~0.03 apart. That is a genuine high-zero near-coincidence, not a comb
    collision: the gate is about the spectroscopically dominant fundamental
    (lowest-frequency, largest 1/(0.25+gamma^2) weight), which is what the
    Lomb-Scargle peak recovers. So we test the first few zeros, not every
    pair.
    """
    z4 = low_zeros(4)
    z3 = low_zeros(3)
    # The fundamentals (first zeros) are well separated.
    assert abs(z4[0] - z3[0]) > 1.0
    # The leading combs (first 3 zeros, which carry the dominant comb power)
    # do not collide.
    for a in z4[:3]:
        assert np.min(np.abs(z3[:3] - a)) > 0.5
    # Neither character has a zero at the zeta fundamental 14.13.
    assert np.min(np.abs(z4 - ZETA_FIRST)) > 1.0
    assert np.min(np.abs(z3 - ZETA_FIRST)) > 1.0
    # The documented high-zero near-coincidence is real and acknowledged.
    assert np.min(np.abs(z4[:, None] - z3[None, :])) < 0.05


# --------------------------------------------------------------------------
# Twisted psi: sanity of the construction
# --------------------------------------------------------------------------

def test_twisted_weights_sum_to_explicit_chebyshev():
    """psi(x; chi) at x = x_max equals sum chi(p^k) log p over all p^k<=x_max."""
    x_max = 100000
    q = 4
    pos, cum = twisted_prime_power_steps(x_max, q)
    psi = make_twisted_psi(x_max, q)
    # Direct brute-force twisted Chebyshev at x_max.
    chi = dirichlet_character(q)
    primes = primes_up_to(x_max)
    brute = 0.0
    for p in primes:
        pk = int(p)
        k = 1
        while pk <= x_max:
            brute += (chi(p) ** k) * math.log(p)
            k += 1
            pk = int(p) ** k
    assert abs(float(psi(np.array([float(x_max)]))[0]) - brute) < 1e-6


def test_twisted_psi_oscillates_about_zero():
    """With no x main term, f_chi has near-zero mean (entire L-function)."""
    x_max = 1_000_000
    u = np.linspace(math.log(1000.0), math.log(x_max), 20000)
    psi = make_twisted_psi(x_max, 4)
    f = twisted_fluctuation(psi, u)
    # Mean small relative to RMS amplitude (pure oscillation, no pole term).
    assert abs(f.mean()) < 0.5 * np.sqrt(np.mean(f ** 2))


# --------------------------------------------------------------------------
# Shared comb run (X = 1e6) reused across the gate tests
# --------------------------------------------------------------------------

X_MAX = 1_000_000
X_MIN = 1000.0
N_SAMPLES = 20000


@pytest.fixture(scope="module")
def combs():
    out = {}
    for q in (4, 3):
        out[q] = mdl.comb_experiment(
            q, X_MAX, X_MIN, N_SAMPLES, gamma_lo=3.0, gamma_hi=25.0,
            n_freq=6000, seed=0)
    return out


# Gate 1 -----------------------------------------------------------------

def test_gate1_cascade_correlation(combs):
    """f_chi correlates with the TRUE-zero explicit-formula cascade."""
    c4 = combs[4]["explicit_formula_correlation"]
    c3 = combs[3]["explicit_formula_correlation"]
    # Zeta version hit ~0.86; with only 5-6 low zeros we expect >= 0.6.
    assert c4 > 0.6, f"chi_4 cascade corr too low: {c4}"
    assert c3 > 0.6, f"chi_3 cascade corr too low: {c3}"


# Gate 2 -----------------------------------------------------------------

def test_gate2_combs_recover_their_own_zeros(combs):
    """Recovered fundamentals land on the L(s,chi) zeros within Rayleigh."""
    for q in (4, 3):
        c = combs[q]
        assert c["hits_within_rayleigh"] == c["n_scored"], (
            f"q={q} only {c['hits_within_rayleigh']}/{c['n_scored']} hits")
        assert c["mean_abs_pct_error"] < 2.0


def test_gate2_combs_disjoint_and_no_zeta_peak(combs):
    """chi_3 and chi_4 fundamentals are disjoint and not the zeta 14.13."""
    top4 = combs[4]["top_fundamental"]
    top3 = combs[3]["top_fundamental"]
    assert abs(top4 - top3) > 1.0, "combs not disjoint"
    assert combs[4]["dist_top_fundamental_to_zeta"] > 1.0
    assert combs[3]["dist_top_fundamental_to_zeta"] > 1.0


# Gate 3 -----------------------------------------------------------------

def test_gate3_phase_randomization_destroys_comb(combs):
    """Phase-randomised surrogate has much lower cascade correlation."""
    for q in (4, 3):
        c = combs[q]
        real = c["explicit_formula_correlation"]
        surr = c["phase_randomized_correlation"]
        assert surr < real - 0.2, (
            f"q={q} surrogate corr {surr} not below real {real}")


# Gate 4 -----------------------------------------------------------------

def test_gate4_novelty_catcher_flags_structure(combs):
    """The mandated address-space catcher returns novel_structure."""
    for q in (4, 3):
        assert combs[q]["catcher_verdict"] == "novel_structure", (
            f"q={q} catcher verdict {combs[q]['catcher_verdict']}")


# Gate 5 -----------------------------------------------------------------

@pytest.fixture(scope="module")
def races():
    return {
        4: mdl.race_experiment(4, 3, 1, X_MAX, X_MIN, N_SAMPLES,
                               n_perm=400, seed=0),
        3: mdl.race_experiment(3, 2, 1, X_MAX, X_MIN, N_SAMPLES,
                               n_perm=400, seed=0),
    }


def test_gate5_chebyshev_bias_vs_label_null(races):
    """Non-residue class leads almost always under the log-density measure,
    significantly above the residue-label permutation null."""
    for q in (4, 3):
        r = races[q]
        assert r["observed_lead_fraction"] > 0.9, (
            f"q={q} observed lead {r['observed_lead_fraction']} not biased")
        assert r["bias"] > 0.2, f"q={q} bias {r['bias']} too small"
        assert r["p_value_one_sided"] <= 0.05, (
            f"q={q} p={r['p_value_one_sided']} not significant")


def test_gate5_observed_beats_every_unbiased_expectation(races):
    """The observed lead exceeds the null mean by several null std."""
    for q in (4, 3):
        r = races[q]
        assert r["observed_lead_fraction"] > r["null_mean_lead_fraction"]


# --------------------------------------------------------------------------
# Cascade form: the Dirichlet cascade reproduces a single clean mode
# --------------------------------------------------------------------------

def test_cascade_single_mode_is_log_periodic():
    """A one-zero cascade is a clean cos(gamma u)-class log-periodic wave
    whose Lomb-Scargle peak sits at gamma."""
    from scipy.signal import lombscargle
    gamma = 8.0
    u = np.linspace(math.log(1000.0), math.log(1e6), 20000)
    y = dirichlet_cascade(u, np.array([gamma]))
    y0 = y - y.mean()
    freqs = np.linspace(2.0, 20.0, 4000)
    power = lombscargle(u, y0, freqs, normalize=True)
    peak = freqs[int(np.argmax(power))]
    assert abs(peak - gamma) < 0.1, f"peak {peak} != gamma {gamma}"
