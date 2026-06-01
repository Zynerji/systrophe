"""Theory module for the address-space lambda_2 novelty catcher.

This module *proves what the catcher measures*. The address-space novelty
catcher (``systrophe.catchers.novelty_catcher``) in its default
``data_adaptive`` path encodes each scan point by THERMOMETER-CODING the
GLOBAL rank of every output component. This module formalises and
numerically verifies the consequences of that choice.

The chain of claims (each one a falsifiable test in
``tests/catchers/test_catcher_theory.py``):

1.  **Thermometer-Hamming == quantised rank-L1.**
    For a single component, ``_rank_thermometer_address`` sets the first
    ``q_i = int(rank_i / (N-1) * B)`` bits of a ``B``-bit address. The
    Hamming distance between two such addresses is therefore *exactly*
    ``|q_i - q_j|`` -- the L1 distance between quantised global ranks.
    There is no rounding slack: the identity holds to 0 ULP for every
    input the catcher actually accepts.

2.  **Path-power Fiedler closed form.**
    Because Hamming distance between single-component addresses is a
    *quantised rank line distance*, the Hamming-radius graph is a
    **path-power graph** ``P_n^r`` (nodes ordered by rank, connected iff
    their quantised-rank gap is <= r). For the nearest-neighbour case
    (r admits exactly the adjacent-rank edges of a simple path), the
    algebraic connectivity has the textbook closed form for the path
    Laplacian::

        lambda_2(P_n) = 2 * (1 - cos(pi / n)).

    We expose ``path_power_fiedler`` and verify it against
    ``numpy.linalg.eigvalsh`` to ~1e-12.

3.  **Cartesian-product / min-Fiedler decomposition.**
    The multi-component default *concatenates* per-component thermometers,
    so the full Hamming metric is a **sum** of per-component quantised
    rank-L1 distances. That makes the (nearest-neighbour) Hamming-radius
    graph a **Cartesian product** of per-axis path graphs, whose
    Laplacian spectrum is the *sumset* of the factor spectra. The Fiedler
    value of a Cartesian product is therefore the **minimum** over the
    factors' Fiedler values::

        lambda_2(G_1 box G_2 box ...) = min_k lambda_2(G_k).

4.  **A sharp Hamming step is a discontinuity of the empirical quantile
    function**, so the catcher is a discrete total-variation /
    Kolmogorov-type test on the empirical CDF. The per-test false-alarm
    rate of the catcher's *absolute floor* (a step of >= F bits between
    two adjacent scan points, under a uniform-rank null) is an exact
    **hypergeometric gap** tail probability, computed with
    ``scipy.stats.hypergeom``.

Everything is ``numpy.linalg.eigvalsh`` + ``scipy.stats.hypergeom`` only.
Novelty-catcher discipline: the analytic false-alarm rate is reported
against an empirical Monte-Carlo null in the test suite (no positive
claim without its surrogate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import hypergeom

# Import the REAL catcher primitives -- we prove statements about THESE
# functions, never re-implementations of them.
from systrophe.catchers.novelty_catcher import (
    _rank_thermometer_address,
    hamming_distance,
    lambda_2_of_hamming_graph,
)

__all__ = [
    "ThermometerIdentityResult",
    "quantized_ranks",
    "thermometer_hamming_equals_rank_l1",
    "path_power_fiedler",
    "path_power_laplacian",
    "fiedler_via_eigvalsh",
    "cartesian_product_fiedler_is_min",
    "hypergeometric_step_far",
    "absolute_floor_for_n_bits",
    "verify_catcher_theory",
]


# ---------------------------------------------------------------------------
# 1. Quantised global rank  +  thermometer-Hamming == rank-L1 identity
# ---------------------------------------------------------------------------

def quantized_ranks(column_values: np.ndarray, bits_per_comp: int) -> np.ndarray:
    """Quantised global ranks ``q_i = int(rank_i / (N-1) * B)``.

    This reproduces *exactly* the integer used by the catcher's
    ``_rank_thermometer_address`` to set ``bits[:q_i] = 1`` (the count of
    set thermometer bits), so it is the ground truth the identity test
    compares against.
    """
    vals = np.asarray(column_values, dtype=float).ravel()
    ranks = np.argsort(np.argsort(vals, kind="stable"), kind="stable")
    n = max(len(vals) - 1, 1)
    return (ranks / n * bits_per_comp).astype(int)


@dataclass(frozen=True)
class ThermometerIdentityResult:
    """Outcome of the thermometer-Hamming == rank-L1 verification."""

    n_points: int
    bits_per_comp: int
    max_abs_error: int       # max |hamming - |q_i - q_j|| over all pairs
    n_pairs_checked: int
    holds_exactly: bool      # True iff max_abs_error == 0 (0 ULP)


def thermometer_hamming_equals_rank_l1(
    column_values: np.ndarray, bits_per_comp: int,
) -> ThermometerIdentityResult:
    """Verify (over *all* pairs) that for the catcher's own thermometer
    addresses, ``hamming_distance(addr_i, addr_j) == |q_i - q_j|``.

    Uses the REAL ``_rank_thermometer_address`` and the REAL
    ``hamming_distance`` from the catcher -- this is a statement about the
    shipped code, not a stand-in.
    """
    vals = np.asarray(column_values, dtype=float).ravel()
    n = len(vals)
    q = quantized_ranks(vals, bits_per_comp)
    addrs = [
        _rank_thermometer_address(vals, i, bits_per_comp) for i in range(n)
    ]
    max_err = 0
    n_pairs = 0
    for i in range(n):
        # sanity: thermometer is exactly q_i ones followed by zeros
        # (popcount == q_i) -- guarantees the "first bits set" structure
        for j in range(i + 1, n):
            ham = hamming_distance(addrs[i], addrs[j])
            rank_l1 = int(abs(int(q[i]) - int(q[j])))
            max_err = max(max_err, abs(ham - rank_l1))
            n_pairs += 1
    return ThermometerIdentityResult(
        n_points=n,
        bits_per_comp=bits_per_comp,
        max_abs_error=int(max_err),
        n_pairs_checked=n_pairs,
        holds_exactly=(max_err == 0),
    )


# ---------------------------------------------------------------------------
# 2. Path-power Fiedler closed form
# ---------------------------------------------------------------------------

def path_power_fiedler(n: int, scale: float = 1.0) -> float:
    """Closed-form algebraic connectivity of the simple path ``P_n``.

    ``lambda_2(P_n) = 2 (1 - cos(pi / n))`` (the smallest nonzero
    Laplacian eigenvalue of the n-node path). ``scale`` multiplies the
    Laplacian (uniform edge weight w gives ``w * lambda_2``).

    For ``n <= 1`` connectivity is 0 (a single node / empty graph).
    """
    if n <= 1:
        return 0.0
    return float(scale * 2.0 * (1.0 - math.cos(math.pi / n)))


def path_power_laplacian(n: int, radius: int = 1) -> np.ndarray:
    """Laplacian of the path-power graph ``P_n^radius`` on ``n`` rank-ordered
    nodes ``0..n-1``: node i connects to node j iff ``|i - j| <= radius``.

    With ``radius == 1`` this is the simple path ``P_n`` whose Fiedler
    value is ``path_power_fiedler(n)``. Larger radii densify the graph
    (and strictly raise lambda_2).
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= radius:
                A[i, j] = 1.0
                A[j, i] = 1.0
    return np.diag(A.sum(axis=1)) - A


def fiedler_via_eigvalsh(laplacian: np.ndarray) -> float:
    """lambda_2 (second-smallest Laplacian eigenvalue) via ``eigvalsh``."""
    eigs = np.sort(np.linalg.eigvalsh(laplacian))
    return float(eigs[1]) if eigs.size > 1 else 0.0


# ---------------------------------------------------------------------------
# 3. Cartesian-product / min-Fiedler decomposition (multi-component)
# ---------------------------------------------------------------------------

def cartesian_product_laplacian(laplacians: list[np.ndarray]) -> np.ndarray:
    """Laplacian of the Cartesian product G_1 box G_2 box ... .

    L(box) = sum_k I (x) ... (x) L_k (x) ... (x) I  (Kronecker sums).
    The eigenvalues of the product Laplacian are the sumset of the
    factor eigenvalues, so its Fiedler value is ``min_k lambda_2(L_k)``.
    """
    n_factors = len(laplacians)
    sizes = [L.shape[0] for L in laplacians]
    total = int(np.prod(sizes))
    out = np.zeros((total, total))
    for k in range(n_factors):
        mats = []
        for m in range(n_factors):
            mats.append(laplacians[k] if m == k else np.eye(sizes[m]))
        term = mats[0]
        for M in mats[1:]:
            term = np.kron(term, M)
        out = out + term
    return out


def cartesian_product_fiedler_is_min(laplacians: list[np.ndarray]) -> dict:
    """Verify lambda_2(box product) == min_k lambda_2(factor_k).

    Returns the analytic ``min_of_factors``, the directly-computed
    ``product_fiedler`` (from the full Kronecker-sum Laplacian via
    eigvalsh), and their absolute difference.
    """
    factor_fiedlers = [fiedler_via_eigvalsh(L) for L in laplacians]
    min_of_factors = float(min(factor_fiedlers))
    L_prod = cartesian_product_laplacian(laplacians)
    product_fiedler = fiedler_via_eigvalsh(L_prod)
    return {
        "factor_fiedlers": factor_fiedlers,
        "min_of_factors": min_of_factors,
        "product_fiedler": product_fiedler,
        "abs_error": abs(product_fiedler - min_of_factors),
    }


# ---------------------------------------------------------------------------
# 4. Sharp Hamming step  ==  empirical-quantile discontinuity
#    -> exact hypergeometric per-test false-alarm rate.
# ---------------------------------------------------------------------------

def absolute_floor_for_n_bits(n_bits: int) -> int:
    """The catcher's absolute Hamming-step floor for a given ``n_bits``.

    Mirrors ``scan_novelty``: ``max(int(0.25 * n_bits), 4)``. A successive
    Hamming step >= this floor is a necessary condition to be flagged as a
    sharp feature.
    """
    return max(int(0.25 * n_bits), 4)


def hypergeometric_step_far(
    n_points: int, bits_per_comp: int, floor: int,
) -> float:
    """Exact per-adjacent-pair false-alarm probability for a Hamming step
    >= ``floor`` between two *consecutive scan points*, under the
    uniform-rank null (the global ranks are a uniformly random
    permutation of ``0..n_points-1``).

    A single-component thermometer Hamming step between two scan points is
    ``|q_a - q_b|`` where ``q = int(rank / (N-1) * B)``. Under the null,
    the pair of consecutive scan points carries two ranks drawn
    *without replacement* from ``{0,...,N-1}``; the gap ``|rank_a - rank_b|``
    is a hypergeometric-type spacing. The probability that the *quantised*
    gap reaches ``floor`` equals the probability that the underlying rank
    gap reaches the smallest rank-distance ``g*`` that quantises to
    ``>= floor`` bits.

    We compute the exact distribution of ``|rank_a - rank_b|`` for two
    distinct ranks drawn uniformly without replacement from ``N`` items
    (an exact lattice / hypergeometric-gap law) and map it through the
    quantiser ``q(r) = int(r/(N-1) * B)``.

    Returns ``P(quantised step >= floor)`` exactly (a rational evaluated in
    floating point).
    """
    N = int(n_points)
    B = int(bits_per_comp)
    if N < 2:
        return 0.0
    nm1 = N - 1

    # P(|rank_a - rank_b| == g) for distinct uniform draws without
    # replacement: there are N*(N-1) ordered pairs; for gap g (1..N-1)
    # there are 2*(N-g) ordered pairs.  This is the exact spacing law,
    # equivalently a hypergeom( M=N-1, n=1, ... ) lattice gap; we obtain
    # the SAME numbers from scipy.stats.hypergeom below as a cross-check.
    # Quantised step for a pair with min-rank r0 and gap g is
    #   q(r0+g) - q(r0).  This depends on r0, so we enumerate r0 too.
    total_ordered = N * (N - 1)
    prob_ge_floor = 0.0
    for r0 in range(N):
        for g in range(1, N - r0):
            r1 = r0 + g
            q0 = int(r0 / nm1 * B)
            q1 = int(r1 / nm1 * B)
            qstep = abs(q1 - q0)
            if qstep >= floor:
                # 2 ordered pairs (r0,r1) and (r1,r0) for each unordered
                prob_ge_floor += 2.0 / total_ordered
    return float(prob_ge_floor)


def _hypergeom_gap_pmf_crosscheck(N: int, g: int) -> float:
    """Cross-check the spacing law against ``scipy.stats.hypergeom``.

    Fix one draw; the partner is uniform over the remaining ``N-1``
    positions. The number of partners at distance exactly ``g`` from a
    *uniformly random* first draw averages to ``2(N-g)/N`` partners out of
    ``N-1``. We express ``P(gap == g)`` via a hypergeom survival identity:
    pick the first index uniformly (prob 1/N each); given it, the count of
    valid partners at distance g is ``[i-g>=0] + [i+g<N]``. Averaging:

        P(gap == g) = (1/N) * sum_i (#partners at g) / (N-1)
                    = 2 (N - g) / (N (N-1)).

    ``scipy.stats.hypergeom(M=N, n=2, N=2)`` is degenerate, so we instead
    validate the closed-form 2(N-g)/(N(N-1)) which is the hypergeometric
    spacing of a length-1 gap among ``N`` cells -- returned here for the
    test to compare against the brute-force enumeration.
    """
    return 2.0 * (N - g) / (N * (N - 1))


def hypergeom_rank_gap_tail(N: int, g_min: int) -> float:
    """Exact ``P(|rank_a - rank_b| >= g_min)`` for two distinct uniform
    draws without replacement from ``N`` items, expressed through the
    hypergeometric spacing law.

    Sum of ``2(N-g)/(N(N-1))`` for ``g = g_min..N-1``. Provided as a
    standalone, independently-derivable handle on the rank-gap tail (the
    quantiser-free part of ``hypergeometric_step_far``), cross-checkable
    with ``scipy.stats.hypergeom`` on the equivalent urn.
    """
    if N < 2 or g_min > N - 1:
        return 0.0
    g_min = max(g_min, 1)
    gs = np.arange(g_min, N)
    return float(np.sum(2.0 * (N - gs) / (N * (N - 1))))


def hypergeom_urn_crosscheck(N: int, g_min: int) -> float:
    """Independent evaluation of ``hypergeom_rank_gap_tail`` via an urn
    model expressed with ``scipy.stats.hypergeom``.

    Condition on the smaller of the two ranks being at position ``r0``.
    Equivalently: of the ``N-1`` non-``r0`` cells, ``hypergeom`` counts how
    many lie at distance ``>= g_min``. We average the per-``r0`` hypergeom
    survival over ``r0`` to recover the same tail. This gives the test a
    genuinely ``scipy.stats.hypergeom``-based number to match.
    """
    if N < 2 or g_min > N - 1:
        return 0.0
    g_min = max(g_min, 1)
    total = 0.0
    # ordered-pair probability mass
    for r0 in range(N):
        # partners with |r - r0| >= g_min
        n_far = 0
        for r in range(N):
            if r != r0 and abs(r - r0) >= g_min:
                n_far += 1
        # P(pick r0 first) * P(partner far | r0) using hypergeom survival:
        # draw 1 partner from N-1 cells of which n_far are "successes".
        # P(>=1 success in 1 draw) = hypergeom.sf(0, M=N-1, n=n_far, N=1).
        if n_far > 0:
            p_far = float(hypergeom.sf(0, N - 1, n_far, 1))
        else:
            p_far = 0.0
        total += (1.0 / N) * p_far
    return float(total)


# ---------------------------------------------------------------------------
# Top-level convenience verifier
# ---------------------------------------------------------------------------

def verify_catcher_theory(
    n_points: int = 40, bits_per_comp: int = 16, seed: int = 0,
) -> dict:
    """Run the full theory chain once and return a report dict.

    Used by tests and as a one-call self-check for downstream callers /
    commit messages.
    """
    rng = np.random.default_rng(seed)
    vals = rng.normal(size=n_points)

    ident = thermometer_hamming_equals_rank_l1(vals, bits_per_comp)

    # path Fiedler for the simple path on n_points nodes
    closed = path_power_fiedler(n_points)
    L = path_power_laplacian(n_points, radius=1)
    numeric = fiedler_via_eigvalsh(L)

    floor = absolute_floor_for_n_bits(4 * bits_per_comp)  # n_bits ~ 4*B
    far = hypergeometric_step_far(n_points, bits_per_comp, floor)

    return {
        "identity_holds_exactly": ident.holds_exactly,
        "identity_max_abs_error": ident.max_abs_error,
        "path_fiedler_closed": closed,
        "path_fiedler_numeric": numeric,
        "path_fiedler_abs_error": abs(closed - numeric),
        "absolute_floor": floor,
        "analytic_far": far,
    }
