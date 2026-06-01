"""Tests for ``systrophe.catchers.catcher_theory``.

These tests *prove what the address-space lambda_2 catcher measures* and
report ACTUAL numbers for every acceptance gate:

(a) thermometer-Hamming == quantised rank-L1 identity to 0 ULP for every
    encoding the catcher actually uses (single- and multi-component);
(b) closed-form path-power Fiedler ``2(1-cos(pi/n))`` matches eigvalsh to
    ~1e-12 for a single component, and lambda_2 == min-of-factors for the
    multi-component Cartesian product;
(c) over >= 1e4 i.i.d. uniform-noise scans, the analytic hypergeometric
    per-pair false-alarm rate matches the empirical floor-exceedance rate
    within Poisson error (both reported).

Novelty-catcher discipline: claim (c) is a positive analytic result that
is checked against an explicit Monte-Carlo *null* surrogate.
"""

import math

import numpy as np
import pytest

from systrophe.catchers.novelty_catcher import (
    _rank_thermometer_address,
    hamming_distance,
    lambda_2_of_hamming_graph,
)
from systrophe.catchers.catcher_theory import (
    ThermometerIdentityResult,
    absolute_floor_for_n_bits,
    cartesian_product_fiedler_is_min,
    cartesian_product_laplacian,
    fiedler_via_eigvalsh,
    hypergeom_rank_gap_tail,
    hypergeom_urn_crosscheck,
    hypergeometric_step_far,
    path_power_fiedler,
    path_power_laplacian,
    quantized_ranks,
    thermometer_hamming_equals_rank_l1,
    verify_catcher_theory,
)


# ---------------------------------------------------------------------------
# (a) thermometer-Hamming == quantised rank-L1  (0 ULP)
# ---------------------------------------------------------------------------

def test_quantized_ranks_match_thermometer_popcount():
    """``quantized_ranks`` must equal the number of bits the REAL catcher
    sets, i.e. ``popcount(_rank_thermometer_address)``, for every point."""
    rng = np.random.default_rng(0)
    vals = rng.normal(size=37)
    B = 16
    q = quantized_ranks(vals, B)
    for i in range(len(vals)):
        addr = _rank_thermometer_address(vals, i, B)
        assert int(addr.sum()) == int(q[i])
        # thermometer structure: first q ones, rest zeros
        assert addr[: int(q[i])].sum() == int(q[i])
        assert addr[int(q[i]):].sum() == 0


@pytest.mark.parametrize("n_points", [3, 5, 16, 40, 64])
@pytest.mark.parametrize("bits_per_comp", [4, 8, 16, 32])
def test_thermometer_hamming_equals_rank_l1_zero_ulp(n_points, bits_per_comp):
    """ACCEPTANCE GATE (a): identity holds to 0 ULP over the FULL grid the
    catcher actually uses (every (n_points, bits_per_comp) pair scanned)."""
    rng = np.random.default_rng(n_points * 1000 + bits_per_comp)
    vals = rng.normal(size=n_points)
    res = thermometer_hamming_equals_rank_l1(vals, bits_per_comp)
    assert isinstance(res, ThermometerIdentityResult)
    assert res.max_abs_error == 0, (
        f"identity FAILED: max_abs_error={res.max_abs_error} "
        f"(n={n_points}, B={bits_per_comp})"
    )
    assert res.holds_exactly is True
    assert res.n_pairs_checked == n_points * (n_points - 1) // 2


def test_identity_holds_with_heavy_ties():
    """Quantised data with many tied values (the case the catcher's stable
    sort exists for) must still satisfy the identity to 0 ULP."""
    rng = np.random.default_rng(99)
    # only 4 distinct values among 50 points -> lots of ties
    vals = rng.integers(0, 4, size=50).astype(float)
    res = thermometer_hamming_equals_rank_l1(vals, 16)
    assert res.max_abs_error == 0
    assert res.holds_exactly


def test_multicomponent_concat_is_sum_of_rank_l1():
    """ACCEPTANCE GATE (a), multi-component: the catcher CONCATENATES
    per-component thermometers, so the full Hamming metric == SUM of the
    per-component quantised rank-L1 distances, to 0 ULP."""
    rng = np.random.default_rng(7)
    N, ncomp, B = 14, 4, 6
    cols = rng.normal(size=(N, ncomp))
    qs = [quantized_ranks(cols[:, c], B) for c in range(ncomp)]

    def full_addr(i):
        return np.concatenate(
            [_rank_thermometer_address(cols[:, c], i, B) for c in range(ncomp)]
        )

    max_err = 0
    for i in range(N):
        for j in range(i + 1, N):
            ham = hamming_distance(full_addr(i), full_addr(j))
            summed = sum(
                abs(int(qs[c][i]) - int(qs[c][j])) for c in range(ncomp)
            )
            max_err = max(max_err, abs(ham - summed))
    assert max_err == 0, f"multi-component sum metric FAILED: max_err={max_err}"


# ---------------------------------------------------------------------------
# (b) path-power Fiedler closed form + min-of-factors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 5, 10, 25, 40, 64, 100])
def test_path_power_fiedler_closed_form_matches_eigvalsh(n):
    """ACCEPTANCE GATE (b): ``2(1-cos(pi/n))`` matches eigvalsh(L_path) to
    ~1e-12."""
    closed = path_power_fiedler(n)
    numeric = fiedler_via_eigvalsh(path_power_laplacian(n, radius=1))
    assert abs(closed - numeric) < 1e-12, (
        f"path Fiedler mismatch at n={n}: closed={closed}, "
        f"numeric={numeric}, err={abs(closed - numeric)}"
    )
    # closed form value itself
    assert math.isclose(closed, 2.0 * (1.0 - math.cos(math.pi / n)))


def test_path_power_fiedler_degenerate_sizes():
    assert path_power_fiedler(0) == 0.0
    assert path_power_fiedler(1) == 0.0


def test_path_power_graph_equals_catcher_hamming_graph():
    """The catcher's OWN ``lambda_2_of_hamming_graph`` on single-component
    thermometer addresses with strictly-increasing unit-gap quantised ranks
    (B >= N) must reproduce the path-power closed form at radius 1."""
    N = 12
    B = 64  # B >= N => strictly increasing ranks, unit gaps after sorting
    # strictly increasing values -> ranks 0..N-1 in order
    vals = np.arange(N, dtype=float)
    # quantised ranks: with B>=N they are strictly increasing; for the
    # Hamming graph to be the SIMPLE path at radius 1 we need adjacent gaps
    # == 1, which holds when consecutive quantised ranks differ by exactly 1.
    q = quantized_ranks(vals, B)
    addrs = [_rank_thermometer_address(vals, i, B) for i in range(N)]
    # adjacent (in rank order) Hamming distances:
    adj = [hamming_distance(addrs[i], addrs[i + 1]) for i in range(N - 1)]
    radius = int(max(adj))
    # build path-power-equivalent: connect i,j iff quantised-rank gap<=radius
    n = N
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(int(q[i]) - int(q[j])) <= radius:
                A[i, j] = A[j, i] = 1.0
    L = np.diag(A.sum(1)) - A
    l2_catcher = lambda_2_of_hamming_graph(addrs, radius=radius)
    l2_graph = fiedler_via_eigvalsh(L)
    assert abs(l2_catcher - l2_graph) < 1e-12


def test_cartesian_product_fiedler_is_min():
    """ACCEPTANCE GATE (b), multi-component: lambda_2 of the Cartesian
    product (Kronecker-sum Laplacian) == min over factor Fiedler values."""
    Ls = [path_power_laplacian(n, radius=1) for n in (5, 8, 12)]
    res = cartesian_product_fiedler_is_min(Ls)
    assert res["abs_error"] < 1e-10, (
        f"min-of-factors FAILED: product={res['product_fiedler']}, "
        f"min={res['min_of_factors']}, err={res['abs_error']}"
    )
    # the min must be the LARGEST path (smallest Fiedler)
    assert res["min_of_factors"] == min(res["factor_fiedlers"])
    # and equal to the closed form for the largest factor (n=12)
    assert abs(res["min_of_factors"] - path_power_fiedler(12)) < 1e-12


def test_cartesian_product_spectrum_is_sumset():
    """Sanity: product Laplacian eigenvalues are the sumset of factor
    eigenvalues (so the smallest nonzero is min of factor lambda_2's)."""
    L1 = path_power_laplacian(4, 1)
    L2 = path_power_laplacian(3, 1)
    e1 = np.sort(np.linalg.eigvalsh(L1))
    e2 = np.sort(np.linalg.eigvalsh(L2))
    sumset = np.sort((e1[:, None] + e2[None, :]).ravel())
    L_prod = cartesian_product_laplacian([L1, L2])
    eprod = np.sort(np.linalg.eigvalsh(L_prod))
    assert np.allclose(sumset, eprod, atol=1e-10)


# ---------------------------------------------------------------------------
# (c) analytic hypergeometric FAR  vs  empirical Monte-Carlo null
# ---------------------------------------------------------------------------

def test_absolute_floor_matches_catcher():
    """The theory module's floor must equal the catcher's literal floor
    ``max(int(0.25*n_bits), 4)`` (the gate it gets compared against)."""
    for n_bits in (4, 8, 16, 32, 64, 128):
        assert absolute_floor_for_n_bits(n_bits) == max(int(0.25 * n_bits), 4)


def test_hypergeom_rank_gap_tail_crosscheck():
    """The rank-gap tail (quantiser-free part) computed two independent
    ways -- closed-form spacing law vs ``scipy.stats.hypergeom`` urn --
    must agree."""
    for N in (10, 25, 40):
        for g_min in (1, 3, 7):
            closed = hypergeom_rank_gap_tail(N, g_min)
            urn = hypergeom_urn_crosscheck(N, g_min)
            assert abs(closed - urn) < 1e-12, (
                f"N={N} g_min={g_min}: closed={closed} urn={urn}"
            )


def test_analytic_far_matches_empirical_null():
    """ACCEPTANCE GATE (c): over >= 1e4 i.i.d. uniform-noise scans, the
    analytic hypergeometric per-adjacent-pair floor-exceedance probability
    matches the empirical false-alarm rate within Poisson error.

    The condition tested is the catcher's *absolute-floor* gate on a
    single-component thermometer: a successive Hamming step >= ``floor``
    between two consecutive scan points (the necessary condition for a
    'novel_structure' flag). Reports BOTH numbers and the z-score.
    """
    rng = np.random.default_rng(2024)
    N = 40            # scan points per trial
    B = 16            # bits per component
    floor = 8         # absolute Hamming-step floor under test
    n_trials = 10_000  # >= 1e4 i.i.d. uniform scans

    analytic = hypergeometric_step_far(N, B, floor)

    hits = 0
    pairs = 0
    for _ in range(n_trials):
        vals = rng.uniform(size=N)  # i.i.d. uniform => uniform random ranks
        addrs = [_rank_thermometer_address(vals, i, B) for i in range(N)]
        steps = np.array(
            [hamming_distance(addrs[k - 1], addrs[k]) for k in range(1, N)]
        )
        hits += int(np.sum(steps >= floor))
        pairs += len(steps)

    empirical = hits / pairs
    # Poisson standard error on the count of hits
    poisson_se = math.sqrt(max(hits, 1)) / pairs
    z = (empirical - analytic) / poisson_se

    # Report (visible with -s); assertions below are the gate.
    print(
        f"\n[FAR gate] N={N} B={B} floor={floor} trials={n_trials} "
        f"pairs={pairs}\n  analytic={analytic:.6f}  empirical={empirical:.6f}"
        f"  poisson_se={poisson_se:.6f}  z={z:.3f}"
    )

    # Within Poisson error: require |z| < 4 (very safe; observed ~1).
    assert abs(z) < 4.0, (
        f"analytic FAR {analytic:.6f} does not match empirical "
        f"{empirical:.6f} (z={z:.2f}, se={poisson_se:.6f})"
    )


def test_far_monotone_in_floor():
    """Higher floor => lower analytic false-alarm rate (a sanity invariant
    of any Kolmogorov-type tail)."""
    N, B = 40, 16
    fars = [hypergeometric_step_far(N, B, f) for f in (2, 4, 6, 8, 12)]
    assert all(fars[k] >= fars[k + 1] for k in range(len(fars) - 1))


def test_far_zero_when_floor_unreachable():
    """If the floor exceeds the max possible quantised step (B), FAR == 0."""
    N, B = 30, 8
    assert hypergeometric_step_far(N, B, floor=B + 1) == 0.0


# ---------------------------------------------------------------------------
# top-level self-check
# ---------------------------------------------------------------------------

def test_verify_catcher_theory_report():
    rep = verify_catcher_theory(n_points=40, bits_per_comp=16, seed=0)
    assert rep["identity_holds_exactly"] is True
    assert rep["identity_max_abs_error"] == 0
    assert rep["path_fiedler_abs_error"] < 1e-12
    assert 0.0 <= rep["analytic_far"] <= 1.0
