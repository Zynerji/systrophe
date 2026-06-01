"""The chronology-horizon flicker as a Perron-Frobenius transfer operator.

Context
-------
``chronology_horizon.chaotic_barrier`` drives the Tipler rotation parameter
``a(t)`` on the rotating-dust Lorenz attractor across the supercritical
threshold ``a = 1/2``.  Whenever ``a(t) > 1/2`` the chronology / Cauchy horizon
exists and the renormalized Boulware stress tensor diverges (CTC-open, the
protection barrier is ON); whenever ``a(t) < 1/2`` there is no horizon (CTC-
closed, barrier OFF).  The barrier therefore *flickers* on a chaotic schedule.

This module recasts that flicker as the action of a coarse-grained
**Perron-Frobenius (transfer) operator** on a two-symbol partition of the
attractor:

    'O'  (open)   <->  a(t) > 1/2   (g_openness > 0, horizon present)
    'C'  (closed) <->  a(t) < 1/2   (g_openness = 0, no horizon)

The symbolic sequence ``...OOCCCOCC...`` is a coarse-grained image of the
deterministic-but-chaotic dynamics.  Counting consecutive symbol pairs gives
the 2x2 row-stochastic transition matrix ``P``; its sub-leading eigenvalue is

    lambda_2(P) = tr(P) - 1 = P_OO + P_CC - 1,

the decay rate of the band-occupation autocorrelation, and

    mean dwell time  = 1 / (1 - lambda_2).

What is tautological vs. what is the result
-------------------------------------------
The *mean* dwell time reproduced from ``1/(1-lambda_2)`` matches the empirical
mean run length **by construction** (a first-order Markov / two-state chain has
its mean dwell pinned by the diagonal transition probabilities).  That match is
reported but is NOT the deliverable.

The non-tautological content is the *deviation* of the full empirical
run-length distribution from the geometric law a first-order Markov chain
implies:

  (a) **KS falsification of the Markov property.**  A genuine first-order chain
      has geometric run lengths ``P(len = k) = (1-p) p^{k-1}``.  We KS-test the
      empirical run lengths against that geometric law (per symbol).  A
      *significant* KS deviation is the expected, interesting outcome: it
      quantifies the **non-Markovianity** of the chronology flicker (the Lorenz
      attractor has memory the 2-state coarse graining cannot capture).

  (b) **Markov-order comparison.**  We re-estimate the operator on 3-symbol
      blocks (order-2 coarse graining) and ask whether the order-2 dwell-time
      prediction is closer to the empirical mean than the order-1 prediction --
      i.e. whether adding memory improves the transfer-operator model.

Null discipline (Systrophe rule: no positive without a surrogate)
-----------------------------------------------------------------
``surrogate_lambda2`` phase-randomizes ``a(t)`` (preserving its power spectrum /
PSD exactly but destroying deterministic phase structure).  Genuine Lorenz
dwell-time persistence must give ``lambda_2`` *larger* than the surrogate's:
the real flicker has band-occupation autocorrelation a PSD-matched random
signal does not.  ``scan_novelty`` on the dwell-time series must not return
``'uniform'`` (the dwell sequence carries address-space structure).

Reuses
------
- ``rotating_dust_lorenz.RotatingDustLorenz`` / ``rotation_parameter_timeseries``
- ``chronology_horizon.A_CRIT`` (the a = 1/2 threshold), ``chaotic_barrier``
- ``dctc_gated_circuit.gate_openness`` (CTC openness g(a), 0 below / ramps above)
- ``systrophe.catchers.novelty_catcher.scan_novelty``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from rotating_dust_lorenz import RotatingDustLorenz, rotation_parameter_timeseries
from chronology_horizon import A_CRIT
from dctc_gated_circuit import gate_openness


# --------------------------------------------------------------------------- #
# 1.  Symbolic coarse-graining of the chaotic a(t) flicker.
# --------------------------------------------------------------------------- #
def lorenz_rotation_series(
    a_center: float = 0.62,
    amp: float = 0.18,
    lorenz_t_max: float = 400.0,
    dt: float = 0.02,
    t_transient: float = 20.0,
    seed_state: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Integrate the rotating-dust Lorenz attractor and map to a(t).

    Mirrors ``chronology_horizon.chaotic_barrier``'s construction (same
    a_center/amp convention, ``rotation_parameter_timeseries`` with
    ``eps = amp / a_center``) but returns a long ``a(t)`` track suitable for
    transfer-operator statistics.

    Returns dict with 't', 'a', 'dt'.
    """
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(
        np.asarray(seed_state, dtype=float),
        t_max=lorenz_t_max,
        dt=dt,
        t_transient=t_transient,
    )
    rot = rotation_parameter_timeseries(
        traj, a0=a_center, eps=amp / a_center, clip_min=0.05
    )
    t = rot["t"] - rot["t"][0]
    return {"t": t, "a": np.asarray(rot["a"], dtype=float), "dt": dt}


def symbolize(a: np.ndarray, threshold: float = A_CRIT) -> np.ndarray:
    """Two-state coarse-graining: 1 = 'O' (CTC-open, a>1/2), 0 = 'C' (closed).

    Uses ``gate_openness`` so the partition is exactly the CTC-openness support:
    g(a) > 0  iff  a > A_CRIT.  (gate_openness ramps over a width but is strictly
    positive precisely above threshold, so the >0 test reproduces a > 1/2.)
    """
    a = np.asarray(a, dtype=float)
    g = np.array([gate_openness(float(ai)) for ai in a])
    return (g > 0.0).astype(np.int8)


# --------------------------------------------------------------------------- #
# 2.  Perron-Frobenius transfer operator from consecutive-pair counts.
# --------------------------------------------------------------------------- #
def transition_matrix(symbols: np.ndarray, n_states: int = 2) -> dict:
    """Estimate the row-stochastic transfer operator P by counting pairs.

    P[i, j] = Prob(next = j | current = i), estimated from consecutive symbol
    pairs via ``numpy.bincount`` on the flattened pair index ``i*n + j``.

    Returns dict with:
      'P'            : n x n row-stochastic matrix,
      'counts'       : raw n x n pair-count matrix,
      'lambda_2'     : sub-leading eigenvalue = tr(P) - 1 (2-state identity),
      'lambda_2_eig' : sub-leading eigenvalue from the actual spectrum (check),
      'mean_dwell'   : 1 / (1 - lambda_2)  (autocorrelation timescale),
      'stationary'   : stationary distribution (left eigenvector, lambda=1).
    """
    s = np.asarray(symbols, dtype=np.int64).ravel()
    if s.size < 2:
        raise ValueError("need at least 2 symbols to count transitions")
    cur, nxt = s[:-1], s[1:]
    flat = cur * n_states + nxt
    counts = np.bincount(flat, minlength=n_states * n_states).reshape(
        n_states, n_states
    ).astype(float)

    row = counts.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero for an unvisited state (leave its row uniform).
    safe = np.where(row == 0.0, 1.0, row)
    P = counts / safe
    empty = (row.ravel() == 0.0)
    if np.any(empty):
        P[empty, :] = 1.0 / n_states

    tr = float(np.trace(P))
    lam2_identity = tr - 1.0  # exact for a 2-state stochastic matrix

    # Independent spectral computation (works for any n; cross-checks the 2-state
    # trace identity and is the honest lambda_2 for the order-2 operator).
    evals = np.linalg.eigvals(P)
    evals = evals[np.argsort(-np.abs(evals))]  # leading first (lambda_1 ~ 1)
    lam2_eig = float(np.real(evals[1])) if evals.size > 1 else 0.0

    mean_dwell = float("inf") if abs(1.0 - lam2_identity) < 1e-15 else 1.0 / (
        1.0 - lam2_identity
    )

    # Stationary distribution: left eigenvector for eigenvalue 1.
    w, V = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(V[:, k])
    pi = pi / pi.sum() if pi.sum() != 0 else np.full(n_states, 1.0 / n_states)

    return {
        "P": P,
        "counts": counts,
        "lambda_2": lam2_identity,
        "lambda_2_eig": lam2_eig,
        "mean_dwell": mean_dwell,
        "stationary": pi,
    }


# --------------------------------------------------------------------------- #
# 3.  Run lengths + KS falsification of the first-order Markov (geometric) law.
# --------------------------------------------------------------------------- #
def run_lengths(symbols: np.ndarray) -> dict:
    """Run (dwell) lengths for each symbol value.

    Returns dict mapping symbol value -> int array of consecutive-run lengths,
    plus 'all' for the pooled lengths and 'values'/'starts' bookkeeping.
    """
    s = np.asarray(symbols, dtype=np.int64).ravel()
    if s.size == 0:
        return {"all": np.array([], dtype=int)}
    change = np.flatnonzero(np.diff(s)) + 1
    bounds = np.concatenate(([0], change, [s.size]))
    lengths = np.diff(bounds)
    values = s[bounds[:-1]]
    out = {"all": lengths, "values": values}
    for v in np.unique(s):
        out[int(v)] = lengths[values == v]
    return out


def geometric_ks_test(symbols: np.ndarray, symbol_value: int) -> dict:
    """KS-test empirical run lengths of one symbol against the geometric law
    implied by the first-order transition matrix.

    A first-order Markov chain has geometric dwell:  P(len = k) = (1-p) p^{k-1}
    with persistence  p = P[i, i]  (self-transition probability of that state).
    The geometric CDF is  F(k) = 1 - p^{floor(k)}  for k >= 1.

    A *significant* KS deviation (small p-value) is the expected, interesting
    outcome: it quantifies non-Markovianity (memory the 2-state coarse graining
    misses).  Returns dict with 'ks_stat', 'p_value', 'p_self', 'n_runs',
    'empirical_mean', 'geometric_mean'.
    """
    tm = transition_matrix(symbols)
    p_self = float(tm["P"][symbol_value, symbol_value])
    rl = run_lengths(symbols)
    lengths = np.asarray(rl.get(int(symbol_value), np.array([], dtype=int)), float)
    n_runs = int(lengths.size)
    if n_runs < 3 or not (0.0 < p_self < 1.0):
        return {
            "ks_stat": float("nan"), "p_value": float("nan"),
            "p_self": p_self, "n_runs": n_runs,
            "empirical_mean": float(lengths.mean()) if n_runs else float("nan"),
            "geometric_mean": (1.0 / (1.0 - p_self)) if 0.0 <= p_self < 1.0
            else float("inf"),
        }

    # Geometric CDF on the support k = 1, 2, 3, ...  (number of trials until the
    # first "leave" event), matching how run lengths are defined.
    def geom_cdf(k):
        k = np.asarray(k, dtype=float)
        out = np.where(k < 1.0, 0.0, 1.0 - p_self ** np.floor(k))
        return out

    ks = stats.kstest(lengths, geom_cdf)
    return {
        "ks_stat": float(ks.statistic),
        "p_value": float(ks.pvalue),
        "p_self": p_self,
        "n_runs": n_runs,
        "empirical_mean": float(lengths.mean()),
        "geometric_mean": float(1.0 / (1.0 - p_self)),
    }


# --------------------------------------------------------------------------- #
# 4.  Markov-order comparison: order-1 (2-symbol) vs order-2 (3-symbol blocks).
# --------------------------------------------------------------------------- #
def block_symbols(symbols: np.ndarray, order: int) -> np.ndarray:
    """Encode overlapping length-``order`` blocks as integer super-states.

    order=1 -> the original 2 symbols; order=2 -> 4 super-states (3-symbol
    transfer operator: super-state = (s_t, s_{t+1}), transition to
    (s_{t+1}, s_{t+2})).
    """
    s = np.asarray(symbols, dtype=np.int64).ravel()
    if order <= 1:
        return s
    # Sliding window of width `order`, base-2 encoded.
    n = s.size - order + 1
    if n <= 0:
        return np.array([], dtype=np.int64)
    blocks = np.zeros(n, dtype=np.int64)
    for k in range(order):
        blocks = blocks * 2 + s[k:k + n]
    return blocks


def order_comparison(symbols: np.ndarray) -> dict:
    """Compare the dwell-time prediction of the order-1 vs order-2 operator.

    The honest target is the empirical mean dwell of the *open* ('O') state.
    For order-1 we use ``mean_dwell`` of the 2-state operator restricted to the
    open-state persistence (1/(1-P_OO)).  For order-2 we build the 4-state block
    operator and predict the open-state dwell from the conditional persistence
    given the previous symbol was also open, P[(O,O)->(O,O)] (the memory term).

    Returns dict with both predictions, the empirical open-state mean, and which
    order is closer (improves) -- 'order2_improves' bool.
    """
    s = np.asarray(symbols, dtype=np.int64).ravel()
    rl = run_lengths(s)
    open_lengths = np.asarray(rl.get(1, np.array([], dtype=int)), float)
    emp_open_mean = float(open_lengths.mean()) if open_lengths.size else float("nan")

    # Order-1 prediction: geometric mean from P_OO.
    tm1 = transition_matrix(s, n_states=2)
    p_oo = float(tm1["P"][1, 1])
    pred1 = 1.0 / (1.0 - p_oo) if p_oo < 1.0 else float("inf")

    # Order-2 prediction: 4-state block operator on (s_t, s_{t+1}); super-state
    # 0b11 = (O,O). Its self-persistence P[(O,O)->(O,O)] = Prob(O | prev two O).
    blk = block_symbols(s, order=2)
    pred2 = float("nan")
    p_oo_oo = float("nan")
    p_cc = float(tm1["P"][0, 0])
    p_cc_cc = float("nan")
    if blk.size >= 2:
        tm2 = transition_matrix(blk, n_states=4)
        P2 = tm2["P"]
        oo = 0b11
        cc = 0b00
        p_oo_oo = float(P2[oo, oo])
        p_cc_cc = float(P2[cc, cc])
        # Conditional geometric mean given memory of one prior open step.
        pred2 = 1.0 / (1.0 - p_oo_oo) if p_oo_oo < 1.0 else float("inf")

    err1 = abs(pred1 - emp_open_mean)
    err2 = abs(pred2 - emp_open_mean)
    improves = bool(np.isfinite(err2) and err2 < err1)
    # The non-Markov memory is strongest in whichever state's conditional
    # persistence departs most from its marginal.
    mem_open = abs(p_oo_oo - p_oo) if np.isfinite(p_oo_oo) else float("nan")
    mem_closed = abs(p_cc_cc - p_cc) if np.isfinite(p_cc_cc) else float("nan")
    return {
        "empirical_open_mean": emp_open_mean,
        "order1_pred": pred1,
        "order2_pred": pred2,
        "p_oo": p_oo,
        "p_oo_given_oo": p_oo_oo,
        "p_cc": p_cc,
        "p_cc_given_cc": p_cc_cc,
        "memory_open": float(mem_open),
        "memory_closed": float(mem_closed),
        "max_memory": float(np.nanmax([mem_open, mem_closed])),
        "order1_abs_err": float(err1),
        "order2_abs_err": float(err2),
        "order2_improves": improves,
    }


# --------------------------------------------------------------------------- #
# 5.  Surrogate null: phase-randomize a(t) (preserve PSD), recompute lambda_2.
# --------------------------------------------------------------------------- #
def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized surrogate: identical power spectrum, scrambled phases.

    Replaces each Fourier phase (except DC / Nyquist) with a uniform random
    phase, then inverse-transforms.  The PSD is preserved exactly; deterministic
    phase structure (the Lorenz folding) is destroyed.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spec.size)
    phases[0] = 0.0  # keep DC real
    if n % 2 == 0:
        phases[-1] = 0.0  # keep Nyquist real for even length
    surr = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    return surr


def surrogate_lambda2(
    a: np.ndarray,
    n_surrogates: int = 40,
    threshold: float = A_CRIT,
    seed: int = 0,
) -> dict:
    """Compare the real dwell-time persistence lambda_2 to a PSD-matched null.

    For each surrogate we phase-randomize ``a(t)`` (preserving PSD), symbolize at
    the SAME threshold, and recompute lambda_2.  Genuine Lorenz band-occupation
    autocorrelation must exceed the surrogate distribution.

    Returns dict with 'lambda_2_real', surrogate mean/std/array, a one-sided
    'p_value' (fraction of surrogates >= real), and 'exceeds_null'.
    """
    a = np.asarray(a, dtype=float)
    real = transition_matrix(symbolize(a, threshold))["lambda_2"]
    rng = np.random.default_rng(seed)
    surr_lam = np.empty(n_surrogates)
    for i in range(n_surrogates):
        sa = phase_randomize(a, rng)
        # Match the surrogate's threshold to its own median offset so the
        # open-fraction is comparable (PSD-preserving surrogates have the same
        # variance; we keep the absolute a=1/2 partition for a fair physical
        # comparison, but recenter to the real series' mean to preserve the
        # occupation fraction the PSD implies).
        sa_shifted = sa - sa.mean() + a.mean()
        surr_lam[i] = transition_matrix(symbolize(sa_shifted, threshold))["lambda_2"]
    surr_mean = float(np.mean(surr_lam))
    surr_std = float(np.std(surr_lam))
    p_value = float((np.sum(surr_lam >= real) + 1) / (n_surrogates + 1))
    return {
        "lambda_2_real": float(real),
        "lambda_2_surrogate_mean": surr_mean,
        "lambda_2_surrogate_std": surr_std,
        "lambda_2_surrogate": surr_lam,
        "p_value": p_value,
        "exceeds_null": bool(real > surr_mean),
        "z_score": float((real - surr_mean) / surr_std) if surr_std > 0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# 6.  Novelty catcher on the dwell-time series (must NOT be 'uniform').
# --------------------------------------------------------------------------- #
def dwell_time_series(symbols: np.ndarray) -> np.ndarray:
    """The ordered sequence of run (dwell) lengths -- the dwell-time series."""
    return np.asarray(run_lengths(symbols)["all"], dtype=float)


def novelty_on_dwell(symbols: np.ndarray, window: int = 8) -> "object":
    """Run ``scan_novelty`` on overlapping windows of the dwell-time series.

    The "parameter" axis indexes successive windows; the output for each window
    is the vector of dwell lengths in it.  A structured dwell sequence yields a
    non-'uniform' verdict.

    Returns the ``NoveltyScanResult`` (has ``.verdict``).
    """
    from systrophe.catchers.novelty_catcher import scan_novelty

    dwell = dwell_time_series(symbols)
    n = dwell.size
    if n < window + 2:
        window = max(2, n // 2)
    n_win = n - window + 1
    starts = np.arange(n_win, dtype=float)

    def output_fn(p: float) -> np.ndarray:
        i = int(round(p))
        i = max(0, min(i, n_win - 1))
        return dwell[i:i + window]

    return scan_novelty(
        starts,
        output_fn,
        parameter_label="dwell_window_start",
    )


# --------------------------------------------------------------------------- #
# 7.  Top-level driver: assemble the full report.
# --------------------------------------------------------------------------- #
@dataclass
class FlickerTransferReport:
    """Container for the full transfer-operator analysis of the flicker."""

    P: np.ndarray
    lambda_2: float
    mean_dwell: float
    stationary: np.ndarray
    fraction_open: float
    n_symbols: int
    ks_open: dict
    ks_closed: dict
    order_cmp: dict
    surrogate: dict
    novelty_verdict: str
    extras: dict = field(default_factory=dict)


def analyze_flicker(
    a_center: float = 0.62,
    amp: float = 0.18,
    lorenz_t_max: float = 400.0,
    dt: float = 0.02,
    n_surrogates: int = 40,
    seed: int = 0,
) -> FlickerTransferReport:
    """End-to-end: drive a(t), symbolize, build the transfer operator, run the
    KS falsification, the Markov-order comparison, the surrogate null, and the
    novelty catcher.  Returns a ``FlickerTransferReport``.
    """
    series = lorenz_rotation_series(
        a_center=a_center, amp=amp, lorenz_t_max=lorenz_t_max, dt=dt
    )
    a = series["a"]
    sym = symbolize(a)
    tm = transition_matrix(sym)
    ks_o = geometric_ks_test(sym, symbol_value=1)
    ks_c = geometric_ks_test(sym, symbol_value=0)
    oc = order_comparison(sym)
    surr = surrogate_lambda2(a, n_surrogates=n_surrogates, seed=seed)
    nov = novelty_on_dwell(sym)

    return FlickerTransferReport(
        P=tm["P"],
        lambda_2=tm["lambda_2"],
        mean_dwell=tm["mean_dwell"],
        stationary=tm["stationary"],
        fraction_open=float(np.mean(sym)),
        n_symbols=int(sym.size),
        ks_open=ks_o,
        ks_closed=ks_c,
        order_cmp=oc,
        surrogate=surr,
        novelty_verdict=str(nov.verdict),
        extras={
            "lambda_2_eig": tm["lambda_2_eig"],
            "counts": tm["counts"],
            "a_min": float(a.min()),
            "a_max": float(a.max()),
        },
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    rep = analyze_flicker()
    np.set_printoptions(precision=4, suppress=True)
    print("Transfer operator P (rows=current C,O; cols=next):")
    print(rep.P)
    print(f"lambda_2 (= tr P - 1)        : {rep.lambda_2:.4f}")
    print(f"lambda_2 (spectral check)    : {rep.extras['lambda_2_eig']:.4f}")
    print(f"mean dwell time (1/(1-l2))   : {rep.mean_dwell:.3f} steps")
    print(f"stationary (C, O)            : {rep.stationary}")
    print(f"fraction open (a>1/2)        : {rep.fraction_open:.3f}")
    print(f"n symbols                    : {rep.n_symbols}")
    print("--- KS falsification of first-order Markov (geometric) law ---")
    print(f"OPEN  : KS={rep.ks_open['ks_stat']:.4f} p={rep.ks_open['p_value']:.2e} "
          f"(emp mean {rep.ks_open['empirical_mean']:.2f} vs geom "
          f"{rep.ks_open['geometric_mean']:.2f})")
    print(f"CLOSED: KS={rep.ks_closed['ks_stat']:.4f} p={rep.ks_closed['p_value']:.2e} "
          f"(emp mean {rep.ks_closed['empirical_mean']:.2f} vs geom "
          f"{rep.ks_closed['geometric_mean']:.2f})")
    print("--- Markov-order comparison (open-state dwell) ---")
    oc = rep.order_cmp
    print(f"empirical open mean : {oc['empirical_open_mean']:.3f}")
    print(f"order-1 pred        : {oc['order1_pred']:.3f}  (err {oc['order1_abs_err']:.3f})")
    print(f"order-2 pred        : {oc['order2_pred']:.3f}  (err {oc['order2_abs_err']:.3f})")
    print(f"order-2 improves    : {oc['order2_improves']}")
    print("--- surrogate null (PSD-matched phase randomization) ---")
    sg = rep.surrogate
    print(f"lambda_2 real       : {sg['lambda_2_real']:.4f}")
    print(f"lambda_2 surrogate  : {sg['lambda_2_surrogate_mean']:.4f} "
          f"+/- {sg['lambda_2_surrogate_std']:.4f}")
    print(f"exceeds null        : {sg['exceeds_null']}  (p={sg['p_value']:.3f}, "
          f"z={sg['z_score']:.2f})")
    print(f"novelty verdict     : {rep.novelty_verdict}")
