"""Detection statistics for the morphic-resonance harness.

Three layers, weakest-assumption first:

  1. :func:`catcher_verdict`         -- model-free address-space novelty
     catcher (Systrophe) on the ordered cost sequence: is there ANY ordered
     structure at all?
  2. :func:`count_vs_time_identifiability` -- the decisive regression:
     cost ~ time + cumulative_count. Reports the partial effect of count
     beyond a secular time-trend AND the collinearity (VIF) between them, so
     a count effect is never claimed when count is indistinguishable from time.
  3. :func:`acausal_test`            -- the CTC fingerprint: does cost depend
     on FUTURE realizations beyond past ones? A causal mechanism (any
     conventional learning) cannot produce this.
"""

from __future__ import annotations

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty


def catcher_verdict(panel, n_bits: int = 32) -> dict:
    """Address-space novelty catcher over the instance-order axis.

    Each instance's (scalar) cost is the output hashed at its order position.
    Returns the catcher verdict + max consecutive Hamming step. This is the
    model-free "is there ordered structure" gate.
    """
    cost = panel.cost
    order = np.arange(len(cost), dtype=float)

    # output_fn returns the running window of cost ending at this index, so
    # the rank-thermometer encoding has multi-component structure to bite on.
    def output_fn(p):
        i = int(round(p))
        i = max(0, min(len(cost) - 1, i))
        lo = max(0, i - 3)
        return cost[lo:i + 1]

    res = scan_novelty(order, output_fn, n_bits=n_bits,
                       data_adaptive=True, parameter_label="instance_order")
    return {
        "verdict": res.verdict,
        "n_sharp_features": len(res.sharp_features),
        "lambda_2_at_radius": {int(k): float(v)
                               for k, v in res.lambda_2_at_radius.items()},
    }


def _ols(X: np.ndarray, y: np.ndarray):
    """Plain OLS with t-stats. X already includes an intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    return beta, se, t, sigma2


def _vif(a: np.ndarray, b: np.ndarray) -> float:
    """Variance-inflation factor between two regressors (collinearity).

    VIF = 1 / (1 - r^2). VIF > ~10 means the two are too collinear to
    attribute an effect to one rather than the other -> ``unidentifiable``.
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    if denom == 0:
        return float("inf")
    r = float((a @ b) / denom)
    r2 = min(r * r, 1.0 - 1e-12)
    return 1.0 / (1.0 - r2)


def count_vs_time_identifiability(panel) -> dict:
    """Regress cost on calendar time and cumulative prior count jointly.

    The morphic claim is that ``cost`` tracks cumulative count *beyond* a
    secular time-trend. This is identifiable only if count and time are not
    collinear (the role of the bursty instantiation schedule).

    Returns standardized partial slopes, t-stats, and the count<->time VIF.
    """
    y = panel.cost
    t = panel.calendar_index
    c = panel.cumulative_prior

    # standardize regressors so slopes are comparable
    def z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x - x.mean()

    tz, cz = z(t), z(c)
    X = np.column_stack([np.ones_like(y), tz, cz])
    beta, se, tstat, _ = _ols(X, y)
    vif = _vif(t, c)
    return {
        "time_slope": float(beta[1]),
        "time_t": float(tstat[1]),
        "count_slope": float(beta[2]),
        "count_t": float(tstat[2]),
        "count_time_vif": float(vif),
        "identifiable": bool(vif < 10.0),
    }


def acausal_test(panel) -> dict:
    """Single-panel acausality probe -- and a demonstration of why it CANNOT
    work for a single form.

    Within one form, ``cumulative_prior + cumulative_future == N-1`` is
    constant, so past and future counts are *perfectly* collinear: any linear
    "future effect" is just the sign-flip of the past effect. The only thing
    that could betray acausality is NONLINEAR curvature in the cost-vs-order
    profile that a causal accumulation cannot produce.

    We therefore absorb the linear trend and a causal past-proximity signal
    into the design matrix, then ask whether a future-proximity signal
    (orthogonalized against all of that) explains any *residual* curvature.
    For every causal mechanism this is ~0 by construction; in practice the
    CTC field's residual curvature is also negligible at one form (see
    ``ctc.solve_ctc_field``), which is exactly why the harness routes the real
    acausality decision to :func:`acausal_across_forms`.
    """
    n = panel.n
    y = panel.cost - panel.cost.mean()
    idx = np.arange(n, dtype=float)
    ls = max(n / 12.0, 1.0)

    def z(x):
        x = np.asarray(x, float) - np.mean(x)
        s = x.std()
        return x / s if s > 0 else x

    fwd = np.array([np.sum(np.exp(-(np.arange(i + 1, n) - i) / ls))
                    for i in range(n)])
    past = np.array([np.sum(np.exp(-(i - np.arange(0, i)) / ls))
                     for i in range(n)])

    # Absorb: intercept + linear ramp (any secular OR linear-count trend) +
    # causal past-proximity. A pure causal/linear model is fully captured here.
    X0 = np.column_stack([np.ones(n), z(idx), z(past)])
    beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    resid = y - X0 @ beta0

    # Future-proximity orthogonalized against everything causal/linear.
    fz = z(fwd)
    proj = X0 @ np.linalg.lstsq(X0, fz, rcond=None)[0]
    fz_orth = fz - proj
    if fz_orth.std() > 0:
        fz_orth = fz_orth / fz_orth.std()

    X = np.column_stack([np.ones(n), fz_orth])
    beta, se, tstat, _ = _ols(X, resid)
    return {
        "future_slope": float(beta[1]),
        "future_t": float(tstat[1]),
        "acausal": bool(abs(tstat[1]) > 3.0),
        "note": "single-form acausality is structurally near-unidentifiable; "
                "see acausal_across_forms",
    }


def acausal_across_forms(forms) -> dict:
    """The identifiable acausality test: across MULTIPLE forms.

    ``forms`` is a list of dicts, one per form, each with::

        early_cost   : mean acquisition cost over the first-k instances
        early_count  : realizations of this form during that early window
        eventual_total : the form's TOTAL realizations over all time (future)

    The CTC-resonance / morphic prediction is that a form's *early* cost
    depends on its *eventual* total (information looped back from the future),
    beyond its early count. Because ``eventual_total`` varies across forms
    independently of ``early_count``, the future effect is now identifiable:
    regress early_cost on (early_count, eventual_total) and read the eventual
    coefficient. For any causal mechanism it is ~0.
    """
    early_cost = np.array([f["early_cost"] for f in forms], float)
    early_count = np.array([f["early_count"] for f in forms], float)
    eventual = np.array([f["eventual_total"] for f in forms], float)

    def z(x):
        x = x - x.mean()
        s = x.std()
        return x / s if s > 0 else x

    ec, ev = z(early_count), z(eventual)
    X = np.column_stack([np.ones(len(forms)), ec, ev])
    beta, se, tstat, _ = _ols(X, early_cost)
    return {
        "early_count_slope": float(beta[1]),
        "early_count_t": float(tstat[1]),
        "eventual_total_slope": float(beta[2]),
        "eventual_total_t": float(tstat[2]),
        "count_eventual_vif": _vif(early_count, eventual),
        "acausal": bool(abs(tstat[2]) > 3.0),
    }
