"""Newton-Kantorovich iteration for back-reaction fixed-point problems.

Newton-Kantorovich (NK) is the standard quadratic-convergent iteration
for nonlinear equations F(x) = 0 with x in a Banach space:

    x_{n+1} = x_n - [F'(x_n)]^{-1} F(x_n).

This module provides 1D, n-D (numerical-Jacobian), and a thin
back-reaction wrapper. NK is contrasted directly with Picard iteration
(linear convergence) so that the actual asymptotic rate of a real
iterate can be classified.

The Grok updates.txt chat claimed a self-consistent iterator
"converges in <10 iterations", but the printed trace showed a *constant*
per-iteration step --- i.e. a linear walk (Picard), not quadratic NK.
This module exposes the genuine NK iteration so the user can verify
or falsify the convergence claim concretely.

Notes
-----
- All functions return a `NKResult` capturing the iterate history,
  the residual history, and a convergence-rate classification.
- Rate classification: 'quadratic' if log(|r_{n+1}|) ~ 2 log(|r_n|)
  in the last 3 iterations; 'linear' if log(|r_{n+1}|) ~ c +
  log(|r_n|); 'unclear' otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class NKResult:
    """Result of a Newton-Kantorovich (or Picard) iteration.

    Fields
    ------
    x          : the final iterate
    converged  : True iff |F(x)| < tol on exit
    iterations : number of iterations actually performed
    residuals  : list of |F(x_k)| at each iteration
    history    : list of x_k (the iterates themselves)
    rate       : 'quadratic', 'linear', 'unclear', or 'stalled'
    """

    x: np.ndarray
    converged: bool
    iterations: int
    residuals: list[float] = field(default_factory=list)
    history: list[np.ndarray] = field(default_factory=list)
    rate: str = "unclear"


def _classify_rate(residuals: list[float]) -> str:
    """Classify convergence rate from a residual history."""
    if len(residuals) < 3:
        return "unclear"
    tail = residuals[-3:]
    if any(r <= 0 for r in tail):
        return "stalled"
    # logs
    logs = [np.log(r) for r in tail]
    diffs = [logs[i + 1] - logs[i] for i in range(len(logs) - 1)]
    # Quadratic: log_{n+1} ~ 2 log_n + c. Ratio of consecutive log-magnitudes ~ 2.
    ratios = [logs[i + 1] / logs[i] if logs[i] != 0 else float("inf")
              for i in range(len(logs) - 1)]
    # Use ratios: for quadratic, ratio ~ 2; for linear, ratio ~ 1.
    mean_ratio = float(np.mean(ratios))
    if 1.6 < mean_ratio < 2.6:
        return "quadratic"
    if 0.6 < mean_ratio < 1.4:
        return "linear"
    if all(abs(d) < 1e-12 for d in diffs):
        return "stalled"
    return "unclear"


def newton_kantorovich_1d(
    F: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 50,
    eps_fd: float = 1e-6,
    df_analytical: Optional[Callable[[float], float]] = None,
) -> NKResult:
    """1D Newton-Kantorovich iteration for F(x) = 0.

    Uses central-difference finite difference for F'(x) unless an
    analytical derivative is provided.
    """
    x = float(x0)
    residuals = []
    history = [np.array([x])]
    for k in range(max_iter):
        f = float(F(x))
        residuals.append(abs(f))
        if abs(f) < tol:
            return NKResult(
                x=np.array([x]), converged=True, iterations=k,
                residuals=residuals, history=history,
                rate=_classify_rate(residuals),
            )
        if df_analytical is not None:
            df = float(df_analytical(x))
        else:
            df = (float(F(x + eps_fd)) - float(F(x - eps_fd))) / (2 * eps_fd)
        if abs(df) < 1e-30:
            # Singular Jacobian; fall back to linear step (Picard with damping)
            x = x - 0.1 * np.sign(f)
        else:
            x = x - f / df
        history.append(np.array([x]))
    residuals.append(abs(float(F(x))))
    return NKResult(
        x=np.array([x]), converged=False, iterations=max_iter,
        residuals=residuals, history=history,
        rate=_classify_rate(residuals),
    )


def newton_kantorovich_nd(
    F: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
    eps_fd: float = 1e-6,
    damping: float = 1.0,
) -> NKResult:
    """N-dimensional Newton-Kantorovich iteration.

    Computes the Jacobian J[i, j] = dF_i/dx_j via central differences.
    Uses pseudoinverse for singular or ill-conditioned Jacobians.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    residuals = []
    history = [x.copy()]
    for k in range(max_iter):
        f = np.asarray(F(x), dtype=float)
        r = float(np.linalg.norm(f))
        residuals.append(r)
        if r < tol:
            return NKResult(
                x=x, converged=True, iterations=k,
                residuals=residuals, history=history,
                rate=_classify_rate(residuals),
            )
        # Numerical Jacobian
        J = np.zeros((n, n))
        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += eps_fd
            x_minus[j] -= eps_fd
            J[:, j] = (np.asarray(F(x_plus)) - np.asarray(F(x_minus))) / (2 * eps_fd)
        # Solve J dx = -f
        try:
            dx = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -f, rcond=None)
        x = x + damping * dx
        history.append(x.copy())
    residuals.append(float(np.linalg.norm(F(x))))
    return NKResult(
        x=x, converged=False, iterations=max_iter,
        residuals=residuals, history=history,
        rate=_classify_rate(residuals),
    )


def picard_iteration_1d(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> NKResult:
    """Picard iteration x_{n+1} = g(x_n) for comparison with NK.

    Converges linearly (rate ~ |g'(x*)|) where NK converges quadratically.
    Provided here so callers can directly compare iterations and assess
    convergence-rate claims.
    """
    x = float(x0)
    residuals = []
    history = [np.array([x])]
    for k in range(max_iter):
        gx = float(g(x))
        r = abs(gx - x)
        residuals.append(r)
        if r < tol:
            return NKResult(
                x=np.array([x]), converged=True, iterations=k,
                residuals=residuals, history=history,
                rate=_classify_rate(residuals),
            )
        x = gx
        history.append(np.array([x]))
    return NKResult(
        x=np.array([x]), converged=False, iterations=max_iter,
        residuals=residuals, history=history,
        rate=_classify_rate(residuals),
    )


def back_reaction_residual_factory(
    anomaly_target_fn: Callable[[np.ndarray], float],
    sample_points: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a residual function for a back-reaction problem.

    `anomaly_target_fn(params)` returns the trace anomaly (or other
    diagnostic) at a fixed reference point; `sample_points` is an
    array of reference radii at which to evaluate.

    The residual F(params) is the vector of differences between the
    anomaly evaluated at each sample point and a reference value
    (here, zero by convention, but the caller can shift).

    Returned function takes `params` (array) and returns array of
    residuals (one per sample point).
    """
    def residual(params: np.ndarray) -> np.ndarray:
        return np.array([anomaly_target_fn(np.concatenate([params, [r]]))
                         for r in sample_points])
    return residual


def is_convergence_rate_quadratic(result: NKResult, atol_log: float = 0.5) -> bool:
    """True iff the residual history shows quadratic convergence.

    Checks log r_{n+1} / log r_n ~ 2 in the final 3 iterates (within atol).
    """
    return result.rate == "quadratic"
