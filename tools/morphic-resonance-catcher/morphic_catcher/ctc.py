"""CTC-resonance model: the morphic field as a self-consistent fixed point.

Motivation
----------
The one place mainstream physics offers a *rigorous* analog of "the present
constrained by a consistency condition that loops through time" is the
closed-timelike-curve / closed-time-path formalism:

  * **Dinos** (``HYPOTHESIS.md`` Step 1, verified): the Mobius temporal loop
    is Schwinger-Keldysh on a finite contour -- a forward sweep and a backward
    sweep averaged to a Picard fixed point ``f = M[f]``.
  * **Systrophe** Deutsch-CTC channel theory: the chronology-respecting state
    is the fixed point ``rho = N[rho]`` of the loop superoperator, reached by
    iteration whose count is governed by the spectral gap ``|lambda_2|``.

This module ports that skeleton to a *classical* "establishment field" over a
temporally ordered ensemble of learners. It is an explicit analogy, not a
derivation: there is no claim that a morphic field exists or that this is the
unique classical limit of Deutsch's condition. The point is to give the
hypothesis a concrete, falsifiable dynamical form.

The fixed point
---------------
Let ``s_i = i / (N-1)`` be the causal seed (fraction of the ensemble realized
before instance ``i``) and ``W`` a temporal coupling kernel. The field
``f`` available to each instance, and the establishment ``e`` it realizes, are
tied by self-consistency ``e == f`` with

    f = (1 - gamma) * s  +  gamma * W @ e            (Keldysh source + loop)

Substituting ``e == f`` gives the resolvent (Schwinger-Dyson) form

    f = (1 - gamma) * (I - gamma * W)^{-1} @ s

solved by Neumann / power iteration ``f_{k+1} = (1-gamma) s + gamma W f_k`` --
i.e. the same Picard iteration Dinos verifies as Keldysh saddle-finding.

The kernel ``W`` mixes a strictly **causal** part (coupling to the past) with
an **acausal** part (coupling to the future -- the closed-timelike-curve seam),
weighted by ``acausal_fraction``:

  * ``acausal_fraction == 0``  -> W is lower-triangular; ``f_i`` depends only on
    the past; the model reduces to ordinary causal cumulative reinforcement
    (statistically identical to ``generate.morphic_field``).
  * ``acausal_fraction > 0``   -> ``f_i`` gains a dependence on realizations
    that occur AFTER it. This future-count dependence is the *only*
    empirically distinctive prediction of the CTC-resonance model, and it is
    exactly what the harness's :func:`morphic_catcher.detect.acausal_test`
    is built to detect (and what no controlled morphic experiment has shown).
"""

from __future__ import annotations

import numpy as np


def _temporal_kernel(n: int, acausal_fraction: float, length_scale: float):
    """Build the substochastic coupling kernel ``W``.

    Row ``i`` couples to past indices ``j < i`` (causal) and, scaled by
    ``acausal_fraction``, to future indices ``j > i`` (the CTC seam). Coupling
    decays exponentially with temporal separation. Rows are normalized so the
    row sum is at most 1 (substochastic), guaranteeing the Neumann series for
    ``(I - gamma W)^{-1}`` converges for ``gamma < 1``.
    """
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :]).astype(float)
    decay = np.exp(-sep / max(length_scale, 1e-9))
    np.fill_diagonal(decay, 0.0)

    past = np.tril(decay, k=-1)
    future = np.triu(decay, k=1)
    W = past + acausal_fraction * future

    row = W.sum(axis=1, keepdims=True)
    row[row == 0.0] = 1.0
    return W / row


def solve_ctc_field(
    n: int = 120,
    acausal_fraction: float = 0.5,
    gamma: float = 0.25,
    length_scale: float = None,
    n_iter: int = 256,
    tol: float = 1e-10,
):
    """Solve the self-consistent establishment field on the loop.

    Returns the field ``f`` normalized to ``[0, 1]`` (the morphic
    "field strength" each instance draws on, in temporal order).

    Implementation is the Picard / Neumann iteration
    ``f <- (1-gamma) s + gamma W f`` to a fixed point -- the classical
    analog of the Dinos Keldysh saddle and the Systrophe D-CTC fixed point.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1)")
    if not (0.0 <= acausal_fraction <= 1.0):
        raise ValueError("acausal_fraction must be in [0, 1]")
    if length_scale is None:
        length_scale = max(n / 12.0, 1.0)

    s = np.arange(n, dtype=float) / max(n - 1, 1)
    W = _temporal_kernel(n, acausal_fraction, length_scale)

    f = s.copy()
    for _ in range(n_iter):
        f_new = (1.0 - gamma) * s + gamma * (W @ f)
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    span = f.max() - f.min()
    if span <= 0:
        return np.zeros(n)
    return (f - f.min()) / span


def ctc_iteration_count(
    n: int = 120, acausal_fraction: float = 0.5, gamma: float = 0.25,
    length_scale: float = None, tol: float = 1e-10, n_iter: int = 1000,
) -> dict:
    """Diagnostic: iterations to convergence vs the spectral-gap prediction.

    Mirrors the Systrophe D-CTC spectral oracle: the empirical fixed-point
    iteration count should scale as ``-log(tol) / log(1/(gamma*|lambda_max(W)|))``.
    Returned for the findings log; not used in the verdict.
    """
    if length_scale is None:
        length_scale = max(n / 12.0, 1.0)
    s = np.arange(n, dtype=float) / max(n - 1, 1)
    W = _temporal_kernel(n, acausal_fraction, length_scale)

    f = s.copy()
    iters = n_iter
    for k in range(n_iter):
        f_new = (1.0 - gamma) * s + gamma * (W @ f)
        if np.max(np.abs(f_new - f)) < tol:
            iters = k + 1
            f = f_new
            break
        f = f_new

    lam = float(np.max(np.abs(np.linalg.eigvals(W))))
    rate = gamma * lam
    predicted = (-np.log(tol) / -np.log(rate)) if 0 < rate < 1 else float("inf")
    return {
        "iterations": iters,
        "spectral_radius_W": lam,
        "contraction_rate": rate,
        "predicted_iterations": predicted,
    }
