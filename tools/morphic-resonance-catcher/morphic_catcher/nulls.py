"""Surrogate / generative nulls for the morphic-resonance harness.

Systrophe rule: no verdict without a null run. Two nulls, matching the ELF
tool's philosophy:

  order_shuffle_null      -- permute instance order. Destroys ALL temporal
                             and count structure while preserving the marginal
                             cost distribution exactly. The model-free null for
                             "is the ordered structure real?".
  independent_learner_null-- the generative null: an i.i.d. panel with the same
                             marginal cost mean/spread and the same schedule.
                             The principled comparison for "is this more than
                             independent learning?".

Both return a p-value for a chosen scalar statistic computed on the real panel
vs its null distribution, with the (+1)/(+1) small-sample correction the ELF
adapter uses.
"""

from __future__ import annotations

import numpy as np

from .generate import Panel


def _p_value(real_stat: float, null_stats: np.ndarray, tail: str = "greater") -> float:
    null = np.asarray(null_stats, float)
    n = len(null)
    if tail == "greater":
        hits = np.sum(null >= real_stat)
    elif tail == "less":
        hits = np.sum(null <= real_stat)
    else:  # two-sided on |stat|
        hits = np.sum(np.abs(null) >= abs(real_stat))
    return float((hits + 1) / (n + 1))


def order_shuffle_null(panel, statistic, n_surrogates: int = 200,
                       seed: int = 0, tail: str = "greater") -> dict:
    """Permute the cost sequence; recompute ``statistic`` on each shuffle.

    ``statistic`` takes a :class:`Panel` and returns a float. The panel handed
    to the statistic on each draw keeps the real calendar/count axes but a
    shuffled cost vector, so any genuine cost<->order coupling is broken.
    """
    rng = np.random.default_rng(seed)
    real = statistic(panel)
    null = np.empty(n_surrogates)
    for k in range(n_surrogates):
        perm = rng.permutation(panel.n)
        shuffled = Panel(panel.cost[perm], panel.calendar_index,
                         panel.cumulative_prior, panel.cumulative_future,
                         meta={"null": "order_shuffle"})
        null[k] = statistic(shuffled)
    return {
        "real_statistic": float(real),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": _p_value(real, null, tail=tail),
        "n_surrogates": n_surrogates,
    }


def independent_learner_null(panel, statistic, n_surrogates: int = 200,
                             seed: int = 0, tail: str = "greater") -> dict:
    """Generative null: i.i.d. panels matched to the real marginal + schedule.

    Each surrogate draws fresh i.i.d. costs with the real panel's mean and
    std, on the real calendar/count axes. Tests whether the statistic exceeds
    what pure independent learning produces.
    """
    rng = np.random.default_rng(seed)
    mu, sd = float(panel.cost.mean()), float(panel.cost.std())
    real = statistic(panel)
    null = np.empty(n_surrogates)
    for k in range(n_surrogates):
        costs = rng.normal(mu, sd if sd > 0 else 1e-6, size=panel.n)
        surr = Panel(costs, panel.calendar_index, panel.cumulative_prior,
                     panel.cumulative_future, meta={"null": "independent"})
        null[k] = statistic(surr)
    return {
        "real_statistic": float(real),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": _p_value(real, null, tail=tail),
        "n_surrogates": n_surrogates,
    }
