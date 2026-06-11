"""Benchmark tasks for reservoir computation.

NARMA-10  — nonlinear system identification (memory + mild nonlinearity).
parity-N  — temporal XOR; *linearly inseparable*, so a linear medium with a
            linear readout cannot solve it (the decisive nonlinearity probe).
memory capacity — the classic linear-memory / nonlinearity tradeoff.
"""
from __future__ import annotations
import numpy as np


def narma10(T: int, rng: np.random.Generator):
    u = rng.random(T) * 0.5
    y = np.zeros(T)
    for t in range(10, T - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t] + 0.1)
        y[t + 1] = np.clip(y[t + 1], -2, 2)
    return u[:, None], y[:, None]


def parity_task(T: int, n_bits: int, rng: np.random.Generator):
    u = rng.integers(0, 2, T)
    y = np.array([
        (-1.0 if (u[max(0, t - n_bits + 1):t + 1].sum() % 2) else 1.0)
        for t in range(T)
    ])
    return (u * 2.0 - 1.0)[:, None], y[:, None]


def memory_capacity(X_tr, X_te, u_tr, u_te, wash, max_k=30):
    from .readout import ridge_fit, ridge_pred
    mc = 0.0
    for k in range(1, max_k + 1):
        ytr = np.roll(u_tr[:, 0], k)
        yte = np.roll(u_te[:, 0], k)
        beta = ridge_fit(X_tr[wash:], ytr[wash:, None])
        p = ridge_pred(X_te[wash:], beta)[:, 0]
        c = np.corrcoef(p, yte[wash:])[0, 1]
        mc += 0.0 if np.isnan(c) else c * c
    return mc
