"""Leaky echo-state reservoirs — the standard model of physical reservoir
computing. All connectivity kinds are rescaled to an identical spectral
radius so that only the *structure* of the medium differs between arms."""
from __future__ import annotations
import numpy as np


def make_W(n: int, kind: str, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Reservoir connectivity matrix, rescaled to spectral radius ``rho``.

    kind:
      'random'  — 10% sparse Gaussian (the control).
      'helical' — cos(omega * dtheta) ring over neuron index angle.
      'mobius'  — helical with an antiperiodic (half-twist) sign factor.
    """
    if kind == "random":
        W = rng.standard_normal((n, n))
        W *= (rng.random((n, n)) < 0.1)
    elif kind == "helical":
        i = np.arange(n)
        dth = 2 * np.pi * (i[:, None] - i[None, :]) / n
        W = np.cos(5.0 * dth)
    elif kind == "mobius":
        i = np.arange(n)
        dth = 2 * np.pi * (i[:, None] - i[None, :]) / n
        twist = np.sign(np.cos(np.pi * (i[:, None] - i[None, :]) / n))
        W = np.cos(5.0 * dth) * twist
    else:
        raise ValueError(f"unknown kind {kind!r}")
    sr = np.max(np.abs(np.linalg.eigvals(W)))
    if sr == 0:
        return W
    return W * (rho / sr)


def run_reservoir(
    u: np.ndarray, n: int, kind: str, nonlinear: bool,
    rng: np.random.Generator, rho: float = 0.95, leak: float = 0.3,
    in_scale: float = 0.5,
) -> np.ndarray:
    """Drive a reservoir with inputs ``u`` (T, din) -> states (T, n)."""
    u = np.atleast_2d(u)
    if u.shape[0] == 1 and u.shape[1] != 1:
        u = u.T
    T, din = u.shape
    W = make_W(n, kind, rho, rng)
    win = (rng.random((n, din)) * 2 - 1) * in_scale
    bias = (rng.random(n) * 2 - 1) * 0.1
    f = np.tanh if nonlinear else (lambda z: z)
    X = np.zeros((T, n))
    x = np.zeros(n)
    for t in range(T):
        x = (1 - leak) * x + leak * f(W @ x + win @ u[t] + bias)
        X[t] = x
    return X


def spectral_radius(W: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(W))))
