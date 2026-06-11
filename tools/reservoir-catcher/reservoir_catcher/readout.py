"""Ridge readout, error metrics, and the independent Systrophe address-space
λ₂ readout (reuses ``systrophe.catchers.novelty_catcher``)."""
from __future__ import annotations
import numpy as np

from systrophe.catchers.novelty_catcher import lambda_2_of_hamming_graph


def ridge_fit(X: np.ndarray, Y: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    A = Xb.T @ Xb + reg * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ Y)


def ridge_pred(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ beta


def nrmse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2) / (np.var(y) + 1e-12)))


def address_lambda2(
    X: np.ndarray, wash: int, max_pts: int = 150,
    rng: np.random.Generator | None = None,
) -> float:
    """λ₂ of the Hamming graph over sign-bit addresses of visited states.

    A higher value means the medium explores a richer, better-connected
    region of state space — an emergence signal independent of task error.
    Uses the canonical Systrophe catcher (`lambda_2_of_hamming_graph`).
    """
    S = X[wash:]
    if rng is None:
        rng = np.random.default_rng(0)
    if len(S) > max_pts:
        S = S[rng.choice(len(S), max_pts, replace=False)]
    addrs = [(s > 0).astype(int) for s in S]
    n = len(addrs)
    if n < 2:
        return 0.0
    # data-driven radius = median pairwise Hamming distance
    A = np.array(addrs)
    dists = (A[:, None, :] != A[None, :, :]).sum(2)
    radius = int(np.median(dists[np.triu_indices(n, 1)]))
    return lambda_2_of_hamming_graph(addrs, radius)
