"""ER=EPR throat width vs which-way fringe visibility: bound-conjecture testbed.

Maldacena-Susskind ER=EPR identifies the entanglement carried by an EPR
pair with an Einstein-Rosen bridge whose throat has Bekenstein-Hawking
entropy S_throat = A_throat / (4 ell_P^2). Operationally, that entropy
is the entanglement of the EPR resource across the bipartition.

This module asks a sharp empirical question: when a which-way ancilla
is teleported through a *Werner-noisy* EPR pair, can fringe visibility
in the post-throat double-slit be upper-bounded by a function of the
throat's entanglement?

Three candidate bounds are tested:

  (A) naive Bekenstein:  V_post  <=  exp( -S_vN(rho_W) )
        -- uses the full bipartite von-Neumann entropy. Equals
           log(4) for the maximally mixed pair, log(2) for the pure
           Bell state -- the wrong direction.

  (B) entanglement-of-formation:  V_post  <=  exp( -E_F(w) )
        -- uses the actual entanglement measure for Werner states,
           which is 1 for the pure Bell state and 0 below w = 1/3.

  (C) Englert-Bekenstein hybrid:  V_post  <=  1 - tanh( E_F(w)
                                                       * sin^2(theta/2) )
        -- the natural blend of Englert's V^2 + D^2 <= 1 and a
           Bekenstein factor: stronger entanglement *and* stronger
           coupling jointly compress visibility.

Model
-----
A Werner pair rho_W(w) = w |Phi+><Phi+| + (1-w) I/4 (w in [0,1])
teleports an ancilla with singlet-fraction fidelity F(w) = (1+w)/2.
For the which-way ancilla
  |0>  (slit U)        and        cos(theta/2)|0> + sin(theta/2)|1>  (slit L)
the post-throat overlap between branches is
    V_post(w, theta) = (1 - w) + w * cos(theta/2).
At w = 1 this reduces to the textbook Englert visibility V = cos(theta/2).
At w = 0 the ancilla is decohered to I/2 -- no which-way information --
and V_post = 1 (full fringes).

The module evaluates V_post against bounds (A), (B), (C) on a grid and
reports per-bound verdicts. The honest expectation: (A) is too weak in
one direction and too tight in another (it does not hold); (C) is the
likeliest survivor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


# ----- helpers -------------------------------------------------------------


def _check_w(w: float) -> None:
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"Werner parameter w must be in [0, 1], got {w}")


def _check_theta(theta: float) -> None:
    if not 0.0 <= theta <= math.pi:
        raise ValueError(f"theta must be in [0, pi], got {theta}")


def _binary_entropy(p: float) -> float:
    """h(p) = -p log p - (1-p) log (1-p), natural log."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


# ----- Werner state entropies ---------------------------------------------


def werner_eigenvalues(w: float) -> np.ndarray:
    """Eigenvalues of rho_W(w) = w |Phi+><Phi+| + (1 - w) I/4."""
    _check_w(w)
    return np.array(
        [(1.0 + 3.0 * w) / 4.0,
         (1.0 - w) / 4.0,
         (1.0 - w) / 4.0,
         (1.0 - w) / 4.0],
        dtype=float,
    )


def werner_vn_entropy(w: float) -> float:
    """Von-Neumann entropy S(rho_W(w)) in nats."""
    eigs = werner_eigenvalues(w)
    safe = np.clip(eigs, 1e-30, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def werner_concurrence(w: float) -> float:
    """Concurrence C(rho_W(w)) = max(0, (3w - 1) / 2)."""
    _check_w(w)
    return float(max(0.0, (3.0 * w - 1.0) / 2.0))


def werner_entanglement_of_formation(w: float) -> float:
    """E_F = h((1 + sqrt(1 - C^2)) / 2) in nats. Wootters' formula."""
    C = werner_concurrence(w)
    if C == 0.0:
        return 0.0
    return _binary_entropy(0.5 * (1.0 + math.sqrt(1.0 - C * C)))


def throat_area_proxy(
    w: float, ell_P: float = 1.0, measure: str = "E_F",
) -> float:
    """A_throat = 4 ell_P^2 * S(w), per Bekenstein-Hawking.

    Parameters
    ----------
    w : float
        Werner parameter.
    ell_P : float
        Planck-length proxy.
    measure : {"vN", "E_F"}
        Which entropy to identify with A / (4 ell_P^2). Defaults to E_F.
    """
    if ell_P <= 0:
        raise ValueError(f"ell_P must be positive, got {ell_P}")
    if measure == "vN":
        S = werner_vn_entropy(w)
    elif measure == "E_F":
        S = werner_entanglement_of_formation(w)
    else:
        raise ValueError(f"unknown measure {measure!r}")
    return float(4.0 * ell_P * ell_P * S)


# ----- post-throat visibility ---------------------------------------------


def post_throat_visibility(w: float, theta: float) -> float:
    """V_post = (1 - w) + w cos(theta/2)."""
    _check_w(w)
    _check_theta(theta)
    return float((1.0 - w) + w * math.cos(theta / 2.0))


# ----- candidate bounds ---------------------------------------------------


def bound_A_vN(w: float, theta: float) -> float:
    """Naive Bekenstein: V <= exp(-S_vN(w))."""
    return float(math.exp(-werner_vn_entropy(w)))


def bound_B_EF(w: float, theta: float) -> float:
    """Refined: V <= exp(-E_F(w))."""
    return float(math.exp(-werner_entanglement_of_formation(w)))


def bound_C_hybrid(w: float, theta: float) -> float:
    """Englert-Bekenstein hybrid:
       V <= 1 - tanh( E_F(w) * sin^2(theta/2) )."""
    EF = werner_entanglement_of_formation(w)
    return float(1.0 - math.tanh(EF * math.sin(theta / 2.0) ** 2))


BOUNDS = {
    "A_naive_vN":   bound_A_vN,
    "B_E_F":        bound_B_EF,
    "C_hybrid":     bound_C_hybrid,
}


# ----- the test -----------------------------------------------------------


@dataclass(frozen=True)
class BoundVerdict:
    """Per-bound verdict on a (w, theta) grid."""

    name: str
    holds: bool                     # V_post <= V_bound everywhere?
    max_overshoot: float            # max(V_post - V_bound); >0 means falsified
    where_overshoot_max: tuple[float, float]  # (w, theta) at the worst point


@dataclass(frozen=True)
class BoundTestResult:
    w_grid: np.ndarray
    theta_grid: np.ndarray
    V_post: np.ndarray
    verdicts: dict[str, BoundVerdict]


def evaluate_bounds(
    w_grid: Iterable[float] | None = None,
    theta_grid: Iterable[float] | None = None,
    bounds: dict[str, Callable[[float, float], float]] | None = None,
) -> BoundTestResult:
    """Evaluate each candidate bound against V_post on a 2D grid.

    Default grid: 21 x 21 over (w, theta) in [0,1] x [0, pi].
    """
    if w_grid is None:
        w_grid = np.linspace(0.0, 1.0, 21)
    if theta_grid is None:
        theta_grid = np.linspace(0.0, math.pi, 21)
    if bounds is None:
        bounds = BOUNDS
    w_arr = np.asarray(list(w_grid), dtype=float)
    th_arr = np.asarray(list(theta_grid), dtype=float)
    V = np.array([
        [post_throat_visibility(float(w), float(t)) for t in th_arr]
        for w in w_arr
    ])
    verdicts: dict[str, BoundVerdict] = {}
    for name, fn in bounds.items():
        VB = np.array([[fn(float(w), float(t)) for t in th_arr] for w in w_arr])
        overshoot = V - VB
        idx = np.unravel_index(int(np.argmax(overshoot)), overshoot.shape)
        max_over = float(overshoot[idx])
        verdicts[name] = BoundVerdict(
            name=name,
            holds=bool(max_over <= 1e-9),
            max_overshoot=max_over,
            where_overshoot_max=(float(w_arr[idx[0]]), float(th_arr[idx[1]])),
        )
    return BoundTestResult(
        w_grid=w_arr, theta_grid=th_arr, V_post=V, verdicts=verdicts,
    )
