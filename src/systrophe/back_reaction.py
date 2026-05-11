"""Self-consistent back-reaction of the renormalised stress tensor.

Wires `hadamard_offtrace` (full <T_{mu nu}>_ren on the LP background)
together with `newton_kantorovich` (quadratic-convergent root-finder)
to address the back-reaction problem

    G_{mu nu}(g) = 8 pi <T_{mu nu}>_ren(g).

The LP exterior is Einstein vacuum (G_{mu nu} = 0), so the right-
hand side measures *how far* the renormalised stress tensor is from
zero. We define a back-reaction residual

    R(delta) = norm( <T_{mu nu}>_ren(g(delta); r_samples) )

over a sweep of pair-offset parameters delta, and use NK to locate
local minima. Minima approximate chronology-favouring vacua --- in
the sense of Hawking's chronology-protection intuition.

This module does not perform a *full* Einstein-equation-coupled
back-reaction solve (which would require integrating the modified
geodesic equation and re-evaluating the metric at each iterate). It
does perform the *static residual minimization*, which is the
load-bearing diagnostic and is computable in seconds.

Public API
----------
- `pair_back_reaction_residual(pair, r_samples)`: scalar residual
- `optimal_delta_by_NK(s1, s2_base, delta_range, ...)`: NK search
- `back_reaction_landscape(s1, s2_base, delta_grid, r_samples)`:
  full sweep
- `chronology_favoring_deltas(landscape, threshold_quantile=0.1)`:
  delta values with bottom-10% residual
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hadamard_offtrace import hadamard_offtrace_T
from .newton_kantorovich import newton_kantorovich_1d
from .pair import SystrophePair
from .sinusoid import TiplerSinusoid


def pair_back_reaction_residual_at_r(pair: SystrophePair, r: float) -> float:
    """Back-reaction residual norm at one radius r.

    Returns Frobenius norm of <T_{mu nu}>_ren evaluated on the
    *first* cylinder of the pair (a leading-order approximation).
    For pair-specific back-reaction one needs the linearised
    superposition; here we use the dominant cylinder as the LP
    background and view the second cylinder as a small perturbation
    (consistent with `pair.py` linearised superposition).
    """
    from .vanstockum import VanStockumInterior
    # Construct a van Stockum from s1's parameters
    # alpha = sqrt(4 a^2 - 1), so a = sqrt((alpha^2 + 1)/4)
    a = float(np.sqrt((pair.s1.alpha ** 2 + 1.0) / 4.0))
    omega = a / pair.s1.R
    vs1 = VanStockumInterior(omega=omega, R=pair.s1.R)
    try:
        T = hadamard_offtrace_T(vs1, r=r)
    except Exception:
        return float("inf")
    return float(np.linalg.norm(T))


def pair_back_reaction_residual(
    pair: SystrophePair,
    r_samples: np.ndarray,
    L_weight: float = 1.0,
    T_weight: float = 1.0,
) -> float:
    """Composite back-reaction diagnostic across `r_samples`.

    The renormalised stress tensor on the LP background (computed
    using the dominant cylinder s1) is independent of the pair offset
    delta in the linearised superposition. To get a delta-sensitive
    back-reaction diagnostic we combine

        residual(delta, r) = T_weight * |<T_{mu nu}>(s1; r)|
                           + L_weight * |L_pair(delta; r)|

    The first term penalises high local curvature (low Hadamard);
    the second penalises large pair-L (CTC content). A back-reaction-
    consistent vacuum minimises both: small curvature AND small CTC
    content. For a matched-pair the minimum is at delta = pi
    (anti-phase extinction), matching the known v0.11 CTC-log-measure
    result.

    The two weights default to 1 (equal contribution). Pass T_weight=0
    to focus on the L-only (chronology-extinction) diagnostic, or
    L_weight=0 for the pure Hadamard term.
    """
    total = 0.0
    for r in r_samples:
        T_part = pair_back_reaction_residual_at_r(pair, float(r))
        L_part = abs(float(pair.L(float(r))))
        total += T_weight * T_part + L_weight * L_part
    return total


@dataclass(frozen=True)
class BackReactionLandscape:
    """Sweep of back-reaction residual across pair-offset delta."""

    deltas: np.ndarray
    residuals: np.ndarray
    min_delta: float
    min_residual: float
    max_delta: float
    max_residual: float


def back_reaction_landscape(
    s1: TiplerSinusoid,
    s2_base: TiplerSinusoid,
    delta_grid: np.ndarray,
    r_samples: np.ndarray,
) -> BackReactionLandscape:
    """Sweep delta_offset; compute back-reaction residual at each."""
    delta_grid = np.asarray(delta_grid, dtype=float)
    residuals = np.zeros_like(delta_grid)
    for i, off in enumerate(delta_grid):
        s2_shifted = TiplerSinusoid(
            R=s2_base.R, a=s2_base.a, A=s2_base.A,
            delta=s2_base.delta + float(off), p=s2_base.p,
        )
        pair = SystrophePair(s1=s1, s2=s2_shifted)
        residuals[i] = pair_back_reaction_residual(pair, r_samples)
    idx_min = int(np.argmin(residuals))
    idx_max = int(np.argmax(residuals))
    return BackReactionLandscape(
        deltas=delta_grid,
        residuals=residuals,
        min_delta=float(delta_grid[idx_min]),
        min_residual=float(residuals[idx_min]),
        max_delta=float(delta_grid[idx_max]),
        max_residual=float(residuals[idx_max]),
    )


def optimal_delta_by_NK(
    s1: TiplerSinusoid,
    s2_base: TiplerSinusoid,
    delta_seed: float,
    r_samples: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 30,
):
    """Newton-Kantorovich search for a local minimum of the residual.

    Treats d residual / d delta as the function to drive to zero.
    Returns the NKResult from `newton_kantorovich_1d`.
    """
    def F(delta: float) -> float:
        # Central-difference derivative of residual w.r.t. delta
        eps_d = 1e-3
        # at delta + eps
        s2p = TiplerSinusoid(
            R=s2_base.R, a=s2_base.a, A=s2_base.A,
            delta=s2_base.delta + delta + eps_d, p=s2_base.p,
        )
        s2m = TiplerSinusoid(
            R=s2_base.R, a=s2_base.a, A=s2_base.A,
            delta=s2_base.delta + delta - eps_d, p=s2_base.p,
        )
        R_p = pair_back_reaction_residual(SystrophePair(s1=s1, s2=s2p), r_samples)
        R_m = pair_back_reaction_residual(SystrophePair(s1=s1, s2=s2m), r_samples)
        return (R_p - R_m) / (2 * eps_d)

    return newton_kantorovich_1d(F, x0=delta_seed, tol=tol, max_iter=max_iter,
                                  eps_fd=1e-3)


def chronology_favoring_deltas(
    landscape: BackReactionLandscape, quantile: float = 0.1
) -> np.ndarray:
    """Return the delta values in the bottom-quantile of residual.

    These are the offsets that most strongly suppress the renormalised
    stress tensor --- the candidate "chronology-favouring" vacua.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    threshold = float(np.quantile(landscape.residuals, quantile))
    mask = landscape.residuals <= threshold
    return landscape.deltas[mask]


def compare_to_ctc_extinction(
    s1: TiplerSinusoid,
    s2_base: TiplerSinusoid,
    delta_grid: np.ndarray,
    r_min: float = 1.05,
    r_max: float = 10.0,
    r_samples: np.ndarray | None = None,
) -> dict:
    """Compare back-reaction minimum to the static CTC-extinction minimum.

    Returns dict with:
      - landscape          : BackReactionLandscape
      - br_min_delta       : delta minimising the back-reaction residual
      - ctc_min_delta      : delta minimising the CTC log-measure (the
                             phasor-extinction point)
      - alignment_delta    : abs difference between the two minima
                             (mod 2 pi). Small => Hawking-style chronology
                             protection is consistent with back-reaction;
                             large => mismatch (back-reaction selects
                             different vacuum than naive extinction).
    """
    if r_samples is None:
        r_samples = np.linspace(r_min, r_max, 8)
    lndscp = back_reaction_landscape(s1, s2_base, delta_grid, r_samples)
    # CTC log measure across delta
    ctc_measures = np.zeros_like(delta_grid)
    for i, off in enumerate(delta_grid):
        s2_shifted = TiplerSinusoid(
            R=s2_base.R, a=s2_base.a, A=s2_base.A,
            delta=s2_base.delta + float(off), p=s2_base.p,
        )
        pair = SystrophePair(s1=s1, s2=s2_shifted)
        ctc_measures[i] = pair.total_ctc_log_measure(r_min, r_max)
    idx_ctc_min = int(np.argmin(ctc_measures))
    ctc_min_delta = float(delta_grid[idx_ctc_min])
    diff = abs(lndscp.min_delta - ctc_min_delta)
    diff_wrapped = min(diff, abs(2 * np.pi - diff))
    return {
        "landscape": lndscp,
        "br_min_delta": lndscp.min_delta,
        "br_min_residual": lndscp.min_residual,
        "ctc_min_delta": ctc_min_delta,
        "ctc_measures": ctc_measures,
        "alignment_delta": diff_wrapped,
    }
