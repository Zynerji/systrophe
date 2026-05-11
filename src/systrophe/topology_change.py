"""Topology-change probabilities for CTC band structure.

In quantum gravity, topology changes (pinch-off, merger, band fusion)
are heuristically suppressed by an instanton action
    P_topology ~ exp(-S_E / hbar)
where S_E is the Euclidean action of the topology-changing geometry.

For the LP supercritical exterior, two relevant transitions:
  (a) CTC-band pinch-off: a band (r_n, r_{n+1}) collapses to zero
      width, disconnecting the exterior.
  (b) Adjacent-band merger: bands (r_n, r_{n+1}) and (r_{n+1}, r_{n+2})
      fuse by absorbing the intermediate F=0 surface.

Module:
- ctc_band_pinch_off_action: Euclidean action for pinching off band n
- ctc_band_merger_action: action for fusing bands n and n+1
- topology_change_probability: P = exp(-S_E)
- preferred_topology_change: which transition has lowest S_E
- pair_extinction_topology_change: pair effect on transition rates
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class TopologyTransition:
    transition_type: str  # "pinch_off" or "merger"
    band_index: int
    r_inner: float
    r_outer: float
    euclidean_action: float
    probability: float


def _chronology_horizons(vs: VanStockumInterior, n: int) -> list[float]:
    if not vs.is_supercritical():
        return []
    R = vs.R
    alpha = vs.alpha
    gamma_c = math.pi - math.atan(alpha)
    zeros = []
    for k in range(1, 200):
        u_k = (k * math.pi - gamma_c) / alpha
        if u_k <= 0:
            continue
        zeros.append(R * math.exp(u_k))
        if len(zeros) >= n + 1:
            break
    return zeros


def ctc_band_pinch_off_action(
    vs: VanStockumInterior, band_index: int,
) -> dict:
    """Action for collapsing band (r_n, r_{n+1}) to zero width.

    Heuristic: S_E ~ |F''(r_mid)| * (r_{n+1} - r_n)^2 / 4.
    """
    zeros = _chronology_horizons(vs, band_index + 1)
    if band_index >= len(zeros) - 1:
        return {
            "transition_type": "pinch_off",
            "band_index": band_index,
            "euclidean_action": float("inf"),
            "available": False,
        }
    r_inner = zeros[band_index]
    r_outer = zeros[band_index + 1]
    r_mid = math.sqrt(r_inner * r_outer)
    eps = 1e-4 * r_mid
    F_plus = float(vs.analytic_exterior_F(np.array([r_mid + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r_mid - eps]))[0])
    F = float(vs.analytic_exterior_F(np.array([r_mid]))[0])
    Fpp = (F_plus + F_minus - 2 * F) / (eps * eps)
    width = r_outer - r_inner
    S_E = abs(Fpp) * width ** 2 / 4.0
    return {
        "transition_type": "pinch_off",
        "band_index": band_index,
        "r_inner": r_inner,
        "r_outer": r_outer,
        "F_pp_at_mid": float(Fpp),
        "band_width": float(width),
        "euclidean_action": float(S_E),
        "available": True,
    }


def ctc_band_merger_action(
    vs: VanStockumInterior, band_index: int,
) -> dict:
    """Action for fusing bands n and n+1 by absorbing F=0 surface at r_{n+1}."""
    zeros = _chronology_horizons(vs, band_index + 2)
    if band_index >= len(zeros) - 2:
        return {
            "transition_type": "merger",
            "band_index": band_index,
            "euclidean_action": float("inf"),
            "available": False,
        }
    r_inner = zeros[band_index]
    r_middle = zeros[band_index + 1]
    r_outer = zeros[band_index + 2]
    eps = 1e-4 * r_middle
    F_plus = float(vs.analytic_exterior_F(np.array([r_middle + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r_middle - eps]))[0])
    Fp = (F_plus - F_minus) / (2 * eps)
    width = r_outer - r_inner
    S_E = abs(Fp) * width / 2.0
    return {
        "transition_type": "merger",
        "band_index": band_index,
        "r_inner": r_inner,
        "r_middle": r_middle,
        "r_outer": r_outer,
        "F_prime_at_middle": float(Fp),
        "euclidean_action": float(S_E),
        "available": True,
    }


def topology_change_probability(euclidean_action: float,
                                  hbar: float = 1.0) -> float:
    """P = exp(-S_E / hbar) (bounded to avoid underflow)."""
    if not math.isfinite(euclidean_action):
        return 0.0
    if euclidean_action / max(hbar, 1e-30) > 700:
        return 0.0
    return float(math.exp(-euclidean_action / max(hbar, 1e-30)))


def preferred_topology_change(
    vs: VanStockumInterior, n_bands: int = 3,
) -> TopologyTransition:
    """Find the transition with the lowest Euclidean action."""
    candidates = []
    for i in range(n_bands):
        po = ctc_band_pinch_off_action(vs, i)
        if po.get("available"):
            candidates.append(po)
        m = ctc_band_merger_action(vs, i)
        if m.get("available"):
            candidates.append(m)
    if not candidates:
        return TopologyTransition(
            transition_type="none", band_index=-1,
            r_inner=float("nan"), r_outer=float("nan"),
            euclidean_action=float("inf"), probability=0.0,
        )
    best = min(candidates, key=lambda c: c["euclidean_action"])
    return TopologyTransition(
        transition_type=best["transition_type"],
        band_index=best["band_index"],
        r_inner=float(best.get("r_inner", float("nan"))),
        r_outer=float(best.get("r_outer", float("nan"))),
        euclidean_action=float(best["euclidean_action"]),
        probability=topology_change_probability(best["euclidean_action"]),
    )


def pair_extinction_topology_change(
    vs: VanStockumInterior, delta: float,
    band_index: int = 0,
) -> dict:
    """For a Systrophe pair at offset delta, the effective F-scale is
    multiplied by (1+cos delta)/2; both pinch-off and merger actions
    scale accordingly. At delta=pi, effective action -> 0, so transition
    becomes "trivially" available (no band, no transition needed).
    """
    extinction = 0.5 * (1.0 + math.cos(delta))
    base = ctc_band_pinch_off_action(vs, band_index)
    if not base.get("available"):
        return {"delta": delta, "extinction_factor": extinction,
                "available": False}
    S_pair = base["euclidean_action"] * extinction ** 2
    P_pair = topology_change_probability(S_pair)
    return {
        "delta": delta,
        "extinction_factor": extinction,
        "S_single": base["euclidean_action"],
        "S_pair": float(S_pair),
        "P_pair": float(P_pair),
        "is_trivial_at_extinction": bool(abs(delta - math.pi) < 1e-9),
    }


def all_topology_transitions(
    vs: VanStockumInterior, n_bands: int = 3,
) -> list[TopologyTransition]:
    """All available topology transitions, sorted by action."""
    transitions = []
    for i in range(n_bands):
        po = ctc_band_pinch_off_action(vs, i)
        if po.get("available"):
            transitions.append(TopologyTransition(
                transition_type="pinch_off",
                band_index=i,
                r_inner=po["r_inner"],
                r_outer=po["r_outer"],
                euclidean_action=po["euclidean_action"],
                probability=topology_change_probability(po["euclidean_action"]),
            ))
        m = ctc_band_merger_action(vs, i)
        if m.get("available"):
            transitions.append(TopologyTransition(
                transition_type="merger",
                band_index=i,
                r_inner=m["r_inner"],
                r_outer=m["r_outer"],
                euclidean_action=m["euclidean_action"],
                probability=topology_change_probability(m["euclidean_action"]),
            ))
    return sorted(transitions, key=lambda t: t.euclidean_action)
