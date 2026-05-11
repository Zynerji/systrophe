"""Quantum information channel embedded in the LP exterior.

Treat a worldline in the LP exterior as a *noisy quantum channel*:
qubits are launched at radius r_A, propagate radially to r_B, and we
ask for the channel capacity. The "noise" comes from the gravitational
red/blue-shift between r_A and r_B (which acts as amplitude damping)
and from photon-sphere caustics / chronology-horizon crossings (which
act as projective measurement).

This module provides:

- redshift_factor: ratio of proper times between two radii;
- amplitude_damping_p: damping parameter as function of redshift;
- channel_capacity_holevo: log_2 of effective channel dimension;
- chronology_horizon_decoherence: how many CHs are crossed and
  per-crossing decoherence;
- closed_timelike_loop_capacity: information capacity *around* a CTC
  loop (vanishing classical capacity; quantum capacity model);
- entanglement_swap_protocol: simple distillation across two radii;
- channel_assisted_clock_sync: send a quantum clock across radii and
  measure the time-dilation residual.

The channel-capacity computations are heuristic: we model the LP
exterior as a graded amplitude-damping channel; closed-form quantum
capacities are not available, so we use the lower bound
Q >= log_2(1 - p) + p log_2 p where p is the damping. Use as a
rough scaling, not a literal bound.
"""

from __future__ import annotations

import math

import numpy as np

from .vanstockum import VanStockumInterior


def redshift_factor(vs: VanStockumInterior, r_A: float, r_B: float) -> float:
    """Gravitational redshift ratio sqrt(-g_tt(r_A) / -g_tt(r_B)).

    For LP, g_tt = -F. We return sqrt(F(r_A)/F(r_B)) (assuming both
    positive, ie. between chronology horizons).
    """
    F_A = float(vs.analytic_exterior_F(np.array([r_A]))[0])
    F_B = float(vs.analytic_exterior_F(np.array([r_B]))[0])
    if F_A <= 0 or F_B <= 0:
        return float("nan")
    return float(math.sqrt(F_A / F_B))


def amplitude_damping_p(redshift: float) -> float:
    """Map redshift ratio z to amplitude-damping p in [0, 1].

    Heuristic: p = 1 - z^2 / (1 + z^2) when z > 1 (blueshift induces
    bit-flip); p = 1 - 1/z^2 / (1 + 1/z^2) when z < 1; both give
    p in [0, 0.5] in both regimes.
    """
    if not math.isfinite(redshift) or redshift <= 0:
        return 1.0
    z = redshift
    if z >= 1.0:
        return float(1.0 - z * z / (1.0 + z * z))
    return float(1.0 - (1.0 / (z * z)) / (1.0 + 1.0 / (z * z)))


def channel_capacity_holevo(p: float) -> float:
    """Heuristic Holevo capacity log_2(d_effective) for amplitude damping.

    Standard result: classical capacity of amplitude-damping is
    log_2(1 + sqrt(1-p)) - h_2((1+sqrt(1-p))/2) where h_2 is binary
    entropy.
    """
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    s = math.sqrt(max(1.0 - p, 0.0))
    arg = (1.0 + s) / 2.0
    if arg <= 0 or arg >= 1:
        return 0.0
    h2 = -arg * math.log2(arg) - (1 - arg) * math.log2(1 - arg)
    return float(max(1.0 - h2, 0.0))


def chronology_horizon_crossings_per_loop(
    vs: VanStockumInterior, r_start: float, r_end: float,
) -> int:
    """Number of F=0 surfaces traversed by a radial path from r_start to r_end."""
    if not vs.is_supercritical():
        return 0
    R = vs.R
    alpha = vs.alpha
    gamma = math.pi - math.atan(alpha)
    n_start = (alpha * math.log(r_start / R) + gamma) / math.pi
    n_end = (alpha * math.log(r_end / R) + gamma) / math.pi
    return int(abs(math.floor(n_end) - math.floor(n_start)))


def chronology_horizon_decoherence(
    vs: VanStockumInterior, r_start: float, r_end: float,
    per_crossing_p: float = 0.5,
) -> dict:
    """Effective channel parameters after crossing N chronology horizons.

    Each F=0 crossing applies an amplitude-damping with parameter
    per_crossing_p. After N crossings, the cumulative damping is
    1 - (1-p)^N.
    """
    n_cross = chronology_horizon_crossings_per_loop(vs, r_start, r_end)
    cumulative_p = float(1.0 - (1.0 - per_crossing_p) ** n_cross)
    capacity = channel_capacity_holevo(cumulative_p)
    return {
        "n_crossings": int(n_cross),
        "cumulative_damping": cumulative_p,
        "channel_capacity_bits": capacity,
    }


def closed_timelike_loop_capacity(
    vs: VanStockumInterior, r_loop: float,
) -> dict:
    """Information capacity *around* a CTC loop.

    Per Deutsch's prescription, a CTC enforces consistency:
       rho_in -> Tr_aux(U(rho_in (x) rho_CTC) U^dagger) = rho_CTC.
    Classical capacity of a self-consistent CTC channel is zero (the
    output equals the fixed point regardless of input). Quantum
    capacity is also zero for generic U. We return both as zero,
    plus a "non-Deutsch" capacity (the Lloyd P-CTC capacity).
    """
    F_r = float(vs.analytic_exterior_F(np.array([r_loop]))[0])
    if F_r >= 0:
        regime = "non-CTC"
    else:
        regime = "CTC"
    deutsch_classical = 0.0
    deutsch_quantum = 0.0
    # Lloyd P-CTC: capacity = log_2(rank of U) for generic U on a CTC
    # In our heuristic, rank = 2 (qubit case).
    pctc_capacity = 1.0 if regime == "CTC" else 0.0
    return {
        "r_loop": r_loop,
        "F_at_loop": F_r,
        "regime": regime,
        "deutsch_classical_capacity": deutsch_classical,
        "deutsch_quantum_capacity": deutsch_quantum,
        "pctc_lloyd_capacity_bits": pctc_capacity,
    }


def entanglement_swap_protocol(
    vs: VanStockumInterior, r_A: float, r_B: float,
) -> dict:
    """Send half of a Bell pair from r_A to r_B; report final fidelity.

    Fidelity decays per the amplitude-damping channel:
        F_final = (1 + sqrt(1 - p)) / 2
    where p is the effective damping over the radial trajectory.
    """
    z = redshift_factor(vs, r_A, r_B)
    p_radial = amplitude_damping_p(z) if math.isfinite(z) else 1.0
    # Add CH-crossing damping
    n_cross = chronology_horizon_crossings_per_loop(vs, r_A, r_B)
    p_ch = 1.0 - (1.0 - 0.5) ** n_cross
    # Combined (independent) damping
    p_total = 1.0 - (1.0 - p_radial) * (1.0 - p_ch)
    fidelity = (1.0 + math.sqrt(max(1.0 - p_total, 0.0))) / 2.0
    return {
        "r_A": r_A,
        "r_B": r_B,
        "redshift_factor": z,
        "p_radial": p_radial,
        "p_ch_crossings": p_ch,
        "p_total": p_total,
        "final_fidelity": float(fidelity),
    }


def channel_assisted_clock_sync(
    vs: VanStockumInterior, r_A: float, r_B: float,
    coord_time: float = 1.0,
) -> dict:
    """Send a quantum clock between r_A and r_B; report time-dilation residual.

    Proper time elapsed at r is dtau = sqrt(F(r)) dt. The differential
    dtau(B) - dtau(A) is the "residual" the clock measures.
    """
    F_A = float(vs.analytic_exterior_F(np.array([r_A]))[0])
    F_B = float(vs.analytic_exterior_F(np.array([r_B]))[0])
    if F_A <= 0 or F_B <= 0:
        return {
            "r_A": r_A, "r_B": r_B,
            "tau_A": float("nan"), "tau_B": float("nan"),
            "residual": float("nan"),
            "behind_CH": True,
        }
    tau_A = float(math.sqrt(F_A) * coord_time)
    tau_B = float(math.sqrt(F_B) * coord_time)
    return {
        "r_A": r_A, "r_B": r_B,
        "tau_A": tau_A, "tau_B": tau_B,
        "residual": tau_B - tau_A,
        "behind_CH": False,
    }


def shannon_classical_capacity_over_radial_path(
    vs: VanStockumInterior, r_min_path: float, r_max_path: float,
    n_path_samples: int = 50,
) -> dict:
    """Classical Shannon capacity along a radial path with multiple CH crossings.

    Returns the cumulative capacity per CH segment.
    """
    if r_min_path >= r_max_path:
        raise ValueError("r_max_path must exceed r_min_path")
    r_path = np.linspace(r_min_path, r_max_path, n_path_samples)
    capacities = []
    for r_i in r_path[1:]:
        z = redshift_factor(vs, r_min_path, float(r_i))
        if not math.isfinite(z):
            capacities.append(0.0)
            continue
        p = amplitude_damping_p(z)
        capacities.append(channel_capacity_holevo(p))
    return {
        "r_path": r_path[1:],
        "capacities": np.array(capacities),
        "mean_capacity": float(np.mean(capacities)),
        "min_capacity": float(np.min(capacities)),
    }
