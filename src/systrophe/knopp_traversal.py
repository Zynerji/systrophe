"""Knopp Drive traversal: end-to-end engineering deliverable.

Given a Knopp-Drive configuration and a destination distance L, this
module returns the full traversal report:
 - apparent crossing time (coordinate time in the LP frame)
 - craft proper time
 - total integrated exotic-matter requirement (zero inside the Tipler
   band, otherwise a finite number)
 - sustained drive power
 - total energy budget (drive power x traversal duration)
 - Pfenning-Ford compatibility flag

The traversal trajectory is the Krasnikov tube's worldline; we
parameterise it as a linear ramp along the bubble axis at constant
apparent velocity v_s. The Tipler CTC band gives a "free" segment in
the middle of the journey where the exotic-matter requirement
vanishes.

This is the headline engineering output that converts the framework's
catcher-verified emergents into an actionable monetary number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .knopp_drive import (
    KnoppDriveBudget,
    KnoppDriveConfig,
    knopp_budget,
)
from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class KnoppTraversalReport:
    distance: float                   # apparent traversal distance (geometric units)
    v_s_apparent: float               # apparent shell velocity
    coord_time_total: float           # external coordinate time elapsed
    proper_time_total: float          # craft proper-time elapsed
    n_band_segments: int              # number of distinct Tipler CTC bands traversed
    inside_band_fraction: float       # fraction of journey inside a band
    exotic_matter_total: float        # integrated |E_neg| over trajectory
    sustained_drive_power: float      # P_drive at saturation
    total_energy_budget: float        # P_drive * coord_time_total
    pfenning_ford_compatible: bool


def _band_indicator(vs: VanStockumInterior, r: float) -> bool:
    """True iff r is inside a Tipler CTC band (tipler tilt >= 1)."""
    from .tipler_krasnikov_hybrid import tipler_tilt_at
    return tipler_tilt_at(vs, r) >= 1.0


def knopp_traversal(
    cfg: KnoppDriveConfig | None = None,
    distance: float = 12.0,
    n_steps: int = 200,
) -> KnoppTraversalReport:
    """Numerically integrate the Knopp Drive traversal over a
    radial path of given distance.

    For simplicity we take the trajectory to be a radial outflow from
    r = R_cylinder + 0.05 to r = R_cylinder + 0.05 + distance, at the
    apparent shell velocity v_s. Each step samples the Knopp budget
    locally.

    Returns a `KnoppTraversalReport` summarising apparent times,
    exotic-matter budget, and power usage.
    """
    if cfg is None:
        cfg = KnoppDriveConfig()
    vs = VanStockumInterior(omega=cfg.omega, R=cfg.R_cylinder)
    r0 = cfg.R_cylinder + 0.05
    rs = np.linspace(r0, r0 + distance, n_steps)
    dr = rs[1] - rs[0]
    # Apparent dt = dr / v_s per step
    dt_per_step = dr / max(cfg.v_s, 1e-12)
    coord_time = float(dt_per_step * n_steps)
    # Proper time is multiplied by the local g_tt factor; for an
    # Alcubierre-style flat interior the craft sits at rest, so
    # proper time = coord time.
    proper_time = coord_time  # in the Alcubierre interior the craft is geodesic
    # Per-step exotic matter
    inside_band = np.array([_band_indicator(vs, float(r)) for r in rs])
    n_band_segments = int(np.sum(np.diff(inside_band.astype(int)) > 0)) + int(
        inside_band[0]
    )
    inside_frac = float(np.mean(inside_band))
    # Step-wise composite |E_neg|
    e_neg_per_step = []
    pf_flags = []
    P_drive = 0.0
    for r in rs:
        b = knopp_budget(replace(cfg, r_orbit=float(r)))
        e_neg_per_step.append(abs(b.composite_E_neg))
        P_drive = b.sustained_drive_power  # constant across the journey
        pf_flags.append(b.pfenning_ford_compatible)
    exotic_total = float(np.trapezoid(np.array(e_neg_per_step), rs))
    energy_total = float(P_drive * coord_time)
    pf_all = all(pf_flags)
    return KnoppTraversalReport(
        distance=float(distance),
        v_s_apparent=float(cfg.v_s),
        coord_time_total=coord_time,
        proper_time_total=proper_time,
        n_band_segments=int(n_band_segments),
        inside_band_fraction=inside_frac,
        exotic_matter_total=exotic_total,
        sustained_drive_power=float(P_drive),
        total_energy_budget=energy_total,
        pfenning_ford_compatible=pf_all,
    )


def knopp_traversal_Q_sweep(
    distance: float = 12.0,
    Q_range: tuple[float, float] = (1.0, 2000.0),
    n_Q: int = 30,
    cfg_base: KnoppDriveConfig | None = None,
    n_steps: int = 80,
) -> dict:
    """Sweep Q and report the (E_total, P-F-ok) curve.

    The Pfenning-Ford bound puts an UPPER limit on Q for a given
    traversal duration: as Q grows, |E_shell| shrinks but tau grows
    linearly; their product must exceed 3/(32 pi^2 sigma^2). For
    short journeys, all Q are OK; for long journeys, high Q fails.

    Catcher runs over the Q sweep on the P-F-failure boolean (1 at
    fail, 0 at OK) -> any flip is an emergent: the bound where the
    optimal cavity becomes incompatible with quantum inequalities.
    """
    if cfg_base is None:
        cfg_base = KnoppDriveConfig()
    Q_grid = np.linspace(*Q_range, n_Q)
    E_totals = np.zeros(n_Q)
    pf_failures = np.zeros(n_Q)
    for i, Q in enumerate(Q_grid):
        cfg = replace(cfg_base, Q=float(Q))
        rep = knopp_traversal(cfg, distance=distance, n_steps=n_steps)
        E_totals[i] = rep.total_energy_budget
        pf_failures[i] = 0.0 if rep.pfenning_ford_compatible else 1.0
    # Find the Q where P-F flips
    flip_idx = -1
    for i in range(1, n_Q):
        if pf_failures[i] != pf_failures[i - 1]:
            flip_idx = i
            break
    from .novelty_catcher import scan_novelty
    def fn(qv):
        idx = int(np.argmin(np.abs(Q_grid - qv)))
        return np.array([float(pf_failures[idx])])
    result = scan_novelty(Q_grid, fn, n_bits=32)
    return {
        "distance": float(distance),
        "Q_grid": Q_grid.tolist(),
        "E_total_grid": E_totals.tolist(),
        "pfenning_ford_failures": pf_failures.tolist(),
        "flip_Q": float(Q_grid[flip_idx]) if flip_idx >= 0 else None,
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }


def summarise_traversal(rep: KnoppTraversalReport) -> str:
    """One-line summary of a Knopp traversal report."""
    return (
        f"Knopp traversal of L={rep.distance:.2f}: "
        f"t_coord={rep.coord_time_total:.2f}, "
        f"t_proper={rep.proper_time_total:.2f}, "
        f"|E_neg|={rep.exotic_matter_total:.3e}, "
        f"inside_band_frac={rep.inside_band_fraction:.2f}, "
        f"E_total={rep.total_energy_budget:.3e}, "
        f"P-F_ok={rep.pfenning_ford_compatible}"
    )
