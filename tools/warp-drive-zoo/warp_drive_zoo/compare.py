"""Head-to-head comparator: NEC at wall + total negative energy + ranking.

The base ``WarpDriveComparison`` ranks drives by their integrated exotic-
matter content. ``compare_drives_with_qi`` extends it with a uniform
Ford-Roman quantum-inequality realizability score
(``qi_normalized_score`` > 1 => QI-forbidden) so warp drives sit on the
SAME axis as the Goedel / Gott / Kerr / van Stockum / wormhole CTC registry
(see ``qi_scorer`` and ``qi_registry``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .drives import WarpDrive
from .qi_registry import score_alcubierre
from .qi_scorer import (
    QIScore,
    score_member,
    t_kk_dust_min,
    t_kk_vacuum_min,
)


@dataclass(frozen=True)
class WarpDriveComparison:
    """Side-by-side warp-drive comparison.

    Attributes
    ----------
    drives : list[str]
        Drive names, in input order.
    nec_at_wall : dict[str, float]
        T_{kk} at each drive's canonical wall sample. Negative => NEC
        violated => exotic matter required at the wall.
    energy_density_at_wall : dict[str, float]
        T_tt at each wall.
    total_negative_energy : dict[str, float]
        Integral of min(T_tt, 0) dV. <= 0; the magnitude is the exotic-
        matter requirement.
    nec_violated : dict[str, bool]
        True iff nec_at_wall < 0.
    ranking_by_exotic_matter : list[tuple[str, float]]
        Drives sorted by |total_negative_energy| ascending — lower is
        less exotic.
    """
    drives: list
    nec_at_wall: dict
    energy_density_at_wall: dict
    total_negative_energy: dict
    nec_violated: dict
    ranking_by_exotic_matter: list


def compare_drives(
    drives: list[WarpDrive],
    box_half_size: float = 3.0,
    n_grid: int = 32,
) -> WarpDriveComparison:
    """Compare a list of drives head-to-head.

    Parameters
    ----------
    drives : list of WarpDrive
        Each must expose name, nec_radial, energy_density,
        total_negative_energy, wall_location.
    box_half_size, n_grid : integration parameters for non-Alcubierre
        drives.

    Returns
    -------
    WarpDriveComparison
    """
    nec_at_wall: dict = {}
    rho_at_wall: dict = {}
    total: dict = {}
    nec_violated: dict = {}
    for d in drives:
        x_w, rho_w = d.wall_location()
        nec_at_wall[d.name] = float(d.nec_radial(x_w, rho_w))
        rho_at_wall[d.name] = float(d.energy_density(x_w, rho_w))
        total[d.name] = float(d.total_negative_energy(box_half_size, n_grid))
        nec_violated[d.name] = bool(nec_at_wall[d.name] < 0)

    ranking = sorted(
        ((name, abs(total[name])) for name in total),
        key=lambda t: t[1],
    )
    return WarpDriveComparison(
        drives=[d.name for d in drives],
        nec_at_wall=nec_at_wall,
        energy_density_at_wall=rho_at_wall,
        total_negative_energy=total,
        nec_violated=nec_violated,
        ranking_by_exotic_matter=ranking,
    )


# ---------------------------------------------------------------------------
# QI-extended comparison: add the uniform Ford-Roman realizability score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarpDriveQIComparison:
    """Warp-drive comparison extended with the Ford-Roman QI realizability axis.

    Attributes
    ----------
    base : WarpDriveComparison
        The classical NEC / exotic-matter comparison.
    qi_scores : dict[str, QIScore]
        Per-drive ``QIScore`` (most-negative T_kk excursion / |QI bound|).
    qi_normalized_score : dict[str, float]
        ``qi_scores[name].qi_normalized_score``; > 1 => QI-forbidden.
    qi_forbidden : dict[str, bool]
        True iff the drive's score > 1.
    ranking_by_qi : list[tuple[str, float]]
        Drives sorted ascending by qi_normalized_score — lower is more
        realizable.
    """

    base: WarpDriveComparison
    qi_scores: dict
    qi_normalized_score: dict
    qi_forbidden: dict
    ranking_by_qi: list


def _drive_qi_score(d: WarpDrive, box_half_size: float, n_grid: int) -> QIScore:
    """Build a ``QIScore`` for any WarpDrive on the uniform QI axis.

    The most-negative T_kk excursion is read from the drive's own
    ``nec_radial`` / ``energy_density`` sampled over the integration box,
    so positive-energy drives (e.g. Bobrick-Martire with m_ADM > 0, Lentz)
    score 0 and Alcubierre scores > 1. The sampling time tau is the
    macroscopic wall scale (the drive's wall location radius, floored to 1).
    Alcubierre routes through the closed-form scorer for an exact number.
    """
    name = getattr(d, "name", "drive")
    if name == "alcubierre":
        v_s = getattr(d, "v_s", 1.0)
        R = getattr(d, "R", 1.0)
        sigma = getattr(d, "sigma", 8.0)
        return score_alcubierre(v_s=v_s, R=R, sigma=sigma)
    # Generic drive: scan T_tt over the box for the most-negative value.
    xs = np.linspace(-box_half_size, box_half_size, max(n_grid, 16))
    rhos = np.linspace(0.01, box_half_size, max(n_grid, 16))
    t_min = 0.0
    for x in xs:
        for r in rhos:
            v = float(d.energy_density(float(x), float(r)))
            if v < t_min:
                t_min = v
    x_w, _ = d.wall_location()
    tau = max(abs(float(x_w)), 1.0)
    kind = "exotic" if t_min < 0.0 else "vacuum"
    return score_member(name, t_kk_dust_min(t_min) if t_min >= 0 else t_min,
                        tau=tau, source_kind=kind)


def compare_drives_with_qi(
    drives: list[WarpDrive],
    box_half_size: float = 3.0,
    n_grid: int = 32,
) -> WarpDriveQIComparison:
    """Run the base comparison and attach the uniform QI realizability score."""
    base = compare_drives(drives, box_half_size=box_half_size, n_grid=n_grid)
    qi_scores: dict = {}
    qi_norm: dict = {}
    qi_forbidden: dict = {}
    for d in drives:
        sc = _drive_qi_score(d, box_half_size, n_grid)
        qi_scores[d.name] = sc
        qi_norm[d.name] = float(sc.qi_normalized_score)
        qi_forbidden[d.name] = bool(sc.qi_forbidden)
    ranking_by_qi = sorted(
        ((name, qi_norm[name]) for name in qi_norm),
        key=lambda t: t[1],
    )
    return WarpDriveQIComparison(
        base=base,
        qi_scores=qi_scores,
        qi_normalized_score=qi_norm,
        qi_forbidden=qi_forbidden,
        ranking_by_qi=ranking_by_qi,
    )
