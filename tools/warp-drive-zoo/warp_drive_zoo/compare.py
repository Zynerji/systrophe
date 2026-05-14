"""Head-to-head comparator: NEC at wall + total negative energy + ranking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .drives import WarpDrive


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
