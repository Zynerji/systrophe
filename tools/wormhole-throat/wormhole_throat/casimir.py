"""Casimir energy density at a wormhole-throat candidate."""

from __future__ import annotations

from dataclasses import dataclass

from systrophe.casimir_throat import (
    brown_maclay_at_lp_point,
    brown_maclay_energy_density,
    compare_throat_to_brown_maclay,
)
from systrophe.vanstockum import VanStockumInterior


@dataclass(frozen=True)
class CasimirThroatReport:
    """Casimir-energy-density diagnostics at one throat radius."""
    r_throat: float
    plate_separation_d: float
    brown_maclay_T_components: dict
    brown_maclay_energy_density_flat: float
    topological_coefficient: float
    topological_to_BM_ratio: float


def casimir_at_throat(omega: float, R: float, r_throat: float,
                      plate_separation_d: float = 1.0,
                      gamma_eff: float = 0.0) -> CasimirThroatReport:
    """Brown-Maclay Casimir tensor at an LP throat + topological coefficient.

    Parameters
    ----------
    omega, R : van Stockum cylinder parameters.
    r_throat : float
        Throat radius (typically a candidate from
        WormholeThroatExplorer.candidate_throats()).
    plate_separation_d : float, default 1.0
        Plate separation d for the Brown-Maclay calculation.
    gamma_eff : float, default 0.0
        Effective holonomy phase for the topological-coefficient
        comparison.
    """
    vs = VanStockumInterior(omega=float(omega), R=float(R))
    bm_at = brown_maclay_at_lp_point(
        vs, r=float(r_throat), d=float(plate_separation_d),
    )
    rho_BM_flat = float(brown_maclay_energy_density(float(plate_separation_d)))
    cmp_ = compare_throat_to_brown_maclay(
        d=float(plate_separation_d), gamma_eff=float(gamma_eff),
    )
    return CasimirThroatReport(
        r_throat=float(r_throat),
        plate_separation_d=float(plate_separation_d),
        brown_maclay_T_components=dict(bm_at),
        brown_maclay_energy_density_flat=rho_BM_flat,
        topological_coefficient=float(cmp_["C_topological"]),
        topological_to_BM_ratio=float(cmp_.get("ratio_C_to_BM", float("nan"))),
    )
