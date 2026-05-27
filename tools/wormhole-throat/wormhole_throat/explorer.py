"""WormholeThroatExplorer: scan + diagnose throats on a van Stockum exterior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from systrophe.geometry.wormhole_throat import (
    build_wormhole_report,
    effective_redshift_function,
    effective_shape_function,
    find_candidate_throats,
    flaring_out_check,
    z3_cover_throat_interpretation,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@dataclass(frozen=True)
class ThroatReport:
    """Compact summary of a single candidate throat."""
    r_throat: float
    b_throat: float
    b_prime: float
    is_throat: bool
    is_flaring_out: bool
    F_throat: float
    redshift_phi: float
    is_horizon_at_throat: bool


class WormholeThroatExplorer:
    """Wrap a VanStockum exterior and expose throat diagnostics.

    Parameters
    ----------
    omega, R : float
        Van Stockum source parameters.
    r_search_min, r_search_max : float, optional
        Throat search range. Defaults 1.05R .. 10R.
    """

    def __init__(self, omega: float, R: float,
                 r_search_min: float | None = None,
                 r_search_max: float | None = None) -> None:
        self.vs = VanStockumInterior(omega=float(omega), R=float(R))
        self.r_search_min = (1.05 * R) if r_search_min is None else float(r_search_min)
        self.r_search_max = (10.0 * R) if r_search_max is None else float(r_search_max)

    @property
    def omega(self) -> float:
        return float(self.vs.omega)

    @property
    def R(self) -> float:
        return float(self.vs.R)

    def candidate_throats(self, n_grid: int = 2001) -> list[float]:
        """Throats are loci where b_eff(r) = r ⇔ L(r) = 0 — i.e. CTC
        band boundaries."""
        return list(find_candidate_throats(
            self.vs, r_min=self.r_search_min, r_max=self.r_search_max,
            n_grid=n_grid,
        ))

    def shape_function(self, r: float) -> float:
        return float(effective_shape_function(self.vs, float(r)))

    def redshift_function(self, r: float) -> float:
        return float(effective_redshift_function(self.vs, float(r)))

    def report(self, r_throat: float) -> ThroatReport:
        """Full diagnostic at one r_throat candidate."""
        w = build_wormhole_report(self.vs, r_throat=float(r_throat))
        return ThroatReport(
            r_throat=float(w.r_throat),
            b_throat=float(w.b_throat),
            b_prime=float(w.b_prime),
            is_throat=bool(w.is_throat),
            is_flaring_out=bool(w.is_flaring_out),
            F_throat=float(w.F_throat),
            redshift_phi=float(w.redshift_phi),
            is_horizon_at_throat=bool(w.is_horizon_at_throat),
        )

    def report_all(self, n_grid: int = 2001) -> list[ThroatReport]:
        """Report on every candidate throat in the search range."""
        return [self.report(r) for r in self.candidate_throats(n_grid=n_grid)]

    def z3_cover_interpretation(self, gamma_eff: float = 0.0,
                                  n_branches: int = 3) -> dict:
        """Z_3 cover quotient interpretation (cylinder axis as throat fixed locus)."""
        return dict(z3_cover_throat_interpretation(
            gamma_eff=float(gamma_eff), n_branches=int(n_branches),
        ))
