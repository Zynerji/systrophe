"""WarpDrive: common-protocol wrapper around 3 canonical warp metrics.

Each drive exposes:
  - `energy_density(x, rho)` — T_tt at a point
  - `nec_radial(x, rho)` — T_{kk} along outward radial null geodesic
  - `total_negative_energy(box_half_size, n_grid)` — numerical integration
    over a cylindrical volume; only the negative-T_tt contributions are
    summed (the "exotic matter requirement").

Wall location is drive-specific:
  - Alcubierre: r_s = sqrt(x² + rho²) = R
  - Bobrick-Martire & Lentz: (x_wall, rho_wall) = (R, 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from systrophe.alcubierre import (
    alcubierre_energy_density,
    alcubierre_NEC_radial,
    alcubierre_total_negative_energy,
)
from systrophe.bobrick_martire import (
    bm_energy_density,
    bm_NEC_radial,
)
from systrophe.lentz_soliton import (
    lentz_energy_density,
    lentz_NEC_radial,
)


class WarpDrive(Protocol):
    """Common API."""
    name: str

    def energy_density(self, x: float, rho: float) -> float: ...
    def nec_radial(self, x: float, rho: float) -> float: ...
    def total_negative_energy(self, box_half_size: float, n_grid: int) -> float: ...
    def wall_location(self) -> tuple[float, float]: ...


def _integrate_negative_T_tt(
    drive: "WarpDrive", box_half_size: float, n_grid: int,
) -> float:
    """Integrate min(T_tt, 0) over a cylindrical box [-L,L] x [0,L].

    Returns a NEGATIVE number (or 0 if the drive's T_tt is non-negative
    everywhere). Volume element is 2 pi rho dx d rho.
    """
    xs = np.linspace(-box_half_size, box_half_size, n_grid)
    rhos = np.linspace(0.01, box_half_size, n_grid)
    dx = xs[1] - xs[0]
    drho = rhos[1] - rhos[0]
    total = 0.0
    for x in xs:
        for r in rhos:
            T_tt = drive.energy_density(float(x), float(r))
            if T_tt < 0:
                total += T_tt * 2.0 * np.pi * r * dx * drho
    return float(total)


# ---------------------------------------------------------------------------
# Alcubierre adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlcubierreDrive:
    """Alcubierre (1994) warp bubble.

    Wall is at r_s = R. Parametrise by ship velocity v_s, wall thickness
    1/sigma, and bubble radius R.
    """
    v_s: float = 1.0
    R: float = 1.0
    sigma: float = 8.0
    name: str = "alcubierre"

    def energy_density(self, x: float, rho: float) -> float:
        r_s = float(np.sqrt(x ** 2 + rho ** 2))
        return float(alcubierre_energy_density(
            r_s=r_s, v_s=self.v_s, R=self.R, sigma=self.sigma,
        ))

    def nec_radial(self, x: float, rho: float) -> float:
        r_s = float(np.sqrt(x ** 2 + rho ** 2))
        return float(alcubierre_NEC_radial(
            r_s=r_s, v_s=self.v_s, R=self.R, sigma=self.sigma,
        ))

    def total_negative_energy(self, box_half_size: float = 3.0,
                              n_grid: int = 40) -> float:
        """Use Alcubierre's analytic numerical integral when available."""
        return float(alcubierre_total_negative_energy(
            v_s=self.v_s, R=self.R, sigma=self.sigma, n_grid=max(n_grid, 200),
        ))

    def wall_location(self) -> tuple[float, float]:
        # Wall is the sphere r_s = R. Canonical wall sample = (R, 0).
        return (self.R, 0.0)


# ---------------------------------------------------------------------------
# Bobrick-Martire adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BobrickMartireDrive:
    """Bobrick-Martire (2021) linearised positive-energy variant.

    m_ADM > 0 respects NEC (no exotic matter). m_ADM < 0 needs exotic.
    """
    m_ADM: float = 1.0
    R: float = 1.0
    sigma: float = 4.0
    name: str = "bobrick_martire"

    def energy_density(self, x: float, rho: float) -> float:
        return float(bm_energy_density(
            x=float(x), rho=float(rho), m_ADM=self.m_ADM,
            R=self.R, sigma=self.sigma,
        ))

    def nec_radial(self, x: float, rho: float) -> float:
        return float(bm_NEC_radial(
            x=float(x), rho=float(rho), m_ADM=self.m_ADM,
            R=self.R, sigma=self.sigma,
        ))

    def total_negative_energy(self, box_half_size: float = 3.0,
                              n_grid: int = 40) -> float:
        return _integrate_negative_T_tt(self, box_half_size, n_grid)

    def wall_location(self) -> tuple[float, float]:
        return (self.R, 0.0)


# ---------------------------------------------------------------------------
# Lentz adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LentzDrive:
    """Lentz (2020) two-soliton positive-energy proposal."""
    v_s: float = 0.7
    sigma: float = 4.0
    rho_1: float = 0.0
    rho_2: float = 1.5
    a_1: float = 1.0
    a_2: float = -1.0
    name: str = "lentz"

    def energy_density(self, x: float, rho: float) -> float:
        return float(lentz_energy_density(
            x=float(x), rho=float(rho), v_s=self.v_s, sigma=self.sigma,
            rho_1=self.rho_1, rho_2=self.rho_2,
            a_1=self.a_1, a_2=self.a_2,
        ))

    def nec_radial(self, x: float, rho: float) -> float:
        return float(lentz_NEC_radial(
            x=float(x), rho=float(rho), v_s=self.v_s, sigma=self.sigma,
            rho_1=self.rho_1, rho_2=self.rho_2,
            a_1=self.a_1, a_2=self.a_2,
        ))

    def total_negative_energy(self, box_half_size: float = 3.0,
                              n_grid: int = 40) -> float:
        return _integrate_negative_T_tt(self, box_half_size, n_grid)

    def wall_location(self) -> tuple[float, float]:
        # Wall is at the soliton outer ring rho_2.
        return (0.0, self.rho_2)
