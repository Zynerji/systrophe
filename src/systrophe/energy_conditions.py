"""Energy-condition diagnostics for the van Stockum dust source.

The van Stockum interior solves Einstein's equations sourced by a
*pressureless* (dust) stress-energy tensor

    T_{\\mu \\nu} = \\rho(r) u_\\mu u_\\nu,
    \\rho(r) = (\\omega^2 / (2\\pi)) \\exp(\\omega^2 r^2),
    u^\\mu = \\partial_t   (comoving frame).

with c = G = 1. We check the four standard pointwise energy conditions
(NEC, WEC, SEC, DEC).

Result (verified analytically and numerically):
  All four conditions are satisfied at every interior point for any
  omega > 0, R > 0. This is a *strong* statement about the Tipler /
  van Stockum CTC structure: the chronology pathology arises purely
  from the geometric idealisations (infinite axial extent, rigid
  rotation, perfect axisymmetry) and NOT from exotic matter. The
  source is positive-energy, causal-flux, satisfies the strong energy
  condition (focusing), and bounded above by the speed of light.

This module exposes:
- proper_energy_density(r): \\rho(r) closed form.
- total_energy_per_unit_length(R, omega): finite integral 1/2 (e^{a^2} - 1).
- check_null_energy_condition(vs, r): NEC verdict per radius.
- check_weak_energy_condition(vs, r): WEC verdict.
- check_strong_energy_condition(vs, r): SEC verdict.
- check_dominant_energy_condition(vs, r): DEC verdict.
- energy_condition_report(vs, r_grid): summary dict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class EnergyConditionReport:
    """Summary of pointwise energy-condition verdicts for a van Stockum source.

    Each verdict is True when the condition is satisfied at every sampled
    radius; False otherwise. The `min_residual` fields record the
    minimum value of the relevant inequality across the sample grid (a
    positive minimum means the condition is satisfied with margin).
    """

    omega: float
    R: float
    a: float
    r_grid: np.ndarray
    rho: np.ndarray
    nec_holds: bool
    wec_holds: bool
    sec_holds: bool
    dec_holds: bool
    nec_min_residual: float
    wec_min_residual: float
    sec_min_residual: float
    dec_min_residual: float
    total_energy_per_unit_length: float


def proper_energy_density(omega: float, r: float | np.ndarray) -> np.ndarray:
    """\\rho(r) = (omega^2 / (2 pi)) * exp(omega^2 r^2). Always non-negative."""
    r_arr = np.asarray(r, dtype=float)
    return (omega ** 2 / (2.0 * np.pi)) * np.exp(omega ** 2 * r_arr ** 2)


def total_energy_per_unit_length(omega: float, R: float) -> float:
    """Energy per unit z-length of the dust column: \\int_0^R rho(r) 2 pi r dr.

    Closed form: 0.5 * (exp((omega R)^2) - 1)  in c = G = 1 units.
    """
    a = omega * R
    return 0.5 * (np.exp(a * a) - 1.0)


def check_null_energy_condition(vs: VanStockumInterior, r: np.ndarray) -> tuple[bool, np.ndarray]:
    """T_{mu nu} k^mu k^nu >= 0 for all null k.

    For dust T = rho u u with rho >= 0, this reduces to rho * (u . k)^2 >= 0,
    which is non-negative for any rho >= 0. We return True and the
    minimum residual rho across the sample grid (a sanity check).
    """
    rho = proper_energy_density(vs.omega, r)
    holds = bool(np.all(rho >= 0.0))
    return holds, rho


def check_weak_energy_condition(vs: VanStockumInterior, r: np.ndarray) -> tuple[bool, np.ndarray]:
    """T_{mu nu} t^mu t^nu >= 0 for all unit timelike t.

    For dust this reduces to rho * (u . t)^2 >= 0; satisfied iff rho >= 0.
    """
    rho = proper_energy_density(vs.omega, r)
    holds = bool(np.all(rho >= 0.0))
    return holds, rho


def check_strong_energy_condition(
    vs: VanStockumInterior, r: np.ndarray, t_dot_u_squared: float = 1.0
) -> tuple[bool, np.ndarray]:
    """Strong energy condition: (T_{mu nu} - 1/2 g_{mu nu} T) t^mu t^nu >= 0.

    For dust T = -rho, so the condition becomes
        rho (u . t)^2 + 1/2 rho (t . t) = rho [(u . t)^2 - 1/2] >= 0
    (using t . t = -1 for a unit timelike vector).

    The inequality (u . t)^2 >= 1 holds for any timelike normalisation
    (Cauchy-Schwarz on a Lorentzian space), so the SEC is satisfied with
    margin (u . t)^2 - 1/2 >= 1/2. We test the worst case
    `t_dot_u_squared = 1` (t aligned with u, gamma = 1) by default.
    """
    rho = proper_energy_density(vs.omega, r)
    residual = rho * (t_dot_u_squared - 0.5)
    holds = bool(np.all(residual >= -1e-12))
    return holds, residual


def check_dominant_energy_condition(
    vs: VanStockumInterior, r: np.ndarray
) -> tuple[bool, np.ndarray]:
    """Dominant energy condition: T_{mu nu} t^mu t^nu >= 0 AND T^mu_nu t^nu causal.

    For dust, T^mu_nu t^nu = rho u^mu (u_nu t^nu). This is a multiple of
    u^mu, which is timelike (norm -1) wherever the dust 4-velocity is
    timelike. Inside the van Stockum interior, u = partial_t with
    g_{tt} = -1 < 0 always, so u is timelike everywhere; DEC holds.
    Returns the residual `rho` (>=0 ensures DEC).
    """
    rho = proper_energy_density(vs.omega, r)
    holds = bool(np.all(rho >= 0.0))
    return holds, rho


def energy_condition_report(
    vs: VanStockumInterior, r_grid: np.ndarray | None = None
) -> EnergyConditionReport:
    """One-shot summary of energy conditions on the interior r in [0, R]."""
    if r_grid is None:
        r_grid = np.linspace(0.0, vs.R, 1001)
    r_grid = np.asarray(r_grid, dtype=float)
    rho = proper_energy_density(vs.omega, r_grid)

    nec, nec_res = check_null_energy_condition(vs, r_grid)
    wec, wec_res = check_weak_energy_condition(vs, r_grid)
    sec, sec_res = check_strong_energy_condition(vs, r_grid)
    dec, dec_res = check_dominant_energy_condition(vs, r_grid)

    return EnergyConditionReport(
        omega=float(vs.omega),
        R=float(vs.R),
        a=float(vs.a),
        r_grid=r_grid,
        rho=rho,
        nec_holds=nec,
        wec_holds=wec,
        sec_holds=sec,
        dec_holds=dec,
        nec_min_residual=float(nec_res.min()),
        wec_min_residual=float(wec_res.min()),
        sec_min_residual=float(sec_res.min()),
        dec_min_residual=float(dec_res.min()),
        total_energy_per_unit_length=float(total_energy_per_unit_length(vs.omega, vs.R)),
    )
