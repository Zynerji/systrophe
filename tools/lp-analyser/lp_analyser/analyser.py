"""LPAnalyser: one entry point for the analytic-CTC physics stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from systrophe.vanstockum import VanStockumInterior
from systrophe.quantum_diagnostics import (
    cauchy_horizon_estimate,
    hawking_temperature_at_horizon,
    surface_gravity_at_horizon,
)
from systrophe.energy_conditions import energy_condition_report
from systrophe.stress_energy_ctc import (
    boulware_stress_tensor,
    chronology_protection_report,
    divergence_rate_at_horizon,
    hartle_hawking_stress_tensor,
    unruh_stress_tensor,
    StressEnergyState,
)
from systrophe.hadamard_modesum import (
    hadamard_chronology_report,
    hadamard_V_0,
    hadamard_V_1,
    trace_anomaly_via_V_1,
)
from systrophe.point_splitting import (
    kretschmann_scalar,
    trace_anomaly_4d_exact,
)


@dataclass
class LPSummary:
    """Compact characterisation of a supercritical Tipler exterior."""
    omega: float
    R: float
    a: float
    alpha: float
    regime: str
    interior_regime: str
    is_supercritical: bool
    has_interior_ctc: bool
    mass_per_unit_length: float
    angular_momentum_per_unit_length: float
    energy_conditions_all_satisfied: bool
    n_cauchy_horizons_in_10R: int
    first_horizon_r: float | None
    first_horizon_surface_gravity: float | None
    first_horizon_hawking_temperature: float | None
    boulware_T_tt_simple_pole_power: float | None       # ~ -1 in supercritical
    chronology_protection_verdict: str | None           # from Phase 2a report
    hadamard_V_1_at_midpoint: float | None              # Phase 2b
    kretschmann_at_midpoint: float | None

    def __repr__(self) -> str:
        return (
            f"LPSummary(omega={self.omega}, R={self.R}, a={self.a:.3f}, "
            f"regime={self.regime!r}, n_horizons={self.n_cauchy_horizons_in_10R}, "
            f"CP={self.chronology_protection_verdict!r})"
        )


class LPAnalyser:
    """Drive the analytic-CTC stack for a single rigidly-rotating dust cylinder.

    Wraps :class:`systrophe.vanstockum.VanStockumInterior` and exposes
    the Phase 1a/2a/2b APIs as methods on a single object.

    Parameters
    ----------
    omega : float
        Rotation rate of the source dust (units 1/length, c=G=1).
    R : float
        Cylinder radius. Source occupies r in [0, R]; exterior at r > R.
    """

    def __init__(self, omega: float, R: float) -> None:
        self.vs = VanStockumInterior(omega=float(omega), R=float(R))

    # ------------------------------------------------------------------
    # Forward the obvious classical-GR properties
    # ------------------------------------------------------------------
    @property
    def omega(self) -> float:
        return float(self.vs.omega)

    @property
    def R(self) -> float:
        return float(self.vs.R)

    @property
    def a(self) -> float:
        """Dimensionless rotation parameter a = omega * R."""
        return float(self.vs.a)

    @property
    def alpha(self) -> float:
        """Log-frequency alpha = sqrt(4 a^2 - 1) (defined only when a > 1/2)."""
        if not self.vs.is_supercritical():
            return float("nan")
        return float(self.vs.alpha)

    @property
    def is_supercritical(self) -> bool:
        return bool(self.vs.is_supercritical())

    # ------------------------------------------------------------------
    # Metric components and CTC structure
    # ------------------------------------------------------------------
    def F(self, r: float | np.ndarray) -> np.ndarray:
        """F(r) = -g_{tt}(r) exterior metric component."""
        return np.asarray(self.vs.analytic_exterior_F(np.asarray(r)))

    def K(self, r: float | np.ndarray) -> np.ndarray:
        """K(r) = g_{t phi}(r)."""
        return np.asarray(self.vs.analytic_exterior_K(np.asarray(r)))

    def L(self, r: float | np.ndarray) -> np.ndarray:
        """L(r) = g_{phi phi}(r); L < 0 marks CTC bands."""
        return np.asarray(self.vs.analytic_exterior_L(np.asarray(r)))

    def cauchy_horizons(self, r_max: float | None = None) -> np.ndarray:
        """Closed-form Cauchy horizons (F = 0 surfaces). Requires
        supercritical exterior. Default sweep goes out to 10*R."""
        return np.asarray(cauchy_horizon_estimate(self.vs))

    def ctc_bands(self, r_min: float = None, r_max: float = None,
                     n_grid: int = 2001) -> list[tuple[float, float]]:
        """Intervals in [r_min, r_max] where L(r) < 0 (CTC bands)."""
        from systrophe.ctc import find_ctc_intervals
        r_min = r_min if r_min is not None else 1.05 * self.R
        r_max = r_max if r_max is not None else 10.0 * self.R
        return find_ctc_intervals(self.L, r_min=r_min, r_max=r_max, n_grid=n_grid)

    # ------------------------------------------------------------------
    # Phase 1a: energy conditions
    # ------------------------------------------------------------------
    def energy_conditions(self) -> Any:
        """Pointwise energy-condition diagnostics (NEC, WEC, SEC, DEC)
        on the dust source. For van Stockum dust these are *all*
        satisfied for any omega > 0 -- CTCs come from idealised
        geometry, not exotic matter."""
        return energy_condition_report(self.vs)

    # ------------------------------------------------------------------
    # Hawking / surface gravity
    # ------------------------------------------------------------------
    def surface_gravity(self, r_horizon: float | None = None) -> float:
        """kappa = |F'(r_H)|/2 at a Cauchy horizon. Defaults to the first."""
        if r_horizon is None:
            r_horizon = float(self.cauchy_horizons()[0])
        return float(surface_gravity_at_horizon(self.vs, r_horizon))

    def hawking_temperature(self, r_horizon: float | None = None) -> float:
        """T_H = kappa / (2 pi)."""
        if r_horizon is None:
            r_horizon = float(self.cauchy_horizons()[0])
        return float(hawking_temperature_at_horizon(self.vs, r_horizon))

    # ------------------------------------------------------------------
    # Phase 2a: renormalised stress-energy
    # ------------------------------------------------------------------
    def stress_tensor(self, r: float, state: str = "boulware",
                         r_horizon: float | None = None) -> dict:
        """<T_{mu nu}>_ren in (t, r) sector for the chosen vacuum state.

        state in {"boulware", "hartle_hawking", "unruh"}.
        """
        if state == "boulware":
            return boulware_stress_tensor(self.vs, r)
        if state == "hartle_hawking":
            return hartle_hawking_stress_tensor(self.vs, r, r_horizon=r_horizon)
        if state == "unruh":
            return unruh_stress_tensor(self.vs, r, r_horizon=r_horizon)
        raise ValueError(
            f"unknown state {state!r}; expected boulware|hartle_hawking|unruh"
        )

    def chronology_protection_scan(self, n_horizons: int = 3,
                                     n_samples: int = 18,
                                     eps_min: float = 5e-4,
                                     eps_max: float = 3e-2) -> Any:
        """Phase 2a: fit the Boulware <T_{tt}> simple-pole divergence at
        the first `n_horizons` Cauchy horizons. Returns a
        :class:`systrophe.stress_energy_ctc.ChronologyProtectionReport`.

        On the canonical omega=2, R=1 exterior the powers come out
        ~ -1.00 at every horizon scanned -- the headline Phase 2a
        result."""
        return chronology_protection_report(
            self.vs, n_horizons=n_horizons, n_samples=n_samples,
            eps_min=eps_min, eps_max=eps_max,
        )

    def divergence_rate(self, r_horizon: float, state: str = "boulware",
                          component: str = "T_tt", n_samples: int = 18) -> Any:
        """Power-law fit |<T_{component}>(r_H + eps)| ~ eps^p at a horizon."""
        state_enum = StressEnergyState(state) if isinstance(state, str) else state
        return divergence_rate_at_horizon(
            self.vs, r_horizon, state=state_enum, component=component,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Phase 2b: Hadamard biparametrix
    # ------------------------------------------------------------------
    def hadamard_V_0(self, r: float) -> float:
        """V_0 = 0 in vacuum + massless + conformal."""
        return float(hadamard_V_0(self.vs, r))

    def hadamard_V_1(self, r: float) -> float:
        """V_1 = K_kretsch / 720 in vacuum-conformal case."""
        return float(hadamard_V_1(self.vs, r))

    def hadamard_chronology_scan(self, n_horizons: int = 3) -> Any:
        """Phase 2b chronology scan: V_1 bounded at every Cauchy horizon."""
        return hadamard_chronology_report(self.vs, n_horizons=n_horizons)

    def trace_anomaly_check(self, r: float) -> dict:
        """Cross-check: 2 V_1 / (8 pi^2) reproduces K / (2880 pi^2)."""
        via_V_1 = trace_anomaly_via_V_1(self.vs, r)
        via_ps = trace_anomaly_4d_exact(self.vs, r)
        rel_diff = abs(via_V_1 - via_ps) / max(abs(via_ps), 1e-30)
        return {
            "trace_anomaly_via_V_1": float(via_V_1),
            "trace_anomaly_via_point_splitting": float(via_ps),
            "relative_difference": float(rel_diff),
        }

    def kretschmann(self, r: float) -> float:
        return float(kretschmann_scalar(self.vs, r))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> LPSummary:
        """One-shot characterisation of the spacetime.

        Reasonably fast (< 1 s for supercritical regimes); produces a
        flat dataclass that captures the load-bearing facts: regime,
        Cauchy horizons, energy conditions, Phase 2a CP verdict, and
        Phase 2b Hadamard V_1 sanity.
        """
        ecr = self.energy_conditions()
        all_ec_ok = bool(
            ecr.nec_holds and ecr.wec_holds
            and ecr.sec_holds and ecr.dec_holds
        )
        horizons = self.cauchy_horizons() if self.vs.is_supercritical() else np.array([])
        first_r = float(horizons[0]) if len(horizons) > 0 else None
        first_kappa = self.surface_gravity() if first_r is not None else None
        first_T_H = self.hawking_temperature() if first_r is not None else None

        cp_power = None
        cp_verdict = None
        V_1_mid = None
        K_mid = None
        if self.vs.is_supercritical() and len(horizons) >= 1:
            try:
                rep = self.chronology_protection_scan(n_horizons=min(2, len(horizons)),
                                                          n_samples=14, eps_min=1e-3)
                cp_power = float(rep.boulware_fits[0].power)
                cp_verdict = rep.verdict
            except Exception:
                pass
            # Midpoint between source boundary and first horizon
            r_mid = 0.5 * (self.R + first_r)
            try:
                V_1_mid = self.hadamard_V_1(r_mid)
                K_mid = self.kretschmann(r_mid)
            except Exception:
                pass

        return LPSummary(
            omega=self.omega,
            R=self.R,
            a=self.a,
            alpha=self.alpha,
            regime=str(self.vs.regime),
            interior_regime=str(self.vs.interior_regime),
            is_supercritical=self.is_supercritical,
            has_interior_ctc=bool(self.vs.has_interior_ctc()),
            mass_per_unit_length=float(self.vs.mass_per_unit_length),
            angular_momentum_per_unit_length=float(self.vs.angular_momentum_per_unit_length),
            energy_conditions_all_satisfied=all_ec_ok,
            n_cauchy_horizons_in_10R=int(len(horizons)),
            first_horizon_r=first_r,
            first_horizon_surface_gravity=first_kappa,
            first_horizon_hawking_temperature=first_T_H,
            boulware_T_tt_simple_pole_power=cp_power,
            chronology_protection_verdict=cp_verdict,
            hadamard_V_1_at_midpoint=V_1_mid,
            kretschmann_at_midpoint=K_mid,
        )
