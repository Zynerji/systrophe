"""Compare Guderley shock divergence to QFTCS Boulware Cauchy-horizon divergence.

This module makes no physical-equivalence claim. It just reports the
two power-law exponents side-by-side and the absolute residual. The
empirical observation is that:

* Guderley (γ=5/3, n=3): ρ-divergence power ≈ -0.905.
* Boulware <T_tt> at a Cauchy horizon of the supercritical Tipler
  exterior: power = -1.000 (universal across horizons; Phase 2a's
  headline result).

So the residual sits near 0.1 across the canonical gas, and the
physical content of the comparison is *that fact*, not a tunable
matching.
"""

from __future__ import annotations

from dataclasses import dataclass

from systrophe.ctc.stress_energy_ctc import divergence_rate_at_horizon
from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate

from .shock import compute_guderley_exponent, density_power_at_focus


@dataclass(frozen=True)
class ShockHorizonComparison:
    """Empirical comparison of the two divergence powers.

    Attributes
    ----------
    gamma, n : float, int
        Guderley parameters.
    beta : float
        Self-similarity exponent.
    guderley_density_power : float
        ρ ~ r^{guderley_density_power} as r → 0 at t_focus.
    horizon_r : float
        Cauchy horizon radius used for QFTCS power.
    qftcs_T_tt_power : float
        Power of <T_tt>_Boulware ~ |r - r_H|^{qftcs_T_tt_power}.
    absolute_residual : float
        |guderley_density_power - qftcs_T_tt_power|.
    interpretation : str
        Short one-liner.
    """
    gamma: float
    n: int
    beta: float
    guderley_density_power: float
    horizon_r: float
    qftcs_T_tt_power: float
    absolute_residual: float
    interpretation: str


def compare_to_cauchy_horizon(
    vs: VanStockumInterior,
    gamma: float = 5.0 / 3.0,
    n: int = 3,
    n_horizon: int = 0,
    n_samples: int = 14,
) -> ShockHorizonComparison:
    """Compute both powers and return the residual.

    Parameters
    ----------
    vs : VanStockumInterior
        Must be supercritical (a > 1/2). Phase 2a Boulware divergence
        is universal here: power → -1.000.
    gamma, n : adiabatic index and geometry index.
    n_horizon : which Cauchy horizon (0 = innermost in r_max=10R sweep).
    n_samples : QFTCS power fit samples per horizon.
    """
    if not vs.is_supercritical():
        raise ValueError("vs must be supercritical (a > 1/2)")
    horizons = cauchy_horizon_estimate(vs)
    if len(horizons) <= n_horizon:
        raise ValueError(
            f"no Cauchy horizon at index {n_horizon}; "
            f"have {len(horizons)} horizons"
        )
    r_h = float(horizons[n_horizon])

    gud_exp = compute_guderley_exponent(gamma=gamma, n=n)
    gud_power = density_power_at_focus(gamma=gamma, n=n, beta=gud_exp.beta)

    fit = divergence_rate_at_horizon(
        vs, r_horizon=r_h, state="boulware", component="T_tt",
        n_samples=n_samples,
    )
    qftcs_power = float(fit.power)
    residual = abs(gud_power - qftcs_power)

    if residual < 0.05:
        interp = "tight empirical match (no physical mechanism implied)"
    elif residual < 0.2:
        interp = "loose empirical proximity"
    else:
        interp = "divergence powers are different regimes"

    return ShockHorizonComparison(
        gamma=float(gamma), n=int(n), beta=float(gud_exp.beta),
        guderley_density_power=float(gud_power),
        horizon_r=r_h,
        qftcs_T_tt_power=qftcs_power,
        absolute_residual=float(residual),
        interpretation=interp,
    )
