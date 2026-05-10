"""QFTCS back-reaction: renormalised stress-energy on the Tipler exterior.

This module implements the simplest non-trivial QFTCS calculation
applicable to the Tipler / Systrophe spacetime: the conformal anomaly
of a massless conformally-coupled scalar field in a 2D effective
description of the radial-temporal slice.

In two dimensions, a free conformally-coupled scalar has trace anomaly

    <T^mu_mu>_renormalised = R / (24 pi),

where R is the 2D Ricci scalar. The full 4D calculation requires
Schwinger-DeWitt or Hadamard point-splitting renormalisation and is
substantially more complex; the 2D version is the canonical pedagogical
diagnostic for chronology protection (Davies 1976; Christensen 1976).

We compute R(r) for the radial-temporal slice of the Tipler metric
and report:

- `radial_temporal_ricci_scalar(vs, r)`: 2D Ricci scalar of the
  (t, r) slice with the induced 2-metric.
- `conformal_anomaly_trace(vs, r)`: <T^mu_mu> = R / (24 pi).
- `vacuum_energy_density_proxy(vs, r)`: T_tt component proxy via
  the conformal anomaly + smooth-vacuum approximation.
- `stress_energy_at_horizon(vs, r_horizon)`: divergence rate of the
  stress-energy at the Cauchy horizon.

The 2D anomaly is itself diverging when R diverges. For the Tipler
exterior, R(r) typically does NOT diverge at F = 0 (the F-zero is a
coordinate effect of the radial-temporal slice, not a curvature
singularity); however the 4D-effective vacuum energy IS expected to
diverge there. We expose both diagnostics.

References
----------
- P. C. W. Davies, S. A. Fulling, W. G. Unruh, *Energy-momentum tensor
  near an evaporating black hole*, Phys. Rev. D 13 (1976) 2720.
- S. M. Christensen, *Vacuum expectation value of the stress tensor in
  an arbitrary curved background: the covariant point-separation
  method*, Phys. Rev. D 14 (1976) 2490.
- N. Birrell, P. C. W. Davies, *Quantum Fields in Curved Space*, CUP,
  1982.
"""

from __future__ import annotations

import numpy as np


def radial_temporal_ricci_scalar(vs, r: float | np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """2D Ricci scalar of the (t, r) slice of the Tipler exterior.

    The 2-metric ds^2 = -F(r) dt^2 + h(r) dr^2 has Ricci scalar
        R_2d = -F''(r) / (F(r) * h(r)) + (F'(r))^2 / (2 F(r)^2 * h(r))
              + F'(r) h'(r) / (2 F(r) * h(r)^2).
    For the Tipler analytic exterior we set h = 1 (leading order),
    reducing to
        R_2d = -F''/F + (F')^2/(2 F^2).
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    r_arr = np.asarray(r, dtype=float)

    def F(rr): return vs.analytic_exterior_F(rr)

    Fp = (F(r_arr + eps) - F(r_arr - eps)) / (2 * eps)
    Fpp = (F(r_arr + eps) - 2 * F(r_arr) + F(r_arr - eps)) / (eps * eps)
    F_safe = np.where(np.abs(F(r_arr)) > 1e-9, F(r_arr), np.sign(F(r_arr) + 1e-300) * 1e-9)
    R = -Fpp / F_safe + (Fp * Fp) / (2.0 * F_safe * F_safe)
    return R


def conformal_anomaly_trace(vs, r: float | np.ndarray) -> np.ndarray:
    """<T^mu_mu>_ren = R_2d / (24 pi) for a conformally-coupled massless scalar.

    The 2D conformal-anomaly result is exact for a massless conformally
    coupled scalar; for the 4D physical case it is a leading-order
    approximation valid in the locally-2D regime near a Cauchy horizon.
    """
    R = radial_temporal_ricci_scalar(vs, r)
    return R / (24.0 * np.pi)


def vacuum_energy_density_proxy(vs, r: float | np.ndarray) -> np.ndarray:
    """T_tt proxy via the 2D Polyakov action.

    For a 2D conformally-coupled scalar in the Boulware vacuum,
    <T_tt>_ren ~ -R_2d / (96 pi). We use this as the leading-order
    proxy for the Tipler radial-temporal slice. Negative values
    indicate vacuum-energy depletion (Casimir-like); divergent values
    near the chronology horizon signal back-reaction breakdown.
    """
    R = radial_temporal_ricci_scalar(vs, r)
    return -R / (96.0 * np.pi)


def stress_energy_at_horizon(vs, r_horizon: float, n_samples: int = 50, eps_min: float = 1e-3) -> dict:
    """Diagnostic: rate of divergence of the stress-energy near a Cauchy horizon.

    Samples the trace anomaly at radii (r_horizon + eps_i) for
    eps_i decreasing geometrically and fits the resulting profile
    to A * eps^p, returning the inferred power p (negative => divergent).
    """
    eps_array = eps_min * (1.5 ** np.arange(n_samples))
    eps_array = eps_array[eps_array < 0.5 * abs(r_horizon)]
    if eps_array.size < 5:
        eps_array = np.linspace(eps_min, eps_min * 10, 10)
    r_samples = r_horizon + eps_array
    trace = np.array([float(conformal_anomaly_trace(vs, r)) for r in r_samples])
    # Fit log|trace| = log|A| + p log(eps)
    log_eps = np.log(eps_array[: trace.size])
    valid = np.abs(trace) > 1e-30
    if valid.sum() < 3:
        return {"power": float("nan"), "samples": list(zip(r_samples.tolist(), trace.tolist()))}
    log_T = np.log(np.abs(trace[valid]))
    p, log_A = np.polyfit(log_eps[valid], log_T, 1)
    return {
        "power": float(p),
        "amplitude_log": float(log_A),
        "n_samples_fit": int(valid.sum()),
        "r_horizon": float(r_horizon),
    }
