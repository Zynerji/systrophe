"""Self-consistent semiclassical Newton-Kantorovich back-reaction for the
Toroidal Knopp Drive.

`knopp_toroidal.py` ships a *dimensional* back-reaction proxy E_BR ~
lambda / M^4 with a free prefactor lambda (default 1e-4). That's a
placeholder, not a derivation. This module replaces it with an actual
Newton-Kantorovich iteration on the band edges under a Polyakov-style
vacuum-stress source, and derives the Q-cavity threshold Q_thr from
first principles (energy-balance between back-reaction flux and cavity
absorption).

Construction
------------

Step 1. Effective 2D radial metric on the toroidal slice (z = 0):

    ds_2D^2  =  -F(rho) dt^2  +  drho^2,
    F(rho)   =  1 - 2 Phi_eff(rho)  =  1 + 4 M / r(rho),
    r(rho)   =  sqrt((d/2)^2 + rho^2).

Step 2. Polyakov-Boulware vacuum stress tensor (the same exact 2D
result used in `stress_energy_ctc.boulware_stress_tensor`):

    <T_tt>_B  =  -(1 / 24 pi) * [ F''(rho) - 0.5 F'(rho)^2 / F(rho) ].

Step 3. Back-reaction shifts the effective tilt:

    T_eff^BR(rho)  =  T_eff^classical(rho)  -  lambda * <T_tt>_B(rho)

(sign convention: positive Polyakov vacuum energy *opposes* the CTC
band, pulling its edges inward, consistent with Hawking chronology
protection).

Step 4. Band-edge equation:
    residual(rho)  =  T_eff^BR(rho)  -  1  =  0.

Solve by `newton_kantorovich_1d` separately for the inner and outer
edges. Reports the shifted band, convergence diagnostics, and the
energy-balance Q-threshold.

Step 5. Q-threshold from first principles:

    Q_thr  =  sqrt( |E_Krasnikov| * omega_0 * A_band / |<T_kk>_B| ),

obtained by setting the back-reaction power flux equal to the cavity
absorption rate. Above Q_thr the cavity dominates and the band
survives; below, the band collapses under semiclassical pressure.

References
----------
- Hiscock-Konkowski (1982) PRD 26, 1225 -- 2D semiclassical back-
  reaction on Schwarzschild as the prototype.
- Davies-Fulling-Unruh (1976) PRD 13, 2720 -- Polyakov stress
  construction.
- Aguilera Katayama (April 2026) -- the toroidal binary framework
  this module quantises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.lp.newton_kantorovich import NKResult, newton_kantorovich_1d


# ----- effective 2D metric on the toroidal slice --------------------------


def F_effective(binary: EffectiveToroidalKerrBinary, rho: float) -> float:
    """F(rho) = 1 - 2 Phi_eff(rho) = 1 + 4 M / sqrt((d/2)^2 + rho^2)."""
    return float(1.0 - 2.0 * binary.phi_newtonian(rho))


def _F_derivatives(
    binary: EffectiveToroidalKerrBinary, rho: float, eps: float = 1e-5,
) -> tuple[float, float, float]:
    """(F, F', F'') via central differences."""
    F_c = F_effective(binary, rho)
    F_p = F_effective(binary, rho + eps)
    F_m = F_effective(binary, rho - eps)
    F_first = (F_p - F_m) / (2.0 * eps)
    F_second = (F_p - 2.0 * F_c + F_m) / (eps * eps)
    return F_c, F_first, F_second


# ----- Polyakov-Boulware vacuum stress -----------------------------------


def t_kk_polyakov_boulware(
    binary: EffectiveToroidalKerrBinary, rho: float, eps: float = 1e-5,
) -> float:
    """<T_tt>_Boulware on the effective 2D toroidal-slice metric.

    Returns NaN when F <= 0 (interior of an effective ergoregion).
    """
    F, Fp, Fpp = _F_derivatives(binary, rho, eps=eps)
    if F <= 0:
        return float("nan")
    return float(-(1.0 / (24.0 * math.pi)) * (Fpp - 0.5 * Fp * Fp / F))


# ----- back-reacted band edges via NK iteration ---------------------------


@dataclass(frozen=True)
class BackreactedBand:
    """Back-reacted toroidal CTC band report."""
    lam: float
    rho_in_classical: Optional[float]
    rho_out_classical: Optional[float]
    rho_in_BR: Optional[float]
    rho_out_BR: Optional[float]
    shift_in: Optional[float]
    shift_out: Optional[float]
    band_width_classical: Optional[float]
    band_width_BR: Optional[float]
    inner_converged: bool
    outer_converged: bool
    inner_iterations: int
    outer_iterations: int
    t_kk_at_inner: float
    t_kk_at_outer: float
    band_closed: bool


def backreacted_band(
    binary: EffectiveToroidalKerrBinary,
    lam: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> BackreactedBand:
    """Solve T_eff^classical(rho) - lambda * <T_kk>_B(rho) = 1 for the
    inner and outer back-reacted band edges.

    `lam` is the dimensionless back-reaction strength. lam = 0
    recovers the classical band edges; lam > 0 pulls the band inward
    (chronology-protection direction). Sufficiently large lam closes
    the band entirely.
    """
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    edges_classical = binary.ctc_band_edges(include_phi=False)
    rho_in_c, rho_out_c = edges_classical
    if rho_in_c is None:
        return BackreactedBand(
            lam=lam,
            rho_in_classical=None, rho_out_classical=None,
            rho_in_BR=None, rho_out_BR=None,
            shift_in=None, shift_out=None,
            band_width_classical=None, band_width_BR=None,
            inner_converged=False, outer_converged=False,
            inner_iterations=0, outer_iterations=0,
            t_kk_at_inner=0.0, t_kk_at_outer=0.0,
            band_closed=True,
        )

    def residual(rho: float) -> float:
        # Guard against NK pushing into the non-physical rho <= 0 region.
        if rho <= 0:
            # Return a large finite value (NK will step back).
            return 1e6
        T = binary.t_eff(rho, include_phi=False)
        Tkk = t_kk_polyakov_boulware(binary, rho)
        if not math.isfinite(Tkk):
            # Push iteration outward when in the strong-field core.
            return 1e6
        return T - lam * Tkk - 1.0

    nk_in = newton_kantorovich_1d(
        residual, x0=rho_in_c, tol=tol, max_iter=max_iter,
    )
    nk_out = newton_kantorovich_1d(
        residual, x0=rho_out_c, tol=tol, max_iter=max_iter,
    )

    rho_in_BR = float(nk_in.x[0]) if nk_in.converged else None
    rho_out_BR = float(nk_out.x[0]) if nk_out.converged else None

    # If the back-reaction is too strong to support a band, the
    # iterations may diverge or collide -- detect that.
    band_closed = (
        not nk_in.converged or not nk_out.converged
        or (rho_in_BR is not None and rho_out_BR is not None
            and rho_in_BR >= rho_out_BR - 1e-9)
    )

    # Diagnostic: <T_kk> values at the classical band edges
    Tkk_in = t_kk_polyakov_boulware(binary, rho_in_c)
    Tkk_out = t_kk_polyakov_boulware(binary, rho_out_c)

    width_c = rho_out_c - rho_in_c
    width_BR = None
    shift_in = None
    shift_out = None
    if rho_in_BR is not None and rho_out_BR is not None and not band_closed:
        width_BR = rho_out_BR - rho_in_BR
        shift_in = rho_in_BR - rho_in_c
        shift_out = rho_out_BR - rho_out_c

    return BackreactedBand(
        lam=lam,
        rho_in_classical=rho_in_c,
        rho_out_classical=rho_out_c,
        rho_in_BR=rho_in_BR,
        rho_out_BR=rho_out_BR,
        shift_in=shift_in,
        shift_out=shift_out,
        band_width_classical=width_c,
        band_width_BR=width_BR,
        inner_converged=nk_in.converged,
        outer_converged=nk_out.converged,
        inner_iterations=nk_in.iterations,
        outer_iterations=nk_out.iterations,
        t_kk_at_inner=float(Tkk_in),
        t_kk_at_outer=float(Tkk_out),
        band_closed=bool(band_closed),
    )


# ----- critical lambda: where the band closes -----------------------------


def critical_lambda(
    binary: EffectiveToroidalKerrBinary,
    lam_min: float = 0.0,
    lam_max: float = 1e6,
    tol: float = 1e-3,
    max_bisect: int = 60,
) -> float:
    """Find lambda_crit at which the back-reacted band first closes.

    Uses bisection on `lam`. Returns +inf if the band never closes for
    lam in [lam_min, lam_max].
    """
    if not binary.has_toroidal_ctc_band(include_phi=False):
        return 0.0  # no band to begin with
    if not backreacted_band(binary, lam=lam_max).band_closed:
        return float("inf")
    lo, hi = lam_min, lam_max
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        if backreacted_band(binary, lam=mid).band_closed:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


# ----- Q-threshold from energy balance -----------------------------------


def q_threshold_from_balance(
    binary: EffectiveToroidalKerrBinary,
    E_krasnikov: float = 1.0,
    omega_0: float = 1.0,
) -> Optional[float]:
    """Q-cavity threshold from energy balance against the BR flux.

        Q_thr  =  sqrt( |E_Krasnikov| * omega_0 * A_band / |<T_kk>_B| ).

    Above Q_thr the cavity exactly absorbs the chronology-protection
    flux at the inner band edge; below, the band collapses.

    Returns None if there is no band (nothing to balance).
    """
    edges = binary.ctc_band_edges(include_phi=False)
    if edges[0] is None:
        return None
    rho_in, rho_out = edges
    A_band = math.pi * (rho_out ** 2 - rho_in ** 2)
    Tkk_in = abs(t_kk_polyakov_boulware(binary, rho_in))
    if Tkk_in < 1e-30:
        return float("inf")
    return float(math.sqrt(E_krasnikov * omega_0 * A_band / Tkk_in))


# ----- combined diagnostic ------------------------------------------------


@dataclass(frozen=True)
class BackreactionDiagnostic:
    binary: EffectiveToroidalKerrBinary
    band_at_lam: BackreactedBand
    lambda_critical: float
    Q_threshold: Optional[float]


def backreaction_diagnostic(
    binary: EffectiveToroidalKerrBinary,
    lam: float = 1.0,
    E_krasnikov: float = 1.0,
    omega_0: float = 1.0,
) -> BackreactionDiagnostic:
    """Full self-consistent BR report: shifted band edges, lambda_crit, Q_thr."""
    band = backreacted_band(binary, lam=lam)
    lam_crit = critical_lambda(binary)
    Q_thr = q_threshold_from_balance(
        binary, E_krasnikov=E_krasnikov, omega_0=omega_0,
    )
    return BackreactionDiagnostic(
        binary=binary,
        band_at_lam=band,
        lambda_critical=lam_crit,
        Q_threshold=Q_thr,
    )


def summarise_backreaction(d: BackreactionDiagnostic) -> str:
    """Human-readable summary."""
    band = d.band_at_lam
    if band.rho_in_classical is None:
        return "No classical band -- nothing to back-react."
    lines = [
        f"Toroidal Knopp NK back-reaction (lambda = {band.lam:.4g}):",
        f"  classical band edges  = [{band.rho_in_classical:.4f}, "
        f"{band.rho_out_classical:.4f}]",
        f"  classical band width  = {band.band_width_classical:.4f}",
        f"  <T_kk>_B at edges     = "
        f"({band.t_kk_at_inner:+.4e}, {band.t_kk_at_outer:+.4e})",
    ]
    if band.band_closed:
        lines.append("  ** BAND CLOSED by back-reaction at this lambda **")
    else:
        lines.append(
            f"  BR band edges         = "
            f"[{band.rho_in_BR:.4f}, {band.rho_out_BR:.4f}]"
        )
        lines.append(
            f"  BR shift (in, out)    = "
            f"({band.shift_in:+.4f}, {band.shift_out:+.4f})"
        )
        lines.append(
            f"  BR band width         = {band.band_width_BR:.4f}"
        )
        lines.append(
            f"  shrink fraction       = "
            f"{1.0 - band.band_width_BR / band.band_width_classical:.4f}"
        )
    lines.append(
        f"  inner NK iters / conv = {band.inner_iterations} / "
        f"{band.inner_converged}"
    )
    lines.append(
        f"  outer NK iters / conv = {band.outer_iterations} / "
        f"{band.outer_converged}"
    )
    lines.append(f"  lambda_critical       = {d.lambda_critical:.4g}")
    lines.append(f"  Q_threshold (energy)  = {d.Q_threshold}")
    return "\n".join(lines)
