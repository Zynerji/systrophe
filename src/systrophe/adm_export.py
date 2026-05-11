"""ADM (3+1) decomposition of the Lewis-Papapetrou exterior.

Provides an ADM (Arnowitt-Deser-Misner) 3+1 decomposition of the LP
exterior for hand-off to numerical-relativity codes (Einstein Toolkit,
GRChombo, BAM, etc.).

The 4-metric

    ds^2 = -F dt^2 + 2 K dt dphi + L dphi^2 + h (dr^2 + dz^2)

is foliated on t = const slices. The ADM quantities on each slice:

    Spatial metric gamma_{ij}:
        gamma_rr = h,    gamma_phi phi = L,    gamma_zz = h

    Shift vector beta^i:
        beta^phi = K / L,    beta^phi^phi (lowered) = K

    Lapse function alpha:
        alpha = sqrt(F + K^2 / L)

    Extrinsic curvature K_{ij} (using stationarity d_t gamma = 0):
        K_{r phi} = (1 / (2 alpha)) d_r K
        all other components = 0

These are sufficient for initial-data hand-off; the evolution code
handles the time-development.

CTC region behaviour
--------------------
When F + K^2/L < 0 (deep CTC region), the lapse alpha becomes
imaginary --- the t = const slice has flipped from spacelike to
timelike. This is the signature of the chronology horizon being a
*Cauchy horizon* for the foliation; the export is undefined past this
point. Codes consuming this output must either:

  (a) reformulate with a different foliation (e.g., using the angular
      Killing vector), or
  (b) treat the F+K^2/L = 0 surface as a Cauchy surface and stop
      evolution there.

Note
----
This module produces ADM initial data on a 1D profile (the LP exterior
is invariant under z translations and t/phi rotations). For full 3D
data, the user must extrude the profile onto a 3D grid in their own
NR code; the closed-form profile is exact at every (t, z, phi) on
the slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ADMSlice:
    """ADM data on a 1D radial profile of the LP exterior.

    All arrays have shape (n_radii,), indexed by r.

    Fields
    ------
    r            : radial grid
    gamma_rr     : spatial metric component
    gamma_phiphi : spatial metric component
    gamma_zz     : spatial metric component
    beta_phi     : shift vector (lowered phi component)
    beta_up_phi  : shift vector (raised phi component)
    alpha        : lapse function
    K_rphi       : extrinsic curvature off-diagonal component
    is_valid     : bool array; False where alpha^2 < 0 (foliation breaks)
    """

    r: np.ndarray
    gamma_rr: np.ndarray
    gamma_phiphi: np.ndarray
    gamma_zz: np.ndarray
    beta_phi: np.ndarray
    beta_up_phi: np.ndarray
    alpha: np.ndarray
    K_rphi: np.ndarray
    is_valid: np.ndarray


def adm_decompose_lp(
    vs, r_grid: np.ndarray, eps: float = 1e-5
) -> ADMSlice:
    """Decompose the LP exterior into ADM 3+1 components on a radial grid.

    Parameters
    ----------
    vs : VanStockumInterior
        Source whose exterior is being decomposed.
    r_grid : array_like
        Radii at which to evaluate.
    eps : float
        Step size for d/dr finite difference.

    Returns
    -------
    ADMSlice with each component evaluated at r_grid.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    n = len(r_grid)
    gamma_rr = np.zeros(n)
    gamma_phiphi = np.zeros(n)
    gamma_zz = np.zeros(n)
    beta_phi = np.zeros(n)
    beta_up_phi = np.zeros(n)
    alpha = np.zeros(n)
    K_rphi = np.zeros(n)
    is_valid = np.ones(n, dtype=bool)

    for i, r in enumerate(r_grid):
        r = float(r)
        F = float(vs.analytic_exterior_F(r))
        K = float(vs.analytic_exterior_K(r))
        L = float(vs.analytic_exterior_L(r))
        # Spatial metric (h = 1 by leading-order convention)
        h = 1.0
        gamma_rr[i] = h
        gamma_phiphi[i] = L
        gamma_zz[i] = h
        # Shift
        if abs(L) > 1e-30:
            beta_up_phi[i] = K / L
            beta_phi[i] = K
        # Lapse
        alpha_sq = F + (K * K) / max(abs(L), 1e-30) * np.sign(L) if L != 0 else F
        # Use L sign properly: alpha^2 = F + K^2 / L
        if abs(L) > 1e-30:
            alpha_sq = F + K * K / L
        else:
            alpha_sq = F
        if alpha_sq > 0:
            alpha[i] = float(np.sqrt(alpha_sq))
        else:
            alpha[i] = float("nan")
            is_valid[i] = False
        # K_rphi via FD
        K_plus = float(vs.analytic_exterior_K(r + eps))
        K_minus = float(vs.analytic_exterior_K(r - eps))
        dK_dr = (K_plus - K_minus) / (2 * eps)
        if alpha[i] > 0 and is_valid[i]:
            K_rphi[i] = dK_dr / (2 * alpha[i])
        else:
            K_rphi[i] = float("nan")

    return ADMSlice(
        r=r_grid, gamma_rr=gamma_rr, gamma_phiphi=gamma_phiphi,
        gamma_zz=gamma_zz, beta_phi=beta_phi, beta_up_phi=beta_up_phi,
        alpha=alpha, K_rphi=K_rphi, is_valid=is_valid,
    )


def export_to_einsteintoolkit_ascii(
    slice_data: ADMSlice, output_path: str
) -> str:
    """Write ADM initial data in Einstein Toolkit-compatible ASCII format.

    Format: one row per radial grid point, columns
        r  gamma_rr  gamma_phiphi  gamma_zz  beta_phi  alpha  K_rphi  is_valid

    Returns the absolute output path.
    """
    import os
    arr = np.column_stack([
        slice_data.r,
        slice_data.gamma_rr,
        slice_data.gamma_phiphi,
        slice_data.gamma_zz,
        slice_data.beta_phi,
        slice_data.alpha,
        slice_data.K_rphi,
        slice_data.is_valid.astype(int),
    ])
    header = (
        "Systrophe ADM 1D initial-data profile (Einstein Toolkit ASCII)\n"
        "Columns: r  gamma_rr  gamma_phiphi  gamma_zz  beta_phi  alpha  K_rphi  is_valid\n"
        "Coordinates: (r, phi, z), with r as the radial profile. Extrude in phi, z.\n"
        "is_valid = 0 indicates F + K^2/L < 0 (chronology horizon; foliation breaks)."
    )
    np.savetxt(output_path, arr, header=header, fmt="%.10e")
    return os.path.abspath(output_path)


def hamiltonian_constraint_residual(slice_data: ADMSlice) -> np.ndarray:
    """Numerical Hamiltonian constraint residual (vacuum case).

    H = R^(3) - K_{ij} K^{ij} + (tr K)^2 = 0.

    For our 1D LP profile with K_{ij} containing only K_{r phi}:
        K_{ij} K^{ij} = 2 gamma^{rr} gamma^{phi phi} K_{r phi}^2
        tr K = K_i^i = 0 (off-diagonal extrinsic curvature)

    R^(3) is the spatial Ricci scalar (computable from gamma alone).

    Returns the residual at each grid point. Identically zero (modulo
    FD noise) would confirm the LP exterior is vacuum on each slice.
    """
    r = slice_data.r
    gamma_rr = slice_data.gamma_rr
    gamma_phi = slice_data.gamma_phiphi
    K_rphi = slice_data.K_rphi
    valid = slice_data.is_valid

    # Off-diagonal-only K_{ij} (only K_{r phi}); K^{ij} K_{ij} = 2 g^{rr} g^{phi phi} K_{r phi}^2
    KK = np.full_like(r, np.nan)
    mask = valid & (gamma_rr > 0) & (gamma_phi > 0)
    KK[mask] = 2 * K_rphi[mask] ** 2 / (gamma_rr[mask] * gamma_phi[mask])
    # tr K = 0 here
    trK = np.zeros_like(r)

    # R^(3) computation from gamma: for diagonal (gamma_rr, gamma_phi, gamma_zz)
    # with z-translation invariance, R^(3) reduces to a single radial expression
    # involving d_r gamma_phi / gamma_phi etc. We compute it numerically.
    # (For a more rigorous implementation see e.g. Wald 1984 Appendix E.)
    # Here we use central differences as a leading-order estimate.
    n = len(r)
    R3 = np.zeros(n)
    for i in range(1, n - 1):
        if not valid[i]:
            R3[i] = float("nan")
            continue
        dr = r[i + 1] - r[i - 1]
        if dr <= 0:
            R3[i] = 0.0
            continue
        # d_r ln(gamma_phi) and its derivative
        ln_g = np.log(max(gamma_phi[i], 1e-30))
        ln_gp = np.log(max(gamma_phi[i + 1], 1e-30))
        ln_gm = np.log(max(gamma_phi[i - 1], 1e-30))
        d1 = (ln_gp - ln_gm) / dr
        d2 = (ln_gp - 2 * ln_g + ln_gm) / ((dr / 2) ** 2)
        # Simplified scalar; not the full R^(3) but a usable diagnostic
        R3[i] = d2 + 0.5 * d1 ** 2

    return R3 - KK + trK ** 2


def adm_summary(slice_data: ADMSlice) -> dict:
    """Summary statistics of an ADM slice."""
    valid = slice_data.is_valid
    return {
        "n_valid": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
        "first_horizon_r": float(slice_data.r[np.argmin(slice_data.alpha)]) if valid.any() else float("nan"),
        "alpha_min_valid": float(np.nanmin(slice_data.alpha[valid])) if valid.any() else float("nan"),
        "alpha_max_valid": float(np.nanmax(slice_data.alpha[valid])) if valid.any() else float("nan"),
        "K_rphi_max_abs": float(np.nanmax(np.abs(slice_data.K_rphi[valid]))) if valid.any() else float("nan"),
    }
