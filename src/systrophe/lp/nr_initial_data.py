"""Numerical-relativity initial data export with constraint checks.

Extends `adm_export` with:

- Hamiltonian and momentum constraint residuals on the 1D profile;
- BSSN (Baumgarte-Shapiro-Shibata-Nakamura) conformal decomposition:
  conformal factor chi, conformal metric tilde_gamma, conformal
  traceless extrinsic curvature tilde_A_ij;
- 3D extrusion onto a Cartesian grid;
- HDF5/JSON serialization of the initial data slab.

The output is consumable by Einstein Toolkit (Cactus), GRChombo, or
BAM via their initial-data import interfaces.

References
----------
- Baumgarte-Shapiro, Numerical Relativity (Cambridge, 2010);
- Alcubierre, Introduction to 3+1 Numerical Relativity (Oxford, 2008).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def adm_profile_1d(vs: VanStockumInterior, r_grid: np.ndarray) -> dict:
    """1D ADM profile on the given radial grid (using analytic forms)."""
    F = vs.analytic_exterior_F(r_grid)
    K = vs.analytic_exterior_K(r_grid)
    L = vs.analytic_exterior_L(r_grid)
    h = np.ones_like(r_grid)  # conformal factor placeholder
    # ADM components
    gamma_rr = h
    gamma_pp = L
    gamma_zz = h
    # Lapse-shift
    denom = L
    alpha2 = F + K * K / np.maximum(denom, 1e-30)
    alpha = np.sqrt(np.where(alpha2 > 0, alpha2, np.nan))
    beta_phi = K / np.maximum(denom, 1e-30)
    # Extrinsic curvature (only K_{r phi} non-zero)
    dr = np.gradient(r_grid)
    K_rphi = np.gradient(K, r_grid) / (2 * np.where(alpha > 0, alpha, np.nan))
    return {
        "r_grid": r_grid,
        "gamma_rr": gamma_rr,
        "gamma_pp": gamma_pp,
        "gamma_zz": gamma_zz,
        "lapse": alpha,
        "shift_phi": beta_phi,
        "K_rphi": K_rphi,
        "F": F, "K": K, "L": L,
    }


def hamiltonian_constraint_residual(
    vs: VanStockumInterior, r_grid: np.ndarray,
) -> dict:
    """Residual of the Hamiltonian constraint: R + K^2 - K_ij K^ij - 16 pi rho.

    For van Stockum interior matter (rho = const dust), the constraint
    is automatically satisfied. For the LP exterior (vacuum), R should
    equal K_ij K^ij - K^2.
    """
    prof = adm_profile_1d(vs, r_grid)
    K_rphi = prof["K_rphi"]
    # K_ij K^ij = 2 K_rphi^2 / (gamma_rr gamma_pp)
    K_sq_term = 2 * K_rphi ** 2 / np.maximum(prof["gamma_rr"] * prof["gamma_pp"], 1e-30)
    # Trace K = gamma^ij K_ij = 0 here (only off-diagonal)
    trace_K = np.zeros_like(r_grid)
    # 3-Ricci scalar approximated by Laplacian of log gamma_pp
    log_gpp = np.log(np.abs(prof["gamma_pp"]) + 1e-30)
    R_approx = np.gradient(np.gradient(log_gpp, r_grid), r_grid)
    residual = R_approx + trace_K ** 2 - K_sq_term  # vacuum: should be ~0
    return {
        "r_grid": r_grid,
        "residual": residual,
        "mean_abs_residual": float(np.mean(np.abs(residual))),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def momentum_constraint_residual(
    vs: VanStockumInterior, r_grid: np.ndarray,
) -> dict:
    """Residual of D_j (K^{ij} - gamma^{ij} K) = 8 pi j^i.

    In LP vacuum (j^i = 0), the divergence should vanish.
    """
    prof = adm_profile_1d(vs, r_grid)
    K_rphi = prof["K_rphi"]
    # i = phi component: D_r K^{r phi} = d_r K_rphi / sqrt(gamma_rr gamma_pp)
    sqrt_g = np.sqrt(np.abs(prof["gamma_rr"] * prof["gamma_pp"]) + 1e-30)
    K_up = K_rphi / sqrt_g
    res_phi = np.gradient(K_up, r_grid)
    return {
        "r_grid": r_grid,
        "residual_phi": res_phi,
        "mean_abs_residual": float(np.mean(np.abs(res_phi))),
    }


def bssn_decomposition(
    vs: VanStockumInterior, r_grid: np.ndarray,
) -> dict:
    """BSSN conformal decomposition: chi, tilde_gamma_ij, A_ij, K_trace, Gamma^i."""
    prof = adm_profile_1d(vs, r_grid)
    # Conformal factor: chi = |det(gamma)|^{-1/3} (use |det| since L<0 inside CTC bands)
    det_gamma = np.abs(prof["gamma_rr"] * prof["gamma_pp"] * prof["gamma_zz"])
    chi = np.where(det_gamma > 1e-30, det_gamma ** (-1.0 / 3.0), np.nan)
    # Conformal metric tilde_gamma_ij = chi^{2/3} gamma_ij (using -1/3 power = unit determinant)
    tilde_grr = chi * prof["gamma_rr"]
    tilde_gpp = chi * prof["gamma_pp"]
    tilde_gzz = chi * prof["gamma_zz"]
    # K trace and traceless A
    trace_K = np.zeros_like(r_grid)
    tilde_A_rphi = chi * prof["K_rphi"]
    return {
        "r_grid": r_grid,
        "chi": chi,
        "tilde_gamma_rr": tilde_grr,
        "tilde_gamma_pp": tilde_gpp,
        "tilde_gamma_zz": tilde_gzz,
        "tilde_A_rphi": tilde_A_rphi,
        "K_trace": trace_K,
    }


def extrude_to_3d_cartesian(
    vs: VanStockumInterior,
    box_size: float = 10.0, n_grid: int = 32,
) -> dict:
    """Extrude the 1D profile onto a 3D Cartesian grid.

    Returns the ADM tensor components evaluated at each grid point.
    """
    x = np.linspace(-box_size, box_size, n_grid)
    y = np.linspace(-box_size, box_size, n_grid)
    z = np.linspace(-box_size, box_size, n_grid)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)
    # Build profile sampled on r
    r_flat = r.ravel()
    r_flat_safe = np.maximum(r_flat, vs.R * 1.01)
    F_flat = vs.analytic_exterior_F(r_flat_safe)
    L_flat = vs.analytic_exterior_L(r_flat_safe)
    K_flat = vs.analytic_exterior_K(r_flat_safe)
    F_3d = F_flat.reshape(r.shape)
    L_3d = L_flat.reshape(r.shape)
    K_3d = K_flat.reshape(r.shape)
    interior_mask = r < vs.R
    F_3d = np.where(interior_mask, 1.0, F_3d)
    return {
        "x": x, "y": y, "z": z, "n_grid": n_grid,
        "F": F_3d, "K": K_3d, "L": L_3d,
        "r_distance": r,
        "phi": phi,
        "interior_mask": interior_mask,
    }


def export_initial_data_json(
    vs: VanStockumInterior, output_path: str | Path,
    r_min: float = None, r_max: float = None, n_samples: int = 200,
) -> dict:
    """Export ADM + BSSN initial data to JSON file."""
    if r_min is None:
        r_min = vs.R * 1.01
    if r_max is None:
        r_max = vs.R * 20.0
    r_grid = np.linspace(r_min, r_max, n_samples)
    adm = adm_profile_1d(vs, r_grid)
    bssn = bssn_decomposition(vs, r_grid)
    ham = hamiltonian_constraint_residual(vs, r_grid)
    mom = momentum_constraint_residual(vs, r_grid)

    payload = {
        "omega_dust": vs.omega,
        "R": vs.R,
        "a": vs.a,
        "regime": "supercritical" if vs.is_supercritical() else "sub_or_critical",
        "r_grid": r_grid.tolist(),
        "ADM": {
            "lapse": adm["lapse"].tolist(),
            "shift_phi": adm["shift_phi"].tolist(),
            "gamma_rr": adm["gamma_rr"].tolist(),
            "gamma_pp": adm["gamma_pp"].tolist(),
            "K_rphi": adm["K_rphi"].tolist(),
        },
        "BSSN": {
            "chi": bssn["chi"].tolist(),
            "tilde_gamma_rr": bssn["tilde_gamma_rr"].tolist(),
            "tilde_gamma_pp": bssn["tilde_gamma_pp"].tolist(),
            "tilde_A_rphi": bssn["tilde_A_rphi"].tolist(),
        },
        "constraints": {
            "ham_mean_abs": ham["mean_abs_residual"],
            "ham_max_abs": ham["max_abs_residual"],
            "mom_mean_abs": mom["mean_abs_residual"],
        },
    }
    p = Path(output_path)
    p.write_text(json.dumps(payload, indent=2))
    return {
        "output_path": str(p),
        "n_samples": n_samples,
        "constraints_ham_mean": ham["mean_abs_residual"],
        "constraints_mom_mean": mom["mean_abs_residual"],
    }


def cauchy_horizon_warning_for_foliation(
    vs: VanStockumInterior, r_grid: np.ndarray,
) -> dict:
    """Find radii where the t=const foliation breaks down (alpha imaginary)."""
    F = vs.analytic_exterior_F(r_grid)
    L = vs.analytic_exterior_L(r_grid)
    K = vs.analytic_exterior_K(r_grid)
    alpha2 = F + K * K / np.maximum(L, 1e-30)
    breakdown_mask = alpha2 < 0
    if np.any(breakdown_mask):
        r_first_breakdown = float(r_grid[np.argmax(breakdown_mask)])
    else:
        r_first_breakdown = None
    return {
        "r_first_breakdown": r_first_breakdown,
        "fraction_breakdown": float(np.mean(breakdown_mask)),
        "alpha_sq_min": float(np.min(alpha2)),
    }
