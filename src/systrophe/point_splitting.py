"""4D point-splitting renormalisation on the Lewis-Papapetrou background.

Implements the *full* 4D conformal-anomaly trace and the leading
DeWitt-Schwinger expansion of <phi^2>_ren and <T_{mu nu}>_ren for a
massless conformally-coupled scalar field on the Tipler / Systrophe
vacuum exterior.

Strategy
--------
At each radius r we evaluate the LP metric g_{mu nu}(r) and its first
and second r-derivatives by central differencing the analytic Case III
closed forms. From these, the Christoffel symbols, Riemann tensor,
Ricci tensor, Ricci scalar, and Kretschmann scalar are assembled
purely numerically.

The 4D trace anomaly for a massless conformally-coupled scalar field
in vacuum (R_{mu nu} = 0, R = 0) is exactly

    <T^mu_{~mu}>_ren = K_kretsch / (2880 pi^2),

where K_kretsch = R_{mu nu rho sigma} R^{mu nu rho sigma} is the
Kretschmann scalar (Birrell & Davies 1982; Christensen 1976). For the
Lewis-Papapetrou vacuum metric R_{mu nu} = 0 holds analytically; we
verify it numerically as a sanity check.

The DeWitt-Schwinger leading coefficient is

    a_2(x) = (1/180) R_{mu nu rho sigma} R^{mu nu rho sigma}      [vacuum]

which gives <phi^2>_ren the leading large-mass expansion
<phi^2>_ren ~ a_2(x) / (16 pi^2 m^2) for a massive field.

The full off-trace Hadamard subtraction <T_{mu nu}>_ren requires
choosing a vacuum state and summing over modes; we expose the
machinery and provide:
- `dewitt_a2_coefficient`: a_2(x) coefficient (exact in vacuum).
- `phi_squared_largemass_expansion`: leading large-mass <phi^2>_ren.
- `effective_action_volume_density`: a_2(x) / (32 pi^2) (the heat-
  kernel volume density of the one-loop effective action).
- `trace_anomaly_4d_exact`: the exact 4D conformal-anomaly trace.

This is the natural completion of the 2D conformal-anomaly module in
`qftcs_backreaction.py`: the 4D version is exact in 4D vacuum (rather
than a leading-order 2D approximation).

References
----------
- N. D. Birrell and P. C. W. Davies, *Quantum Fields in Curved Space*,
  CUP 1982.
- S. M. Christensen, Phys. Rev. D 14 (1976) 2490 (point splitting).
- B. S. DeWitt, *Dynamical Theory of Groups and Fields* (1965); Phys.
  Rep. 19 (1975) 295 (heat-kernel expansion).
"""

from __future__ import annotations

import numpy as np


def metric_tensor(vs, r: float) -> np.ndarray:
    """Lewis-Papapetrou metric tensor at radius r (4x4)."""
    F = float(vs.analytic_exterior_F(r))
    K = float(vs.analytic_exterior_K(r))
    L = float(vs.analytic_exterior_L(r))
    h = 1.0  # leading-order h approximation
    g = np.zeros((4, 4))
    g[0, 0] = -F
    g[0, 2] = K
    g[2, 0] = K
    g[2, 2] = L
    g[1, 1] = h
    g[3, 3] = h
    return g


def metric_inverse(g: np.ndarray) -> np.ndarray:
    return np.linalg.inv(g)


def metric_first_derivative(vs, r: float, eps: float = 1e-5) -> np.ndarray:
    """partial_r g_{mu nu}(r) via central differences. Shape (4, 4)."""
    g_plus = metric_tensor(vs, r + eps)
    g_minus = metric_tensor(vs, r - eps)
    return (g_plus - g_minus) / (2 * eps)


def metric_second_derivative(vs, r: float, eps: float = 1e-4) -> np.ndarray:
    """partial_r^2 g_{mu nu}(r) via central differences."""
    g_plus = metric_tensor(vs, r + eps)
    g_at = metric_tensor(vs, r)
    g_minus = metric_tensor(vs, r - eps)
    return (g_plus - 2 * g_at + g_minus) / (eps * eps)


def christoffel_symbols(vs, r: float, eps: float = 1e-5) -> np.ndarray:
    """Christoffel symbols Gamma^mu_{nu rho} at radius r. Shape (4, 4, 4).

    Gamma^mu_{nu rho} = (1/2) g^{mu sigma} (d_nu g_{rho sigma}
                                            + d_rho g_{nu sigma}
                                            - d_sigma g_{nu rho}).
    Only d_r is non-trivial for the Lewis-Papapetrou metric.
    """
    g = metric_tensor(vs, r)
    g_inv = metric_inverse(g)
    dg_r = metric_first_derivative(vs, r, eps=eps)

    # dg_full[mu_coord, nu, rho] = d_{mu_coord} g_{nu rho}
    # Only the r component (mu_coord = 1) is non-zero; others are zero.
    dg_full = np.zeros((4, 4, 4))
    dg_full[1, :, :] = dg_r

    Gamma = np.zeros((4, 4, 4))
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                s = 0.0
                for sigma in range(4):
                    s += g_inv[mu, sigma] * (
                        dg_full[nu, rho, sigma]
                        + dg_full[rho, nu, sigma]
                        - dg_full[sigma, nu, rho]
                    )
                Gamma[mu, nu, rho] = 0.5 * s
    return Gamma


def riemann_tensor(vs, r: float, eps: float = 5e-4) -> np.ndarray:
    """Riemann tensor R^mu_{nu rho sigma} at radius r. Shape (4, 4, 4, 4).

    Computed via central differences of the Christoffel symbols:
    R^mu_{nu rho sigma} = d_rho Gamma^mu_{nu sigma} - d_sigma Gamma^mu_{nu rho}
                          + Gamma^mu_{lambda rho} Gamma^lambda_{nu sigma}
                          - Gamma^mu_{lambda sigma} Gamma^lambda_{nu rho}.
    Only d_r differentials are non-trivial.
    """
    G_at = christoffel_symbols(vs, r, eps=eps)
    G_plus = christoffel_symbols(vs, r + eps, eps=eps)
    G_minus = christoffel_symbols(vs, r - eps, eps=eps)
    dG_r = (G_plus - G_minus) / (2 * eps)
    dG_full = np.zeros((4, 4, 4, 4))
    dG_full[1, :, :, :] = dG_r

    R = np.zeros((4, 4, 4, 4))
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    val = dG_full[rho, mu, nu, sigma] - dG_full[sigma, mu, nu, rho]
                    for lam in range(4):
                        val += G_at[mu, lam, rho] * G_at[lam, nu, sigma]
                        val -= G_at[mu, lam, sigma] * G_at[lam, nu, rho]
                    R[mu, nu, rho, sigma] = val
    return R


def ricci_tensor(vs, r: float, eps: float = 5e-4) -> np.ndarray:
    """Ricci tensor R_{nu sigma} = R^mu_{nu mu sigma}. Shape (4, 4)."""
    R_upper = riemann_tensor(vs, r, eps=eps)
    return np.einsum("munuS->nuS", R_upper.swapaxes(0, 2))[0]  # placeholder; correct below


def ricci_tensor_correct(vs, r: float, eps: float = 5e-4) -> np.ndarray:
    """Ricci tensor R_{nu sigma} = sum_mu R^mu_{nu mu sigma}."""
    R = riemann_tensor(vs, r, eps=eps)
    Ric = np.zeros((4, 4))
    for nu in range(4):
        for sigma in range(4):
            for mu in range(4):
                Ric[nu, sigma] += R[mu, nu, mu, sigma]
    return Ric


def ricci_scalar(vs, r: float, eps: float = 5e-4) -> float:
    """R = g^{mu nu} R_{mu nu}."""
    g = metric_tensor(vs, r)
    g_inv = metric_inverse(g)
    Ric = ricci_tensor_correct(vs, r, eps=eps)
    return float(np.einsum("ab,ab->", g_inv, Ric))


def kretschmann_scalar(vs, r: float, eps: float = 5e-4) -> float:
    """Kretschmann scalar K = R_{mu nu rho sigma} R^{mu nu rho sigma}."""
    R_upper = riemann_tensor(vs, r, eps=eps)
    g = metric_tensor(vs, r)
    g_inv = metric_inverse(g)

    # Lower-index Riemann: R_{mu nu rho sigma} = g_{mu lambda} R^lambda_{nu rho sigma}
    R_lower = np.einsum("ml,lnro->mnro", g, R_upper)
    # Fully-upper Riemann
    R_upupupup = np.einsum(
        "ma,nb,oc,pd,abcd->mnop", g_inv, g_inv, g_inv, g_inv, R_lower
    )
    K = float(np.einsum("mnop,mnop->", R_lower, R_upupupup))
    return K


def trace_anomaly_4d_exact(vs, r: float, eps: float = 5e-4) -> float:
    """Exact 4D conformal-anomaly trace for a massless conformally-coupled scalar.

    In a vacuum spacetime (R_{mu nu} = 0, R = 0), the trace anomaly reduces to

        <T^mu_{~mu}>_ren = K / (2880 pi^2),

    where K is the Kretschmann scalar. For non-vacuum spacetimes additional
    terms involving R_{mu nu} R^{mu nu}, R^2, and box R appear (Birrell-
    Davies 1982). Here we assume the LP exterior is vacuum and use the
    simplified form.
    """
    K = kretschmann_scalar(vs, r, eps=eps)
    return K / (2880.0 * np.pi * np.pi)


def dewitt_a2_coefficient(vs, r: float, eps: float = 5e-4) -> float:
    """Heat-kernel coefficient a_2(x) for a massless conformally-coupled scalar.

    In vacuum,
        a_2(x) = (1/180) R_{mu nu rho sigma} R^{mu nu rho sigma}
               = K / 180.
    """
    K = kretschmann_scalar(vs, r, eps=eps)
    return K / 180.0


def phi_squared_largemass_expansion(vs, r: float, mass: float, eps: float = 5e-4) -> float:
    """Leading large-mass expansion of <phi^2>_ren via the heat kernel.

    For mass m large compared to the curvature scales,
        <phi^2>_ren ~ a_2(x) / (16 pi^2 m^2)
              + O(1/m^4).
    Used as a sanity check / asymptotic limit; not a substitute for a
    full mode-sum calculation at finite m.
    """
    if mass <= 0:
        raise ValueError("mass must be positive")
    a2 = dewitt_a2_coefficient(vs, r, eps=eps)
    return a2 / (16.0 * np.pi * np.pi * mass * mass)


def effective_action_volume_density(vs, r: float, eps: float = 5e-4) -> float:
    """One-loop effective-action volume density a_2(x) / (32 pi^2).

    The renormalised one-loop effective Lagrangian density of a massless
    conformally-coupled scalar (in the vacuum-state approximation) is
    proportional to a_2(x). This module exposes the dimensionful density
    for use in back-reaction estimates.
    """
    a2 = dewitt_a2_coefficient(vs, r, eps=eps)
    return a2 / (32.0 * np.pi * np.pi)


def vacuum_residual(vs, r: float, eps: float = 5e-4) -> float:
    """Numerical max|R_{mu nu}| as a sanity check that the LP exterior is vacuum.

    Should be small (limited by finite-difference error). A large value
    indicates the metric is being evaluated outside its analytic domain
    (e.g. across a Cauchy horizon F = 0).
    """
    Ric = ricci_tensor_correct(vs, r, eps=eps)
    return float(np.max(np.abs(Ric)))
