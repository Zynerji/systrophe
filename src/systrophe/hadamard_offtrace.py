"""Off-trace components of the renormalised stress-energy tensor.

The `point_splitting.py` module computes the renormalised *trace*
<T^mu_{~mu}>_ren = K / (2880 pi^2) exactly on a 4D vacuum background.
This module adds the *full* tensor <T_{mu nu}>_ren by exposing the
locally-determined geometric structures and their trace decomposition.

Mathematical content
--------------------
For a massless conformally-coupled scalar field on a 4D *vacuum*
background (R_{mu nu} = 0, R = 0), one valid local closed form
satisfying the conformal anomaly is

    <T_{mu nu}>_ren = (1 / 2880 pi^2)  R_{mu rho sigma tau} R_nu^{rho sigma tau}.

This tensor:
- is symmetric in (mu, nu) (by Riemann index permutations);
- has trace g^{mu nu} R_{mu rho sigma tau} R_nu^{rho sigma tau} = K
  (Kretschmann scalar), so the trace anomaly K / (2880 pi^2) is
  recovered exactly;
- depends only on local Riemann curvature (no state choice).

Caveat
------
A *physical* renormalised <T_{mu nu}>_ren depends on the choice of
vacuum state (Boulware, Hartle-Hawking, Unruh). The locally-determined
formula above contains the *state-independent* geometric piece;
state-dependent corrections involve mode sums over the chosen vacuum
and cannot be written as local curvature polynomials.

This module exposes the geometric piece and its trace decomposition.
Comparison to the trace anomaly is a sanity check; comparison to a
specific vacuum state requires additional infrastructure.

References
----------
- S. M. Christensen, Phys. Rev. D 14 (1976) 2490 (point splitting).
- S. M. Christensen, Phys. Rev. D 17 (1978) 946 (back-reaction).
- N. D. Birrell and P. C. W. Davies, *Quantum Fields in Curved Space*
  Section 6.2 (1982).
- Y. Decanini and A. Folacci, Phys. Rev. D 78 (2008) 044025
  (covariant Hadamard expansion).
"""

from __future__ import annotations

import numpy as np

from .point_splitting import (
    kretschmann_scalar,
    metric_inverse,
    metric_tensor,
    riemann_tensor,
)

_ANOMALY_CONST = 1.0 / (2880.0 * np.pi * np.pi)


def riemann_squared_tensor(vs, r: float, eps: float = 5e-4) -> np.ndarray:
    """Compute R_{mu rho sigma tau} R_nu^{rho sigma tau} at radius r.

    The four-index Riemann tensor is contracted on three indices to
    produce a symmetric (4, 4) tensor.

    Returns
    -------
    np.ndarray of shape (4, 4)
        Symmetric tensor with trace equal to the Kretschmann scalar.
    """
    R_upper = riemann_tensor(vs, r, eps=eps)
    g = metric_tensor(vs, r)
    g_inv = metric_inverse(g)

    # Lower-index Riemann
    R_lower = np.einsum("ml,lnrs->mnrs", g, R_upper)
    # R_mu^{rho sigma tau} = g^{rho a} g^{sigma b} g^{tau c} R_{mu a b c}
    R_one_low = np.einsum("ra,sb,tc,mabc->mrst", g_inv, g_inv, g_inv, R_lower)

    # Contract R_{mu rho sigma tau} R_nu^{rho sigma tau}
    T = np.einsum("mrst,nrst->mn", R_lower, R_one_low)
    # Symmetrise (numerically) - tiny asymmetry from FD noise
    return 0.5 * (T + T.T)


def hadamard_offtrace_T(vs, r: float, eps: float = 5e-4) -> np.ndarray:
    """Locally-determined geometric piece of <T_{mu nu}>_ren.

    Formula:
        <T_{mu nu}>_geom = (1 / 2880 pi^2) R_{mu rho sigma tau} R_nu^{rho sigma tau}.

    Trace equals the Kretschmann scalar / (2880 pi^2), exactly matching
    the trace anomaly.

    Returns
    -------
    np.ndarray of shape (4, 4)
    """
    T_R2 = riemann_squared_tensor(vs, r, eps=eps)
    return _ANOMALY_CONST * T_R2


def hadamard_T_trace(T: np.ndarray, vs, r: float) -> float:
    """Compute g^{mu nu} T_{mu nu} at radius r."""
    g = metric_tensor(vs, r)
    g_inv = metric_inverse(g)
    return float(np.einsum("ab,ab->", g_inv, T))


def hadamard_T_traceless_part(T: np.ndarray, vs, r: float) -> np.ndarray:
    """Trace-free part of a (0, 2) tensor T_{mu nu}.

    T_traceless_{mu nu} = T_{mu nu} - (1/4) g_{mu nu} (g^{ab} T_{ab}).
    """
    g = metric_tensor(vs, r)
    trace = hadamard_T_trace(T, vs, r)
    return T - 0.25 * g * trace


def trace_decomposition(vs, r: float, eps: float = 5e-4) -> dict:
    """Decompose the locally-determined <T_{mu nu}>_ren into trace + traceless.

    Returns dict with:
      - T_total      : the full (4, 4) tensor
      - T_trace      : scalar trace value
      - T_trace_part : the (1/4) g_{mu nu} * trace piece
      - T_traceless  : the traceless piece T_{mu nu} - (1/4) g_{mu nu} trace
      - trace_anomaly_check : trace_part value minus K/(2880 pi^2);
                              should be ~ 0 by construction.
    """
    T = hadamard_offtrace_T(vs, r, eps=eps)
    trace = hadamard_T_trace(T, vs, r)
    g = metric_tensor(vs, r)
    T_trace_part = 0.25 * g * trace
    T_traceless = T - T_trace_part
    K = kretschmann_scalar(vs, r, eps=eps)
    expected_trace = K * _ANOMALY_CONST
    return {
        "T_total": T,
        "T_trace": trace,
        "T_trace_part": T_trace_part,
        "T_traceless": T_traceless,
        "trace_anomaly_check": abs(trace - expected_trace),
        "kretschmann": K,
    }


def energy_density_in_static_frame(vs, r: float, eps: float = 5e-4) -> float:
    """Energy density rho = <T_{tt}> in the LP coordinate frame.

    Returns the (tt) component of the locally-determined <T_{mu nu}>_ren.
    Sign convention: positive rho means positive energy density.
    """
    T = hadamard_offtrace_T(vs, r, eps=eps)
    g = metric_tensor(vs, r)
    # In LP coords, g_{tt} = -F. Coordinate-frame rho is T_{tt}/(-g_{tt}) = T_{tt}/F.
    F = -g[0, 0]
    if F <= 0:
        # We've crossed the chronology horizon; energy density not
        # well-defined in this frame.
        return float("nan")
    return float(T[0, 0] / F)


def hadamard_T_diagonal_components(vs, r: float, eps: float = 5e-4) -> dict:
    """Return the four diagonal components T_{tt}, T_{rr}, T_{phi phi}, T_{zz}.

    These are the principal-axis pressures and energy density (after
    sign/frame-rotation conventions).
    """
    T = hadamard_offtrace_T(vs, r, eps=eps)
    return {
        "T_tt": float(T[0, 0]),
        "T_rr": float(T[1, 1]),
        "T_phi_phi": float(T[2, 2]),
        "T_zz": float(T[3, 3]),
        "T_t_phi": float(T[0, 2]),
    }
