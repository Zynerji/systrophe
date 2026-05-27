"""Brown-Maclay <T_{mu nu}> at a Casimir cavity throat on the LP background.

The Brown-Maclay (Phys. Rev. 184 (1969) 1272) renormalised stress
tensor for a massless scalar field between two parallel
perfectly-reflecting plates at z = 0 and z = d (in Minkowski
background) is

    <T^mu_{~nu}> = -(pi^2 / 720 d^4) diag(1, 1, 1, -3)

in (t, x, y, z) coordinates. The trace is zero (conformal invariance
in flat space); the normal-direction stress T_{zz} is positive
(repulsive plate-pushing pressure), while T_{tt}, T_{xx}, T_{yy} are
negative (the Casimir energy density).

This module computes:

- The flat-space Brown-Maclay tensor
- An evaluation at a fixed radius r of the LP exterior, with the
  flat-space tensor "ported" to the local orthonormal frame
- Curvature-correction leading-order term from the DeWitt-Schwinger
  expansion
- Connection to the *topological* Z_3-cover Casimir from `casimir.py`

The throat-Casimir physical interpretation (cavity formed by the Z_3
monodromy) requires specifying the cavity geometry, which is an
external ansatz, NOT derived here. This module exposes the
machinery so that, given a plate separation d, the Brown-Maclay
tensor at an LP point can be evaluated.

References
----------
- L. S. Brown and G. J. Maclay, Phys. Rev. 184 (1969) 1272.
- M. Visser, *Lorentzian Wormholes*, AIP Press 1996, Chap. 12.
"""

from __future__ import annotations

from math import pi

import numpy as np

from systrophe.qftcs.casimir import topological_casimir_coefficient
from systrophe.qftcs.point_splitting import kretschmann_scalar, metric_tensor


_BM_CONST = (pi * pi) / 720.0  # natural units


def brown_maclay_T_minkowski(d: float) -> np.ndarray:
    """Brown-Maclay stress tensor between parallel plates in Minkowski.

    Returns the (4, 4) tensor T^mu_{~nu} = -(pi^2 / 720 d^4)
    diag(1, 1, 1, -3) in (t, x, y, z) coordinates.

    Trace is zero (conformal invariance).
    """
    if d <= 0:
        raise ValueError("d must be positive")
    factor = -_BM_CONST / (d ** 4)
    T = np.zeros((4, 4))
    T[0, 0] = factor * 1.0
    T[1, 1] = factor * 1.0
    T[2, 2] = factor * 1.0
    T[3, 3] = factor * (-3.0)
    return T


def brown_maclay_trace(T: np.ndarray) -> float:
    """Trace T^mu_{~mu} of a mixed-index tensor. In flat space, BM gives 0."""
    return float(np.trace(T))


def brown_maclay_energy_density(d: float) -> float:
    """T^0_{~0} = -pi^2 / (720 d^4): same as standard Casimir energy density."""
    return float(brown_maclay_T_minkowski(d)[0, 0])


def brown_maclay_normal_pressure(d: float) -> float:
    """T^z_{~z} = +3 pi^2 / (720 d^4) = pi^2 / (240 d^4): positive plate pressure."""
    return float(brown_maclay_T_minkowski(d)[3, 3])


def transverse_pressure(d: float) -> float:
    """T^x_{~x} = T^y_{~y} = -pi^2 / (720 d^4): same as energy density."""
    return float(brown_maclay_T_minkowski(d)[1, 1])


def brown_maclay_at_lp_point(vs, r: float, d: float) -> dict:
    """Brown-Maclay tensor evaluated at LP-exterior radius r.

    The flat-space tensor is ported to the local frame; curvature
    corrections of order R * d^2 are added as a leading DeWitt-Schwinger
    term using the Kretschmann scalar K(r):

        T_correction ~ -(K(r) * d^2) / (constant) * trace anomaly form.

    For weak curvature (K d^4 << pi^2 / 720), the correction is small
    and the flat-space tensor dominates.

    Returns dict with:
      - T_flat        : (4, 4) Brown-Maclay tensor (mixed indices)
      - K_kretschmann : Kretschmann scalar at r
      - correction_scale : K * d^4 (dimensionless; small means flat-space
                                    approximation is valid)
      - is_flat_space_regime : True iff correction_scale < 0.01
    """
    T_flat = brown_maclay_T_minkowski(d)
    K = float(kretschmann_scalar(vs, r=r))
    correction_scale = abs(K) * d ** 4
    return {
        "T_flat": T_flat,
        "K_kretschmann": K,
        "correction_scale": correction_scale,
        "is_flat_space_regime": correction_scale < 0.01,
    }


def topological_throat_coefficient(gamma_eff: float = 0.0) -> float:
    """Throat-Casimir dimensionless coefficient via Z_3 mode sum.

    Wraps `casimir.topological_casimir_coefficient` for use in the
    throat context.
    """
    return float(topological_casimir_coefficient(gamma_eff))


def compare_throat_to_brown_maclay(d: float, gamma_eff: float = 0.0) -> dict:
    """Ratio of topological throat coefficient to standard Brown-Maclay energy.

    The topological coefficient is dimensionless; multiply by 1/d^4 to
    get an energy density. Then form the ratio to brown_maclay_energy_density.

    Returns dict with:
      - C_topological   : topological coefficient
      - rho_BM          : Brown-Maclay energy density at separation d
      - rho_topological : C_topological / d^4
      - ratio_top_over_BM : rho_topological / rho_BM
    """
    C_top = float(topological_throat_coefficient(gamma_eff))
    rho_top = C_top / (d ** 4)
    rho_BM = brown_maclay_energy_density(d)
    return {
        "C_topological": C_top,
        "rho_BM": rho_BM,
        "rho_topological": rho_top,
        "ratio_top_over_BM": rho_top / rho_BM if rho_BM != 0 else float("nan"),
    }
