"""Van Stockum (1937) rotating-dust interior metric.

Exact closed-form interior of an infinite rigidly-rotating dust cylinder
with angular velocity omega, in cylindrical coordinates (t, r, phi, z):

    ds^2 = -dt^2 + 2 omega r^2 dt dphi
           + r^2 (1 - omega^2 r^2) dphi^2
           + exp(-omega^2 r^2) (dr^2 + dz^2)

References
----------
- W. J. van Stockum, Proc. Roy. Soc. Edin. 57 (1937) 135.
- F. J. Tipler, Phys. Rev. D 9 (1974) 2203.
- W. B. Bonnor, J. Phys. A 13 (1980) 2121.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np


@dataclass(frozen=True)
class VanStockumInterior:
    """Single rigidly-rotating dust cylinder, van Stockum interior solution.

    Parameters
    ----------
    omega : float
        Angular velocity of the dust (units of 1/length, c=G=1).
    R : float
        Cylinder radius. Source occupies r in [0, R].
    """

    omega: float
    R: float

    def __post_init__(self) -> None:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.omega < 0:
            raise ValueError("omega must be non-negative; flip phi orientation otherwise")

    @property
    def a(self) -> float:
        """Dimensionless rotation parameter a = omega * R."""
        return self.omega * self.R

    @property
    def alpha(self) -> float:
        """Tipler exterior log-frequency: alpha = sqrt(4 a^2 - 1) for a > 1/2."""
        v = 4.0 * self.a * self.a - 1.0
        if v <= 0:
            raise ValueError(
                f"alpha is real only for a = omega*R > 1/2 (got a = {self.a:.4f})."
            )
        return float(np.sqrt(v))

    def is_supercritical(self) -> bool:
        """True iff a > 1/2 (Tipler oscillatory exterior, Bonnor Case III)."""
        return self.a > 0.5

    def metric(self, r: float) -> np.ndarray:
        """Return 4x4 metric tensor at radius r (interior, r <= R).

        Order: (t, r, phi, z).
        """
        if r < 0:
            raise ValueError("r must be non-negative")
        if r > self.R:
            raise ValueError(f"r = {r} is outside cylinder; use exterior metric")
        w = self.omega
        g = np.zeros((4, 4))
        g[0, 0] = -1.0
        g[0, 2] = w * r * r
        g[2, 0] = w * r * r
        g[2, 2] = r * r * (1.0 - w * w * r * r)
        g[1, 1] = exp(-w * w * r * r)
        g[3, 3] = exp(-w * w * r * r)
        return g

    def gphiphi(self, r: float) -> float:
        """g_{phi phi}(r) interior; negative iff omega*r > 1 (CTC condition)."""
        return r * r * (1.0 - self.omega * self.omega * r * r)

    def has_interior_ctc(self) -> bool:
        """True iff the interior contains a CTC region (omega * R > 1)."""
        return self.a > 1.0

    def proper_circumference(self, r: float) -> float:
        """Proper length of a closed phi-orbit at fixed (t, r, z), interior.

        sqrt(|g_{phi phi}|) * 2 pi. Drops to zero at r = 1/omega and is the
        signature of the CTC threshold.
        """
        return 2.0 * np.pi * float(np.sqrt(abs(self.gphiphi(r))))

    def determinant_tphi_block(self, r: float) -> float:
        """Determinant of the (t, phi) sub-block; equals -r^2 throughout.

        Useful invariant: g_tt g_{phi phi} - g_{t phi}^2 = -r^2.
        """
        g = self.metric(r)
        return g[0, 0] * g[2, 2] - g[0, 2] * g[2, 0]

    @property
    def regime(self) -> str:
        """Bonnor exterior regime label: 'subcritical', 'critical', or 'supercritical'."""
        if self.a > 0.5 + 1e-12:
            return "supercritical"
        if self.a > 0.5 - 1e-9:
            return "critical"
        return "subcritical"

    def analytic_exterior_F(self, r: float | np.ndarray) -> np.ndarray:
        """Analytic exterior F(r) = -g_{tt}(r) in any Bonnor regime.

        Dispatches by regime:

        - **Supercritical** (a > 1/2):  F(r) = (r/R) sin(alpha u + gamma)/sin gamma
            with alpha = sqrt(4a^2 - 1), gamma = pi - arctan(alpha).
        - **Critical** (a = 1/2):  F(r) = (r/R) (1 - u).
        - **Subcritical** (a < 1/2):  F(r) = (r/R) [cosh(beta u) - sinh(beta u)/beta]
            with beta = sqrt(1 - 4a^2).
        """
        r_arr = np.asarray(r, dtype=float)
        u = np.log(r_arr / self.R)
        regime = self.regime
        if regime == "supercritical":
            alpha = self.alpha
            gamma = np.pi - np.arctan(alpha)
            return (r_arr / self.R) * np.sin(alpha * u + gamma) / np.sin(gamma)
        if regime == "critical":
            return (r_arr / self.R) * (1.0 - u)
        # subcritical
        beta = float(np.sqrt(1.0 - 4.0 * self.a * self.a))
        return (r_arr / self.R) * (np.cosh(beta * u) - np.sinh(beta * u) / beta)

    def analytic_exterior_K(self, r: float | np.ndarray) -> np.ndarray:
        """Analytic exterior K(r) = g_{t phi}(r) in any Bonnor regime.

        Dispatches by regime:

        - **Supercritical**:  K(r) = (r/alpha)[((alpha^2-1)/2)sin(.) - alpha cos(.)].
        - **Critical**:  K(r) = (r/2)(1 + u).
        - **Subcritical**:  K(r) = a r [cosh(beta u) + sinh(beta u)/beta].
        """
        r_arr = np.asarray(r, dtype=float)
        u = np.log(r_arr / self.R)
        regime = self.regime
        if regime == "supercritical":
            alpha = self.alpha
            gamma = np.pi - np.arctan(alpha)
            theta = alpha * u + gamma
            return (r_arr / alpha) * (
                ((alpha * alpha - 1.0) / 2.0) * np.sin(theta) - alpha * np.cos(theta)
            )
        if regime == "critical":
            return (r_arr / 2.0) * (1.0 + u)
        # subcritical
        beta = float(np.sqrt(1.0 - 4.0 * self.a * self.a))
        return self.a * r_arr * (np.cosh(beta * u) + np.sinh(beta * u) / beta)

    def analytic_exterior_L(self, r: float | np.ndarray) -> np.ndarray:
        """Analytic exterior L(r) = g_{phi phi}(r) in any Bonnor regime.

        Dispatches by regime:

        - **Supercritical**:  L(r) = (r R sin gamma / alpha^2) [ Q sin(.)
            + alpha (alpha^2-1) cos(.) ],  Q = alpha^2 - (alpha^2-1)^2/4.
        - **Critical**:  L(r) = (r R / 4) (3 + u).
        - **Subcritical**:  L(r) = r R (1 - a^2 S_+(u)^2) / S_-(u),
            where S_+- = cosh(beta u) +/- sinh(beta u)/beta.
            Equivalently L = (r^2 - K^2)/F; the constraint FL+K^2 = r^2
            holds identically.
        """
        r_arr = np.asarray(r, dtype=float)
        u = np.log(r_arr / self.R)
        regime = self.regime
        if regime == "supercritical":
            alpha = self.alpha
            gamma = np.pi - np.arctan(alpha)
            theta = alpha * u + gamma
            Q = alpha * alpha - ((alpha * alpha - 1.0) ** 2) / 4.0
            return (
                (r_arr * self.R * np.sin(gamma) / (alpha * alpha))
                * (Q * np.sin(theta) + alpha * (alpha * alpha - 1.0) * np.cos(theta))
            )
        if regime == "critical":
            return (r_arr * self.R / 4.0) * (3.0 + u)
        # subcritical: hyperbolic closed form
        beta = float(np.sqrt(1.0 - 4.0 * self.a * self.a))
        S_plus = np.cosh(beta * u) + np.sinh(beta * u) / beta
        S_minus = np.cosh(beta * u) - np.sinh(beta * u) / beta
        return r_arr * self.R * (1.0 - self.a * self.a * S_plus * S_plus) / S_minus

    def tipler_sinusoid_L(self):
        """TiplerSinusoid matched to the analytic g_{phi phi}(r) (CTC-relevant).

        Returns the log-periodic representation of L = g_{phi phi} in the
        Bonnor Case III exterior. CTC bands of the single cylinder are the
        negative bands of this sinusoid.
        """
        from .sinusoid import TiplerSinusoid

        if not self.is_supercritical():
            raise ValueError("TiplerSinusoid_L construction requires a > 1/2")
        alpha = self.alpha
        gamma = np.pi - np.arctan(alpha)
        Q = alpha * alpha - ((alpha * alpha - 1.0) ** 2) / 4.0
        M = np.sqrt(Q * Q + alpha * alpha * (alpha * alpha - 1.0) ** 2)
        psi = np.arctan2(Q, alpha * (alpha * alpha - 1.0))
        A_L = float(self.R * np.sin(gamma) * M / (alpha * alpha))
        delta_L = float(gamma - psi)
        # Wrap delta_L to (-pi, pi]
        delta_L = float(np.mod(delta_L + np.pi, 2 * np.pi) - np.pi)
        return TiplerSinusoid(R=float(self.R), a=float(self.a), A=A_L, delta=delta_L, p=0.0)

    def tipler_sinusoid_F(self):
        """TiplerSinusoid matched to F = -g_{tt} for r > R (ergosurface-relevant).

        Analytic Case III F = (r/R) sin(alpha u + gamma)/sin gamma maps to
            p = 0, A = 1/(R sin gamma), delta = gamma - pi/2.
        F < 0 marks ergoregions (t-Killing vector spacelike), NOT CTCs.
        For CTC analysis use tipler_sinusoid_L instead.
        """
        from .sinusoid import TiplerSinusoid

        if not self.is_supercritical():
            raise ValueError("TiplerSinusoid_F construction requires a > 1/2")
        alpha = self.alpha
        gamma = np.pi - np.arctan(alpha)
        return TiplerSinusoid(
            R=float(self.R),
            a=float(self.a),
            A=float(1.0 / (self.R * np.sin(gamma))),
            delta=float(gamma - np.pi / 2),
            p=0.0,
        )

    # Backward-compatible alias; defaults to the CTC-relevant L sinusoid.
    def tipler_sinusoid(self):
        """Default TiplerSinusoid: matched to L = g_{phi phi} (CTC-relevant).

        Use tipler_sinusoid_F() for the F = -g_{tt} envelope (ergoregion).
        """
        return self.tipler_sinusoid_L()

    def integrate_exterior(
        self,
        r_max: float,
        n_samples: int = 4001,
        rtol: float = 1e-10,
        atol: float = 1e-13,
    ):
        """Numerically integrate the vacuum exterior outward to r_max.

        Returns an LPSolution. Works for any a (sub/critical/super-critical);
        the analytic form is only available for a > 1/2.
        """
        from .lewis_papapetrou import integrate_lp_exterior

        return integrate_lp_exterior(
            omega_dust=float(self.omega),
            R=float(self.R),
            r_max=float(r_max),
            n_samples=int(n_samples),
            rtol=float(rtol),
            atol=float(atol),
        )


def vanstockum_interior_metric(omega: float, r: float | np.ndarray) -> dict:
    """Functional convenience: return metric components as dict at given r.

    Accepts scalar or array r. Components: g_tt, g_tphi, g_phiphi, g_rr, g_zz.
    """
    r = np.asarray(r, dtype=float)
    w = float(omega)
    return {
        "g_tt": np.full_like(r, -1.0),
        "g_tphi": w * r * r,
        "g_phiphi": r * r * (1.0 - w * w * r * r),
        "g_rr": np.exp(-w * w * r * r),
        "g_zz": np.exp(-w * w * r * r),
    }
