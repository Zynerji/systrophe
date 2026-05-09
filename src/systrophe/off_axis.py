"""Off-axis Systrophe pair: two parallel-axis cylinders linearly superposed.

Two co-rotating dust cylinders whose symmetry axes are parallel but
displaced by a perpendicular distance d break the joint axisymmetry.
Their leading-order vacuum exterior is constructed by:

  1. Computing each single-cylinder metric perturbation
        h_{mu nu}^{(i)} = g_{mu nu}^{(i)} - eta_{mu nu}
     (each in its own polar frame, then in shared Cartesian).
  2. Translating each to a common Cartesian frame.
  3. Summing the perturbations:  h^{tot} = h^{(1)} + h^{(2)}.
  4. Recovering the joint metric: g^{tot} = eta + h^{tot}.

This is leading-order in G; nonlinear cross-terms appear at O(G^2)
and are not modelled here. Each individual cylinder is treated by
the analytic Case III closed forms (supercritical only).

Coordinate convention
---------------------
The pair uses Cartesian coordinates (t, x, y, z) with both cylinder
axes parallel to z. Cylinder 1 is centered at the origin; cylinder 2
is centered at (d, 0). Polar coordinates around cylinder 1 are
(r1, phi1) = (sqrt(x^2 + y^2), atan2(y, x)).

Public API
----------
``OffAxisPair(cyl1, cyl2, separation)`` evaluates joint metric
components at any (x, y) and provides convenience methods for
identifying CTC regions on a 2D grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


def _cartesian_h(F: np.ndarray, K: np.ndarray, L: np.ndarray,
                 r: np.ndarray, phi: np.ndarray) -> dict:
    """Cartesian perturbation h_{mu nu} = g_{mu nu} - eta_{mu nu} for one cylinder.

    Inputs are the cylinder's local polar (r, phi) and metric components
    (F, K, L). The cylinder's local conformal factor h_metric(r) is
    approximated by 1 (the asymptotic Minkowski value); this is exact
    for the t, phi block and a leading-order approximation for the
    spatial block. The returned components are perturbations only;
    summing across cylinders is well defined.

    Returns a dict with keys h_tt, h_tx, h_ty, h_xx, h_yy, h_xy.
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    r_safe = np.where(np.abs(r) > 1e-300, r, 1.0e-300)

    h_tt = -F + 1.0  # g_tt = -F so h_tt = g_tt - eta_tt = -F - (-1) = 1 - F
    # off-diagonals from g_{t phi} = K -> g_{tx} = -K sin(phi) / r, g_{ty} = K cos(phi) / r
    h_tx = -K * sin_phi / r_safe
    h_ty = K * cos_phi / r_safe
    # spatial block: g_xx, g_yy, g_xy from g_phiphi = L (set g_rr = 1)
    # eta_xx = eta_yy = 1 in Cartesian.
    g_xx = (cos_phi ** 2) + L * (sin_phi ** 2) / (r_safe ** 2)
    g_yy = (sin_phi ** 2) + L * (cos_phi ** 2) / (r_safe ** 2)
    g_xy = sin_phi * cos_phi - L * sin_phi * cos_phi / (r_safe ** 2)
    h_xx = g_xx - 1.0
    h_yy = g_yy - 1.0
    h_xy = g_xy
    return {
        "h_tt": h_tt,
        "h_tx": h_tx,
        "h_ty": h_ty,
        "h_xx": h_xx,
        "h_yy": h_yy,
        "h_xy": h_xy,
    }


@dataclass(frozen=True)
class OffAxisPair:
    """Two parallel-axis co-rotating cylinders, linearised superposition.

    Parameters
    ----------
    cyl1, cyl2 : VanStockumInterior
        The two source cylinders. Both must be supercritical (a > 1/2)
        for the analytic Case III closed forms to apply.
    separation : float
        Perpendicular distance between the parallel axes. Must be
        positive; for separation = 0 use the co-axial SystrophePair.
    """

    cyl1: VanStockumInterior
    cyl2: VanStockumInterior
    separation: float

    def __post_init__(self) -> None:
        if not isinstance(self.cyl1, VanStockumInterior):
            raise TypeError("cyl1 must be a VanStockumInterior")
        if not isinstance(self.cyl2, VanStockumInterior):
            raise TypeError("cyl2 must be a VanStockumInterior")
        if not (self.cyl1.is_supercritical() and self.cyl2.is_supercritical()):
            raise ValueError("Both cylinders must be supercritical (a > 1/2)")
        if self.separation <= 0:
            raise ValueError("separation must be positive (use SystrophePair for co-axial)")

    def _metric_components_cyl(
        self, vs: VanStockumInterior, r: np.ndarray, phi: np.ndarray
    ) -> dict:
        """Evaluate the cylinder's analytic-Case-III F, K, L; clip below R.

        Inside a cylinder (r < R) we use the analytic interior closed form
        for F, K, L; outside we use the Case III analytic exterior. The
        analytic_exterior_F/K/L are only valid for r > R, but the
        van Stockum interior also has closed-form components that we can
        substitute. For points inside cylinder 2 evaluated in cylinder 1's
        polar frame, the perturbation is the local interior contribution.
        """
        F = np.zeros_like(r, dtype=float)
        K = np.zeros_like(r, dtype=float)
        L = np.zeros_like(r, dtype=float)
        outside = r >= vs.R
        inside = ~outside
        if np.any(outside):
            F[outside] = vs.analytic_exterior_F(r[outside])
            K[outside] = vs.analytic_exterior_K(r[outside])
            L[outside] = vs.analytic_exterior_L(r[outside])
        if np.any(inside):
            # Interior: F=1, K = omega r^2, L = r^2 (1 - (omega r)^2)
            r_in = r[inside]
            F[inside] = 1.0
            K[inside] = vs.omega * r_in ** 2
            L[inside] = r_in ** 2 * (1.0 - vs.omega ** 2 * r_in ** 2)
        return {"F": F, "K": K, "L": L}

    def cartesian_metric_perturbation(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Total Cartesian perturbation h^{tot}(x, y) summed across both cylinders.

        At each (x, y), evaluate cylinder 1's contribution at local polar
        (r1, phi1) and cylinder 2's at (r2, phi2) where r2 = sqrt((x - d)^2 + y^2),
        phi2 = atan2(y, x - d). Sum the Cartesian perturbations.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r1 = np.sqrt(x * x + y * y)
        phi1 = np.arctan2(y, x)
        r2 = np.sqrt((x - self.separation) ** 2 + y * y)
        phi2 = np.arctan2(y, x - self.separation)

        m1 = self._metric_components_cyl(self.cyl1, r1, phi1)
        h1 = _cartesian_h(m1["F"], m1["K"], m1["L"], r1, phi1)
        m2 = self._metric_components_cyl(self.cyl2, r2, phi2)
        h2 = _cartesian_h(m2["F"], m2["K"], m2["L"], r2, phi2)

        return {k: h1[k] + h2[k] for k in h1.keys()}

    def cartesian_metric(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Total joint metric in Cartesian. Returns dict of g_{mu nu}."""
        h = self.cartesian_metric_perturbation(x, y)
        # eta in Cartesian: g_tt = -1, g_xx = g_yy = 1, off-diag = 0
        return {
            "g_tt": -1.0 + h["h_tt"],
            "g_tx": h["h_tx"],
            "g_ty": h["h_ty"],
            "g_xx": 1.0 + h["h_xx"],
            "g_yy": 1.0 + h["h_yy"],
            "g_xy": h["h_xy"],
        }

    def has_local_ctc(self, x: float, y: float) -> bool:
        """Check whether a CTC exists locally at point (x, y).

        A CTC at fixed (t, x, y, z) along the angular direction phi1
        (around cylinder 1's axis) requires g_{phi1 phi1} < 0 in the
        joint metric. We compute g_{phi phi} at (x, y) using the
        Jacobian.
        """
        # Use 1-element arrays so np.asarray works in _metric_components_cyl
        g = self.cartesian_metric(np.atleast_1d(x), np.atleast_1d(y))
        r1 = float(np.sqrt(x * x + y * y))
        if r1 < 1e-12:
            return False  # at origin; degenerate
        cos_phi = float(x) / r1
        sin_phi = float(y) / r1
        # g_phi1phi1 in cyl-1 polar: r1^2 * (sin^2 phi g_xx + cos^2 phi g_yy - 2 sin cos g_xy)
        g_phiphi = float(
            r1 * r1
            * (sin_phi ** 2 * g["g_xx"][0] + cos_phi ** 2 * g["g_yy"][0]
               - 2.0 * sin_phi * cos_phi * g["g_xy"][0])
        )
        return g_phiphi < 0.0

    def ctc_map_2d(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        nx: int = 121,
        ny: int = 121,
    ) -> dict:
        """Compute the CTC map (boolean grid) over a 2D rectangle.

        Returns dict with 'x', 'y', 'r1', 'r2', 'g_phiphi_cyl1' (CTC
        diagnostic in cyl-1 polar), and 'is_ctc' boolean grid.
        """
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        g = self.cartesian_metric(X, Y)
        R1 = np.sqrt(X * X + Y * Y)
        R2 = np.sqrt((X - self.separation) ** 2 + Y * Y)
        # Avoid singularity at origin
        cos_phi = np.where(R1 > 1e-12, X / np.where(R1 > 1e-12, R1, 1.0), 1.0)
        sin_phi = np.where(R1 > 1e-12, Y / np.where(R1 > 1e-12, R1, 1.0), 0.0)
        g_phiphi = R1 * R1 * (
            sin_phi ** 2 * g["g_xx"]
            + cos_phi ** 2 * g["g_yy"]
            - 2.0 * sin_phi * cos_phi * g["g_xy"]
        )
        return {
            "x": x,
            "y": y,
            "r1": R1,
            "r2": R2,
            "g_phiphi_cyl1": g_phiphi,
            "is_ctc": g_phiphi < 0.0,
        }
