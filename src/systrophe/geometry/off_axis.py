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

from systrophe.geometry.vanstockum import VanStockumInterior


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
    # Floor |r| away from zero on the cylinder axis. The floor must survive
    # being squared below (r_safe ** 2): 1e-300 squared underflows to 0.0 in
    # float64 and reintroduces the divide-by-zero it was meant to prevent
    # (invalid-value warnings + inf/nan at any grid point on a cylinder axis).
    # 1e-150 squared is 1e-300, still a normal float, so the spatial block
    # stays finite.
    r_safe = np.where(np.abs(r) > 1e-150, r, 1.0e-150)

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

    def integrate_test_particle(
        self,
        x0: float,
        y0: float,
        vx0: float,
        vy0: float,
        t_max: float,
        n_samples: int = 2001,
        kappa: float = 1.0,
    ) -> dict:
        """Numerically integrate a test-particle trajectory in the joint metric.

        Treats the linearised joint metric as a 2+1D system in (t, x, y)
        with z held fixed. The geodesic equations are derived from the
        Cartesian metric components via the Christoffel symbols computed
        by finite differencing of `cartesian_metric`. This is the
        standard machinery for joint-metric trajectories where no
        Killing-vector reduction is available.

        Parameters
        ----------
        x0, y0 : float
            Initial Cartesian position.
        vx0, vy0 : float
            Initial coordinate velocity dx/dt, dy/dt.
        t_max : float
            Final coordinate time.
        n_samples : int
            Output grid density.
        kappa : float
            +1 (timelike) or 0 (null). The initial 4-velocity is
            normalised accordingly.

        Returns
        -------
        dict with 't', 'x', 'y', 'tau' arrays.

        Notes
        -----
        This is leading-order in G (the metric perturbation). For
        long-time integrations near a CTC band the linearisation may
        break down. Used here primarily as a diagnostic tool for
        identifying CTC encounters along a trajectory.
        """
        from scipy.integrate import solve_ivp

        eps = 1e-6 * max(abs(x0), abs(y0), 1.0)

        def metric_at(x, y):
            g = self.cartesian_metric(np.atleast_1d(x), np.atleast_1d(y))
            return {k: float(v[0]) for k, v in g.items()}

        def metric_grad(x, y):
            """Approximate partial_x and partial_y of g_{ij} at (x, y)."""
            mp = metric_at(x + eps, y)
            mm = metric_at(x - eps, y)
            d_x = {k: (mp[k] - mm[k]) / (2 * eps) for k in mp}
            mp = metric_at(x, y + eps)
            mm = metric_at(x, y - eps)
            d_y = {k: (mp[k] - mm[k]) / (2 * eps) for k in mp}
            return d_x, d_y

        def state_dot(coord_t, state):
            x, y, ux, uy = state
            ut = 1.0  # parameterised by coord t (so ut = dt/dt = 1)
            d_x, d_y = metric_grad(x, y)
            # Christoffel for 2+1D (t, x, y) with the joint metric.
            # Acceleration: d^2 x^i / dt^2 = -Gamma^i_{ab} u^a u^b / (u^t)^2
            # Approximation: drop the d_t terms (stationary metric assumed).
            g = metric_at(x, y)
            # Index map: t=0, x=1, y=2. We compute a^x and a^y assuming u^t = 1.
            # Christoffel symbols of the second kind (rough):
            # Gamma^i_{jk} = 1/2 g^{i l} (d_j g_{lk} + d_k g_{lj} - d_l g_{jk})
            # We restrict to 2+1D and only the x, y spatial accelerations.
            # For simplicity and stability, integrate using a stationary-
            # metric approximation: a^i = -1/(2) (d_i g_{tt} + 2 d_i g_{tj} u^j
            # + d_i g_{jk} u^j u^k) / g_{ii} (approx.)
            ax_num = -0.5 * (
                d_x["g_tt"]
                + 2 * d_x["g_tx"] * ux + 2 * d_x["g_ty"] * uy
                + d_x["g_xx"] * ux * ux + 2 * d_x["g_xy"] * ux * uy + d_x["g_yy"] * uy * uy
            )
            ay_num = -0.5 * (
                d_y["g_tt"]
                + 2 * d_y["g_tx"] * ux + 2 * d_y["g_ty"] * uy
                + d_y["g_xx"] * ux * ux + 2 * d_y["g_xy"] * ux * uy + d_y["g_yy"] * uy * uy
            )
            ax = ax_num / max(abs(g["g_xx"]), 1e-9)
            ay = ay_num / max(abs(g["g_yy"]), 1e-9)
            return [ux, uy, ax, ay]

        sol = solve_ivp(
            state_dot,
            t_span=(0.0, t_max),
            y0=[x0, y0, vx0, vy0],
            t_eval=np.linspace(0.0, t_max, n_samples),
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"off-axis test particle integration failed: {sol.message}")

        # Compute proper time along the trajectory by integrating ds = sqrt(-ds^2)
        # for kappa=1 (timelike), or ds = 0 for kappa=0 (null).
        x_arr, y_arr = sol.y[0], sol.y[1]
        ux_arr, uy_arr = sol.y[2], sol.y[3]
        dtau_dt = np.zeros_like(sol.t)
        for i, (x, y, ux, uy) in enumerate(zip(x_arr, y_arr, ux_arr, uy_arr)):
            g = metric_at(x, y)
            ds2 = (
                g["g_tt"]
                + 2 * g["g_tx"] * ux + 2 * g["g_ty"] * uy
                + g["g_xx"] * ux * ux + 2 * g["g_xy"] * ux * uy + g["g_yy"] * uy * uy
            )
            if kappa > 0:
                dtau_dt[i] = float(np.sqrt(max(-ds2, 0.0)))
            else:
                dtau_dt[i] = 0.0
        # cumulative trapezoidal
        tau = np.concatenate(([0.0], np.cumsum(0.5 * (dtau_dt[1:] + dtau_dt[:-1]) * np.diff(sol.t))))

        return {"t": sol.t, "x": x_arr, "y": y_arr, "tau": tau}

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

    # -----------------------------------------------------------------
    # Phase 3b: quantitative orbit + topology diagnostics
    # -----------------------------------------------------------------

    def ergosurface_2d(
        self,
        x_min: float, x_max: float, y_min: float, y_max: float,
        nx: int = 121, ny: int = 121,
    ) -> dict:
        """Locate the joint ergosurface g_tt = 0 in the (x, y) plane.

        Returns dict with grid + g_tt scalar + is_ergoregion boolean
        (g_tt > 0 means t-Killing vector spacelike: inside ergoregion).
        """
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        g = self.cartesian_metric(X, Y)
        return {
            "x": x,
            "y": y,
            "g_tt": np.asarray(g["g_tt"]),
            "is_ergoregion": np.asarray(g["g_tt"]) > 0.0,
        }

    def ctc_region_topology(
        self,
        x_min: float, x_max: float, y_min: float, y_max: float,
        nx: int = 121, ny: int = 121,
    ) -> dict:
        """Classify the 2D CTC region topology.

        Counts connected components, identifies whether each component
        is simply-connected or contains holes (genus 0 vs > 0 in 2D),
        and reports per-component areas (in grid units).

        Strategy: floodfill on the boolean CTC mask via scipy.ndimage.
        """
        from scipy.ndimage import label
        ctc_map = self.ctc_map_2d(x_min, x_max, y_min, y_max, nx=nx, ny=ny)
        mask = np.asarray(ctc_map["is_ctc"])
        labels, n_components = label(mask)
        component_areas = []
        for k in range(1, n_components + 1):
            area = int(np.sum(labels == k))
            component_areas.append(area)

        # Detect simply-connectedness by flooding the *complement*: components
        # of NOT(ctc) that don't touch the boundary indicate holes.
        outer = np.zeros_like(mask, dtype=bool)
        outer[0, :] = True
        outer[-1, :] = True
        outer[:, 0] = True
        outer[:, -1] = True
        non_ctc = ~mask
        non_ctc_labels, n_non = label(non_ctc)
        boundary_touching = set()
        for k in range(1, n_non + 1):
            if np.any((non_ctc_labels == k) & outer):
                boundary_touching.add(k)
        n_holes = max(0, n_non - len(boundary_touching))

        return {
            "n_components": int(n_components),
            "component_areas": component_areas,
            "n_holes": int(n_holes),
            "ctc_fraction": float(np.mean(mask)),
            "topology_summary": (
                "empty" if n_components == 0
                else "simply_connected" if (n_components == 1 and n_holes == 0)
                else "multi_component" if (n_components > 1 and n_holes == 0)
                else "with_holes" if n_holes > 0
                else "complex"
            ),
        }

    def trace_anomaly_2d_sector(self, x: float, y: float, eps: float = 1e-3) -> float:
        """4D trace anomaly proxy at (x, y) using only the 2D (t, x or y) sector.

        Builds a 2D effective metric from the joint Cartesian metric and
        evaluates R_2D / (24 pi). This is a *leading-order* indicator of
        local QFTCS back-reaction in the off-axis pair; the full 4D
        Hadamard subtraction would require a substantial extension.
        """
        # Choose the radial direction along the y axis (perpendicular to
        # the separation axis x) as the "2D" coordinate.
        def F_at(yy: float) -> float:
            g = self.cartesian_metric(np.atleast_1d(x), np.atleast_1d(yy))
            return float(-g["g_tt"][0])
        F = F_at(y)
        F_p = (F_at(y + eps) - F_at(y - eps)) / (2.0 * eps)
        F_pp = (F_at(y + eps) - 2.0 * F + F_at(y - eps)) / (eps * eps)
        if abs(F) < 1e-9:
            return float("inf")
        # 2D Ricci scalar of ds^2 = -F dt^2 + dy^2 (assuming h_yy = 1 at this order)
        R_2D = -F_pp / F + (F_p * F_p) / (2.0 * F * F)
        return float(R_2D / (24.0 * np.pi))

    def geodesic_completeness_test(
        self,
        x_starts: tuple,
        y_starts: tuple,
        vx0: float = 0.1,
        vy0: float = 0.1,
        t_max: float = 100.0,
        n_samples: int = 501,
        escape_radius: float = 100.0,
    ) -> list:
        """Test whether timelike test particles escape to infinity from
        several initial conditions.

        Returns a list of dicts (one per starting condition) with:
          - 'reaches_escape': bool, whether sqrt(x^2 + y^2) > escape_radius
          - 'final_radius'  : maximum radius reached
          - 'enters_ctc'   : whether the path enters the CTC region
          - 'success'       : whether the integration completed
        """
        results = []
        for x0, y0 in zip(x_starts, y_starts):
            try:
                traj = self.integrate_test_particle(
                    x0=float(x0), y0=float(y0), vx0=vx0, vy0=vy0,
                    t_max=t_max, n_samples=n_samples, kappa=1.0,
                )
                final_r = float(np.max(np.sqrt(traj["x"] ** 2 + traj["y"] ** 2)))
                # Check if path enters CTC by sampling has_local_ctc at intervals
                ctc_hit = False
                for i in range(0, len(traj["x"]), max(1, len(traj["x"]) // 20)):
                    if self.has_local_ctc(float(traj["x"][i]), float(traj["y"][i])):
                        ctc_hit = True
                        break
                results.append({
                    "x0": float(x0), "y0": float(y0),
                    "reaches_escape": bool(final_r >= escape_radius),
                    "final_radius": final_r,
                    "enters_ctc": ctc_hit,
                    "success": True,
                })
            except Exception as exc:
                results.append({
                    "x0": float(x0), "y0": float(y0),
                    "reaches_escape": False,
                    "final_radius": float("nan"),
                    "enters_ctc": False,
                    "success": False,
                    "error": str(exc),
                })
        return results
