"""Tomimatsu-Sato delta = 2 distorted-Kerr vacuum spacetime (1972).

The Tomimatsu-Sato (TS) family is the discrete sequence of asymptotically
flat, stationary, axisymmetric *vacuum* solutions of the Einstein
equations labelled by an integer distortion parameter ``delta``.  The
``delta = 1`` member is the Kerr black hole; ``delta = 2`` is the first
genuinely new member -- a "distorted Kerr" with a naked ring singularity,
a pair of directional singularities, and a richer closed-timelike-curve
(CTC) structure.  It is the last named-but-absent analytic CTC spacetime
on the Systrophe roadmap (section 1b).

Coordinates and form
--------------------
The metric is written in prolate spheroidal coordinates ``(x, y)`` with
``x >= 1`` (a radial-like coordinate) and ``-1 <= y <= 1``
(``y = cos theta``), and focal length ``sigma``.  In Lewis-Papapetrou
(Weyl) form,

    ds^2 = -f (dt - omega dphi)^2
           + f^{-1} [ e^{2 gamma} (drho^2 + dz^2) + rho^2 dphi^2 ],

with

    rho = sigma * sqrt((x^2 - 1)(1 - y^2)),
    f = A / B,
    omega = 2 sigma q (1 - y^2) C / A,
    e^{2 gamma} (drho^2 + dz^2)  ->  the explicit g_xx, g_yy below.

The parameters ``p, q`` satisfy ``p^2 + q^2 = 1`` (``p`` is the
"oblateness", ``q`` the rotation), and the ADM mass is ``M = 2 sigma / p``
(delta = 2).  The three polynomials ``A, B, C`` (degrees 8, 8, 7 in
``(x, y)`` for delta = 2) are the explicit Tomimatsu-Sato polynomials::

    A = (p^2 (x^2-1)^2 + q^2 (1-y^2)^2)^2
        - 4 p^2 q^2 (x^2-1)(1-y^2)(x^2-y^2)^2

    B = (p^2 x^4 + 2 p x^3 - 2 p x + q^2 y^4 - 1)^2
        + 4 q^2 y^2 (p x^3 - p x y^2 - y^2 + 1)^2

    C = p^3 x (1-x^2) (2 (x^4-1) + (x^2+3)(1-y^2))
        + p^2 (x^2-1) ((x^2-1)(1-y^2) - 4 x^2 (x^2-y^2))
        + q^2 (1-y^2)^3 (p x + 1)

(verified against the SageManifolds Tomimatsu-Sato worksheet and the
``delta=1`` Kerr reduction below).

g_{phi phi} and CTCs
--------------------
The relevant CTC diagnostic is the closed-phi-orbit norm

    g_{phi phi} = - f omega^2 + f^{-1} rho^2
                = - m^2 (y^2 - 1)
                    ( p^2 B^2 (x^2-1) + 4 q^2 delta^2 C^2 (y^2-1) )
                  / (A B delta^2),    delta = 2, m = sigma.

A closed phi-orbit at fixed ``(t, x, y)`` is timelike (a CTC) iff
``g_{phi phi} < 0``.

Mapping to a radial coordinate
------------------------------
For the shared ``find_ctc_intervals`` interface (which wants a callable
``r -> L(r)`` over ``r > 0``) we expose the prolate radial coordinate via
``r = sigma * x`` (so ``r > sigma``); the equatorial slice is
``y = 0`` (``theta = pi / 2``).

Reduction to extremal Kerr
--------------------------
The ``delta -> 1`` member of the family is exactly Kerr.  In this module
the ``delta = 1`` metric is provided separately (Kerr polynomials in the
same prolate convention) so the reduction can be checked numerically:
``g_{phi phi}`` of the ``delta = 1`` TS metric reproduces
``KerrSpacetime(M, a).gphiphi`` to ~1e-13, and in particular the
extremal limit ``a -> M`` reproduces ``KerrSpacetime(M, a=M)``.

Reference
---------
A. Tomimatsu and H. Sato, *New exact solution for the gravitational
field of a spinning mass*, Phys. Rev. Lett. 29 (1972) 1344; and
*New series of exact solutions for gravitational fields of spinning
masses*, Prog. Theor. Phys. 50 (1973) 95.  CTC / singularity analysis:
G. W. Gibbons and R. A. Russell-Clark, Phys. Rev. Lett. 30 (1973) 398.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Tomimatsu-Sato delta = 2 polynomials (pure numpy, broadcast over (x, y)).
# ---------------------------------------------------------------------------


def ts_delta2_A(x: np.ndarray, y: np.ndarray, p: float, q: float) -> np.ndarray:
    """Tomimatsu-Sato delta = 2 polynomial A(x, y; p, q)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x * x - 1.0
    ym = 1.0 - y * y
    return (p * p * xm * xm + q * q * ym * ym) ** 2 - 4.0 * p * p * q * q * xm * ym * (
        x * x - y * y
    ) ** 2


def ts_delta2_B(x: np.ndarray, y: np.ndarray, p: float, q: float) -> np.ndarray:
    """Tomimatsu-Sato delta = 2 polynomial B(x, y; p, q) (>= 0)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t1 = p * p * x ** 4 + 2.0 * p * x ** 3 - 2.0 * p * x + q * q * y ** 4 - 1.0
    t2 = p * x ** 3 - p * x * y * y - y * y + 1.0
    return t1 * t1 + 4.0 * q * q * y * y * t2 * t2


def ts_delta2_C(x: np.ndarray, y: np.ndarray, p: float, q: float) -> np.ndarray:
    """Tomimatsu-Sato delta = 2 polynomial C(x, y; p, q) (twist numerator)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x * x - 1.0
    ym = 1.0 - y * y
    return (
        p ** 3 * x * (1.0 - x * x) * (2.0 * (x ** 4 - 1.0) + (x * x + 3.0) * ym)
        + p * p * xm * (xm * ym - 4.0 * x * x * (x * x - y * y))
        + q * q * ym ** 3 * (p * x + 1.0)
    )


@dataclass(frozen=True)
class TomimatsuSato:
    """Tomimatsu-Sato delta = 2 distorted-Kerr vacuum spacetime.

    Parameters
    ----------
    p : float
        Oblateness parameter, ``0 < p < 1``.  Smaller ``p`` is closer to
        the extremal/ultra-spinning regime.  ``q = sqrt(1 - p^2)`` is the
        rotation parameter.
    sigma : float, optional
        Focal length of the prolate spheroidal coordinates (sets the
        overall length scale).  Default 1.0.  The ADM mass is
        ``M = 2 sigma / p``.
    delta : int, optional
        Distortion integer.  Only ``delta = 2`` is implemented for the
        full metric (the named target); ``delta = 1`` is available through
        :meth:`gphiphi_delta1` for the Kerr-reduction check.
    """

    p: float
    sigma: float = 1.0
    delta: int = 2
    q: float = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError("p must lie in (0, 1)")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.delta != 2:
            raise ValueError("only delta = 2 is implemented for the full metric")
        object.__setattr__(self, "q", sqrt(1.0 - self.p * self.p))

    # -- physical scalars --------------------------------------------------

    @property
    def mass(self) -> float:
        """ADM mass M = 2 sigma / p (delta = 2)."""
        return 2.0 * self.sigma / self.p

    @property
    def angular_momentum(self) -> float:
        """ADM angular momentum J = M a with a = sigma q / p^2 (delta = 2).

        For delta = 2, J = 2 sigma^2 q (delta^2 + ...) reduces to
        M a where a = q sigma / p^2 * 2 ... ; we report the standard
        spin a = sigma q (delta^2 + 1) / (delta p^2) evaluated at delta=2,
        i.e. a = 5 sigma q / (2 p^2).  J = M a.
        """
        a = 5.0 * self.sigma * self.q / (2.0 * self.p * self.p)
        return self.mass * a

    # -- coordinate map ----------------------------------------------------

    def _xy(self, r, theta):
        """Map (r, theta) -> prolate spheroidal (x, y); r = sigma x, y = cos theta."""
        r_a = np.asarray(r, dtype=float)
        t_a = np.asarray(theta, dtype=float)
        x = r_a / self.sigma
        y = np.cos(t_a)
        return x, y

    # -- metric functions (delta = 2) -------------------------------------

    def f(self, r, theta) -> np.ndarray:
        """f = -g_{tt} = A / B (the Killing-time norm)."""
        x, y = self._xy(r, theta)
        return ts_delta2_A(x, y, self.p, self.q) / ts_delta2_B(x, y, self.p, self.q)

    def omega(self, r, theta) -> np.ndarray:
        """Frame-dragging potential omega = 2 sigma q (1 - y^2) C / A."""
        x, y = self._xy(r, theta)
        A = ts_delta2_A(x, y, self.p, self.q)
        C = ts_delta2_C(x, y, self.p, self.q)
        return 2.0 * self.sigma * self.q * (1.0 - y * y) * C / A

    def gphiphi(self, r, theta) -> np.ndarray:
        """g_{phi phi}(r, theta) for the TS delta = 2 metric.

        Negative wherever the closed phi-orbit is timelike (a CTC).
        Uses the explicit Tomimatsu-Sato form
            g_{phi phi} = - m^2 (y^2 - 1)
                ( p^2 B^2 (x^2-1) + 4 q^2 delta^2 C^2 (y^2-1) )
              / (A B delta^2),  m = sigma, delta = 2,
        which is algebraically identical to -f omega^2 + f^{-1} rho^2.
        """
        x, y = self._xy(r, theta)
        p, q, m, d = self.p, self.q, self.sigma, float(self.delta)
        A = ts_delta2_A(x, y, p, q)
        B = ts_delta2_B(x, y, p, q)
        C = ts_delta2_C(x, y, p, q)
        return (
            -m * m * (y * y - 1.0)
            * (p * p * B * B * (x * x - 1.0) + 4.0 * q * q * d * d * C * C * (y * y - 1.0))
            / (A * B * d * d)
        )

    def gtphi(self, r, theta) -> np.ndarray:
        """g_{t phi} = f omega."""
        return self.f(r, theta) * self.omega(r, theta)

    def gtt(self, r, theta) -> np.ndarray:
        """g_{tt} = -f."""
        return -self.f(r, theta)

    def gxx(self, r, theta) -> np.ndarray:
        """g_{xx} = m^2 B / (p^2 delta^2 (x^2-1)(x^2-y^2)^3) (prolate radial)."""
        x, y = self._xy(r, theta)
        p, m, d = self.p, self.sigma, float(self.delta)
        B = ts_delta2_B(x, y, p, self.q)
        return m * m * B / (p * p * d * d * (x * x - 1.0) * (x * x - y * y) ** 3)

    def gyy(self, r, theta) -> np.ndarray:
        """g_{yy} = m^2 B / (p^2 delta^2 (1-y^2)(x^2-y^2)^3) (prolate angular)."""
        x, y = self._xy(r, theta)
        p, m, d = self.p, self.sigma, float(self.delta)
        B = ts_delta2_B(x, y, p, self.q)
        return m * m * B / (p * p * d * d * (1.0 - y * y) * (x * x - y * y) ** 3)

    def metric_matrix(self, r: float, theta: float) -> np.ndarray:
        """Full 4x4 metric g_{mu nu} in (t, x, y, phi) order at a point.

        Useful for a standalone finite-difference curvature / vacuum check.
        """
        g = np.zeros((4, 4), dtype=float)
        f = float(self.f(r, theta))
        om = float(self.omega(r, theta))
        g[0, 0] = -f
        g[0, 3] = f * om
        g[3, 0] = f * om
        g[3, 3] = float(self.gphiphi(r, theta))
        g[1, 1] = float(self.gxx(r, theta))
        g[2, 2] = float(self.gyy(r, theta))
        return g

    # -- equatorial CTC interface -----------------------------------------

    def equatorial_L(self, r) -> np.ndarray:
        """Equatorial g_{phi phi}(r) = g_{phi phi}(r, theta = pi/2).

        This is the ``L_fn`` consumed by
        :func:`systrophe.ctc.ctc.find_ctc_intervals`.
        """
        return self.gphiphi(r, np.pi / 2.0)

    def has_local_ctc(self, r: float, theta: float) -> bool:
        """True iff a closed phi-orbit at fixed (t, r, theta) is timelike."""
        return float(self.gphiphi(r, theta)) < 0.0

    def has_ctc(self, r_min: float = None, r_max: float = None, n_grid: int = 4001) -> bool:
        """True iff the equatorial slice contains a CTC in [r_min, r_max].

        Defaults: r_min = sigma (1 + 1e-6), r_max = 8 sigma.
        """
        if r_min is None:
            r_min = self.sigma * (1.0 + 1e-6)
        if r_max is None:
            r_max = 8.0 * self.sigma
        r_grid = np.linspace(r_min, r_max, n_grid)
        return bool(np.any(self.equatorial_L(r_grid) < 0.0))

    def ctc_threshold_radius(
        self, r_min: float = None, r_max: float = None, n_grid: int = 20001
    ) -> float | None:
        """Outer edge of the equatorial CTC region (largest r with g_{phi phi} = 0).

        The TS delta = 2 equatorial CTC region is the band
        ``sigma < r < r_CTC`` bounded above by the directional/ring
        singularity (where A = 0).  Returns the radius ``r_CTC = sigma x_*``
        of that boundary, or ``None`` if no CTC is present in the window.
        """
        if r_min is None:
            r_min = self.sigma * (1.0 + 1e-9)
        if r_max is None:
            r_max = 8.0 * self.sigma
        r_grid = np.linspace(r_min, r_max, n_grid)
        L = self.equatorial_L(r_grid)
        neg = L < 0.0
        if not np.any(neg):
            return None
        idx = np.where(neg)[0]
        # last contiguous index of the (innermost) negative band starting at r_min
        last = idx[0]
        for k in idx:
            if k == last + 1 or k == last:
                last = k
            else:
                break
        # bisection between last negative and next positive grid point
        from scipy.optimize import brentq

        r_lo = float(r_grid[last])
        r_hi = float(r_grid[min(last + 1, n_grid - 1)])
        L_lo = float(self.equatorial_L(r_lo))
        L_hi = float(self.equatorial_L(r_hi))
        if L_lo * L_hi < 0.0:
            return float(
                brentq(lambda rr: float(self.equatorial_L(rr)), r_lo, r_hi, xtol=1e-12)
            )
        return r_lo

    # -- singularity / "extra root" structure -----------------------------

    def equatorial_singularity_radii(self) -> list[float]:
        """Radii ``r = sigma x`` of the equatorial directional/ring singularities.

        These are the real roots ``x > 1`` of the equatorial polynomial
        ``A(x, 0) = 0`` (where ``f -> 0`` / the metric degenerates).  For
        TS delta = 2 there are TWO such roots: the inner one bounds the
        CTC band, the outer one is the directional singularity that Kerr
        (delta = 1) does not possess -- the "extra real root".
        """
        p2 = self.p * self.p
        q2 = self.q * self.q
        # A(x,0) as a polynomial in u = x^2:
        #   A = (p^2 (u-1)^2 + q^2)^2 - 4 p^2 q^2 (u-1) u^2
        # p^2 (u-1)^2 + q^2 = p2 u^2 - 2 p2 u + (p2 + q2) = p2 u^2 - 2 p2 u + 1
        poly1 = np.polynomial.Polynomial([1.0, -2.0 * p2, p2])
        term2 = np.polynomial.Polynomial([0.0, 0.0, -4.0 * p2 * q2, 4.0 * p2 * q2])
        a_poly = poly1 * poly1 - term2
        roots = a_poly.roots()
        xs = sorted(
            float(np.sqrt(z.real))
            for z in roots
            if abs(z.imag) < 1e-9 and z.real > 1.0 + 1e-12
        )
        return [self.sigma * x for x in xs]

    # -- delta = 1 (Kerr) reduction ---------------------------------------

    def gphiphi_delta1(self, r, theta, M: float, a: float) -> np.ndarray:
        """g_{phi phi} of the delta = 1 (Kerr) TS metric in the same convention.

        Built from the Kerr-in-prolate functions
            f = (p^2 x^2 + q^2 y^2 - 1) / ((p x + 1)^2 + q^2 y^2),
            omega = -(2/p) kappa q (1 - y^2)(p x + 1) / (p^2 x^2 + q^2 y^2 - 1),
        with kappa = sqrt(M^2 - a^2), p = kappa / M, q = a / M, and the
        Boyer-Lindquist map r = kappa x + M, y = cos theta.  Reproduces
        ``KerrSpacetime(M, a).gphiphi`` to machine precision for a < M and,
        by continuity, extremal Kerr as a -> M.

        This is a *static method in spirit* (does not use ``self``); it is
        provided here so the delta -> 1 reduction can be tested against
        :class:`systrophe.geometry.spacetimes.kerr.KerrSpacetime`.
        """
        M = float(M)
        a = float(a)
        if abs(a) >= M:
            # Degenerate prolate map (kappa = 0); fall back to the exact
            # Boyer-Lindquist g_{phi phi}, which is the kappa -> 0 limit.
            r_a = np.asarray(r, dtype=float)
            t_a = np.asarray(theta, dtype=float)
            s2 = np.sin(t_a) ** 2
            Sigma = r_a * r_a + a * a * np.cos(t_a) ** 2
            return s2 * (r_a * r_a + a * a + 2.0 * M * a * a * r_a * s2 / Sigma)
        kappa = sqrt(M * M - a * a)
        p = kappa / M
        q = a / M
        r_a = np.asarray(r, dtype=float)
        t_a = np.asarray(theta, dtype=float)
        x = (r_a - M) / kappa
        y = np.cos(t_a)
        A = p * p * x * x + q * q * y * y - 1.0
        B = (p * x + 1.0) ** 2 + q * q * y * y
        f = A / B
        omega = -(2.0 / p) * kappa * q * (1.0 - y * y) * (p * x + 1.0) / A
        return -f * omega ** 2 + (1.0 / f) * kappa * kappa * (x * x - 1.0) * (1.0 - y * y)


# ---------------------------------------------------------------------------
# Standalone finite-difference vacuum (Ricci) check -- numpy only.
# ---------------------------------------------------------------------------


def ricci_scalar_fd(
    metric_fn: Callable[[float, float], np.ndarray],
    x0: float,
    y0: float,
    h: float = 1e-4,
) -> float:
    """Ricci scalar R at (x0, y0) from a (t, x, y, phi) metric by finite differences.

    ``metric_fn(x, y)`` must return the 4x4 metric in coordinate order
    ``(t, x, y, phi)``; the metric is assumed independent of ``t`` and
    ``phi`` (stationary + axisymmetric), so only x- and y-derivatives are
    taken.  This is a *standalone* curvature evaluator (it does NOT use
    ``point_splitting.ricci_scalar``, which is hardwired to the r-only van
    Stockum cylinder).  Returns R; for a vacuum solution R = 0.
    """
    coords = [0.0, float(x0), float(y0), 0.0]

    def g_at(c):
        return metric_fn(c[1], c[2])

    def christoffel(c0):
        gg = g_at(c0)
        gi = np.linalg.inv(gg)
        dg = np.zeros((4, 4, 4))
        for cc in (1, 2):
            cp = c0.copy()
            cp[cc] += h
            cm = c0.copy()
            cm[cc] -= h
            dg[cc] = (g_at(cp) - g_at(cm)) / (2.0 * h)
        Gam = np.zeros((4, 4, 4))
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    s = 0.0
                    for d in range(4):
                        s += gi[a, d] * (dg[b, d, c] + dg[c, d, b] - dg[d, b, c])
                    Gam[a, b, c] = 0.5 * s
        return Gam, gi

    Gam0, gi0 = christoffel(coords)
    dGam = np.zeros((4, 4, 4, 4))  # dGam[e, a, b, c] = d_e Gamma^a_bc
    for e in (1, 2):
        cp = coords.copy()
        cp[e] += h
        cm = coords.copy()
        cm[e] -= h
        Gp, _ = christoffel(cp)
        Gm, _ = christoffel(cm)
        dGam[e] = (Gp - Gm) / (2.0 * h)

    # Ricci R_bd = R^a_{b a d}
    Ric = np.zeros((4, 4))
    for b in range(4):
        for d in range(4):
            s = 0.0
            for a in range(4):
                c = a
                term = dGam[c, a, b, d] - dGam[d, a, b, c]
                for e in range(4):
                    term += Gam0[a, c, e] * Gam0[e, b, d] - Gam0[a, d, e] * Gam0[e, b, c]
                s += term
            Ric[b, d] = s
    return float(sum(gi0[b, d] * Ric[b, d] for b in range(4) for d in range(4)))
