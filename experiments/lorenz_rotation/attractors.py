"""A registry of dissipative chaotic flows usable as Tipler-cylinder rotation laws.

The headline result (rotating_dust_lorenz.py) showed that the *only* place a strange
attractor can live in this system is the dissipative matter sector, and that the lowest
modal truncation of a differentially-rotating dust column **is** the Lorenz system. That
construct -- a chaotic rotation a(t) driving a time-varying CTC structure -- is generic:
*any* dissipative chaotic flow can serve as the rotation law. This module collects several
and exposes a uniform interface so the same diagnostics (Lyapunov spectrum, correlation
dimension) and the same CTC bridge apply to all of them.

Honesty note on physical pedigree
----------------------------------
Only **Lorenz** carries the first-principles Saltzman derivation from rotating-dust
convection (sigma = nu/kappa, r = Ra/Ra_c, b = 4/(1+a^2)). Rossler, Chen and Halvorsen are
**phenomenological** alternative rotation laws included to test the *universality* of the
construct (does the CTC bridge / catcher onset work for any dissipative attractor?), not
because they are derived from dust physics. They are labelled accordingly.

Each flow exposes:  rhs(t, s), jacobian(s), integrate(...), name, default_s0,
t_transient, divergence (constant float or NaN if state-dependent), mean_divergence(traj),
and a bifurcation descriptor (param_name, sub_value, chaos_value) for the onset scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp


class ChaoticFlow:
    """Base class: shared integrator + numerical divergence helpers."""

    name: str = "flow"
    default_s0: np.ndarray = np.array([1.0, 1.0, 1.0])
    t_transient: float = 30.0
    physical: bool = False  # True only for the derived rotating-dust Lorenz
    bifurcation: dict = {}  # {param, sub_value, chaos_value, scan}

    def rhs(self, t: float, s: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def jacobian(self, s: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @property
    def divergence(self) -> float:
        """Constant divergence if the flow has one; else NaN (state-dependent)."""
        return float("nan")

    def mean_divergence(self, traj: np.ndarray) -> float:
        """Time-average of tr(J) along a trajectory. Equals sum of Lyapunov
        exponents for an ergodic flow (exact theorem) -- the right comparison
        for state-dependent-divergence systems."""
        return float(np.mean([np.trace(self.jacobian(s)) for s in traj]))

    def integrate(
        self, s0: np.ndarray, t_max: float, dt: float = 0.01,
        rtol: float = 1e-9, atol: float = 1e-12, t_transient: float = 0.0,
    ) -> dict:
        t_eval = np.arange(0.0, t_max + dt, dt)
        t_eval = t_eval[t_eval <= t_max]
        if t_eval.size == 0 or t_eval[-1] < t_max:
            t_eval = np.append(t_eval, t_max)
        with np.errstate(over="ignore", invalid="ignore"):
            sol = solve_ivp(self.rhs, (0.0, t_max), np.asarray(s0, dtype=float),
                            t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
        t, s = sol.t, sol.y.T
        # Robust to unbounded parameter regimes: keep only the finite prefix and
        # flag divergence instead of crashing the caller (e.g. an onset scan).
        finite = np.all(np.isfinite(s), axis=1)
        diverged = (not sol.success) or (not finite.all())
        if diverged:
            cut = int(np.argmin(finite)) if not finite.all() else len(s)
            t, s = t[:cut], s[:cut]
        if t_transient > 0.0 and len(t):
            keep = t >= t_transient
            t, s = t[keep], s[keep]
        return {"t": t, "s": s, "diverged": bool(diverged)}


@dataclass
class RosslerFlow(ChaoticFlow):
    """Rossler (1976) -- single-scroll, period-doubling route to chaos.
    Phenomenological rotation law (not dust-derived)."""

    a: float = 0.2
    b: float = 0.2
    c: float = 5.7

    def __post_init__(self):
        self.name = "Rossler"
        self.default_s0 = np.array([1.0, 1.0, 1.0])
        self.t_transient = 100.0
        self.physical = False
        # c is the canonical bifurcation knob (period-doubling cascade -> chaos)
        self.bifurcation = {"param": "c", "sub_value": 3.0, "chaos_value": 5.7,
                            "scan": np.linspace(2.5, 6.0, 24)}

    def rhs(self, t, s):
        x, y, z = s
        return np.array([-y - z, x + self.a * y, self.b + z * (x - self.c)])

    def jacobian(self, s):
        x, y, z = s
        return np.array([[0.0, -1.0, -1.0],
                         [1.0, self.a, 0.0],
                         [z, 0.0, x - self.c]])

    def with_param(self, value):
        return RosslerFlow(a=self.a, b=self.b, c=float(value))


@dataclass
class ChenFlow(ChaoticFlow):
    """Chen (1999) -- a topological dual of Lorenz, double-scroll.
    Phenomenological rotation law (not dust-derived)."""

    a: float = 35.0
    b: float = 3.0
    c: float = 28.0

    def __post_init__(self):
        self.name = "Chen"
        self.default_s0 = np.array([-0.1, 0.5, -0.6])
        self.t_transient = 30.0
        self.physical = False
        self.bifurcation = {"param": "c", "sub_value": 20.0, "chaos_value": 28.0,
                            "scan": np.linspace(18.0, 30.0, 24)}

    @property
    def divergence(self) -> float:
        return float(-self.a + self.c - self.b)  # constant

    def rhs(self, t, s):
        x, y, z = s
        return np.array([self.a * (y - x),
                         (self.c - self.a) * x - x * z + self.c * y,
                         x * y - self.b * z])

    def jacobian(self, s):
        x, y, z = s
        return np.array([[-self.a, self.a, 0.0],
                         [self.c - self.a - z, self.c, -x],
                         [y, x, -self.b]])

    def with_param(self, value):
        return ChenFlow(a=self.a, b=self.b, c=float(value))


@dataclass
class HalvorsenFlow(ChaoticFlow):
    """Halvorsen cyclically-symmetric attractor.
    Phenomenological rotation law (not dust-derived)."""

    a: float = 1.4

    def __post_init__(self):
        self.name = "Halvorsen"
        self.default_s0 = np.array([-5.0, 0.0, 0.0])
        self.t_transient = 30.0
        self.physical = False
        # Halvorsen is only bounded for a in roughly [1.1, 1.7]; below that the
        # flow diverges (not an attractor). Scan within the bounded regime.
        self.bifurcation = {"param": "a", "sub_value": 1.15, "chaos_value": 1.4,
                            "scan": np.linspace(1.1, 1.7, 24)}

    @property
    def divergence(self) -> float:
        return float(-3.0 * self.a)  # constant

    def rhs(self, t, s):
        x, y, z = s
        return np.array([
            -self.a * x - 4.0 * y - 4.0 * z - y * y,
            -self.a * y - 4.0 * z - 4.0 * x - z * z,
            -self.a * z - 4.0 * x - 4.0 * y - x * x,
        ])

    def jacobian(self, s):
        x, y, z = s
        return np.array([[-self.a, -4.0 - 2.0 * y, -4.0],
                         [-4.0, -self.a, -4.0 - 2.0 * z],
                         [-4.0 - 2.0 * x, -4.0, -self.a]])

    def with_param(self, value):
        return HalvorsenFlow(a=float(value))


class LorenzFlow(ChaoticFlow):
    """Registry wrapper for the physically-derived rotating-dust Lorenz system.

    Delegates the dynamics to a (frozen) RotatingDustLorenz `core`, so the
    Saltzman derivation and parameter semantics live in one place while this
    object carries the registry metadata used by the suite and onset scan.
    """

    def __init__(self, sigma: float = 10.0, r: float = 28.0, b: float = 8.0 / 3.0):
        from rotating_dust_lorenz import RotatingDustLorenz

        self.core = RotatingDustLorenz(sigma=sigma, r=r, b=b)
        self.name = "Lorenz (rotating-dust)"
        self.default_s0 = np.array([1.0, 1.0, 1.0])
        self.t_transient = 20.0
        self.physical = True
        self.bifurcation = {"param": "r", "sub_value": 15.0, "chaos_value": 28.0,
                            "scan": np.linspace(0.5, 40.0, 24)}

    def rhs(self, t, s):
        return self.core.rhs(t, s)

    def jacobian(self, s):
        return self.core.jacobian(s)

    @property
    def divergence(self) -> float:
        return self.core.divergence

    def with_param(self, value):
        return LorenzFlow(sigma=self.core.sigma, r=float(value), b=self.core.b)


def lorenz_flow() -> "LorenzFlow":
    """Return the physically-derived rotating-dust Lorenz flow (canonical chaos)."""
    return LorenzFlow()


def registry() -> list:
    """All chaotic rotation laws, Lorenz first (the derived one)."""
    return [lorenz_flow(), RosslerFlow(), ChenFlow(), HalvorsenFlow()]
