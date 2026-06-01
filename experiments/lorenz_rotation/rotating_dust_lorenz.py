"""First-principles Lorenz-class rotation for a van Stockum / Tipler dust column.

Motivation
----------
A *single rigidly-rotating* van Stockum cylinder is stationary, axisymmetric and
z-independent, so its geodesics carry four conserved quantities (E, ell, p_z,
normalisation): the motion is **integrable** and cannot be chaotic. Moreover
geodesic motion is Hamiltonian, so by Liouville's theorem phase-space volume is
conserved and a *strange attractor is impossible* in pure geodesic dynamics.

A Lorenz attractor is intrinsically **dissipative**. The only place dissipation
lives in this system is the *matter sector*: the dust column itself, once we
allow it to rotate **differentially** and to transport angular momentum / heat
through an effective viscosity. This is not a stretch — the Lorenz '63 system is
historically a three-mode Galerkin truncation of a *rotating, convecting fluid*
(Saltzman 1962), and van Stockum dust **is** a rotating self-gravitating fluid.

Derivation (Saltzman--Lorenz for a rotating self-gravitating dust annulus)
--------------------------------------------------------------------------
Work in a local meridional patch (x ~ radial, z ~ axial) of the column. Boussinesq
perturbations on a differentially-rotating background with a maintained effective-
buoyancy gradient obey, for stream function psi (u = d_z psi, w = -d_x psi) and
buoyancy field theta,

    d_t (lap psi) + J(psi, lap psi) = nu lap^2 psi + g_eff alpha d_x theta
    d_t theta     + J(psi, theta)   = kappa lap theta + (DeltaTheta / H) d_x psi

with J the Jacobian (advection). g_eff is the effective gravity (self-gravity plus
centrifugal imbalance of the differential rotation — the Solberg--Hoiland restoring
term); DeltaTheta/H is the maintained destabilising gradient. This is structurally
*identical* to Rayleigh--Benard convection. Non-dimensionalising on (H, H^2/kappa)
and retaining the Saltzman three modes

    psi   ~  X(t) sin(pi a x) sin(pi z)
    theta ~  Y(t) cos(pi a x) sin(pi z) - Z(t) sin(2 pi z)

gives the Lorenz system

    X' = sigma (Y - X)
    Y' = r X - Y - X Z
    Z' = X Y - b Z

with the parameters carrying a concrete rotating-dust meaning:

    sigma = nu / kappa                      (Prandtl analog: viscosity / transport)
    r     = Ra / Ra_c                       (drive: how supercritical the rotation is)
    b     = 4 / (1 + a^2)                   (overturning-cell aspect factor)

and Ra = g_eff alpha DeltaTheta H^3 / (nu kappa), Ra_c = pi^4 (1+a^2)^3 / a^2.
The classic chaotic point sigma=10, b=8/3, r=28 corresponds to a^2 = 1/2 (cell
aspect a = 1/sqrt(2)), viscosity ten times the angular-momentum diffusivity, and a
differential-rotation drive 28x critical.

Coupling back to the metric
---------------------------
X(t) is the amplitude of the meridional overturning circulation, which redistributes
angular momentum and so modulates the column's surface angular velocity. The minimal
linear coupling is

    a_rot(t) = a0 * (1 + eps * X(t) / X_rms),

i.e. the Tipler rotation parameter a = omega R itself wanders on the Lorenz attractor.
This is the "complex rotation like a Lorenz attractor" — induced in the (dissipative)
matter sector, and inherited *adiabatically* by the (conservative) spacetime, valid
when the Lorenz timescale far exceeds the orbital/light-crossing time of a CTC band.

References
----------
- E. N. Lorenz, J. Atmos. Sci. 20 (1963) 130.
- B. Saltzman, J. Atmos. Sci. 19 (1962) 329.
- A. Solberg / L. Hoiland rotational stability criterion (baroclinic analog).
- W. J. van Stockum, Proc. Roy. Soc. Edin. 57 (1937) 135.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------- #
# Core Lorenz-class vector field with rotating-dust parameter semantics.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RotatingDustLorenz:
    """Lorenz-class reduction of a differentially-rotating dissipative dust column.

    Parameters
    ----------
    sigma : float
        Prandtl analog nu / kappa (viscosity / angular-momentum transport).
    r : float
        Drive ratio Ra / Ra_c (how supercritical the differential rotation is).
    b : float
        Cell aspect factor 4 / (1 + a_cell^2).
    """

    sigma: float = 10.0
    r: float = 28.0
    b: float = 8.0 / 3.0

    def __post_init__(self) -> None:
        if self.sigma <= 0 or self.b <= 0:
            raise ValueError("sigma and b must be positive")

    # -- physically-motivated constructors ---------------------------------- #
    @classmethod
    def from_dust_parameters(
        cls,
        viscosity: float,
        transport: float,
        drive_supercriticality: float,
        cell_aspect: float,
    ) -> "RotatingDustLorenz":
        """Build from rotating-dust physical quantities.

        viscosity (nu), transport (kappa): give sigma = nu / kappa.
        drive_supercriticality: r = Ra / Ra_c directly.
        cell_aspect (a_cell): gives b = 4 / (1 + a_cell^2).
        """
        if transport <= 0:
            raise ValueError("transport (kappa) must be positive")
        return cls(
            sigma=viscosity / transport,
            r=drive_supercriticality,
            b=4.0 / (1.0 + cell_aspect * cell_aspect),
        )

    @property
    def cell_aspect(self) -> float:
        """Recover the overturning-cell aspect a_cell from b = 4/(1+a^2)."""
        return float(np.sqrt(max(4.0 / self.b - 1.0, 0.0)))

    @property
    def r_critical_hopf(self) -> float:
        """Hopf bifurcation drive r_H above which the symmetric fixed points
        lose stability (onset of sustained chaos for the classic sigma, b).

        r_H = sigma (sigma + b + 3) / (sigma - b - 1), valid for sigma > b + 1.
        """
        denom = self.sigma - self.b - 1.0
        if denom <= 0:
            return float("inf")
        return float(self.sigma * (self.sigma + self.b + 3.0) / denom)

    def fixed_points(self) -> dict:
        """Return the three fixed points (origin and the convecting pair C+-)."""
        out = {"origin": np.zeros(3)}
        if self.r > 1.0:
            c = float(np.sqrt(self.b * (self.r - 1.0)))
            out["C_plus"] = np.array([c, c, self.r - 1.0])
            out["C_minus"] = np.array([-c, -c, self.r - 1.0])
        return out

    # -- vector field & Jacobian -------------------------------------------- #
    def rhs(self, t: float, s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array(
            [self.sigma * (y - x), self.r * x - y - x * z, x * y - self.b * z]
        )

    def jacobian(self, s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array(
            [
                [-self.sigma, self.sigma, 0.0],
                [self.r - z, -1.0, -x],
                [y, x, -self.b],
            ]
        )

    @property
    def divergence(self) -> float:
        """Phase-space contraction rate tr(J) = -(sigma + 1 + b) < 0.

        Constant and negative: the flow is **dissipative** (volumes shrink),
        which is exactly why an attractor can exist here but never in the
        conservative geodesic sector.
        """
        return -(self.sigma + 1.0 + self.b)

    # -- integration -------------------------------------------------------- #
    def integrate(
        self,
        s0: np.ndarray,
        t_max: float,
        dt: float = 0.01,
        rtol: float = 1e-9,
        atol: float = 1e-12,
        t_transient: float = 0.0,
    ) -> dict:
        """Integrate the Lorenz-class flow.

        Returns dict with 't' and 's' (n_steps x 3). If t_transient > 0, the
        initial transient is integrated but discarded from the returned arrays.
        """
        t_eval = np.arange(0.0, t_max + dt, dt)
        t_eval = t_eval[t_eval <= t_max]
        if t_eval.size == 0 or t_eval[-1] < t_max:
            t_eval = np.append(t_eval, t_max)
        sol = solve_ivp(
            self.rhs, (0.0, t_max), np.asarray(s0, dtype=float),
            t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Lorenz integration failed: {sol.message}")
        t, s = sol.t, sol.y.T
        if t_transient > 0.0:
            keep = t >= t_transient
            t, s = t[keep], s[keep]
        return {"t": t, "s": s}


# --------------------------------------------------------------------------- #
# Diagnostics: Lyapunov spectrum, Kaplan-Yorke dimension.
# --------------------------------------------------------------------------- #
def lyapunov_spectrum(
    model: RotatingDustLorenz,
    s0: np.ndarray,
    t_max: float = 1000.0,
    dt: float = 0.005,
    renorm_every: int = 20,
    t_transient: float = 20.0,
) -> dict:
    """Full Lyapunov spectrum via the Benettin / Gram-Schmidt QR method.

    Integrates the flow together with an orthonormal frame of tangent vectors
    using a fast fixed-step RK4 on the coupled (state + tangent) system,
    re-orthonormalising every ``renorm_every`` steps and accumulating the log
    growth rates.

    Returns dict with 'exponents' (sorted desc), 'sum' (should ~= divergence),
    'kaplan_yorke_dim', and 'kolmogorov_sinai' (sum of positive exponents).
    """
    n = 3
    s = np.asarray(s0, dtype=float).copy()
    Q = np.eye(n)
    if t_transient > 0:
        s = model.integrate(s, t_transient, dt=dt)["s"][-1]

    def coupled_rhs(state, U):
        ds = model.rhs(0.0, state)
        dU = model.jacobian(state) @ U
        return ds, dU

    def rk4_step(state, U, h):
        k1s, k1U = coupled_rhs(state, U)
        k2s, k2U = coupled_rhs(state + 0.5 * h * k1s, U + 0.5 * h * k1U)
        k3s, k3U = coupled_rhs(state + 0.5 * h * k2s, U + 0.5 * h * k2U)
        k4s, k4U = coupled_rhs(state + h * k3s, U + h * k3U)
        state = state + (h / 6.0) * (k1s + 2 * k2s + 2 * k3s + k4s)
        U = U + (h / 6.0) * (k1U + 2 * k2U + 2 * k3U + k4U)
        return state, U

    n_steps = int(t_max / dt)
    log_sums = np.zeros(n)
    U = Q.copy()
    step = 0
    while step < n_steps:
        for _ in range(renorm_every):
            s, U = rk4_step(s, U, dt)
            step += 1
            if step >= n_steps:
                break
        Q, Rm = np.linalg.qr(U)
        diag = np.diag(Rm)
        signs = np.sign(diag)
        signs[signs == 0] = 1.0
        Q = Q * signs
        log_sums += np.log(np.abs(diag))
        U = Q

    exps = np.sort(log_sums / (n_steps * dt))[::-1]

    # Kaplan-Yorke dimension
    cumsum = np.cumsum(exps)
    ky = 0.0
    if exps[0] <= 0:
        ky = 0.0
    else:
        k = 0
        for j in range(n):
            if cumsum[j] >= 0:
                k = j + 1
            else:
                break
        if k >= n:
            ky = float(n)
        elif k == 0:
            ky = 0.0
        else:
            ky = k + cumsum[k - 1] / abs(exps[k])

    try:
        div = float(model.divergence)
    except Exception:
        div = float("nan")  # state-dependent divergence (e.g. Rossler) -> NaN

    return {
        "exponents": exps,
        "sum": float(np.sum(exps)),
        "divergence": div,
        "kaplan_yorke_dim": float(ky),
        "kolmogorov_sinai": float(np.sum(exps[exps > 0])),
    }


def largest_lyapunov_rk4(
    model: RotatingDustLorenz,
    s0: np.ndarray,
    t_max: float = 150.0,
    dt: float = 0.005,
    d0: float = 1e-9,
    t_transient: float = 20.0,
) -> float:
    """Fast largest-Lyapunov estimate via fixed-step RK4 two-trajectory rescaling.

    Cheap enough for parameter scans (no per-step solve_ivp). Returns the
    leading exponent in the model's time units.
    """
    def rk4(state, h):
        k1 = model.rhs(0.0, state)
        k2 = model.rhs(0.0, state + 0.5 * h * k1)
        k3 = model.rhs(0.0, state + 0.5 * h * k2)
        k4 = model.rhs(0.0, state + h * k3)
        return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    s = np.asarray(s0, dtype=float).copy()
    n_trans = int(t_transient / dt)
    n_steps = int(t_max / dt)
    log_growth = 0.0
    done = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_trans):
            s = rk4(s, dt)
            if not np.all(np.isfinite(s)):
                return float("inf")  # unbounded (non-attractor) parameter regime
        sp = s.copy()
        sp[0] += d0
        for _ in range(n_steps):
            s = rk4(s, dt)
            sp = rk4(sp, dt)
            if not (np.all(np.isfinite(s)) and np.all(np.isfinite(sp))):
                # unbounded parameter regime (not an attractor): report +inf
                return float("inf")
            diff = sp - s
            dist = float(np.linalg.norm(diff))
            if dist == 0.0:
                continue
            log_growth += np.log(dist / d0)
            sp = s + (d0 / dist) * diff
            done += 1
    return float(log_growth / (done * dt)) if done else float("nan")


def largest_lyapunov_two_trajectory(
    model: RotatingDustLorenz,
    s0: np.ndarray,
    t_max: float = 200.0,
    dt: float = 0.01,
    d0: float = 1e-8,
    t_transient: float = 20.0,
) -> float:
    """Largest Lyapunov exponent via the simple two-trajectory rescaling method.

    Cheaper, independent cross-check of the QR spectrum's leading exponent.
    """
    s = np.asarray(s0, dtype=float).copy()
    if t_transient > 0:
        s = model.integrate(s, t_transient, dt=dt)["s"][-1]
    sp = s + d0 * np.array([1.0, 0.0, 0.0])
    n_steps = int(t_max / dt)
    log_growth = 0.0
    for _ in range(n_steps):
        s = model.integrate(s, dt, dt=dt, t_transient=0.0)["s"][-1]
        sp = model.integrate(sp, dt, dt=dt, t_transient=0.0)["s"][-1]
        diff = sp - s
        dist = float(np.linalg.norm(diff))
        if dist == 0.0:
            continue
        log_growth += np.log(dist / d0)
        sp = s + (d0 / dist) * diff  # rescale back to d0 along current direction
    return float(log_growth / (n_steps * dt))


def correlation_dimension(
    model: RotatingDustLorenz,
    s0: np.ndarray,
    t_max: float = 600.0,
    dt: float = 0.01,
    t_transient: float = 50.0,
    n_points: int = 4000,
    n_eps: int = 24,
) -> dict:
    """Grassberger--Procaccia correlation dimension D2 of the attractor.

    An *independent* fractal-dimension estimate (no tangent dynamics), used to
    cross-check the Kaplan--Yorke dimension from the Lyapunov spectrum. For the
    classic Lorenz attractor D2 ~ 2.05.

    Returns dict with 'D2', the (log eps, log C) scaling points, and the fit
    window used.
    """
    from scipy.spatial.distance import pdist

    traj = model.integrate(s0, t_max, dt=dt, t_transient=t_transient)["s"]
    # subsample to a manageable point cloud; coarse spacing reduces temporal
    # correlation (a poor-man's Theiler window).
    idx = np.linspace(0, len(traj) - 1, min(n_points, len(traj))).astype(int)
    pts = traj[idx]
    d = pdist(pts)  # all pairwise Euclidean distances
    d = d[d > 0]
    dmin, dmax = np.percentile(d, 0.5), np.percentile(d, 99)
    eps = np.logspace(np.log10(dmin), np.log10(dmax), n_eps)
    n_pairs = len(d)
    C = np.array([np.count_nonzero(d < e) / n_pairs for e in eps])
    valid = C > 0
    log_eps = np.log(eps[valid])
    log_C = np.log(C[valid])
    # Select the scaling region by correlation-integral *value* (a fixed band
    # that sits below saturation and above finite-N depletion), not by index.
    # C in [3e-3, 0.15] is roughly two decades of clean power law for Lorenz.
    band = (C[valid] >= 3e-3) & (C[valid] <= 0.15)
    if band.sum() < 4:  # fall back to central index window
        lo, hi = int(0.2 * len(log_eps)), int(0.6 * len(log_eps))
        band = np.zeros(len(log_eps), dtype=bool)
        band[lo:hi] = True
    slope, intercept = np.polyfit(log_eps[band], log_C[band], 1)
    return {
        "D2": float(slope),
        "log_eps": log_eps.tolist(),
        "log_C": log_C.tolist(),
        "n_fit_points": int(band.sum()),
        "n_points": len(pts),
    }


# --------------------------------------------------------------------------- #
# Map the Lorenz state onto the Tipler rotation parameter a(t).
# --------------------------------------------------------------------------- #
def rotation_parameter_timeseries(
    traj: dict, a0: float = 1.5, eps: float = 0.2, clip_min: float = 0.55
) -> dict:
    """Map an integrated Lorenz trajectory to a Tipler rotation parameter a(t).

        a(t) = a0 * (1 + eps * X(t) / X_rms),  clipped to stay supercritical.

    Returns dict with 't', 'a', and bookkeeping. clip_min keeps a > 1/2 so the
    exterior stays Bonnor Case III (oscillatory) throughout.
    """
    t = traj["t"]
    X = traj["s"][:, 0]
    x_rms = float(np.sqrt(np.mean(X * X)))
    if x_rms == 0.0:
        x_rms = 1.0
    a = a0 * (1.0 + eps * X / x_rms)
    a_clipped = np.clip(a, clip_min, None)
    return {
        "t": t,
        "a": a_clipped,
        "a_raw": a,
        "x_rms": x_rms,
        "n_clipped": int(np.sum(a < clip_min)),
        "a_min": float(np.min(a_clipped)),
        "a_max": float(np.max(a_clipped)),
    }
