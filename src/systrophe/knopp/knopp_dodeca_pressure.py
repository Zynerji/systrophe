"""Pressure and gradient fields inside the saturated horn torus.

Third module of the dodeca-resonator series (alignment -> first principles ->
pressure). Question: given the full-spectrum standing-wave field that
saturates the tube interior at face-lock, what pressures and gradients does
it exert WITHIN the torus volume -- and how do they scale with the
rising-frequency feedback?

Derivation (model units, coherent saturated field)
--------------------------------------------------
The interior field is an incoherent superposition of standing modes along
the 6 dodecahedron face axes a_j (the icosahedral interference pattern the
demo renders):

    psi_j(r, t) = A_j cos(k a_j . r + chi_j) cos(omega_j t)

For each standing mode the time-averaged potential and kinetic energy
densities are proportional to cos^2(k a_j . r) and sin^2(k a_j . r). Two
standard results follow:

1. LANGEVIN RADIATION PRESSURE on a rigid boundary is the local
   time-averaged energy density: in these units
       I(r) = sum_j (A_j^2 / 2) cos^2(k a_j . r + chi_j)
   evaluated at the wall. Mean interior pressure has the ceiling
   I_ceiling = sum_j A_j^2 / 2, set by saturation -- it CANNOT grow past
   full spectrum.

2. The GOR'KOV-FORM ACOUSTIC POTENTIAL for gradient forces on test matter
   is the difference of potential and kinetic densities, and cos^2 - sin^2
   collapses to a half-wavelength lattice:
       U(r)      = sum_j (A_j^2 / 2) cos(2 (k a_j . r + chi_j))
       grad U(r) = - sum_j A_j^2 k a_j sin(2 (k a_j . r + chi_j))
   Both are ANALYTIC -- no numerics needed for the field itself.

Scaling laws (derived, then verified numerically here):
   - |grad U|_max  ~ k * sum_j A_j^2      -> gradients grow LINEARLY in k
   - trap stiffness U'' at a node ~ k^2   -> traps tighten QUADRATICALLY
   while the pressure ceiling is k-independent. So once the interior is
   saturated, the rising-frequency feedback loop keeps paying off in
   GRADIENTS and TRAP STIFFNESS, not in pressure -- the operationally
   meaningful output of the "increasing in frequency" hypothesis step.

Directionality carries over from first-principles D5: the wall-pressure
dipole along the heading is sat * eps (the C_5/icosahedral field alone has
no m = 1 moment); verified here on the actual wall distribution.

Caveats: same as the rest of the series -- coherent-pump proxy in model
units, no SI claim, no vacuum free lunch. The catcher runs on the k-scan
of the maximum gradient (mandatory).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.knopp.knopp_dodeca_alignment import (
    DEFAULT_R,
    FACE_LOCK_DEG,
    alignment_state,
    dodecahedron_face_axes,
    feedback_equilibrium,
    horn_torus_sdf,
    sweep_rotation,
)

#: demo parity: interior wavenumber rises with saturation, kw = 14 + 24 sat
K_BASE, K_RISE = 14.0, 24.0


def wavenumber(sat: float) -> float:
    """Interior mode wavenumber at a given saturation (demo parity)."""
    if not 0.0 <= sat <= 1.0:
        raise ValueError("sat must be in [0, 1]")
    return K_BASE + K_RISE * sat


def axis_amplitudes(beta_deg: float, Q: float = 60.0) -> tuple[np.ndarray, float]:
    """Map the 24-mode comb onto the 6 face axes (4 modes per axis).

    Returns (A_j^2 array of 6, saturation). Face-lock drives all axes near
    full amplitude; point-lock leaves each axis with one comb line plus weak
    background.
    """
    fb = feedback_equilibrium(beta_deg, Q=Q)
    amps = fb.saturation.amplitudes
    a2 = np.array([float(np.mean(amps[4 * j:4 * j + 4] ** 2))
                   for j in range(6)])
    return a2, fb.saturation.saturation


def field_axes(beta_deg: float) -> np.ndarray:
    """The 6 face axes in the world frame at sweep angle beta."""
    return dodecahedron_face_axes() @ sweep_rotation(beta_deg).T


def radiation_pressure(points: np.ndarray, axes: np.ndarray,
                       a2: np.ndarray, k: float) -> np.ndarray:
    """Langevin pressure proxy I(r) = sum_j (A_j^2/2) cos^2(k a_j . r)."""
    proj = np.asarray(points, dtype=float) @ np.asarray(axes, dtype=float).T
    return np.cos(k * proj) ** 2 @ (np.asarray(a2) / 2.0)


def gorkov_potential(points: np.ndarray, axes: np.ndarray,
                     a2: np.ndarray, k: float) -> np.ndarray:
    """U(r) = sum_j (A_j^2/2) cos(2 k a_j . r)."""
    proj = np.asarray(points, dtype=float) @ np.asarray(axes, dtype=float).T
    return np.cos(2.0 * k * proj) @ (np.asarray(a2) / 2.0)


def gorkov_gradient(points: np.ndarray, axes: np.ndarray,
                    a2: np.ndarray, k: float) -> np.ndarray:
    """grad U(r) = - sum_j A_j^2 k a_j sin(2 k a_j . r) -- exact."""
    p = np.asarray(points, dtype=float)
    ax = np.asarray(axes, dtype=float)
    proj = p @ ax.T
    s = np.sin(2.0 * k * proj) * np.asarray(a2)[None, :] * k
    return -s @ ax


def trap_stiffness(a2: np.ndarray, k: float) -> float:
    """Curvature of U at a potential minimum along one axis: 2 A_j^2 k^2.

    From U_j = (A^2/2) cos(2k x): U'' = -2 A^2 k^2 cos(2k x), so at the
    minimum (cos = -1) the restoring stiffness is +2 A^2 k^2 -- quadratic
    in the feedback frequency. Summed over axes.
    """
    return float(2.0 * np.sum(a2) * k ** 2)


def tube_interior_points(R: float = DEFAULT_R, n: int = 40,
                         extent: float = 1.35) -> np.ndarray:
    """Regular grid restricted to the tube interior (horn-torus SDF < 0)."""
    g = np.linspace(-extent, extent, n)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return pts[horn_torus_sdf(pts, R) < 0.0]


def wall_points(R: float = DEFAULT_R, n: int = 48, shell: float = 0.04) -> np.ndarray:
    """Interior points within `shell` of the wall (where pressure pushes)."""
    g = np.linspace(-1.35, 1.35, n)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    d = horn_torus_sdf(pts, R)
    return pts[(d < 0.0) & (d > -shell)]


def wall_pressure_dipole(beta_deg: float, eps: float, theta0: float = 0.0,
                         Q: float = 60.0, R: float = DEFAULT_R) -> float:
    """m=1 moment of wall pressure along the heading.

    The icosahedral field is modulated by the horn-twist steering lobe
    (1 + eps cos(phi - theta0)); the dipole of the unmodulated field
    vanishes (D5), so the result is ~ sat * eps * <I>_wall.
    """
    if eps < 0:
        raise ValueError("eps must be >= 0")
    w = wall_points(R)
    a2, sat = axis_amplitudes(beta_deg, Q)
    I = radiation_pressure(w, field_axes(beta_deg), a2, wavenumber(sat))
    phi = np.arctan2(w[:, 0], w[:, 2])
    lobe = 1.0 + eps * np.cos(phi - theta0)
    return float(np.mean(I * lobe * np.cos(phi - theta0)) * 2.0)


# ----- report -----------------------------------------------------------------


@dataclass(frozen=True)
class PressureReport:
    """Pressure/gradient landscape of the tube interior at one orientation."""
    beta_deg: float
    saturation: float
    k: float
    pressure_ceiling: float      # sum A_j^2 / 2 (k-independent)
    mean_interior_pressure: float
    max_interior_pressure: float
    mean_wall_pressure: float
    fill_fraction: float         # interior above half the face-lock ceiling
    max_gradient: float
    gradient_k_exponent: float   # fitted: ~1 (gradients linear in k)
    stiffness_k_exponent: float  # analytic: 2 (traps quadratic in k)
    wall_dipole_eps0: float      # ~0: saturated field alone is directionless
    wall_dipole: float           # ~ sat * eps * <I>_wall
    catcher_verdict: str


def pressure_report(
    beta_deg: float = FACE_LOCK_DEG, Q: float = 60.0, eps: float = 0.22,
    R: float = DEFAULT_R, n_grid: int = 40,
) -> PressureReport:
    """Derive the interior pressure/gradient landscape; catcher on the k-scan."""
    a2, sat = axis_amplitudes(beta_deg, Q)
    axes = field_axes(beta_deg)
    k = wavenumber(sat)
    pts = tube_interior_points(R, n_grid)
    wall = wall_points(R)

    I = radiation_pressure(pts, axes, a2, k)
    I_wall = radiation_pressure(wall, axes, a2, k)
    grad = gorkov_gradient(pts, axes, a2, k)
    gmag = np.linalg.norm(grad, axis=1)

    # ceiling reference is the FACE-LOCK ceiling so fill fractions compare
    a2_face, _ = axis_amplitudes(FACE_LOCK_DEG, Q)
    ceiling_face = float(np.sum(a2_face) / 2.0)

    # k-scan: max gradient vs k (expect exponent 1); catcher on the scan
    ks = np.linspace(K_BASE, K_BASE + K_RISE, 12)
    gmax_k = np.array([
        float(np.max(np.linalg.norm(gorkov_gradient(pts, axes, a2, kk), axis=1)))
        for kk in ks])
    g_exp = float(np.polyfit(np.log(ks), np.log(gmax_k), 1)[0])
    s_exp = float(np.polyfit(
        np.log(ks), np.log([trap_stiffness(a2, kk) for kk in ks]), 1)[0])
    table = dict(zip(ks, gmax_k))

    def fn(kk: float) -> np.ndarray:
        key = min(table, key=lambda x: abs(x - kk))
        return np.array([table[key]])

    catch = scan_novelty(ks, fn, n_bits=32, parameter_label="wavenumber_k")

    return PressureReport(
        beta_deg=float(beta_deg),
        saturation=float(sat),
        k=float(k),
        pressure_ceiling=float(np.sum(a2) / 2.0),
        mean_interior_pressure=float(np.mean(I)),
        max_interior_pressure=float(np.max(I)),
        mean_wall_pressure=float(np.mean(I_wall)),
        fill_fraction=float(np.mean(I > 0.5 * ceiling_face)),
        max_gradient=float(np.max(gmag)),
        gradient_k_exponent=g_exp,
        stiffness_k_exponent=s_exp,
        wall_dipole_eps0=wall_pressure_dipole(beta_deg, 0.0, Q=Q, R=R),
        wall_dipole=wall_pressure_dipole(beta_deg, eps, Q=Q, R=R),
        catcher_verdict=catch.verdict,
    )


def summarise_pressure(r: PressureReport) -> str:
    """Human-readable summary."""
    lines = [
        f"Interior pressure/gradient landscape at beta = {r.beta_deg:.1f} deg "
        f"(sat {r.saturation:.2f}, k {r.k:.1f})",
        f"  pressure ceiling (sum A^2/2)  = {r.pressure_ceiling:.3f} "
        f"(k-independent)",
        f"  interior pressure mean/max    = {r.mean_interior_pressure:.3f} / "
        f"{r.max_interior_pressure:.3f}",
        f"  wall pressure mean            = {r.mean_wall_pressure:.3f}",
        f"  fill fraction (>1/2 ceiling)  = {r.fill_fraction:.3f}",
        f"  max |grad U|                  = {r.max_gradient:.2f}",
        f"  gradient ~ k^{r.gradient_k_exponent:.3f}, "
        f"stiffness ~ k^{r.stiffness_k_exponent:.3f}",
        f"  wall dipole eps=0 / eps       = {r.wall_dipole_eps0:.2e} / "
        f"{r.wall_dipole:.3f}",
        f"  catcher: {r.catcher_verdict}",
    ]
    return "\n".join(lines)
