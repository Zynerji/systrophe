"""Scale materiality + emergent phenomena of the locked dodeca resonator.

Seventh module of the series. User questions 2026-06-09: "does the dodeca
scale really matter to any of this? are there any new phenomena that can be
derived from this project?"

S1. SCALE MATERIALITY AUDIT (honest answer: mostly NO).
    Face-lock saturation is 1.00 at EVERY scale from 0.06 to 0.80
    (CV = 0.000) and collapse holds until the body starts swallowing the
    tube (~0.6+). The crystal order X meanwhile swings by a factor ~20
    (CV ~ 0.7) across the same range. So the Casimir/saturation/collapse
    mechanism is SCALE-FREE and the dodeca scale matters ONLY through the
    quasicrystal registry (which rung of the sqrt(5) ladder you sit on).
    CAVEAT, stated plainly: part of the scale-freeness is a modelling
    artifact -- the PFA drive is self-calibrated against the vertex-up
    configuration at each scale, which normalises away absolute gap
    physics. The registry dependence x = k r_in s is geometric and real
    within the model; the flat saturation curve is partly by construction.

S2. ROTATIONAL FREQUENCY COMB with a C5 time-domain selection rule (NEW).
    Spin the locked lattice at Omega: a fixed interior point sees intensity
    lines at EXACT multiples of 5*Omega -- the spatial C5 selection
    (m = 0 mod 5) transfers to the time domain (rotational/orbital-
    angular-momentum sidebands with all n != 0 mod 5 cancelled by
    symmetry). Tilting the spin axis breaks C5 and demultiplies the comb
    to Omega spacing. Measured: untilted lines [5,10,15,20,25]*Omega;
    tilted 4 deg: [1,2,4,5,6,7]*Omega. A 5x comb-spacing jump is a sharp,
    in-principle-observable signature of axis alignment.

S3. CARGO IN THE CHRONOLOGY REGION. At supercritical spin the CTC band
    (rho > 1/Omega) overlaps the outer trap shells: a computable fraction
    of the lattice's cargo sites sit inside the band, and the conveyor's
    grip (Omega_max ~ 12 >> window) can carry matter into and out of it.

S4. THE LOCK HOLDS ITSELF: registry torque, holding range, hysteresis (NEW).
    If the orientation responds to a registry torque kappa * dX/dbeta, the
    multi-well landscape X(beta) (lock at 37.4 deg, secondary wells near
    54/64/74 deg) gives: holding torque ~ 1.6 kappa (lock stays within
    1 deg), max slope |dX/dbeta| ~ 13 per rad, and a quasi-static external-
    torque sweep shows ~25 deg stick-slip jumps with ~49 deg of
    forward/backward hysteresis. Crystal "locking" is not just a name:
    the registry landscape pins the orientation against perturbation.

Catcher (mandatory): on the forward torque-sweep trace beta(tau) -- the
stick-slip jumps are genuine transitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.interpolate import CubicSpline

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.knopp.knopp_dodeca_alignment import (
    FACE_LOCK_DEG,
    alignment_state,
    horn_torus_sdf,
    mode_saturation,
    normalised_drive,
)
from systrophe.knopp.knopp_dodeca_crystal import crystal_order
from systrophe.knopp.knopp_dodeca_pressure import (
    axis_amplitudes,
    gorkov_potential,
    wavenumber,
)
from systrophe.knopp.knopp_dodeca_rotation import spun_axes


# ----- S1: scale materiality ----------------------------------------------------


def scale_materiality_audit(scales: np.ndarray | None = None) -> dict:
    """Saturation/collapse vs crystal order across the scale range."""
    if scales is None:
        scales = np.linspace(0.06, 0.80, 13)
    scales = np.asarray(scales, dtype=float)
    sats, cols, xs = [], [], []
    for s in scales:
        ref = alignment_state(0.0, s)
        st = alignment_state(FACE_LOCK_DEG, s)
        d = normalised_drive(st, 2.0 * ref.drive_raw / ref.area_factor)
        sr = mode_saturation(d, st.vertex_align, st.face_align, st.spiral)
        sats.append(sr.saturation)
        cols.append(sr.collapse)
        xs.append(crystal_order(FACE_LOCK_DEG, float(s)).order)
    sats, cols, xs = map(np.array, (sats, cols, xs))
    cv = lambda a: float(np.std(a) / max(np.mean(a), 1e-12))
    return {
        "scales": scales, "saturation": sats, "collapse": cols,
        "crystal_X": xs, "cv_saturation": cv(sats), "cv_crystal_X": cv(xs),
        "mechanism_scale_free": bool(cv(sats) < 0.01),
        "registry_scale_sensitive": bool(cv(xs) > 0.3),
    }


# ----- S2: rotational comb -------------------------------------------------------


def rotational_comb(
    Omega: float = 1.0, tilt_deg: float = 0.0,
    r0: np.ndarray | None = None, n_t: int = 8192, n_periods: int = 16,
    threshold: float = 0.02,
) -> np.ndarray:
    """Significant spectral lines (rad/time) of I(r0, t) under spin.

    C5 time-domain selection: untilted -> lines at multiples of 5*Omega
    only; tilted -> Omega-spaced sidebands appear.
    """
    if Omega <= 0:
        raise ValueError("Omega must be positive")
    if r0 is None:
        r0 = np.array([0.31, 0.14, 0.05])
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    T = n_periods * 2.0 * math.pi / Omega
    t = np.linspace(0.0, T, n_t, endpoint=False)
    I = np.empty(n_t)
    for i, ti in enumerate(t):
        ax = spun_axes(math.degrees(Omega * ti), tilt_deg)
        I[i] = float(np.sum(a2 / 2.0 * np.cos(k * (ax @ r0)) ** 2))
    F = np.abs(np.fft.rfft(I - I.mean()))
    freqs = np.fft.rfftfreq(n_t, T / n_t) * 2.0 * math.pi
    return freqs[F > threshold * F.max()]


def comb_spacing(Omega: float = 1.0, tilt_deg: float = 0.0) -> float:
    """Lowest significant line: 5*Omega untilted, Omega tilted."""
    lines = rotational_comb(Omega, tilt_deg)
    return float(lines[0]) if len(lines) else 0.0


# ----- S3: cargo in the chronology region ------------------------------------------


def _trap_points(n: int = 150, extent: float = 1.35,
                 depth_cut: float = -0.3) -> np.ndarray:
    """Deep Gor'kov minima across the WHOLE tube (cargo sites)."""
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    from systrophe.knopp.knopp_dodeca_pressure import field_axes
    axes = field_axes(FACE_LOCK_DEG)
    g = np.linspace(-extent, extent, n)
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1)
    U = gorkov_potential(P.reshape(-1, 3), axes, a2, k).reshape(n, n, n)
    m = np.ones((n, n, n), bool)
    m[[0, -1], :, :] = m[:, [0, -1], :] = m[:, :, [0, -1]] = False
    for s in product((-1, 0, 1), repeat=3):
        if s == (0, 0, 0):
            continue
        m &= U < np.roll(U, s, (0, 1, 2))
    ceiling = float(np.sum(a2) / 2.0)
    deep = m & (U < depth_cut * ceiling)
    pts = P[deep]
    return pts[horn_torus_sdf(pts) < 0.0]


def ctc_cargo_fraction(Omega: float, trap_pts: np.ndarray | None = None) -> float:
    """Fraction of cargo (trap) sites inside the CTC band rho > 1/Omega."""
    if Omega <= 0:
        raise ValueError("Omega must be positive")
    if trap_pts is None:
        trap_pts = _trap_points()
    rho = np.hypot(trap_pts[:, 0], trap_pts[:, 2])
    return float(np.mean(rho > 1.0 / Omega))


# ----- S4: registry torque, holding range, hysteresis --------------------------------


def registry_torque_landscape(
    betas: np.ndarray | None = None, scale: float = 0.212,
) -> dict:
    """X(beta) landscape, its wells, and the holding slope max|dX/dbeta|."""
    if betas is None:
        betas = np.linspace(15.0, 75.0, 61)
    betas = np.asarray(betas, dtype=float)
    X = np.array([crystal_order(float(b), scale).order for b in betas])
    dX = np.gradient(X, np.radians(betas))
    peaks = [float(betas[i]) for i in range(1, len(betas) - 1)
             if X[i] > X[i - 1] and X[i] > X[i + 1]]
    return {"betas": betas, "X": X, "peaks_deg": peaks,
            "max_slope_per_rad": float(np.max(np.abs(dX)))}


def orientation_hysteresis(
    landscape: dict | None = None, tau_max: float = 2.5, n_tau: int = 81,
) -> dict:
    """Quasi-static external-torque sweep over the registry landscape.

    Orientation settles to a local maximum of X(beta) + tau*beta (unit
    registry stiffness kappa = 1). Multi-well landscape -> stick-slip
    jumps and forward/backward hysteresis; the lock holds within 1 deg
    for |tau| < holding_tau ~ max slope near the lock.
    """
    if landscape is None:
        landscape = registry_torque_landscape()
    b_rad = np.radians(landscape["betas"])
    sp = CubicSpline(b_rad, landscape["X"])
    lo, hi = float(b_rad[0]), float(b_rad[-1])

    def settle(b: float, tau: float) -> float:
        for _ in range(4000):
            b = min(max(b + 5e-4 * (sp(b, 1) + tau), lo), hi)
        return b

    taus = np.linspace(-tau_max, tau_max, n_tau)
    fwd = []
    b = math.radians(FACE_LOCK_DEG)
    for tau in taus:
        b = settle(b, float(tau))
        fwd.append(math.degrees(b))
    bwd = []
    for tau in taus[::-1]:
        b = settle(b, float(tau))
        bwd.append(math.degrees(b))
    fwd, bwd = np.array(fwd), np.array(bwd[::-1])
    near = np.abs(fwd - FACE_LOCK_DEG) < 1.0
    holding = float(np.max(taus[near])) if np.any(near) else 0.0
    return {"taus": taus, "beta_forward": fwd, "beta_backward": bwd,
            "max_separation_deg": float(np.max(np.abs(fwd - bwd))),
            "max_jump_deg": float(np.max(np.abs(np.diff(fwd)))),
            "holding_tau": holding}


# ----- report ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmergentReport:
    """Scale materiality + the new derived phenomena."""
    mechanism_scale_free: bool
    cv_saturation: float
    cv_crystal_X: float
    comb_spacing_aligned: float      # = 5 Omega (C5 time selection)
    comb_spacing_tilted: float       # = Omega (sidebands restored)
    ctc_cargo_at_1p2: float          # cargo fraction in the band at Omega=1.2
    registry_wells_deg: tuple[float, ...]
    holding_tau: float               # lock holds within 1 deg below this
    hysteresis_separation_deg: float
    stick_slip_jump_deg: float
    catcher_verdict: str


def emergent_report() -> EmergentReport:
    """Compute everything; catcher on the stick-slip torque sweep."""
    audit = scale_materiality_audit()
    land = registry_torque_landscape()
    hyst = orientation_hysteresis(land)

    table = dict(zip(hyst["taus"], hyst["beta_forward"]))

    def fn(tau: float) -> np.ndarray:
        key = min(table, key=lambda v: abs(v - tau))
        return np.array([table[key]])

    catch = scan_novelty(hyst["taus"], fn, n_bits=32,
                         parameter_label="external_torque_tau")
    return EmergentReport(
        mechanism_scale_free=audit["mechanism_scale_free"],
        cv_saturation=audit["cv_saturation"],
        cv_crystal_X=audit["cv_crystal_X"],
        comb_spacing_aligned=comb_spacing(1.0, 0.0),
        comb_spacing_tilted=comb_spacing(1.0, 4.0),
        ctc_cargo_at_1p2=ctc_cargo_fraction(1.2),
        registry_wells_deg=tuple(land["peaks_deg"]),
        holding_tau=hyst["holding_tau"],
        hysteresis_separation_deg=hyst["max_separation_deg"],
        stick_slip_jump_deg=hyst["max_jump_deg"],
        catcher_verdict=catch.verdict,
    )


def summarise_emergent(r: EmergentReport) -> str:
    """Human-readable summary."""
    lines = [
        "Scale materiality + emergent phenomena",
        f"  S1 scale: mechanism scale-free {r.mechanism_scale_free} "
        f"(CV sat {r.cv_saturation:.3f} vs CV X {r.cv_crystal_X:.3f}) -- "
        f"scale matters only through the registry",
        f"  S2 rotational comb: spacing {r.comb_spacing_aligned:.2f} Omega "
        f"aligned (C5 time selection) vs {r.comb_spacing_tilted:.2f} Omega "
        f"tilted",
        f"  S3 chronology cargo: {100 * r.ctc_cargo_at_1p2:.0f}% of trap "
        f"sites inside the CTC band at Omega = 1.2",
        f"  S4 self-holding lock: wells at "
        f"{tuple(round(w, 1) for w in r.registry_wells_deg)} deg; holding "
        f"tau {r.holding_tau:.2f}; hysteresis "
        f"{r.hysteresis_separation_deg:.0f} deg, stick-slip jumps "
        f"{r.stick_slip_jump_deg:.0f} deg",
        f"  catcher (torque sweep): {r.catcher_verdict}",
    ]
    return "\n".join(lines)
