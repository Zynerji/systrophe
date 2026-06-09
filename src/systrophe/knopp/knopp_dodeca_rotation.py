"""Rotating the locked dodecahedron: conveyor, frame drag, counter-helicity,
cornered spirals, and the Tipler window.

Sixth module of the dodeca-resonator series. User questions 2026-06-09:
  Q1 can trapped matter be moved by rotating the dodeca?
  Q2 what happens with frame drag if it spins about the throat axis?
  Q3 tilted before spinning: do the horn tips spin oppositely -- a
     "non-mirror reversed" spiral pair?
  Q4 what if the face spirals trace a round-cornered pentagon instead of a
     perfect spiral?
  Q5 where do Tipler-cylinder concepts plug in?

Derivations (model units; the GR mapping in R2 is an ANALOG mapping)
--------------------------------------------------------------------
R1. LATTICE CONVEYOR (Q1: yes). The Gor'kov potential depends on position
    only through the body-frame projections a_j . r, so spinning the
    dodecahedron by psi about the throat axis rotates the ENTIRE trap
    lattice rigidly: U_spun(r) = U_lock(R_psi^T r), an exact identity.
    The crystal lock survives the spin (the aligned axis IS the throat
    axis; the pump ring is azimuthally symmetric, so ring coherence is
    psi-invariant). Trapped cargo follows the lattice up to the grip
    bound: a unit-mass particle at radius rho stays trapped while the
    centripetal demand Omega^2 rho < g_max, i.e.
        Omega_max = sqrt(g_max / rho)  (~12 at rho = R, g_max ~ 99).
    The locked crystal is an azimuthal conveyor for matter.

R2 + R5. FRAME DRAG / TIPLER WINDOW (Q2, Q5). The spinning saturated field
    column along the throat is the van Stockum / Tipler-cylinder analog:
    rigid rotation parameter a = Omega, effective dust density = the mean
    interior pressure u = ceiling/2. In van Stockum, CTCs open where
    g_phiphi = rho^2 (1 - a^2 rho^2) < 0, i.e. rho > 1/Omega, and the
    interior density the solution NEEDS is 8 pi mu = 4 a^2 e^{a^2 rho^2}.
    AT the CTC radius rho = 1/a the exponent is exactly 1, giving a
    CLOSED-FORM supercritical window:
        Omega_lower = 1/(2R)            (CTC radius inside the tube)
        Omega_upper = sqrt(2 pi u / e)  (field supplies the needed density)
    At lock (u = 1.27, R = 0.66): Omega in (0.76, 1.71) -- and the
    conveyor's grip bound (~12) is far above it: the conveyor can sweep
    the whole Tipler band. Caveats are mandatory: Tipler's theorem needs
    an INFINITE cylinder (finite ones require exotic structure); and
    Systrophe Phase 2a already showed the renormalised Boulware stress
    diverges at the chronology horizon of supercritical Tipler spacetimes
    (Hawking chronology protection) -- the same divergence applies to this
    analog the moment the window is entered. Model units only; no SI claim.

R3. COUNTER-HELICITY HORN PAIR (Q3). The intensity field is exactly
    inversion-symmetric, I(-r) = I(r) (cos^2 is even), so the lower-horn
    pattern is the INVERSION image of the upper -- parity twins, not
    mirror twins ("non-mirror reversed", as asked). Globally both horn
    patterns co-rotate rigidly at the same Omega (the m=5 lobe phase
    advances by 5 psi on BOTH rings, same sign). But the D2 spiral
    handedness, as seen looking INTO each horn from outside, is OPPOSITE:
    the horn twist phi = tau y is odd in y, so the upper face carries a
    +tau winding and the lower face a -tau winding from its own viewpoint.
    A small tilt delta before spinning adds an m=1 sideband (wobble) on
    each pump ring with amplitude linear in delta, frequency-split from
    the m=5 lobes -- the tips precess as a counter-helical pair.

R4. ROUND-CORNERED PENTAGON SPIRAL (Q4: corners are a FEATURE). Along a
    chirped arm the five corner crossings occur at the SAME chirped rate
    as the corrugation itself, so the pentagon corners multiply the source
    by (1 + a5 cos(2 chi + const)) -- SECOND-HARMONIC generation of the
    chirp. The coupling comb extends upward: at a5 ~ 0.8 the mean coupling
    of modes 25-48 roughly doubles (0.028 -> 0.059) while the fundamental
    band 1-24 even gains slightly. Too sharp a corner (a5 -> 1) carves new
    nulls in the high band; moderate rounding is optimal.

Catcher (mandatory): runs on the Omega scan of the Tipler supercriticality
margin -- the window edges are genuine transitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.knopp.knopp_dodeca_alignment import DEFAULT_R, FACE_LOCK_DEG
from systrophe.knopp.knopp_dodeca_crystal import pump_ring
from systrophe.knopp.knopp_dodeca_pressure import (
    axis_amplitudes,
    field_axes,
    gorkov_gradient,
    gorkov_potential,
    radiation_pressure,
    tube_interior_points,
    wavenumber,
)

E_CONST = math.e


def _roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def spun_axes(psi_deg: float, tilt_deg: float = 0.0) -> np.ndarray:
    """Field axes after tilting by delta about x, then spinning psi about y."""
    A = field_axes(FACE_LOCK_DEG)
    M = _roty(math.radians(psi_deg)) @ _rotx(math.radians(tilt_deg))
    return A @ M.T


# ----- R1: conveyor -----------------------------------------------------------


def conveyor_rigidity_residual(psi_deg: float = 25.0, n_pts: int = 200) -> float:
    """max |U_spun(R_psi r) - U_lock(r)|: exact 0 = the lattice co-rotates."""
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    rng = np.random.default_rng(7)
    pts = rng.uniform(-0.8, 0.8, size=(n_pts, 3))
    R = _roty(math.radians(psi_deg))
    u_lock = gorkov_potential(pts, field_axes(FACE_LOCK_DEG), a2, k)
    u_spun = gorkov_potential(pts @ R.T, spun_axes(psi_deg), a2, k)
    return float(np.max(np.abs(u_spun - u_lock)))


def ring_coherence_under_spin(psi_deg: float) -> float:
    """Pump-ring coherence with the dodeca spun by psi (lock survival)."""
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    I = radiation_pressure(pump_ring(0.212), spun_axes(psi_deg), a2, k)
    return max(0.0, 1.0 - float(np.std(I)) / max(float(np.mean(I)), 1e-12))


def max_conveyor_rate(rho: float, R: float = DEFAULT_R) -> float:
    """Grip-limited spin rate: unit-mass cargo at radius rho stays trapped
    while Omega^2 rho < g_max -> Omega_max = sqrt(g_max/rho)."""
    if rho <= 0:
        raise ValueError("rho must be positive")
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    pts = tube_interior_points(R, n=32)
    gmax = float(np.max(np.linalg.norm(
        gorkov_gradient(pts, field_axes(FACE_LOCK_DEG), a2, k), axis=1)))
    return math.sqrt(gmax / rho)


# ----- R2 + R5: van Stockum / Tipler window -------------------------------------


def mean_interior_pressure() -> float:
    """u = ceiling/2 exactly (spatial average of cos^2)."""
    a2, _ = axis_amplitudes(FACE_LOCK_DEG)
    return float(np.sum(a2) / 4.0)


def van_stockum_window(u: float | None = None,
                       R: float = DEFAULT_R) -> tuple[float, float]:
    """(Omega_lower, Omega_upper) of the supercritical Tipler-analog window.

    Lower: CTC radius 1/Omega must fit inside the tube (rho < 2R).
    Upper: at the CTC radius the needed dust density is 8 pi mu = 4 Omega^2 e,
    i.e. mu = Omega^2 e / (2 pi); the field supplies u, so
    Omega_upper = sqrt(2 pi u / e). Closed form.
    """
    if u is None:
        u = mean_interior_pressure()
    if u <= 0 or R <= 0:
        raise ValueError("u and R must be positive")
    return 1.0 / (2.0 * R), math.sqrt(2.0 * math.pi * u / E_CONST)


def tipler_assessment(Omega: float, u: float | None = None,
                      R: float = DEFAULT_R) -> dict:
    """Assess the spinning interior against the van Stockum CTC criterion."""
    if Omega <= 0:
        raise ValueError("Omega must be positive")
    if u is None:
        u = mean_interior_pressure()
    r_ctc = 1.0 / Omega
    in_tube = r_ctc < 2.0 * R
    mu_req = Omega ** 2 * E_CONST / (2.0 * math.pi)
    lo, hi = van_stockum_window(u, R)
    return {
        "Omega": float(Omega),
        "r_ctc": float(r_ctc),
        "ctc_radius_in_tube": bool(in_tube),
        "mu_required_at_band": float(mu_req),
        "mu_available": float(u),
        "supercritical": bool(lo < Omega < hi),
        "window": (float(lo), float(hi)),
        "conveyor_can_reach": bool(Omega < max_conveyor_rate(R)),
        "caveats": (
            "Tipler's theorem needs an infinite cylinder; finite columns "
            "require exotic structure. Systrophe Phase 2a: renormalised "
            "Boulware <T_tt> diverges at the chronology horizon of "
            "supercritical Tipler spacetimes (Hawking chronology "
            "protection). Model units; no SI claim."),
    }


# ----- R3: counter-helicity horn pair ---------------------------------------------


def inversion_symmetry_residual(n_pts: int = 300) -> float:
    """max |I(-r) - I(r)|: exactly 0 (cos^2 even) -- parity twins."""
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.8, 0.8, size=(n_pts, 3))
    ax = field_axes(FACE_LOCK_DEG)
    return float(np.max(np.abs(
        radiation_pressure(pts, ax, a2, k)
        - radiation_pressure(-pts, ax, a2, k))))


def horn_spiral_handedness(y_sign: int, tau: float = 0.8) -> int:
    """Winding sense of the D2 spiral seen looking INTO each horn.

    The horn twist is phi = tau * y (odd in y). Viewed from outside the
    upper horn (looking along -y) the apparent azimuth is phi; from outside
    the lower horn (looking along +y) it is -phi. Winding = sign of
    d(phi_apparent)/d|y|: opposite for the two horns.
    """
    if y_sign not in (1, -1):
        raise ValueError("y_sign must be +1 or -1")
    # d phi/d|y| = tau * y_sign; apparent flip for the lower viewpoint = y_sign
    return int(np.sign(tau * y_sign * y_sign) * (1 if y_sign > 0 else -1))


def ring_lobe_phase(y_sign: int, psi_deg: float, tilt_deg: float = 0.0,
                    scale: float = 0.212, n_phi: int = 720) -> dict:
    """m=5 lobe phase and m=1 wobble amplitude on the +-y pump rings."""
    if y_sign not in (1, -1):
        raise ValueError("y_sign must be +1 or -1")
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    ring = pump_ring(scale, n=n_phi)
    ring = ring * np.array([1.0, float(y_sign), 1.0])
    I = radiation_pressure(ring, spun_axes(psi_deg, tilt_deg), a2, k)
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    c5 = np.sum(I * np.exp(-5j * phi)) / n_phi
    c1 = np.sum(I * np.exp(-1j * phi)) / n_phi
    return {"phase_m5": float(np.angle(c5)), "amp_m5": float(np.abs(c5)),
            "amp_m1": float(np.abs(c1))}


# ----- R4: round-cornered pentagon spiral -------------------------------------------


def cornered_spiral_source(rho: np.ndarray, a5: float, chirp_q: float = 60.0,
                           rho_edge: float = 0.8) -> np.ndarray:
    """Spiral corrugation whose arms trace a round-cornered pentagon.

    Along an arm the five corner crossings occur at the chirp rate itself,
    so corners multiply the corrugation by (1 + a5 cos(2 chi + const)):
    second-harmonic generation. a5 = 0 recovers the perfect spiral.
    """
    if not 0.0 <= a5 <= 1.0:
        raise ValueError("a5 must be in [0, 1]")
    rho = np.asarray(rho, dtype=float)
    chi = chirp_q * np.sqrt(rho)
    return (rho <= rho_edge) * (1.0 + np.cos(chi)) \
        * (1.0 + a5 * np.cos(2.0 * chi + 0.7))


def cornered_coupling(a5: float, chirp_q: float = 60.0,
                      n_modes: int = 48, n_grid: int = 8192) -> np.ndarray:
    """|c_n| of the cornered-spiral source on the radial sin basis."""
    rho = np.linspace(0.0, 1.0, n_grid)
    s = cornered_spiral_source(rho, a5, chirp_q)
    n = np.arange(1, n_modes + 1)
    modes = np.sin(math.pi * np.outer(n, rho))
    return np.abs(np.trapezoid(modes * s[None, :], rho, axis=1))


def high_band_gain(a5: float) -> float:
    """Mean coupling of modes 25-48 relative to the perfect spiral (a5=0)."""
    base = float(np.mean(cornered_coupling(0.0)[24:]))
    return float(np.mean(cornered_coupling(a5)[24:])) / base


# ----- report ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RotationReport:
    """Answers to the five rotation questions, computed."""
    conveyor_rigidity_residual: float    # 0: lattice co-rotates exactly
    lock_coherence_under_spin: float     # ~1: lock survives the spin
    conveyor_omega_max: float            # grip-limited spin rate at rho=R
    tipler_window: tuple[float, float]   # (Omega_lower, Omega_upper)
    conveyor_covers_window: bool
    inversion_residual: float            # 0: horns are parity twins
    handedness_upper: int
    handedness_lower: int                # opposite: counter-helical pair
    corotation_phase_slip: float         # |delta(m5 phase) - 5 psi| both rings
    tilt_sideband_ratio: float           # m1(delta)/m1(0): wobble from tilt
    corner_gain_a5_08: float             # high-band gain at a5 = 0.8
    catcher_verdict: str


def rotation_report(psi_deg: float = 20.0, tilt_deg: float = 4.0) -> RotationReport:
    """Compute the full rotation picture; catcher on the Omega scan."""
    u = mean_interior_pressure()
    lo, hi = van_stockum_window(u)
    om_max = max_conveyor_rate(DEFAULT_R)

    omegas = np.linspace(0.2, 3.0, 29)
    margin = {}
    for om in omegas:
        t = tipler_assessment(float(om), u)
        margin[float(om)] = (t["mu_available"] - t["mu_required_at_band"]
                             if t["ctc_radius_in_tube"] else -t["Omega"])

    def fn(om: float) -> np.ndarray:
        key = min(margin, key=lambda v: abs(v - om))
        return np.array([margin[key]])

    catch = scan_novelty(omegas, fn, n_bits=32,
                         parameter_label="spin_Omega")

    # lobes only exist off the equi-phase lock ring -> track on TILTED rings;
    # with the e^{-im phi} convention a +psi rotation advances phase by -m psi
    up0 = ring_lobe_phase(1, 0.0, tilt_deg)
    up1 = ring_lobe_phase(1, psi_deg, tilt_deg)
    dn0 = ring_lobe_phase(-1, 0.0, tilt_deg)
    dn1 = ring_lobe_phase(-1, psi_deg, tilt_deg)
    expect = (-5.0 * math.radians(psi_deg)) % (2.0 * math.pi)

    def slip(p1, p0):
        d = (p1 - p0) % (2.0 * math.pi)
        return min(abs(d - expect), abs(d - expect + 2 * math.pi),
                   abs(d - expect - 2 * math.pi))

    phase_slip = max(slip(up1["phase_m5"], up0["phase_m5"]),
                     slip(dn1["phase_m5"], dn0["phase_m5"]))
    m1_flat = ring_lobe_phase(1, psi_deg, 0.0)["amp_m1"]
    m1_tilt = ring_lobe_phase(1, psi_deg, tilt_deg)["amp_m1"]

    return RotationReport(
        conveyor_rigidity_residual=conveyor_rigidity_residual(),
        lock_coherence_under_spin=min(
            ring_coherence_under_spin(p) for p in (0.0, 14.0, 33.0, 61.0)),
        conveyor_omega_max=float(om_max),
        tipler_window=(float(lo), float(hi)),
        conveyor_covers_window=bool(om_max > hi),
        inversion_residual=inversion_symmetry_residual(),
        handedness_upper=horn_spiral_handedness(1),
        handedness_lower=horn_spiral_handedness(-1),
        corotation_phase_slip=float(phase_slip),
        tilt_sideband_ratio=float(m1_tilt / max(m1_flat, 1e-15)),
        corner_gain_a5_08=high_band_gain(0.8),
        catcher_verdict=catch.verdict,
    )


def summarise_rotation(r: RotationReport) -> str:
    """Human-readable summary."""
    lines = [
        "Rotating the locked dodecahedron",
        f"  R1 conveyor: lattice co-rotates exactly (residual "
        f"{r.conveyor_rigidity_residual:.1e}); lock coherence under spin "
        f"{r.lock_coherence_under_spin:.3f}; Omega_max ~ "
        f"{r.conveyor_omega_max:.1f}",
        f"  R2/R5 Tipler window: Omega in ({r.tipler_window[0]:.3f}, "
        f"{r.tipler_window[1]:.3f}); conveyor covers it: "
        f"{r.conveyor_covers_window}",
        f"  R3 horns: parity twins (inversion residual "
        f"{r.inversion_residual:.1e}); handedness {r.handedness_upper:+d}/"
        f"{r.handedness_lower:+d} (counter-helical); co-rotation phase slip "
        f"{r.corotation_phase_slip:.2e}; tilt m=1 sideband x"
        f"{r.tilt_sideband_ratio:.1f}",
        f"  R4 cornered spiral: high-band (25-48) gain x"
        f"{r.corner_gain_a5_08:.2f} at a5 = 0.8",
        f"  catcher (Omega scan): {r.catcher_verdict}",
    ]
    return "\n".join(lines)
