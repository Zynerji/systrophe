"""First-principles derivation of the dodeca-in-horn-torus spiral hypothesis.

Companion to knopp_dodeca_alignment (which ENCODED the user's hypothesis and
tested its consequences). This module asks: which steps of the hypothesis
FOLLOW from established physics, which are corrected, and which remain
assumptions? Five derivations, each verified numerically:

D1. Point vs face Casimir (proximity force approximation).
    From the parallel-plate energy per area u(d) = -K/d^3 (K = pi^2 hbar c/720;
    model units K=1), PFA integrates u over the local gap profile:
      - point of curvature radius r_v at gap d0:  E_pt = pi K r_v / d0^2
        (exact integral of the paraboloid gap d0 + rho^2/2r_v)
      - flat face of area A parallel at gap d0:   E_face = K A / d0^3
    Ratio E_face/E_pt = A/(pi r_v d0) -> face-lock DOMINATES point-lock for
    every gap below the crossover d* = A/(pi r_v). DERIVED.

D2. The five-armed Fermat spiral.
    Near the pinch the inner-horn wall is the cusp y(rho) = sqrt(2 R rho).
    The Knopp drive already carries a horn TWIST (steering eps, theta_0):
    azimuth advances with height along a generator, phi = phi_0 + tau * y.
    Projecting the twisted generators onto the face plane:
        rho(phi) = (phi - phi_0)^2 / (2 R tau^2)
    -- a Fermat spiral (rho ~ phi^2). The pentagon's C_5v symmetry restricts
    the contact pattern's azimuthal Fourier content to m = 0 (mod 5); the
    lowest nontrivial component is m = 5: FIVE ARMS. DERIVED (given twist).

D3. Area/contact enhancement magnitude.
    Conformal spiral contact of fraction f of face area A at gap g_c versus
    the bare cusp annulus gives eta = f A g(rho)-integral ratio. The SCALING
    is derived; the MAGNITUDE depends on (f, g_c), which need a material
    model of the warp shell that does not exist. The x7 used in the demo is
    a calibration, not a derivation -- this module computes which (f, g_c)
    pairs it corresponds to. ASSUMPTION, made explicit.

D4. Broadband coupling from the chirped spiral aperture.
    Mode-coupling c_n = <source | mode_n> on a radial sin basis:
      - point source: no spectral nulls but uniformly WEAK (amplitude ~ area)
        -> 0/24 modes ring even at maximum drive
      - flat pentagonal piston (edge at rho_e): c_n = (1 - cos(n pi rho_e))/(n pi)
        has EXACT NULLS (rho_e = 0.8: every 5th mode) and a 1/n tail
        -> 7/24 however hard it is driven
      - STATIC spiral corrugation: chirped phase chi = q sqrt(rho) fills the
        piston nulls but leaves accidental weak modes -> 16/24
      - FREQUENCY-SWEPT spiral (the hypothesis's "increasing in frequency
        via feedback loop"): as the chirp rate q sweeps, each mode is pumped
        when the local wavenumber passes its resonance; with cavity lifetime
        1/kappa = sqrt(Q)/kappa_0 longer than the sweep period the mode
        STORES that energy, so the effective coupling is the max over the
        sweep -> 24/24. FULL SPECTRUM.
    Full-spectrum saturation therefore needs ALL THREE hypothesis
    ingredients jointly: the face's amplitude, the spiral's chirp, and the
    rising-frequency feedback (with Q-storage). None alone suffices. DERIVED.

D5. Directional collapse needs the m=1 channel.
    The saturated C_5 field has zero dipole moment: integral of
    cos(5 phi) cos(phi) over the shell vanishes. A net directional stress
    requires an m = 1 component, and the only m = 1 structure in the drive
    is the horn-twist steering lobe eps*cos(phi - theta_0). Hence
        collapse dipole = saturation * eps  (exactly, to leading order),
    i.e. full-spectrum saturation alone is DIRECTIONLESS; the heading
    selectivity comes from the existing steering twist. CORRECTION to the
    hypothesis (and to the first demo encoding, since patched).

Catcher: per the Systrophe rule, scan_novelty runs on the D1 gap scan and
its verdict is part of the report.

Unchanged caveat: all of this treats the Casimir gap as a coherent pump
(geometry + mode structure). Vacuum fluctuations are not a free energy
reservoir; see knopp_toroidal_casimir_dodecahedron for why Q^N vacuum
amplification is rejected (Brown-Maclay ~O(10) bound).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty

DEFAULT_R = 0.66


# ----- D1: proximity-force approximation, point vs face ---------------------


def pfa_point_energy(d0: float, r_v: float) -> float:
    """|Casimir| energy of a paraboloid tip (curvature radius r_v) at gap d0.

    PFA: E = K integral 2 pi rho drho / (d0 + rho^2/(2 r_v))^3
           = pi K r_v / d0^2   (model units K = 1).
    """
    if d0 <= 0 or r_v <= 0:
        raise ValueError("d0 and r_v must be positive")
    return math.pi * r_v / d0 ** 2


def pfa_point_energy_numeric(d0: float, r_v: float, rho_max: float = 50.0,
                             n: int = 200_000) -> float:
    """Direct quadrature of the D1 point integral (validates the closed form)."""
    rho = np.linspace(0.0, rho_max, n)
    gap = d0 + rho ** 2 / (2.0 * r_v)
    return float(np.trapezoid(2.0 * math.pi * rho / gap ** 3, rho))


def pfa_face_energy(d0: float, area: float) -> float:
    """|Casimir| energy of a parallel flat face of given area at gap d0."""
    if d0 <= 0 or area <= 0:
        raise ValueError("d0 and area must be positive")
    return area / d0 ** 3


def face_point_crossover_gap(area: float, r_v: float) -> float:
    """Gap below which the flat face beats the point: d* = A/(pi r_v)."""
    return area / (math.pi * r_v)


# ----- D2: Fermat spiral from the twisted cusp ------------------------------


def twisted_cusp_projection(
    y: np.ndarray, tau: float, R: float = DEFAULT_R, phi_0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a twisted inner-horn generator onto the face plane.

    Cusp wall: y = sqrt(2 R rho)  =>  rho = y^2/(2R).
    Twist:     phi = phi_0 + tau * y.
    Returns (rho, phi) along the generator -- the contact curve.
    """
    y = np.asarray(y, dtype=float)
    if np.any(y < 0) or tau == 0:
        raise ValueError("y must be >= 0 and tau nonzero")
    return y ** 2 / (2.0 * R), phi_0 + tau * y


def spiral_exponent(tau: float = 0.8, R: float = DEFAULT_R) -> float:
    """Fit rho ~ phi^p along the projected generator; Fermat spiral => p = 2."""
    y = np.linspace(0.05, 1.0, 400)
    rho, phi = twisted_cusp_projection(y, tau, R)
    p = np.polyfit(np.log(phi), np.log(rho), 1)[0]
    return float(p)


def pentagon_azimuthal_spectrum(
    tau: float = 0.8, R: float = DEFAULT_R, rho_ring: float = 0.05,
    arm_width: float = 0.18, n_phi: int = 4096,
) -> np.ndarray:
    """Azimuthal power spectrum |P_m|^2 of the C_5-seeded contact pattern.

    Five generators seeded at the pentagon's C_5 angles 2 pi j/5 produce a
    pattern whose nonzero Fourier content sits at m = 0 (mod 5).
    """
    y_ring = math.sqrt(2.0 * R * rho_ring)
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    pattern = np.zeros(n_phi)
    for j in range(5):
        centre = 2.0 * math.pi * j / 5.0 + tau * y_ring
        d = np.angle(np.exp(1j * (phi - centre)))
        pattern += np.exp(-d ** 2 / (2.0 * arm_width ** 2))
    spec = np.abs(np.fft.rfft(pattern - pattern.mean())) ** 2
    return spec / max(spec.max(), 1e-300)


# ----- D3: contact-area enhancement (scaling derived, magnitude not) --------


def cusp_gap_profile(rho: np.ndarray, h: float, R: float = DEFAULT_R) -> np.ndarray:
    """|vertical gap| between the face plane at height h and the cusp wall."""
    rho = np.asarray(rho, dtype=float)
    y_wall = np.sqrt(np.clip(2.0 * R * rho - rho ** 2, 0.0, None))
    return np.abs(y_wall - h)


def bare_cusp_pfa(h: float, R: float = DEFAULT_R, rho_max: float = 0.30,
                  g0: float = 0.02, n: int = 60_000) -> float:
    """PFA energy of the UNDEFORMED cusp wall over the face annulus."""
    rho = np.linspace(1e-6, rho_max, n)
    g = cusp_gap_profile(rho, h, R)
    return float(np.trapezoid(2.0 * math.pi * rho / (g + g0) ** 3, rho))


def conformal_contact_pfa(area: float, f: float, g_c: float) -> float:
    """PFA energy if a fraction f of the face conforms at uniform gap g_c."""
    if not 0.0 < f <= 1.0 or g_c <= 0:
        raise ValueError("f in (0,1], g_c > 0 required")
    return f * area / g_c ** 3


def gap_for_target_enhancement(
    target: float, h: float, area: float, f: float = 0.3,
    R: float = DEFAULT_R, g0: float = 0.02,
) -> float:
    """The conformal gap g_c that the demo's area factor silently assumes.

    Solves conformal_contact_pfa(area, f, g_c) = target * bare_cusp_pfa(h).
    This is the honest content of D3: the x_target factor is a (f, g_c)
    calibration, not a derivation.
    """
    bare = bare_cusp_pfa(h, R, g0=g0)
    return float((f * area / (target * bare)) ** (1.0 / 3.0))


# ----- D4: coupling spectra (point / piston / spiral) ------------------------


def coupling_spectrum(
    source: str, n_modes: int = 24, rho_edge: float = 0.8,
    chirp_q: float = 60.0, n_grid: int = 8192,
) -> np.ndarray:
    """|c_n| of a source against radial modes sin(n pi rho) on [0, 1].

    source: "point" (narrow gaussian, weight = its tiny area),
            "piston" (flat face, exact nulls at n = 5k for rho_edge = 0.8),
            "spiral" (piston x (1 + cos(q sqrt(rho))) chirped corrugation).
    """
    rho = np.linspace(0.0, 1.0, n_grid)
    if source == "point":
        s = np.exp(-(rho - 0.02) ** 2 / (2 * 0.008 ** 2))
        s *= 0.02 / np.trapezoid(s, rho)
    elif source == "piston":
        s = (rho <= rho_edge).astype(float)
    elif source == "spiral":
        s = (rho <= rho_edge) * (1.0 + np.cos(chirp_q * np.sqrt(rho)))
    else:
        raise ValueError(f"unknown source {source!r}")
    n = np.arange(1, n_modes + 1)
    modes = np.sin(math.pi * np.outer(n, rho))
    return np.abs(np.trapezoid(modes * s[None, :], rho, axis=1))


def piston_coupling_closed_form(n: int, rho_edge: float = 0.8) -> float:
    """c_n = (1 - cos(n pi rho_edge)) / (n pi): exact nulls at n rho_edge even."""
    return (1.0 - math.cos(n * math.pi * rho_edge)) / (n * math.pi)


def swept_spiral_coupling(
    n_modes: int = 24, q_lo: float = 12.0, q_hi: float = 90.0, n_q: int = 40,
) -> np.ndarray:
    """Effective coupling of the frequency-swept spiral: max over the sweep.

    Valid when the sweep period is shorter than the cavity storage time
    1/kappa = sqrt(Q)/kappa_0 (see sweep_storage_condition), so each mode
    integrates the pump while the chirp passes through its resonance.
    """
    qs = np.linspace(q_lo, q_hi, n_q)
    cs = np.array([coupling_spectrum("spiral", n_modes=n_modes, chirp_q=q)
                   for q in qs])
    return cs.max(axis=0)


def sweep_storage_condition(t_sweep: float, Q: float, kappa0: float = 1.5) -> bool:
    """True when cavity storage outlives the sweep: t_sweep < sqrt(Q)/kappa0."""
    if t_sweep <= 0 or Q <= 0:
        raise ValueError("t_sweep and Q must be positive")
    return t_sweep < math.sqrt(Q) / kappa0


def spectral_metrics(c: np.ndarray) -> dict:
    """Total power, weakest coupling, and participation ratio of a spectrum."""
    c = np.asarray(c, dtype=float)
    p = c ** 2
    total = float(np.sum(p))
    pr = float(total ** 2 / max(np.sum(p ** 2), 1e-300))
    return {"total_power": total, "min_coupling": float(np.min(c)),
            "participation_ratio": pr}


def modes_above_threshold(
    c: np.ndarray, drive: float, Q: float = 60.0,
    gamma: float = 2.2, kappa0: float = 1.5,
) -> int:
    """Count of modes with steady amplitude > 1/2.

    Mode ODE a' = G(1-a) - kappa a with G = gamma*drive*kappa_n and
    kappa = kappa0/sqrt(Q): a* > 1/2 iff G > kappa. Couplings are
    normalised to the strongest piston line so the three sources compete
    on one scale.
    """
    if not 0 <= drive <= 1 or Q <= 0:
        raise ValueError("drive in [0,1] and Q > 0 required")
    ref = float(np.max(coupling_spectrum("piston", n_modes=len(c))))
    kn = np.asarray(c, dtype=float) / ref
    return int(np.sum(gamma * drive * kn > kappa0 / math.sqrt(Q)))


# ----- D5: collapse dipole ----------------------------------------------------


def collapse_dipole(
    sat: float, eps: float, theta0: float = 0.0, beta5: float = 0.5,
    n_phi: int = 4096,
) -> float:
    """m=1 moment of the shell radiation stress.

    sigma(phi) = sat * (1 + beta5 cos(5 phi)) * (1 + eps cos(phi - theta0));
    dipole = (1/pi) integral sigma cos(phi - theta0) dphi = sat * eps exactly
    (the C_5 term has no m = 1 content). eps = 0 => zero dipole however
    saturated the field: collapse direction comes from the steering twist.
    """
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    sigma = sat * (1.0 + beta5 * np.cos(5.0 * phi)) \
        * (1.0 + eps * np.cos(phi - theta0))
    return float(np.sum(sigma * np.cos(phi - theta0)) * (2 * math.pi / n_phi)
                 / math.pi)


# ----- report -----------------------------------------------------------------


@dataclass(frozen=True)
class FirstPrinciplesReport:
    """Which hypothesis steps derive, which correct, which stay assumptions."""
    # D1
    crossover_gap: float            # face beats point below this gap
    pfa_closed_form_error: float    # numeric vs analytic point integral
    # D2
    spiral_exponent: float          # Fermat spiral => 2.0
    dominant_azimuthal_mode: int    # C_5 selection => 5
    # D3 (assumption made explicit)
    conformal_gap_for_x7: float     # g_c the demo's x7 silently assumes (f=0.3)
    # D4
    piston_null_modes: int          # exact spectral nulls in 24 (=> 4)
    point_modes_ringing: int
    piston_modes_ringing: int
    spiral_static_modes_ringing: int
    spiral_swept_modes_ringing: int
    full_spectrum_needs_all_three: bool   # face + spiral + frequency sweep
    storage_outlives_sweep: bool
    # D5
    dipole_at_eps0: float           # = 0: saturation alone is directionless
    dipole_sat_eps: float           # = sat*eps
    # catcher
    catcher_verdict: str


def derive_all(
    area: float = 0.30, r_v: float = 0.05, h: float = 0.20,
    drive_point: float = 0.33, drive_face: float = 0.82, Q: float = 60.0,
) -> FirstPrinciplesReport:
    """Run all five derivations at demo-parity parameters."""
    # D1 + catcher on the gap scan of the face/point ratio
    d0_scan = np.logspace(-3, 0, 60)
    ratio = np.array([pfa_face_energy(d, area) / pfa_point_energy(d, r_v)
                      for d in d0_scan])
    table = dict(zip(d0_scan, ratio))

    def fn(d: float) -> np.ndarray:
        key = min(table, key=lambda x: abs(x - d))
        return np.array([table[key]])

    catch = scan_novelty(d0_scan, fn, n_bits=32, parameter_label="gap_d0")
    err = abs(pfa_point_energy_numeric(0.05, r_v)
              - pfa_point_energy(0.05, r_v)) / pfa_point_energy(0.05, r_v)

    # D2
    spec = pentagon_azimuthal_spectrum()
    dominant_m = int(np.argmax(spec[1:]) + 1)

    # D4
    c_pt = coupling_spectrum("point")
    c_pi = coupling_spectrum("piston")
    c_sp = coupling_spectrum("spiral")
    c_sw = swept_spiral_coupling()
    nulls = int(np.sum(c_pi < 1e-3 * np.max(c_pi)))
    n_pt = modes_above_threshold(c_pt, drive_point, Q)
    n_pi = modes_above_threshold(c_pi, drive_face, Q)
    n_sp = modes_above_threshold(c_sp, drive_face, Q)
    n_sw = modes_above_threshold(c_sw, drive_face, Q)

    return FirstPrinciplesReport(
        crossover_gap=face_point_crossover_gap(area, r_v),
        pfa_closed_form_error=float(err),
        spiral_exponent=spiral_exponent(),
        dominant_azimuthal_mode=dominant_m,
        conformal_gap_for_x7=gap_for_target_enhancement(7.0, h, area),
        piston_null_modes=nulls,
        point_modes_ringing=n_pt,
        piston_modes_ringing=n_pi,
        spiral_static_modes_ringing=n_sp,
        spiral_swept_modes_ringing=n_sw,
        full_spectrum_needs_all_three=bool(
            n_sw == 24 and n_sp < 24 and n_pi < 24 and n_pt < 24),
        storage_outlives_sweep=sweep_storage_condition(t_sweep=2.0, Q=Q),
        dipole_at_eps0=collapse_dipole(1.0, 0.0),
        dipole_sat_eps=collapse_dipole(0.9, 0.22),
        catcher_verdict=catch.verdict,
    )


def summarise_first_principles(r: FirstPrinciplesReport) -> str:
    """Human-readable derivation verdicts."""
    lines = [
        "First-principles audit of the dodeca-spiral hypothesis",
        f"  D1 DERIVED   face beats point below gap d* = {r.crossover_gap:.3f}"
        f" (PFA closed form vs quadrature: {r.pfa_closed_form_error:.2e})",
        f"  D2 DERIVED   twisted cusp projects as rho ~ phi^"
        f"{r.spiral_exponent:.3f} (Fermat spiral); C5 selects m = "
        f"{r.dominant_azimuthal_mode} (five arms)",
        f"  D3 ASSUMED   demo's x7 area factor = conformal contact at g_c = "
        f"{r.conformal_gap_for_x7:.4f} (f = 0.3); needs a shell material model",
        f"  D4 DERIVED   piston has {r.piston_null_modes} exact nulls /24;"
        f" modes ringing point/piston/spiral/swept = {r.point_modes_ringing}/"
        f"{r.piston_modes_ringing}/{r.spiral_static_modes_ringing}/"
        f"{r.spiral_swept_modes_ringing}; full spectrum needs face+spiral+"
        f"sweep jointly: {r.full_spectrum_needs_all_three}"
        f" (storage outlives sweep: {r.storage_outlives_sweep})",
        f"  D5 CORRECTED collapse dipole = sat*eps (= "
        f"{r.dipole_sat_eps:.3f} at sat 0.9, eps 0.22); eps = 0 gives "
        f"{r.dipole_at_eps0:.1e} -- saturation alone is directionless",
        f"  catcher: {r.catcher_verdict}",
    ]
    return "\n".join(lines)
