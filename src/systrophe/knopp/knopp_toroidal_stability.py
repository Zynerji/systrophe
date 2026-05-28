"""Stability and gravitational-wave signature of the Toroidal Knopp Drive binary.

The framework's most speculative claim is that two near-extremal,
maximally counter-rotating Kerr black holes hold together long enough
to host the toroidal CTC band. This module quantifies that claim:

(A) Orbital lifetime against GW emission (Peters 1964 quadrupole
    formula, with the leading spin-spin correction for the antiparallel
    maximal configuration).

(B) GW signature: peak frequency, dimensionless strain at detector
    distance, LIGO/LISA detectability.

(C) Effective band lifetime: the toroidal CTC band exists only while
    the binary survives, so the band's lifetime equals the inspiral
    timescale.

All formulas are classical-GR (post-Newtonian-2 at most). The maximally
counter-rotating spin configuration is itself a speculative regime --
real astrophysical black holes have chi <~ 0.998 and unconstrained spin
orientations -- so the absolute numbers should be read as upper bounds
on what could exist if the configuration is actually formed.

References
----------
- P. C. Peters (1964) PR 136, B1224 -- orbital decay from GW emission.
- L. Kidder (1995) PRD 52, 821 -- spin-spin and spin-orbit corrections
  to inspiralling binaries.
- M. Maggiore (2007) "Gravitational Waves Vol. 1" -- standard reference
  for the strain formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary


# ----- physical constants (geometric units, G = c = 1) --------------------
# Reuse standard SI conversion factors so users can interpret results in
# physical units when M is expressed in solar masses.

M_SUN_KG = 1.98892e30
M_SUN_METER = 1.4767e3            # GM_sun / c^2  ~ 1.4767 km
M_SUN_SECOND = 4.925e-6           # GM_sun / c^3  ~ 4.925 microsec
M_SUN_TO_HZ = 1.0 / M_SUN_SECOND  # 1/M_sun in seconds -> Hz
PARSEC_METER = 3.0857e16

# Detector approximate sensitive bands
LIGO_BAND_HZ = (10.0, 5000.0)     # ground-based, advanced LIGO
LISA_BAND_HZ = (1e-4, 1e-1)       # space-based
PTA_BAND_HZ = (1e-9, 1e-6)        # pulsar timing arrays


# ----- orbital + GW formulas (geometric units) ---------------------------


def reduced_mass(binary: EffectiveToroidalKerrBinary) -> float:
    """mu = m1 m2 / (m1 + m2) = M/2 for equal-mass binary."""
    return float(binary.M / 2.0)


def total_mass(binary: EffectiveToroidalKerrBinary) -> float:
    """M_tot = m1 + m2 = 2 M for equal-mass binary."""
    return float(2.0 * binary.M)


def orbital_frequency(binary: EffectiveToroidalKerrBinary) -> float:
    """Kepler: Omega_orb = sqrt(M_tot / r^3) at separation d.

    Returns angular frequency in geometric units (1 / M-units of time).
    """
    return float(math.sqrt(total_mass(binary) / binary.d ** 3))


def gw_frequency(binary: EffectiveToroidalKerrBinary) -> float:
    """GW peak frequency at inspiral: f_GW = Omega_orb / pi (the dominant
    quadrupole l=m=2 mode emits at 2*Omega_orb / (2 pi))."""
    return float(orbital_frequency(binary) / math.pi)


def gw_luminosity_quadrupole(binary: EffectiveToroidalKerrBinary) -> float:
    """Peters-Mathews quadrupole luminosity for a circular equal-mass binary:

        dE/dt = -(32/5) (G^4/c^5) (m1 m2)^2 (m1+m2) / r^5
              = -(64/5) M^5 / r^5    (equal-mass, G = c = 1).

    Returns positive number (energy loss rate per unit time).
    """
    return float((64.0 / 5.0) * binary.M ** 5 / binary.d ** 5)


def orbital_energy(binary: EffectiveToroidalKerrBinary) -> float:
    """Newtonian orbital energy E_orb = -M^2 / (2 d) (equal-mass)."""
    return float(-binary.M ** 2 / (2.0 * binary.d))


def time_to_merger(binary: EffectiveToroidalKerrBinary) -> float:
    """Peters merger time for a circular equal-mass binary:

        t_merge = (5/256) c^5 d^4 / (G^3 M_tot mu^2)
                = (5/256) d^4 / M^3        (equal-mass, G = c = 1, M_tot = 2M, mu = M/2).
    """
    return float((5.0 / 256.0) * binary.d ** 4 / binary.M ** 3)


def spin_spin_energy(binary: EffectiveToroidalKerrBinary) -> float:
    """Leading-order spin-spin coupling energy for antiparallel maximal spins.

    For S1 = +chi M^2 axis, S2 = -chi M^2 axis (antiparallel):
        U_SS = (G/c^2 r^3)[ 3 (S1 . n)(S2 . n) - S1 . S2 ]
    With spins along the binary axis (= n direction):
        S1 . n = chi M^2,   S2 . n = -chi M^2
        S1 . S2 = -chi^2 M^4
    So:
        U_SS = (1/r^3)[ 3(-chi^2 M^4) - (-chi^2 M^4) ]
             = -2 chi^2 M^4 / r^3   (geometric units).

    Negative = attractive bonus on top of Newtonian -M^2/(2r).
    """
    return float(-2.0 * binary.chi ** 2 * binary.M ** 4 / binary.d ** 3)


def spin_spin_correction_fraction(binary: EffectiveToroidalKerrBinary) -> float:
    """|U_SS| / |E_orb| -- relative size of the spin-spin correction."""
    E_orb = abs(orbital_energy(binary))
    if E_orb < 1e-30:
        return float("inf")
    return float(abs(spin_spin_energy(binary)) / E_orb)


def corrected_merger_time(binary: EffectiveToroidalKerrBinary) -> float:
    """Merger time including the SS correction (leading-order approximation).

    The SS coupling adds an attractive U_SS ~ -2 chi^2 M^4 / r^3 term to
    the orbital energy. The leading-order correction multiplies t_merge
    by (1 + delta)^{-1} where delta = |U_SS| / |E_Newt_orb|. For tight
    near-extremal binaries this can be O(1) -- the configuration is
    *barely* perturbative.
    """
    t_quad = time_to_merger(binary)
    delta = spin_spin_correction_fraction(binary)
    # Conservative correction: shrink merger time by 1 + delta
    return float(t_quad / (1.0 + delta))


# ----- physical units conversion ----------------------------------------


def gw_frequency_in_hz(
    binary: EffectiveToroidalKerrBinary, M_solar: float = 1.0,
) -> float:
    """GW frequency in Hz, assuming binary.M is in units of M_solar.

    f_Hz = f_geom / (M_solar * GM_sun/c^3).
    """
    f_geom = gw_frequency(binary)
    # f_geom is in units of 1/M-time, M-time = M_solar * GM_sun/c^3
    # so f_Hz = f_geom / (M_solar * GM_sun/c^3)
    return float(f_geom / (M_solar * M_SUN_SECOND))


def time_to_merger_in_seconds(
    binary: EffectiveToroidalKerrBinary, M_solar: float = 1.0,
) -> float:
    """Merger time in SI seconds."""
    return float(corrected_merger_time(binary) * M_solar * M_SUN_SECOND)


def gw_strain(
    binary: EffectiveToroidalKerrBinary, distance_m: float,
    M_solar: float = 1.0,
) -> float:
    """Characteristic dimensionless GW strain at observer distance D (in metres).

        h ~ (4 G / c^4) (mu / D) (G M_tot / (r c^2))
          = 4 (mu / D) (M_tot / r)            (geometric units in M_solar)

    Converting to SI: h ~ (M_solar * GM_sun/c^2 / D_m) * (4 mu_geom * M_tot_geom / r_geom)
    where the leading factor is the dimensionless small-angle conversion.
    """
    if distance_m <= 0:
        raise ValueError(f"distance_m must be positive, got {distance_m}")
    mu = reduced_mass(binary)
    M_tot = total_mass(binary)
    r = binary.d
    # In geometric units h ~ 4 mu * M_tot / (r * D).  D in geometric units = D_m / (M_solar * GM_sun/c^2).
    D_geom = distance_m / (M_solar * M_SUN_METER)
    return float(4.0 * mu * M_tot / (r * D_geom))


# ----- detectability ----------------------------------------------------


DetectorName = Literal["LIGO", "LISA", "PTA"]


def _detector_band(name: DetectorName) -> tuple[float, float]:
    return {"LIGO": LIGO_BAND_HZ, "LISA": LISA_BAND_HZ, "PTA": PTA_BAND_HZ}[name]


def in_detector_band(
    binary: EffectiveToroidalKerrBinary, detector: DetectorName,
    M_solar: float = 1.0,
) -> bool:
    """Is the binary's GW frequency in the named detector's sensitive band?"""
    f_Hz = gw_frequency_in_hz(binary, M_solar=M_solar)
    lo, hi = _detector_band(detector)
    return bool(lo <= f_Hz <= hi)


def detector_classification(
    binary: EffectiveToroidalKerrBinary, M_solar: float = 1.0,
) -> str:
    """Which detector (if any) sees this binary's GW emission?"""
    for det in ("LIGO", "LISA", "PTA"):
        if in_detector_band(binary, det, M_solar=M_solar):  # type: ignore[arg-type]
            return det
    return "out-of-band"


# ----- combined diagnostic ---------------------------------------------


@dataclass(frozen=True)
class ToroidalStabilityReport:
    """Full stability / GW report for a candidate Toroidal Knopp binary."""
    binary: EffectiveToroidalKerrBinary
    M_solar: float
    # Orbital
    orbital_frequency_geom: float
    orbital_energy_geom: float
    spin_spin_energy_geom: float
    spin_spin_correction_fraction: float
    # Merger
    merger_time_geom: float
    merger_time_seconds: float
    # GW signature
    gw_frequency_geom: float
    gw_frequency_hz: float
    gw_luminosity_geom: float
    # Detectability
    detector_band: str
    band_lifetime_vs_ctc_window: float  # ratio merger_time / orbital_period
    has_band: bool


def stability_report(
    binary: EffectiveToroidalKerrBinary, M_solar: float = 1.0,
) -> ToroidalStabilityReport:
    """Full classical-GR stability + GW signature report."""
    f_geom = gw_frequency(binary)
    P_geom = gw_luminosity_quadrupole(binary)
    t_merge_g = corrected_merger_time(binary)
    t_merge_s = time_to_merger_in_seconds(binary, M_solar=M_solar)
    f_hz = gw_frequency_in_hz(binary, M_solar=M_solar)
    det = detector_classification(binary, M_solar=M_solar)
    # band lifetime in units of orbital period
    Omega = orbital_frequency(binary)
    T_orb = 2.0 * math.pi / Omega
    n_orbits = t_merge_g / T_orb if T_orb > 0 else 0.0
    has_band = binary.has_toroidal_ctc_band(include_phi=False)
    return ToroidalStabilityReport(
        binary=binary,
        M_solar=float(M_solar),
        orbital_frequency_geom=float(Omega),
        orbital_energy_geom=float(orbital_energy(binary)),
        spin_spin_energy_geom=float(spin_spin_energy(binary)),
        spin_spin_correction_fraction=float(
            spin_spin_correction_fraction(binary)
        ),
        merger_time_geom=float(t_merge_g),
        merger_time_seconds=float(t_merge_s),
        gw_frequency_geom=float(f_geom),
        gw_frequency_hz=float(f_hz),
        gw_luminosity_geom=float(P_geom),
        detector_band=str(det),
        band_lifetime_vs_ctc_window=float(n_orbits),
        has_band=bool(has_band),
    )


def summarise_stability(r: ToroidalStabilityReport) -> str:
    """Human-readable summary."""
    lines = [
        f"Toroidal Knopp binary stability + GW signature",
        f"  M = {r.binary.M} M_sun-equiv,  d = {r.binary.d} M,  chi = {r.binary.chi}",
        f"  CTC band exists?       {r.has_band}",
        "",
        f"Orbital:",
        f"  Omega_orb           = {r.orbital_frequency_geom:.4e}  (1/M)",
        f"  E_orb (Newt)        = {r.orbital_energy_geom:+.4e}  M",
        f"  E_SS (spin-spin)    = {r.spin_spin_energy_geom:+.4e}  M",
        f"  |E_SS|/|E_orb|      = {r.spin_spin_correction_fraction:.3f}",
        "",
        f"GW signature:",
        f"  f_GW                = {r.gw_frequency_geom:.4e}  (1/M)",
        f"  f_GW (M={r.M_solar} M_sun) = {r.gw_frequency_hz:.4e} Hz",
        f"  P_GW (quadrupole)   = {r.gw_luminosity_geom:.4e}  (1/M^2)",
        f"  detector band       = {r.detector_band}",
        "",
        f"Lifetime:",
        f"  t_merger (geom)     = {r.merger_time_geom:.4e}  M",
        f"  t_merger (seconds)  = {r.merger_time_seconds:.4e}  s",
        f"  n_orbits to merger  = {r.band_lifetime_vs_ctc_window:.4e}",
    ]
    return "\n".join(lines)
