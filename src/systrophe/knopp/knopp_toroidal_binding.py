"""Non-gravitational binding mechanism catalogue for the Toroidal Knopp binary.

What we're testing
------------------
The stability falsification (knopp_toroidal_stability) showed that
quadrupole GW emission collapses the maximal counter-rotating Kerr
binary in << 1 orbit at the working configuration (d = 2M). For the
toroidal Knopp Drive framework to survive, SOME additional binding
mechanism would have to compensate -- either by reducing the GW
emission, opposing the inspiral with an extra force, or creating a
stable equilibrium at small d.

This module catalogues candidate binding mechanisms and computes the
required coupling strength to offset the GW emission for the working
configuration. The honest answer for each is whether it (i) exists in
the SM / known beyond-SM physics, (ii) couples to BHs at all, and
(iii) at the required strength.

The five mechanisms catalogued
------------------------------
1. **Attractive Yukawa scalar** with coupling alpha and mass mu:
   V_scalar(r) = -alpha * G * M^2 exp(-mu * r) / r.
   Does NOT halt inspiral -- just rescales effective G. Even at
   alpha = O(1), inspiral proceeds (just at a modified rate).
   VERDICT: doesn't open a stable equilibrium.

2. **Short-range repulsion** (hard core) with V_rep(r) = epsilon /
   r^n. Creates a potential minimum at some r_eq if n is large enough.
   Requires HUGE epsilon and n >= 6 for r_eq ~ 2M. The natural
   physical analog (Casimir repulsion between BHs?) is repulsive only
   under specific topologies that don't apply here.
   VERDICT: physically speculative; no SM candidate.

3. **Modified GW emission** via scalar-tensor gravity (dilaton coupling).
   In Brans-Dicke with omega_BD, GW emission has an extra dipole
   contribution from the SCALAR mode that ACCELERATES inspiral.
   Going to omega_BD < 0 could in principle suppress GW emission, but
   omega_BD > 40000 from observation -- not in the right direction
   and astrophysically constrained.
   VERDICT: wrong sign; observational bounds rule it out.

4. **Dark photon ("U(1)_D")** coupling to BH dark charge q_D. Long-
   range repulsion mediated by a massless dark photon scales as
   F = q_D^2 / r^2. For F to balance GW back-reaction at r = 2M,
   q_D must be of order M (geometric units), corresponding to a
   dark fine-structure constant alpha_D ~ 1. Possible in dark-sector
   models but requires BHs to be "charged" under U(1)_D -- no known
   mechanism to put dark charge on a Kerr horizon.
   VERDICT: model-building exists; coupling to physical BHs unclear.

5. **Quantum-area-quantization rigidity** (LQG-inspired): if BH area
   is quantized (A_n = 8 pi gamma l_P^2 sqrt(n(n+1))), there could
   be a "snapping" potential between BHs of different area quantum
   numbers, creating a stable d when areas are quantum-locked.
   Requires the area gap to be O(M) at d = 2M, which means
   sqrt(n(n+1)) ~ M^2 / l_P^2 ~ 10^77 for stellar BHs -- the area
   gap is fractional 1/(2n), exponentially small compared to required
   force.
   VERDICT: quantitatively too weak by many OOM.

Result
------
None of the five candidates derives a binding mechanism strong enough,
of the right sign, AND with a physically motivated coupling to Kerr
BHs at maximal spin. The framework's rescue-path (b) is closed at the
level of catalogued mechanisms -- closing it requires NEW physics
beyond what's in the SM + leading beyond-SM proposals.

This is itself a useful (negative) result: the Toroidal Knopp Drive
cannot be rescued by any currently-conjectured non-gravitational
binding mechanism without invoking truly exotic structure (e.g., a
dedicated "BH binding force" with no other observational signature).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ----- shared inspiral-power criterion -----------------------------------


def gw_inspiral_power(M: float, d: float) -> float:
    """Quadrupole GW power radiated by an equal-mass binary at separation d.

        dE/dt = (64/5) M^5 / d^5   (G = c = 1).
    """
    return float(64.0 / 5.0 * M ** 5 / d ** 5)


def gw_inspiral_force(M: float, d: float) -> float:
    """Effective "drag" force on the orbit from GW emission.

    dE/dt = F_drag * v_orbital,   v_orb = sqrt(M_tot / d) = sqrt(2M/d).
    F_drag = dE/dt / v_orb = (64/5) M^5 / d^5 / sqrt(2M/d)
           = (64/5) M^(9/2) / (d^(9/2) sqrt(2)).
    """
    v = math.sqrt(2.0 * M / d)
    return float(gw_inspiral_power(M, d) / v)


# ----- mechanism 1: attractive Yukawa scalar -----------------------------


@dataclass(frozen=True)
class YukawaMechanism:
    name: str = "Attractive Yukawa scalar (mass mu)"
    description: str = "V(r) = -alpha M^2 exp(-mu r) / r"
    requires_BSM: bool = True   # technically: a new scalar
    rescues_inspiral: bool = False
    coupling_required: Optional[float] = None
    notes: str = (
        "An attractive Yukawa rescales effective G but does not change"
        " the dynamics qualitatively: orbit still inspirals, just on a"
        " modified timescale. To HALT inspiral, the Yukawa would need"
        " to create a stable equilibrium -- but a purely attractive"
        " potential has none (orbit always shrinks). VERDICT: cannot"
        " rescue the framework."
    )


def yukawa_required_coupling(M: float, d: float, mu: float) -> float:
    """The alpha for the Yukawa force F_Y = -alpha M^2 exp(-mu r) / r^2 (1 + mu r)
    to equal F_drag at r = d. Returns the magnitude of alpha.

    Useful as a benchmark: the alpha needed to formally cancel inspiral
    AT a single radius. Doesn't produce a stable equilibrium because
    F_Y is monotone in r.
    """
    F_drag = gw_inspiral_force(M, d)
    yukawa_unit_force = M ** 2 * math.exp(-mu * d) * (1.0 + mu * d) / d ** 2
    if yukawa_unit_force <= 0:
        return float("inf")
    return float(F_drag / yukawa_unit_force)


# ----- mechanism 2: short-range repulsion (hard core) --------------------


@dataclass(frozen=True)
class HardCoreMechanism:
    name: str = "Short-range repulsion (hard core)"
    description: str = "V_rep(r) = epsilon / r^n"
    requires_BSM: bool = True
    rescues_inspiral: bool = True   # IF tuned correctly
    coupling_required: Optional[float] = None
    notes: str = (
        "A repulsive 1/r^n core balanced against Newtonian gravity"
        " gives a stable equilibrium at r_eq = (n epsilon / M^2)^(1/(n-1))."
        " For r_eq = 2M (the working configuration), epsilon = 2^(n-1)"
        " M^(n+1) / n. At n = 6 this requires epsilon ~ 5 M^7. No"
        " known SM or beyond-SM force has this structure between black"
        " holes. Lacks a physical candidate."
    )


def hard_core_required_epsilon(M: float, d_eq: float, n: int = 6) -> float:
    """epsilon required for the V_rep(r) = epsilon / r^n + V_Newt(r) =
    -M^2/(2r) + epsilon/r^n potential to have its minimum at r = d_eq.

    Equilibrium: dV/dr = 0
       M^2 / (2 r^2) - n epsilon / r^(n+1) = 0
       -> epsilon = M^2 r^(n-1) / (2 n)
    At r = d_eq: epsilon = M^2 d_eq^(n-1) / (2 n).

    Also requires d^2 V/dr^2 > 0 at r_eq -> n > 1. Always true.
    """
    if n < 2:
        raise ValueError("n must be >= 2 for a stable equilibrium")
    return float(M ** 2 * d_eq ** (n - 1) / (2 * n))


# ----- mechanism 3: scalar-tensor / Brans-Dicke modified GW emission ----


@dataclass(frozen=True)
class ScalarTensorMechanism:
    name: str = "Modified GW emission (scalar-tensor)"
    description: str = "Brans-Dicke omega_BD, dipole scalar mode"
    requires_BSM: bool = True
    rescues_inspiral: bool = False
    coupling_required: Optional[float] = None
    notes: str = (
        "Brans-Dicke theory adds a scalar mode whose dipole emission"
        " ACCELERATES inspiral (Wagoner 1970): dE/dt_BD = dE/dt_GR * (1 +"
        " 2/(3 + 2 omega_BD)). For omega_BD > 0 this strengthens emission"
        " (wrong direction). For omega_BD < 0 (ghost regime) the formula"
        " formally suppresses emission, but ghost scalars are forbidden"
        " by stability requirements and ruled out observationally"
        " (Cassini: omega_BD > 40000). VERDICT: wrong sign + observational"
        " bounds preclude the rescue."
    )


def brans_dicke_emission_factor(omega_BD: float) -> float:
    """L_GW^BD / L_GW^GR = 1 + 2 / (3 + 2 omega_BD).

    omega_BD > 0:  factor > 1 (faster inspiral)
    omega_BD < -3/2: factor diverges/ghost regime.
    """
    denom = 3.0 + 2.0 * omega_BD
    if abs(denom) < 1e-12:
        return float("inf")
    return float(1.0 + 2.0 / denom)


# ----- mechanism 4: dark photon (U(1)_D) ---------------------------------


@dataclass(frozen=True)
class DarkPhotonMechanism:
    name: str = "Dark photon U(1)_D"
    description: str = "Repulsive 1/r^2 from BH dark charge q_D"
    requires_BSM: bool = True
    rescues_inspiral: bool = True   # IF dark charges exist & couple
    coupling_required: Optional[float] = None
    notes: str = (
        "A massless dark photon with BH dark charge q_D gives a repulsive"
        " 1/r^2 force. To create a stable equilibrium at r_eq matching"
        " Newtonian gravity, q_D = M (geometric units), i.e., alpha_D"
        " = q_D^2 / M^2 ~ 1. Open questions: (a) can Kerr BHs carry"
        " dark charge (no-hair forbids it for SM gauge fields); (b)"
        " what couples a massless U(1)_D to the BH horizon; (c) does"
        " this not show up in BH X-ray spectra. Dark-sector models with"
        " 'hidden' gauge fields exist but are not parametrized for BH"
        " binding."
    )


def dark_photon_required_charge(M: float, d: float) -> float:
    """q_D such that the dark Coulomb F_D = q_D^2 / r^2 balances Newtonian
    gravity F_N = M^2 / r^2 at r = d.

    F_D = F_N  ->  q_D^2 = M^2  ->  q_D = M.
    Independent of d (both forces scale as 1/r^2).
    """
    return float(M)


# ----- mechanism 5: LQG area quantization rigidity ----------------------


@dataclass(frozen=True)
class LQGAreaMechanism:
    name: str = "LQG area-quantization rigidity"
    description: str = "Discrete BH area gap creates 'snapping' potential"
    requires_BSM: bool = True
    rescues_inspiral: bool = False
    coupling_required: Optional[float] = None
    notes: str = (
        "LQG area spectrum A_n = 8 pi gamma l_P^2 sqrt(n(n+1)). For a"
        " Schwarzschild BH of mass M, the quantum number is n ~ M^2 /"
        " l_P^2. Adjacent levels differ by Delta A / A ~ 1/(2n), which"
        " for stellar BHs (M ~ 10 M_sun) is fractional ~10^(-77). The"
        " 'force' from area-quantization rigidity is dE_BH/dr ~"
        " (T_Hawking) * (dA/dr) * (Delta A / A) and is exponentially"
        " smaller than the required binding. VERDICT: too weak by many"
        " orders of magnitude to balance GW emission."
    )


def lqg_area_gap_fractional(M_solar: float, gamma: float = 0.2375) -> float:
    """Fractional area gap (Delta A / A) for a BH of mass M_solar.

    A_n = 8 pi gamma l_P^2 sqrt(n(n+1)).  For large n:  A_n ~ 8 pi gamma l_P^2 n.
    Delta A / A_n ~ 1/n. For Schwarzschild M_solar, A = 16 pi M_solar^2 ~
    16 pi (M_solar / l_P)^2 l_P^2. So n ~ 2 (M_solar / l_P)^2 / gamma.
    Fractional gap ~ 1/n ~ gamma / (2 (M_solar / l_P)^2).
    M_solar/l_P ~ 1.4767e3 m / 1.616e-35 m ~ 9.1e37.
    """
    M_over_lP = M_solar * 9.1e37  # in Planck units
    n = 2.0 * M_over_lP ** 2 / gamma
    return float(1.0 / n)


# ----- catalogue + verdict ----------------------------------------------


# ----- mechanism 6: f(R) higher-derivative gravity ---------------------


@dataclass(frozen=True)
class FRGravityMechanism:
    name: str = "f(R) higher-derivative gravity"
    description: str = "L_grav = R + alpha R^2 -- Ricci-squared correction"
    requires_BSM: bool = True
    rescues_inspiral: bool = False
    coupling_required: Optional[float] = None
    notes: str = (
        "f(R) = R + alpha R^2 gravity introduces a propagating scalar"
        " (the 'scalaron') of mass m_s ~ 1/sqrt(alpha). In the binary"
        " inspiral, the scalaron emits additional GW dipole radiation"
        " (ACCELERATING inspiral, wrong sign) plus modifies the static"
        " Newtonian potential by a Yukawa correction. Neither effect"
        " creates a stable equilibrium at d = 2M. Stronger constraint:"
        " Solar-system tests bound alpha < (M_sun)^2 -- way too small"
        " to affect stellar-mass binary dynamics."
        " VERDICT: wrong sign and bounded out."
    )


def fr_gravity_yukawa_range(alpha: float) -> float:
    """Range of the scalaron Yukawa in f(R) = R + alpha R^2:
        m_scalaron ~ 1/sqrt(6 alpha), range = 1/m = sqrt(6 alpha)."""
    if alpha <= 0:
        return float("inf")
    return float(math.sqrt(6.0 * alpha))


# ----- mechanism 7: braneworld extra dimensions ----------------------


@dataclass(frozen=True)
class BraneworldMechanism:
    name: str = "Braneworld extra dimensions"
    description: str = "RS/ADD modified gravity at sub-Hubble scales"
    requires_BSM: bool = True
    rescues_inspiral: bool = False
    coupling_required: Optional[float] = None
    notes: str = (
        "Randall-Sundrum and ADD braneworld scenarios modify gravity at"
        " sub-millimetre scales. For TWO black holes in a binary with"
        " separation d ~ 2M (km-scale for stellar BHs), modifications"
        " are *negligible* -- braneworld effects appear at scales of"
        " ~10^-5 m at most (current bounds). Even if the modifications"
        " were significant, they ENHANCE gravitational attraction at"
        " short range (ADD: F ~ 1/r^(2+n) for n extra dimensions),"
        " ACCELERATING inspiral rather than halting it."
        " VERDICT: irrelevant at BH-binary scales; wrong sign if it"
        " were."
    )


def add_gravity_enhancement_factor(d: float, R_compact: float = 1e-5,
                                    n_extra: int = 2) -> float:
    """Force enhancement at separation d in ADD with R_compact compact
    radius and n_extra extra dimensions. F_ADD/F_Newton = 1 +
    (R_compact/d)^n_extra in the asymptotic regime.

    For BH binary at d ~ 2M ~ 3 km and R_compact ~ 10^-5 m:
        ratio = (10^-5 / 3000)^2 ~ 10^-17 -- negligible.
    """
    if d <= 0 or R_compact <= 0:
        return 1.0
    return float(1.0 + (R_compact / d) ** n_extra)


# ----- mechanism 8: parity-violating BH "chirality charge" ----------


@dataclass(frozen=True)
class ParityChargeMechanism:
    name: str = "Parity-violating BH chirality charge"
    description: str = "Chern-Simons coupling: gravitomagnetic + dyon-like"
    requires_BSM: bool = True
    rescues_inspiral: bool = False
    coupling_required: Optional[float] = None
    notes: str = (
        "Chern-Simons gravity adds a parity-violating term theta * R*R"
        " * to the Einstein-Hilbert action. Spinning BHs can acquire a"
        " 'chirality charge' Q_CS ~ M^2 chi / (Planck scale). For"
        " antiparallel spins, Q_CS_1 + Q_CS_2 = 0 -- the two BHs'"
        " chirality charges CANCEL in the leading multipole, giving NO"
        " net binding. (For PARALLEL spins the chirality charges add"
        " and give a non-zero effect, but parallel-spin binaries don't"
        " have the toroidal CTC band.) "
        " VERDICT: the framework's antiparallel requirement DEFEATS"
        " the parity-violation rescue."
    )


def chirality_charge_sum_antiparallel(M: float, chi: float = 1.0) -> float:
    """Net chirality charge for ANTIPARALLEL spins: Q_1 + Q_2 = 0.

    The Chern-Simons charge of a Kerr BH is Q_CS ~ chi * M^2 (sign
    follows spin axis). Antiparallel: Q_1 = +chi M^2, Q_2 = -chi M^2.
    Sum = 0 exactly.
    """
    return 0.0


# ----- mechanism 9: vacuum-energy "Casimir" suppression of GW emission --


@dataclass(frozen=True)
class CasimirGWMechanism:
    name: str = "Vacuum-energy GW suppression"
    description: str = "Casimir-style negative-energy cavity between BHs"
    requires_BSM: bool = False     # uses standard QFT
    rescues_inspiral: bool = False
    coupling_required: Optional[float] = None
    notes: str = (
        "Inspired by the original Knopp Drive's Q-cavity feedback idea."
        " A Casimir-like vacuum-energy density between the two BHs"
        " could in principle modify the local stress-energy and the GW"
        " emission rate. The Casimir energy density scales as |E_Cas|"
        " ~ -hbar c / d^4. For d ~ 2M ~ km, this is ~10^-69 J/m^3 --"
        " 60+ orders of magnitude smaller than the GW back-reaction"
        " stress-energy. The Pfenning-Ford quantum inequality further"
        " bounds the integrated effect to NOT exceed |E_GW| * tau."
        " VERDICT: quantitatively too weak by many OOM."
    )


# Replace CATALOGUE with the full nine-mechanism set
CATALOGUE = (
    YukawaMechanism(),
    HardCoreMechanism(),
    ScalarTensorMechanism(),
    DarkPhotonMechanism(),
    LQGAreaMechanism(),
    FRGravityMechanism(),
    BraneworldMechanism(),
    ParityChargeMechanism(),
    CasimirGWMechanism(),
)


@dataclass(frozen=True)
class BindingVerdict:
    mechanism_name: str
    rescues_inspiral: bool
    coupling_required: Optional[float]
    physical_candidate: bool
    notes: str


def survey_binding_mechanisms(
    M: float = 1.0, d: float = 2.0, n_hardcore: int = 6,
    mu_yukawa: float = 0.0,
) -> list[BindingVerdict]:
    """Compute required coupling for each catalogued mechanism."""
    out = []
    # Yukawa
    out.append(BindingVerdict(
        mechanism_name=YukawaMechanism().name,
        rescues_inspiral=False,
        coupling_required=yukawa_required_coupling(M, d, mu_yukawa),
        physical_candidate=False,
        notes=YukawaMechanism().notes,
    ))
    # Hard core
    eps = hard_core_required_epsilon(M, d, n=n_hardcore)
    out.append(BindingVerdict(
        mechanism_name=HardCoreMechanism().name,
        rescues_inspiral=True,
        coupling_required=eps,
        physical_candidate=False,
        notes=HardCoreMechanism().notes,
    ))
    # Scalar-tensor
    out.append(BindingVerdict(
        mechanism_name=ScalarTensorMechanism().name,
        rescues_inspiral=False,
        coupling_required=None,
        physical_candidate=False,
        notes=ScalarTensorMechanism().notes,
    ))
    # Dark photon
    qD = dark_photon_required_charge(M, d)
    out.append(BindingVerdict(
        mechanism_name=DarkPhotonMechanism().name,
        rescues_inspiral=True,
        coupling_required=qD,
        physical_candidate=False,
        notes=DarkPhotonMechanism().notes,
    ))
    # LQG
    gap = lqg_area_gap_fractional(M_solar=M)
    out.append(BindingVerdict(
        mechanism_name=LQGAreaMechanism().name,
        rescues_inspiral=False,
        coupling_required=gap,
        physical_candidate=False,
        notes=LQGAreaMechanism().notes,
    ))
    # f(R) gravity
    out.append(BindingVerdict(
        mechanism_name=FRGravityMechanism().name,
        rescues_inspiral=False,
        coupling_required=None,
        physical_candidate=False,
        notes=FRGravityMechanism().notes,
    ))
    # Braneworld
    out.append(BindingVerdict(
        mechanism_name=BraneworldMechanism().name,
        rescues_inspiral=False,
        coupling_required=add_gravity_enhancement_factor(d=d) - 1.0,
        physical_candidate=False,
        notes=BraneworldMechanism().notes,
    ))
    # Parity charge
    out.append(BindingVerdict(
        mechanism_name=ParityChargeMechanism().name,
        rescues_inspiral=False,
        coupling_required=chirality_charge_sum_antiparallel(M),
        physical_candidate=False,
        notes=ParityChargeMechanism().notes,
    ))
    # Casimir GW suppression
    out.append(BindingVerdict(
        mechanism_name=CasimirGWMechanism().name,
        rescues_inspiral=False,
        coupling_required=None,
        physical_candidate=False,
        notes=CasimirGWMechanism().notes,
    ))
    return out


# ----- rescuability landscape -------------------------------------------


@dataclass(frozen=True)
class RescuabilityPoint:
    """A single point in the (force_law, range, sign) rescue-parameter
    space, with verdict on whether it could in principle rescue."""
    name: str               # short label
    sign: str               # "attractive" or "repulsive"
    power: float            # F = C / r^power
    range_M: float          # characteristic range in units of M (inf = long-range)
    rescues_inspiral: bool
    creates_stable_equilibrium: bool
    notes: str


def rescuability_landscape(
    M: float = 1.0, d: float = 2.0,
) -> list[RescuabilityPoint]:
    """Map the (sign, power, range) space of binding force laws onto a
    yes/no verdict.

    Stable equilibrium requirements:
      - A potential V(r) must have a local minimum at r = d_eq with
        V''(d_eq) > 0.
      - With Newtonian gravity -GM^2/r and an extra force +C/r^p
        (repulsive p > 1) or -C/r^p (attractive p > 1):
        for the COMBINATION to have a local minimum:
          attractive extra: no minimum (both forces inward), inspiral
          repulsive extra:  needs p > 1 (steeper than 1/r); minimum at
                            r_eq satisfying  GM^2 / r^2 = C p / r^(p+1)
                            -> r_eq = (C p / GM^2)^(1/(p-1)).
      - Stability requires V'' > 0 at r_eq, i.e., p > 1.
      - Range: if force is short-range (Yukawa), the minimum exists
        only if range > d_eq.

    This function classifies a representative grid of points.
    """
    landscape: list[RescuabilityPoint] = []
    # Attractive long-range Newtonian: the baseline. No equilibrium.
    landscape.append(RescuabilityPoint(
        name="Newtonian gravity",
        sign="attractive", power=2.0, range_M=float("inf"),
        rescues_inspiral=False,
        creates_stable_equilibrium=False,
        notes="The reference: -GM^2/r alone gives no equilibrium.",
    ))
    # Repulsive long-range 1/r^2 (dark photon-like): tunable equilibrium.
    landscape.append(RescuabilityPoint(
        name="Repulsive 1/r^2 (dark Coulomb)",
        sign="repulsive", power=2.0, range_M=float("inf"),
        rescues_inspiral=True,
        creates_stable_equilibrium=False,   # marginal: forces balance,
                                            # no second derivative
        notes=("Cancels Newtonian gravity exactly at all r if Q^2 = GM^2."
               " This is a 'free-floating' marginal equilibrium with no"
               " restoring force -- not a true stable point."),
    ))
    # Repulsive short-range 1/r^p (hard core, p > 2): true minimum.
    for p in (3.0, 4.0, 6.0, 12.0):
        landscape.append(RescuabilityPoint(
            name=f"Repulsive 1/r^{int(p)} (hard core)",
            sign="repulsive", power=p, range_M=float("inf"),
            rescues_inspiral=True,
            creates_stable_equilibrium=True,
            notes=(f"Creates a stable equilibrium at r_eq with"
                   f" V''(r_eq) > 0. Required coupling scales as"
                   f" M^2 d^{int(p-1)}. For p = {int(p)}, d = 2 M:"
                   f" epsilon ~ {hard_core_required_epsilon(M, d, n=int(p)):.2e}."),
        ))
    # Attractive long-range bonus (Yukawa, alpha < 1): no equilibrium.
    landscape.append(RescuabilityPoint(
        name="Attractive Yukawa (mu -> 0)",
        sign="attractive", power=2.0, range_M=1e3,
        rescues_inspiral=False,
        creates_stable_equilibrium=False,
        notes="Adds to Newton, accelerates inspiral.",
    ))
    # Short-range attractive: doesn't help in our regime either.
    landscape.append(RescuabilityPoint(
        name="Short-range attractive 1/r^4",
        sign="attractive", power=4.0, range_M=1.0,
        rescues_inspiral=False,
        creates_stable_equilibrium=False,
        notes="Even with finite range, attractive force gives no"
              " equilibrium for two bound masses.",
    ))
    # PHYSICALLY-MOTIVATED candidates: which fall in the rescue regime?
    # Only repulsive 1/r^p with p > 2 + sufficient strength does.
    return landscape


def summarise_landscape(landscape: list[RescuabilityPoint]) -> str:
    n_rescue = sum(1 for p in landscape if p.rescues_inspiral)
    n_stable = sum(1 for p in landscape if p.creates_stable_equilibrium)
    lines = [
        "Rescuability landscape:",
        f"  total force-law points:               {len(landscape)}",
        f"  rescues inspiral (force balance):     {n_rescue}",
        f"  creates a stable equilibrium V'' > 0: {n_stable}",
        "",
    ]
    for p in landscape:
        marker = "  ***" if p.creates_stable_equilibrium else "     "
        lines.append(
            f"{marker} {p.name} ({p.sign}, 1/r^{p.power})"
        )
        lines.append(f"        range = {p.range_M} M")
        lines.append(f"        rescues = {p.rescues_inspiral}, "
                     f"stable_eq = {p.creates_stable_equilibrium}")
    lines.append("")
    lines.append(
        "The only force-law class that creates a true stable equilibrium"
    )
    lines.append(
        "is REPULSIVE, SHORT-RANGE, with power-law steeper than 1/r^2"
    )
    lines.append(
        "(equivalently, F ~ 1/r^(p+1) with p >= 2; potential 1/r^p with"
    )
    lines.append(
        "p > 1). No known SM or beyond-SM force on Kerr BHs has this"
    )
    lines.append(
        "structure with the required coupling at d = 2 M."
    )
    return "\n".join(lines)


def summarise_binding_survey(verdicts: list[BindingVerdict]) -> str:
    n_candidates = sum(1 for v in verdicts if v.physical_candidate)
    n_rescues = sum(1 for v in verdicts if v.rescues_inspiral)
    lines = [
        "Non-gravitational binding mechanism survey:",
        f"  total catalogued mechanisms:        {len(verdicts)}",
        f"  could in principle rescue inspiral: {n_rescues}",
        f"  have physical candidate in SM/beyond: {n_candidates}",
        "",
    ]
    for v in verdicts:
        coup = (f"{v.coupling_required:.3e}" if v.coupling_required is not None
                else "n/a")
        lines.append(
            f"  - {v.mechanism_name}"
        )
        lines.append(
            f"      rescues?   {v.rescues_inspiral}"
            f"     coupling: {coup}"
            f"     candidate? {v.physical_candidate}"
        )
    lines.append("")
    if n_candidates == 0:
        lines.append(
            f"  VERDICT: rescue path (b) is CLOSED at the level of "
            f"currently-catalogued mechanisms. None of the {len(verdicts)} "
            "candidates couples to Kerr BHs at the right magnitude with "
            "a physically-motivated mechanism. Opening this path "
            "requires NEW physics beyond what's in the SM + leading "
            "beyond-SM proposals."
        )
    else:
        lines.append(
            f"  VERDICT: {n_candidates} candidate(s) flagged as "
            "physically possible. Inspect notes for required coupling "
            "and observational constraints."
        )
    return "\n".join(lines)
