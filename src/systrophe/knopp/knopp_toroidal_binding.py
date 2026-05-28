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


CATALOGUE = (
    YukawaMechanism(),
    HardCoreMechanism(),
    ScalarTensorMechanism(),
    DarkPhotonMechanism(),
    LQGAreaMechanism(),
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
    return out


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
            "  VERDICT: rescue path (b) is CLOSED at the level of "
            "currently-catalogued mechanisms. None of the five "
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
