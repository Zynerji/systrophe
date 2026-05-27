"""Full traversable-wormhole map: classical, Casimir, and quantum (ER=EPR) routes.

Maps every known route to a traversable wormhole and its cost, to settle whether
ANY of them beats the ~(c^4/G)*size energy wall that warp metrics hit.

Route 1 - Classical Morris-Thorne
    Needs exotic (NEC-violating) matter at the throat,
        |rho_exotic| = b'(r_t)/(8 pi r_t^2)   (exotic_matter_accounting),
    total exotic energy ~ (c^4/G) * r_t * O(1). QI-bounded. ~Jupiter for a metre
    throat.

Route 2 - Casimir-supported (Visser thin-shell)
    Use the Brown-Maclay Casimir vacuum (casimir_throat) as the exotic source.
    exotic_matter_accounting shows the required plate separation
    d_req = (pi^3 r_t^2 / 90 b')^{1/4} ~ 0.7 r_t does NOT fit inside the throat:
    static Casimir is quantitatively insufficient.

Route 3 - Quantum ER=EPR / Gao-Jafferis-Wall
    Two entangled black holes + a double-trace boundary coupling produce a
    NEGATIVE averaged null energy that props the wormhole open (GJW 2017,
    Maldacena-Qi 2018; demonstrated for one qubit on a quantum processor 2022).
    The negative energy is sourced by ENTANGLEMENT, not a slab of exotic matter
    -- so it removes the "unobtainium" problem. BUT by Ryu-Takayanagi
    (S = Area/4 G_N; the HHmL ads_cft_entanglement machinery) a throat of area A
    requires entanglement entropy S = A/(4 ell_P^2). For a macroscopic throat
    this is the Bekenstein-Hawking entropy of a same-size black hole
    (~(r_t/ell_P)^2 ebits ~ 1e70 for a metre throat), and the dual mass is
    M ~ (c^2/G) r_t/2 -- the SAME ~(c^4/G) r_t wall, in entanglement language.

Headline (honest)
-----------------
The wormhole ENERGY wall and the ER=EPR ENTANGLEMENT wall are DUAL: a traversable
throat of radius r_t costs either ~(c^4/G) r_t of exotic energy OR ~(r_t/ell_P)^2
ebits of pre-shared entanglement, both equivalent to assembling a black hole of
radius r_t (~Jupiter for a metre). For INFORMATION (a few qubits) the quantum
wormhole genuinely works; for MATTER/ship transport every route hits the same
Bekenstein-bounded wall. The wormhole does not beat it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.quantum_info.er_epr_pair import wormhole_throat_area
from systrophe.qftcs.exotic_matter_accounting import (
    morris_thorne_exotic_density,
    required_plate_separation,
)
from systrophe.knopp.knopp_ratchet import _C_SI, _G_SI, _GEOM_ENERGY_PER_METRE_J, _JUPITER_MASS_KG
from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.warp_geometry import PLANCK_LENGTH_M

_LN2 = math.log(2.0)


def morris_thorne_exotic_energy_J(
    r_throat_m: float, db_dr_at_throat: float = 0.5
) -> float:
    """Classical exotic-matter energy of a Morris-Thorne throat (SI joules).

    |rho| = b'/(8 pi r_t^2); integrated over a throat shell of volume
    ~ 4 pi r_t^2 * r_t gives geometrized E ~ b' r_t / 2; x c^4/G for joules.
    """
    rho = morris_thorne_exotic_density(r_throat_m, db_dr_at_throat)  # length^-2
    e_geom_m = rho * 4.0 * math.pi * r_throat_m ** 2 * r_throat_m   # length
    return float(abs(e_geom_m) * _GEOM_ENERGY_PER_METRE_J)


def casimir_throat_fits(r_throat_m: float, db_dr_at_throat: float = 0.5) -> dict:
    """Can a static Casimir cavity supply the throat exotic matter? (geometry).

    Returns the required plate separation and whether it fits within the throat
    (d_req <= r_t). Per exotic_matter_accounting it does not (d_req ~ 0.7 r_t and
    the cavity cannot be both the throat wall and fit inside it self-consistently
    once QI is imposed).
    """
    d_req = required_plate_separation(r_throat_m, db_dr_at_throat)
    return {
        "d_required_m": float(d_req),
        "d_to_throat_ratio": float(d_req / r_throat_m),
        "fits": bool(d_req <= 0.1 * r_throat_m),  # need d << r_t for a real cavity
    }


def erepr_entanglement_ebits(r_throat_m: float) -> float:
    """Pre-shared entanglement (ebits) for an ER=EPR throat of radius r_throat.

    Ryu-Takayanagi: S = A/(4 ell_P^2). Spherical throat A = 4 pi r_t^2, so
    S_nats = pi r_t^2 / ell_P^2; ebits = S_nats / ln 2.
    """
    A = 4.0 * math.pi * r_throat_m ** 2
    s_nats = A / (4.0 * PLANCK_LENGTH_M ** 2)
    return float(s_nats / _LN2)


def dual_black_hole_mass_energy_J(r_throat_m: float) -> float:
    """Mass-energy of a Schwarzschild black hole of horizon radius r_throat.

    r_s = 2 G M / c^2 => M c^2 = c^4 r_t / (2 G). This is the ER=EPR dual of the
    entanglement requirement and matches the classical exotic energy to O(1).
    """
    return float(_C_SI ** 4 * r_throat_m / (2.0 * _G_SI))


@dataclass(frozen=True)
class WormholeMapReport:
    r_throat_m: float
    # Route 1: classical Morris-Thorne
    classical_exotic_energy_J: float
    classical_jupiter_masses: float
    # Route 2: Casimir
    casimir_fits: bool
    casimir_d_to_throat_ratio: float
    # Route 3: quantum ER=EPR
    erepr_entanglement_ebits: float
    dual_black_hole_mass_energy_J: float
    dual_jupiter_masses: float
    energy_entanglement_duality_ratio: float   # classical_E / dual_BH_E (~O(1))
    # Transport capability (honest)
    transports_information: bool                # yes (a few qubits), quantum route
    transports_macroscopic_matter: bool         # no for any route at lab budgets
    lab_source_J: float
    residual_oom: float
    beats_the_wall: bool                        # False
    novelty_verdict: str
    novelty_n_sharp: int


def map_wormhole(
    r_throat_m: float = 1.0,
    db_dr_at_throat: float = 0.5,
    lab_source_J: float = 1e-15,
) -> WormholeMapReport:
    """Map all three wormhole routes for a throat of radius r_throat_m."""
    e_classical = morris_thorne_exotic_energy_J(r_throat_m, db_dr_at_throat)
    cas = casimir_throat_fits(r_throat_m, db_dr_at_throat)
    s_ebits = erepr_entanglement_ebits(r_throat_m)
    e_dual = dual_black_hole_mass_energy_J(r_throat_m)

    duality = e_classical / e_dual if e_dual > 0 else float("nan")
    residual = math.log10(e_classical / max(lab_source_J, 1e-300))

    # catcher over throat radius -> classical exotic energy (power law -> smooth)
    rt_grid = np.logspace(-15, 2, 40)

    def fn(rt: float) -> np.ndarray:
        return np.array([morris_thorne_exotic_energy_J(rt, db_dr_at_throat)])

    nov = scan_novelty(rt_grid, fn, n_bits=32, parameter_label="r_throat")

    return WormholeMapReport(
        r_throat_m=float(r_throat_m),
        classical_exotic_energy_J=float(e_classical),
        classical_jupiter_masses=float(e_classical / (_JUPITER_MASS_KG * _C_SI ** 2)),
        casimir_fits=bool(cas["fits"]),
        casimir_d_to_throat_ratio=float(cas["d_to_throat_ratio"]),
        erepr_entanglement_ebits=float(s_ebits),
        dual_black_hole_mass_energy_J=float(e_dual),
        dual_jupiter_masses=float(e_dual / (_JUPITER_MASS_KG * _C_SI ** 2)),
        energy_entanglement_duality_ratio=float(duality),
        transports_information=True,    # GJW: ~O(1) qubits per protocol run
        transports_macroscopic_matter=False,
        lab_source_J=float(lab_source_J),
        residual_oom=float(residual),
        beats_the_wall=False,
        novelty_verdict=nov.verdict,
        novelty_n_sharp=len(nov.sharp_features),
    )


def summarise_wormhole_map(r: WormholeMapReport) -> str:
    return (
        f"WormholeMap r_t={r.r_throat_m} m: "
        f"classical_exotic={r.classical_exotic_energy_J:.2e} J "
        f"({r.classical_jupiter_masses:.2g} Jupiter); "
        f"casimir_fits={r.casimir_fits}; "
        f"ER=EPR entanglement={r.erepr_entanglement_ebits:.2e} ebits "
        f"(dual BH {r.dual_jupiter_masses:.2g} Jupiter); "
        f"energy/entanglement duality={r.energy_entanglement_duality_ratio:.2f}; "
        f"info={r.transports_information}, matter={r.transports_macroscopic_matter}; "
        f"BEATS_WALL={r.beats_the_wall}"
    )
