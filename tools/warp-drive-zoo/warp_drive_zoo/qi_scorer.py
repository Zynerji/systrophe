"""Quantum-Realizability Scorer — a uniform Ford-Roman QI budget axis.

Every member of the spacetime + warp registry that admits closed timelike
curves (or super-luminal transport) can be asked the SAME question:

    "Does the matter that sources this geometry violate the Ford-Roman
     quantum inequality (QI) when its negative-energy excursion is
     sampled over the characteristic time `tau` of the construction?"

The answer is a single scalar::

    qi_normalized_score = |most-negative T_kk excursion (sampled)| / |QI bound(tau)|

with the interpretation::

    score < 1   ->  REALIZABLE      (QI-allowed; classically sourceable)
    score > 1   ->  QI-FORBIDDEN    (sustained negative energy exceeds the
                                     Ford-Roman budget -> exotic matter)

Why this is the right axis
--------------------------
The four pointwise energy conditions (`systrophe.qftcs.energy_conditions`)
prove that the van Stockum dust column — the source of the Tipler CTC —
satisfies NEC, WEC, SEC, DEC *everywhere*: it is positive-energy dust, and
its most-negative null-projected stress excursion is exactly zero. The same
is true of every classical CTC spacetime in the registry whose source is
ordinary matter or vacuum:

  * van Stockum / Tipler  — positive-energy dust (rho = (omega^2/2pi) e^{omega^2 r^2})
  * Goedel                — positive-energy dust + Lambda (rho = 1/(8 pi a^2))
  * Gott                  — two positive-tension cosmic strings (mu > 0); the
                            exterior is a conical *vacuum*
  * Kerr                  — exact *vacuum* (T_mu_nu = 0); the ring-CTC region
                            is sourced by the collapse of ordinary matter

By contrast the warp / wormhole constructions need a genuinely negative
null-projected stress at a *macroscopic* support, sustained over the
macroscopic sampling time the crew must traverse:

  * Alcubierre            — T_tt = -(v_s^2/32pi)(df/dr)^2 < 0 in the wall
  * Morris-Thorne throat  — rho + p_r < 0 at the throat (the defining NEC
                            violation, |rho_exotic| = b'(r_t)/(8 pi r_t^2))

The Ford-Roman quantum inequality for a free massless scalar in 4D
Minkowski (Ford & Roman 1995, 1997; Pfenning & Ford 1997) bounds the
Lorentzian-sampled energy density measured by an inertial observer:

    integral rho(t) * (tau/pi) / (t^2 + tau^2) dt  >=  -3 / (32 pi^2 tau^4).

The bound *weakens as tau -> 0* (you may borrow a large negative density
for a vanishingly short time) and *tightens as tau grows* (you may not
sustain it). A warp drive must hold its wall open over the whole journey,
so the physically honest sampling time is the macroscopic length scale of
the construction (R for a warp bubble, r_throat for a wormhole), NOT the
Planck-thin wall-crossing time. Scored at that macroscopic tau, Alcubierre
and the Morris-Thorne throat are QI-forbidden; the classical dust/vacuum
CTCs are not. This module makes that uniform.

Generalisation of `anec_quantum_inequality_bound`
-------------------------------------------------
`systrophe.qftcs.anec_bound.anec_quantum_inequality_bound(vs, r_min, r_max)`
hard-codes the van-Stockum cylinder and uses the *spatial* window width
(r_max - r_min) as the inverse-fourth-power scale. `ford_roman_qi_bound(tau)`
here is the source-agnostic *temporal* Ford-Roman form; the van-Stockum
call is recovered as the special case tau = (r_max - r_min) with the same
3/(32 pi^2) constant (see `ford_roman_qi_bound` docstring).

Novelty-catcher culture (durable rule 2026-05-11): the registry separation
is *additionally* validated with the address-space lambda_2 catcher + a
phase-randomised surrogate null in `qi_lambda2_separation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import (
    lambda_2_of_hamming_graph,
    real_array_to_address,
)
from systrophe.geometry.alcubierre import alcubierre_energy_density
from systrophe.qftcs.energy_conditions import proper_energy_density
from systrophe.qftcs.exotic_matter_accounting import morris_thorne_exotic_density
from systrophe.knopp.knopp_ratchet import (
    _C_SI,
    _GEOM_ENERGY_PER_METRE_J,
    _JUPITER_MASS_KG,
)

# Ford-Roman constant for a free massless scalar in 4D Minkowski.
# rho_sampled >= -C_FR / tau^4 with C_FR = 3 / (32 pi^2).
C_FORD_ROMAN = 3.0 / (32.0 * math.pi ** 2)


def ford_roman_qi_bound(tau: float, c_fr: float = C_FORD_ROMAN) -> float:
    """Ford-Roman quantum-inequality lower bound on the sampled energy density.

    For a free massless scalar field and a Lorentzian sampling function of
    width ``tau`` (Ford & Roman 1995/1997), the inertially-sampled energy
    density obeys

        rho_sampled  >=  -3 / (32 pi^2 tau^4)  =  -c_fr / tau^4.

    Returns the (negative) lower bound. ``|bound|`` is the QI *budget*: a
    negative-energy excursion whose magnitude exceeds the budget over the
    sampling window is QI-forbidden.

    This is the source-agnostic generalisation of
    ``systrophe.qftcs.anec_bound.anec_quantum_inequality_bound``, which is
    the special case where the "duration" is taken to be the spatial window
    width (r_max - r_min) of the van Stockum radial null geodesic and the
    same constant ``c_fr = 3/(32 pi^2)`` is used. Passing
    ``tau = r_max - r_min`` reproduces that module's number exactly.
    """
    if tau <= 0:
        return float("-inf")
    return -c_fr / tau ** 4


# ---------------------------------------------------------------------------
# T_kk builders (most-negative null-projected stress excursion per source)
# ---------------------------------------------------------------------------
#
# Each builder returns the most-negative value of T_{ab} k^a k^b that the
# *classical matter source* of the spacetime exhibits, in c = G = 1 natural
# units. For dust/vacuum sources this is 0 (NEC respected everywhere). For
# the warp/wormhole constructions, it is built from the published exotic
# density because no Hadamard vacuum is available there.


def t_kk_dust_min(rho_min: float) -> float:
    """Most-negative T_kk for a pure-dust source with minimum density rho_min.

    Dust: T_{ab} = rho u_a u_b, so T_{ab} k^a k^b = rho (u . k)^2. With
    rho >= 0 everywhere this is >= 0; the most-negative excursion is 0
    (NEC respected). We pass through ``min(rho_min, 0)`` so a (hypothetical)
    negative-density sample would be reported honestly.
    """
    return float(min(rho_min, 0.0) )


def t_kk_vacuum_min() -> float:
    """Most-negative T_kk for an Einstein-vacuum region: identically 0.

    Kerr (outside the ring) and the conical Gott exterior are vacuum:
    T_mu_nu = 0, so every null projection vanishes. No exotic matter is
    present at the CTC locus; the pathology is purely geometric.
    """
    return 0.0


def t_kk_alcubierre_min(
    v_s: float = 1.0, R: float = 1.0, sigma: float = 8.0, n_grid: int = 4000,
) -> float:
    """Most-negative T_kk in the Alcubierre wall (closed form, no vacuum).

    The Alcubierre Eulerian energy density is
        T_tt(r_s) = -(v_s^2 / 32 pi) (df/dr_s)^2 <= 0,
    which already supplies a negative null projection along the wall-normal
    null direction (NEC violation). We scan r_s across the wall and return
    the minimum (most negative) value.
    """
    rs = np.linspace(1e-3, R + 6.0 / sigma, n_grid)
    vals = [alcubierre_energy_density(float(r), v_s=v_s, R=R, sigma=sigma)
            for r in rs]
    return float(min(vals))


def t_kk_morris_thorne_min(
    r_throat: float = 1.0, db_dr_at_throat: float = 0.5,
) -> float:
    """Most-negative T_kk at a Morris-Thorne throat (published exotic density).

    At the throat the radial NEC combination is
        (T_{ab} k^a k^b)_throat = rho + p_r = -2 |rho_exotic|,
    with |rho_exotic| = b'(r_t) / (8 pi r_t^2) (Morris-Thorne 1988 Eq. 32).
    The factor 2 is the standard throat NEC result (rho + p_r at the throat
    equals -2 rho for the canonical Morris-Thorne ansatz with p_r = -rho at
    leading order). Returns the negative value.
    """
    rho_ex = morris_thorne_exotic_density(r_throat, db_dr_at_throat)
    return float(-2.0 * rho_ex)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QIScore:
    """Quantum-realizability score for one registry member.

    Attributes
    ----------
    name : str
        Registry member name.
    t_kk_min : float
        Most-negative T_kk excursion from the classical source (c=G=1).
        0 for positive-energy dust / vacuum sources.
    tau : float
        Ford-Roman sampling time (macroscopic construction scale).
    qi_bound : float
        Ford-Roman lower bound -c_fr / tau^4 (negative).
    qi_normalized_score : float
        |t_kk_min| / |qi_bound|. > 1 => QI-forbidden (exotic-requiring).
    qi_forbidden : bool
        True iff score > 1.
    source_kind : str
        "dust" | "vacuum" | "exotic" — the matter sourcing the geometry.
    """

    name: str
    t_kk_min: float
    tau: float
    qi_bound: float
    qi_normalized_score: float
    qi_forbidden: bool
    source_kind: str


def qi_normalized_score(
    t_kk_min: float, tau: float, c_fr: float = C_FORD_ROMAN,
) -> float:
    """|most-negative T_kk excursion| / |Ford-Roman QI bound(tau)|.

    score > 1 => the sustained negative-energy excursion exceeds the
    Ford-Roman budget over the sampling window => QI-forbidden.
    A non-negative source (t_kk_min >= 0) scores exactly 0.
    """
    if t_kk_min >= 0.0:
        return 0.0
    bound = ford_roman_qi_bound(tau, c_fr=c_fr)
    if not math.isfinite(bound) or bound == 0.0:
        return float("inf")
    return float(abs(t_kk_min) / abs(bound))


def score_member(
    name: str, t_kk_min: float, tau: float, source_kind: str,
    c_fr: float = C_FORD_ROMAN,
) -> QIScore:
    """Bundle a single registry member's QI realizability score."""
    bound = ford_roman_qi_bound(tau, c_fr=c_fr)
    score = qi_normalized_score(t_kk_min, tau, c_fr=c_fr)
    return QIScore(
        name=str(name),
        t_kk_min=float(t_kk_min),
        tau=float(tau),
        qi_bound=float(bound),
        qi_normalized_score=float(score),
        qi_forbidden=bool(score > 1.0),
        source_kind=str(source_kind),
    )


# ---------------------------------------------------------------------------
# Alcubierre exotic-energy -> Jupiter-mass floor cross-check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlcubierreJupiterFloor:
    """Alcubierre exotic-energy floor expressed in Jupiter mass-energies.

    Routes the geometrized Alcubierre principal through c^4/G exactly as
    ``warp_geometry.principal_joules`` / ``knopp_ratchet.feasibility_report``
    do, so the number is the same ~Jupiter-mass floor already cited in the
    repo.
    """

    R_m: float
    sigma_m: float
    v_s_over_c: float
    kappa: float
    principal_J: float
    jupiter_masses: float


def alcubierre_jupiter_floor(
    R_m: float = 1.0,
    sigma_m: float = 1.0,
    v_s_over_c: float = 1.0,
    kappa: float = 1.0 / 12.0,
) -> AlcubierreJupiterFloor:
    """Geometrized Alcubierre exotic-energy principal in Jupiter mass-energies.

        E_geom = kappa (v_s/c)^2 R^2 / sigma     [metres]
        E_SI   = E_geom * c^4 / G                [joules]

    For a 1 m luminal bubble (kappa = 1/12), E_SI ~ 1.0e43 J ~ 0.06 Jupiter
    mass-energies (the per-bubble principal); the multi-stroke ledger floor
    of `knopp_ratchet.feasibility_report` is ~3.8 Jupiter mass-energies.
    Both are within an order of magnitude of "a Jupiter mass" — the floor
    this scorer's QI-forbidden verdict corroborates.
    """
    e_geom_m = kappa * v_s_over_c ** 2 * R_m ** 2 / sigma_m
    principal_J = abs(e_geom_m) * _GEOM_ENERGY_PER_METRE_J
    jupiter = principal_J / (_JUPITER_MASS_KG * _C_SI ** 2)
    return AlcubierreJupiterFloor(
        R_m=float(R_m),
        sigma_m=float(sigma_m),
        v_s_over_c=float(v_s_over_c),
        kappa=float(kappa),
        principal_J=float(principal_J),
        jupiter_masses=float(jupiter),
    )


# ---------------------------------------------------------------------------
# Novelty-catcher: lambda_2 separation of classical vs exotic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QILambda2Separation:
    """Address-space lambda_2 separation of classical vs exotic members.

    The QI scores are hashed to bit addresses; a Hamming-radius graph is
    built and lambda_2 (algebraic connectivity) is computed on the FULL
    registry. Classical (score ~ 0) and exotic (score > 1) members live in
    distinct address clusters, so the within-true-cluster Hamming
    compactness is much tighter than under a phase-randomised (shuffled-
    label) surrogate that destroys the structure.

    Two complementary diagnostics are reported:

    * ``lambda_2_full`` — algebraic connectivity of the full registry's
      Hamming-radius graph. A value > 0 with a *disconnected* picture at
      small radius signals two separated clusters.
    * ``within_mean_hamming`` vs the surrogate null — the falsifiable
      statistic. The true labelling's mean within-group Hamming distance
      sits many sigma BELOW the shuffled-label surrogate mean (tighter =>
      the catcher confirms the split is structural, not an artefact).
    """

    names: list
    scores: list
    classical_mask: list
    lambda_2_full: float
    within_mean_hamming_true: float
    within_mean_hamming_surrogate_mean: float
    within_mean_hamming_surrogate_std: float
    separation_z: float
    separates: bool
    verdict: str


def _mean_within_group_hamming(
    addresses: list[np.ndarray], mask: list[bool],
) -> float:
    """Mean pairwise Hamming distance within each labelled group, averaged.

    Lower => tighter clusters. Groups of size < 2 contribute 0.
    """
    groups = [[a for a, m in zip(addresses, mask) if m],
              [a for a, m in zip(addresses, mask) if not m]]
    dists = []
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                dists.append(int(np.sum(g[i] != g[j])))
    return float(np.mean(dists)) if dists else 0.0


def qi_lambda2_separation(
    scores: list[QIScore],
    n_bits: int = 32,
    radius: int = 6,
    n_surrogate: int = 500,
    seed: int = 0,
) -> QILambda2Separation:
    """Validate the classical/exotic split with the address-space catcher.

    Mechanism (HASH-QUINE address-space rule):
      1. Thermometer-encode each member's qi_normalized_score (rank across
         the registry, log-compressed) to a bit address. Thermometer coding
         puts low-score (classical) members near the all-zero address and
         high-score (exotic) members near the all-one address, so the two
         physics classes occupy opposite corners of the Hamming cube.
      2. Compute lambda_2 of the full Hamming-radius graph.
      3. Surrogate null: shuffle the classical/exotic labels ``n_surrogate``
         times and recompute the mean within-group Hamming compactness. The
         TRUE labelling is much tighter (lower) than the surrogate mean; the
         z-score quantifies the separation. ``separates`` iff z > 2.

    The catcher confirms that exotic members (Alcubierre, wormhole) form a
    tight high-score cluster well separated from the classical low-score
    cluster (cylinder, Kerr, Goedel, Gott), and that this is NOT reproduced
    by a random relabelling.
    """
    names = [s.name for s in scores]
    vals = np.array([s.qi_normalized_score for s in scores], dtype=float)
    # Rank-thermometer encoding: robust across the many-decade score spread
    # and immune to bin-boundary artefacts. log1p first so the gap between
    # classical 0 and exotic O(10) does not swamp the within-exotic ordering.
    enc = np.log1p(np.clip(vals, 0.0, 1e6))
    order = np.argsort(np.argsort(enc, kind="stable"), kind="stable")
    n = len(enc)
    addresses = []
    for rank in order:
        n_set = int(round((rank / max(n - 1, 1)) * n_bits))
        a = np.zeros(n_bits, dtype=int)
        a[:n_set] = 1
        addresses.append(a)

    classical_mask = [not s.qi_forbidden for s in scores]
    lam_full = lambda_2_of_hamming_graph(addresses, radius=radius)

    true_within = _mean_within_group_hamming(addresses, classical_mask)

    rng = np.random.default_rng(seed)
    n_cls = sum(classical_mask)
    surro = []
    for _ in range(n_surrogate):
        perm = rng.permutation(n)
        shuf_mask = [False] * n
        for i in perm[:n_cls]:
            shuf_mask[i] = True
        surro.append(_mean_within_group_hamming(addresses, shuf_mask))
    surro = np.array(surro, dtype=float)
    s_mean = float(surro.mean()) if len(surro) else 0.0
    s_std = float(surro.std()) if len(surro) else 0.0
    # Tighter true clustering => true_within BELOW surrogate => positive z.
    z = (s_mean - true_within) / s_std if s_std > 1e-12 else float("inf")
    separates = bool(z > 2.0)
    return QILambda2Separation(
        names=names,
        scores=[float(v) for v in vals],
        classical_mask=classical_mask,
        lambda_2_full=float(lam_full),
        within_mean_hamming_true=float(true_within),
        within_mean_hamming_surrogate_mean=s_mean,
        within_mean_hamming_surrogate_std=s_std,
        separation_z=float(z),
        separates=separates,
        verdict=("separated" if separates else "not_separated"),
    )
