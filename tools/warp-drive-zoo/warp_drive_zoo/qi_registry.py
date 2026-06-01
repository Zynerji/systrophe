"""Registry adapters: score Goedel / Gott / Kerr / van Stockum / wormhole /
Alcubierre on the SAME Ford-Roman quantum-inequality realizability axis.

Each adapter knows the matter source of its spacetime, builds the most-
negative null-projected stress excursion T_kk_min from it (closed form
where no Hadamard vacuum is available), picks the macroscopic Ford-Roman
sampling time tau, and returns a uniform ``QIScore``.

Classical CTC spacetimes (positive-energy dust or vacuum) score < 1
(realizable); the warp / wormhole constructions score > 1 (QI-forbidden).
"""

from __future__ import annotations

import numpy as np

from systrophe.geometry.spacetimes.godel import GodelUniverse
from systrophe.geometry.spacetimes.gott import GottPair
from systrophe.geometry.spacetimes.kerr import KerrSpacetime
from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.qftcs.energy_conditions import proper_energy_density

from .qi_scorer import (
    QIScore,
    score_member,
    t_kk_alcubierre_min,
    t_kk_dust_min,
    t_kk_morris_thorne_min,
    t_kk_vacuum_min,
)


def score_van_stockum(
    vs: VanStockumInterior | None = None,
    omega: float = 0.8,
    R: float = 1.0,
    n_grid: int = 400,
) -> QIScore:
    """van Stockum / Tipler dust column.

    Source: pressureless dust, rho(r) = (omega^2/2pi) e^{omega^2 r^2} >= 0
    everywhere (energy_conditions proves NEC/WEC/SEC/DEC all hold). The
    most-negative T_kk excursion from the source is therefore 0; score = 0.
    Sampling time tau = R (cylinder radius, the macroscopic CTC scale).
    """
    if vs is None:
        vs = VanStockumInterior(omega=omega, R=R)
    rs = np.linspace(1e-3, vs.R, n_grid)
    rho = proper_energy_density(vs.omega, rs)
    t_kk_min = t_kk_dust_min(float(rho.min()))
    return score_member("van_stockum", t_kk_min, tau=float(vs.R),
                        source_kind="dust")


def score_godel(g: GodelUniverse | None = None, a: float = 1.0) -> QIScore:
    """Goedel rotating-dust universe.

    Source: positive-energy dust rho = 1/(8 pi a^2) > 0 plus a cosmological
    constant (which is not exotic matter). Most-negative source T_kk = 0;
    score = 0. Sampling time tau = a (the fundamental length / CTC scale,
    r_CTC = arcsinh(1) ~ 0.88 in these units, O(a)).
    """
    if g is None:
        g = GodelUniverse(a=a)
    t_kk_min = t_kk_dust_min(float(g.dust_density))
    return score_member("godel", t_kk_min, tau=float(g.a),
                        source_kind="dust")


def score_gott(
    gp: GottPair | None = None, mu: float = 0.1, v: float = 0.95,
) -> QIScore:
    """Gott two-cosmic-string pair.

    Source: two parallel cosmic strings, each positive mass per unit length
    mu > 0 (positive tension; the string core satisfies the WEC). The
    *exterior* through which the CTC threads is a conical Einstein vacuum
    (T_mu_nu = 0). Most-negative T_kk along the CTC = 0; score = 0.
    Sampling time tau ~ the conical impact-parameter scale; we use the
    string separation proxy 1 (geometrised) — the score is 0 regardless of
    tau because the exterior is vacuum.
    """
    if gp is None:
        gp = GottPair(mu=mu, v=v)
    # The CTC lives in the vacuum exterior between the strings.
    t_kk_min = t_kk_vacuum_min()
    return score_member("gott", t_kk_min, tau=1.0, source_kind="vacuum")


def score_kerr(
    k: KerrSpacetime | None = None, M: float = 1.0, a: float = 1.5,
) -> QIScore:
    """Kerr rotating black hole / over-extremal naked time machine.

    Source: exact Einstein *vacuum* (T_mu_nu = 0). The ring-CTC region is
    sourced by the collapse of ordinary (positive-energy) matter, and the
    CTC locus itself is vacuum. Most-negative T_kk = 0; score = 0.
    Sampling time tau ~ M (the mass / ring scale).
    """
    if k is None:
        k = KerrSpacetime(M=M, a=a)
    t_kk_min = t_kk_vacuum_min()
    return score_member("kerr", t_kk_min, tau=float(k.M), source_kind="vacuum")


def score_alcubierre(
    v_s: float = 1.0, R: float = 1.0, sigma: float = 8.0,
    tau: float | None = None,
) -> QIScore:
    """Alcubierre warp bubble.

    Source: NO classical matter / no Hadamard vacuum available — the wall
    requires exotic stress-energy. T_kk_min is built from the closed-form
    Alcubierre density T_tt = -(v_s^2/32pi)(df/dr)^2 < 0. The honest
    sampling time is the MACROSCOPIC bubble radius R (the crew must hold the
    wall open across the whole journey, not just the Planck-thin wall
    crossing 1/sigma), so tau = R by default.
    """
    if tau is None:
        tau = R
    t_kk_min = t_kk_alcubierre_min(v_s=v_s, R=R, sigma=sigma)
    return score_member("alcubierre", t_kk_min, tau=float(tau),
                        source_kind="exotic")


def score_wormhole_throat(
    r_throat: float = 1.0, db_dr_at_throat: float = 0.5,
    tau: float | None = None,
) -> QIScore:
    """Morris-Thorne traversable wormhole throat.

    Source: NO vacuum — a true Morris-Thorne throat needs LOCAL exotic
    matter (the defining NEC violation rho + p_r < 0). T_kk_min is built
    from the published exotic density |rho_exotic| = b'(r_t)/(8 pi r_t^2)
    (Morris-Thorne 1988 Eq. 32). Sampling time tau = r_throat (the
    macroscopic throat scale over which the negative energy is sustained).
    """
    if tau is None:
        tau = r_throat
    t_kk_min = t_kk_morris_thorne_min(r_throat=r_throat,
                                      db_dr_at_throat=db_dr_at_throat)
    return score_member("wormhole_throat", t_kk_min, tau=float(tau),
                        source_kind="exotic")


def score_full_registry(
    omega: float = 0.8,
    godel_a: float = 1.0,
    gott_mu: float = 0.1,
    gott_v: float = 0.95,
    kerr_M: float = 1.0,
    kerr_a: float = 1.5,
    alcubierre_v_s: float = 1.0,
    alcubierre_R: float = 1.0,
    alcubierre_sigma: float = 8.0,
    wormhole_r_throat: float = 1.0,
) -> list[QIScore]:
    """Score every registry member on the uniform QI realizability axis.

    Returns a list of ``QIScore`` (classical members first, exotic last),
    suitable for ``qi_lambda2_separation``.
    """
    return [
        score_van_stockum(omega=omega),
        score_godel(a=godel_a),
        score_gott(mu=gott_mu, v=gott_v),
        score_kerr(M=kerr_M, a=kerr_a),
        score_alcubierre(v_s=alcubierre_v_s, R=alcubierre_R,
                         sigma=alcubierre_sigma),
        score_wormhole_throat(r_throat=wormhole_r_throat),
    ]
