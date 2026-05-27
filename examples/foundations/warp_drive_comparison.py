"""Warp-drive comparison: Alcubierre vs Krasnikov vs Lentz vs Bobrick-Martire vs Knopp Drive.

For a fixed mission profile (Earth-Mars equivalent, L=0.52 geometric
units), tabulate the integrated exotic-matter requirement and the
sustained drive power across the five warp-drive families. The Knopp
Drive is expected to dominate by orders of magnitude inside any
Tipler CTC band.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from systrophe.geometry.alcubierre import (
    alcubierre_total_negative_energy,
    pfenning_ford_quantum_bound,
)
from systrophe.geometry.bobrick_martire import bm_NEC_radial
from systrophe.knopp.knopp_drive import KnoppDriveConfig
from systrophe.knopp.knopp_traversal import knopp_traversal
from systrophe.geometry.krasnikov_tube import krasnikov_NEC_radial


def alcubierre_metric_E_neg(v_s: float, R: float, sigma: float) -> float:
    return abs(alcubierre_total_negative_energy(v_s, R, sigma))


def krasnikov_E_neg(alpha: float, x_range: tuple[float, float] = (-3.0, 3.0),
                    n_x: int = 201) -> float:
    xs = np.linspace(*x_range, n_x)
    vals = np.array([abs(krasnikov_NEC_radial(float(x), t=1.0,
                                                alpha=alpha)) for x in xs])
    return float(np.trapezoid(vals, xs))


def lentz_E_neg(v_s: float) -> float:
    """Lentz claim: subluminal soliton has positive T_tt everywhere.
    Approximated by zero integrated negative energy in the subluminal
    regime, and a small finite floor at v_s > 1.
    """
    if v_s < 1.0:
        return 0.0  # Lentz claim
    return 0.01 * (v_s - 1.0)  # nominal floor above c


def bobrick_martire_E_neg(m_ADM: float,
                            x_range: tuple[float, float] = (-2.0, 2.0),
                            rho_range: tuple[float, float] = (0.0, 3.0),
                            n_x: int = 21, n_rho: int = 21) -> float:
    """Linearised B-M integrated NEC. Negative iff m_ADM < 0."""
    if m_ADM >= 0:
        return 0.0
    xs = np.linspace(*x_range, n_x)
    rhos = np.linspace(*rho_range, n_rho)
    total = 0.0
    for x in xs:
        for rho in rhos:
            total += abs(bm_NEC_radial(float(x), float(rho), m_ADM=m_ADM))
    return float(total / (n_x * n_rho))


def knopp_E_neg(L: float, Q: float = 100.0, epsilon: float = 0.2) -> float:
    cfg = KnoppDriveConfig(Q=Q, epsilon_horn=epsilon)
    rep = knopp_traversal(cfg, distance=L, n_steps=80)
    return rep.exotic_matter_total


def main() -> None:
    L = 0.52  # Earth-Mars equivalent
    sigma = 4.0
    R_bubble = 1.0
    v_s = 1.0
    alpha_wall = 4.0
    m_ADM = -1.0
    Q = 100.0

    print("=" * 70)
    print(f"Warp-drive comparison: L = {L} (Earth-Mars equivalent)")
    print("=" * 70)
    print(f"  Common parameters: sigma={sigma}, R_bubble={R_bubble}, "
          f"v_s={v_s}")
    print()

    rows = [
        ("Alcubierre (1994)",
         alcubierre_metric_E_neg(v_s=v_s, R=R_bubble, sigma=sigma),
         "Original FTL bubble. NEC violation scales as v_s^2 sigma^3."),
        ("Krasnikov tube (1995)",
         krasnikov_E_neg(alpha=alpha_wall),
         "Permanent 1+1D corridor. NEC ~ alpha^2 in wall."),
        ("Lentz soliton (2021, subluminal claim)",
         lentz_E_neg(v_s=0.5),
         "Claim: NEC >= 0 everywhere at subluminal v_s."),
        ("Lentz soliton (superluminal v_s=2.0)",
         lentz_E_neg(v_s=2.0),
         "Cross-term turns on; small NEC floor."),
        ("Bobrick-Martire (m_ADM=-1)",
         bobrick_martire_E_neg(m_ADM=m_ADM),
         "Generalised class; NEC sign follows m_ADM."),
        ("Bobrick-Martire (m_ADM=+1)",
         bobrick_martire_E_neg(m_ADM=+1.0),
         "Positive m_ADM = positive E density (no NEC violation)."),
        (f"Knopp Drive (Q={Q})",
         knopp_E_neg(L, Q=Q),
         f"Composite: Tipler-gated + Krasnikov + Q-feedback + horn."),
        (f"Knopp Drive (Q=500)",
         knopp_E_neg(L, Q=500.0),
         "Same composite at higher feedback Q."),
    ]

    print(f"{'Family':46s} {'|E_neg| total':16s}  Comment")
    print("-" * 85)
    for label, e, comment in rows:
        print(f"  {label:44s} {e:14.4e}    {comment[:36]}")
    print()

    print("=" * 70)
    print("Headline shortcut")
    print("=" * 70)
    knopp_val = rows[-1][1]
    alc_val = rows[0][1]
    if knopp_val > 0:
        ratio = alc_val / knopp_val
        print(f"  Knopp Drive (Q=500) requires {ratio:.2e}x LESS exotic")
        print(f"  matter than Alcubierre at L = {L}.")
    elif knopp_val == 0.0:
        print(f"  Knopp Drive (Q=500) requires ZERO exotic matter at L = {L}.")
        print(f"  Alcubierre requires {alc_val:.4e} units (geometric).")
        print(f"  This is the HEADLINE Knopp shortcut: when the journey lies")
        print(f"  entirely inside a Tipler CTC band, the exotic-matter")
        print(f"  requirement vanishes by construction.")

    out_path = Path(__file__).parent / "warp_drive_comparison_results.json"
    out_path.write_text(json.dumps({
        "distance": L,
        "comparison": [{"family": r[0], "E_neg": r[1], "comment": r[2]}
                       for r in rows],
        "knopp_advantage": {
            "knopp_E_neg": knopp_val,
            "alcubierre_E_neg": alc_val,
            "ratio": "infinity (Knopp = 0)" if knopp_val == 0 else (alc_val / knopp_val),
        },
    }, indent=2, default=str))
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
