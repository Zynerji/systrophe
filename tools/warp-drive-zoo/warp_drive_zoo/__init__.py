"""warp-drive-zoo: unified API for Alcubierre / Bobrick-Martire / Lentz warp drives.

Each Systrophe warp-drive module ships its own metric components, energy
density, and NEC_radial — but with different parameter signatures (r_s
for Alcubierre, (x, rho) for the others). This tool wraps the three
canonical drives behind a common protocol so they can be compared
head-to-head:

  - Sample NEC_radial at the wall location
  - Compute total negative exotic-matter content
  - Rank by exotic-matter requirement

The 3 drives:
  - **Alcubierre** (1994): canonical NEC-violating warp bubble.
  - **Bobrick-Martire** (2021): linearised positive-energy variant
    (NEC respected for m_ADM > 0; violated for m_ADM < 0).
  - **Lentz** (2020): two-soliton positive-energy proposal; soliton
    shear vs. v_s² competition.
"""

from __future__ import annotations

from .drives import (
    WarpDrive,
    AlcubierreDrive,
    BobrickMartireDrive,
    LentzDrive,
)
from .compare import (
    WarpDriveComparison,
    compare_drives,
    WarpDriveQIComparison,
    compare_drives_with_qi,
)
from .qi_scorer import (
    C_FORD_ROMAN,
    QIScore,
    QILambda2Separation,
    AlcubierreJupiterFloor,
    ford_roman_qi_bound,
    qi_normalized_score,
    score_member,
    t_kk_dust_min,
    t_kk_vacuum_min,
    t_kk_alcubierre_min,
    t_kk_morris_thorne_min,
    alcubierre_jupiter_floor,
    qi_lambda2_separation,
)
from .qi_registry import (
    score_van_stockum,
    score_godel,
    score_gott,
    score_kerr,
    score_alcubierre,
    score_wormhole_throat,
    score_full_registry,
)

__all__ = [
    "WarpDrive",
    "AlcubierreDrive",
    "BobrickMartireDrive",
    "LentzDrive",
    "WarpDriveComparison",
    "compare_drives",
    "WarpDriveQIComparison",
    "compare_drives_with_qi",
    # QI scorer
    "C_FORD_ROMAN",
    "QIScore",
    "QILambda2Separation",
    "AlcubierreJupiterFloor",
    "ford_roman_qi_bound",
    "qi_normalized_score",
    "score_member",
    "t_kk_dust_min",
    "t_kk_vacuum_min",
    "t_kk_alcubierre_min",
    "t_kk_morris_thorne_min",
    "alcubierre_jupiter_floor",
    "qi_lambda2_separation",
    # registry adapters
    "score_van_stockum",
    "score_godel",
    "score_gott",
    "score_kerr",
    "score_alcubierre",
    "score_wormhole_throat",
    "score_full_registry",
]
