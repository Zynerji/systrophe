"""wormhole-throat: scan + diagnose van Stockum CTC-band-boundary throats.

Wraps `systrophe.wormhole_throat` and `systrophe.casimir_throat` in
an LPAnalyser-style explorer:

  - find candidate wormhole throats (loci of L=0)
  - audit flaring-out condition b'(r) < 1
  - Brown-Maclay Casimir energy density at the throat for a chosen
    plate separation
  - Z_3-cover interpretation diagnostic
"""

from __future__ import annotations

from .explorer import (
    WormholeThroatExplorer,
    ThroatReport,
)
from .casimir import (
    CasimirThroatReport,
    casimir_at_throat,
)

__all__ = [
    "WormholeThroatExplorer",
    "ThroatReport",
    "CasimirThroatReport",
    "casimir_at_throat",
]
