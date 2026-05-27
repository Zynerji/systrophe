"""Additional analytically-tractable CTC spacetimes.

Each module here implements one closed-form CTC spacetime with
shared interfaces for metric components, CTC region detection,
and circular-orbit utilities. Currently:

- godel: Goedel rotating-dust universe (Goedel 1949)

Planned (see ROADMAP.md):
- gott: Gott pair (cosmic strings, 1991)
- kerr_inner: Kerr ring-singularity inner region
"""

from .godel import GodelUniverse, godel_ctc_radius
from .gott import GottPair, gott_critical_velocity, gott_critical_mu
from .kerr import KerrSpacetime
from .line_singularity import LineSingularity
from .cosmic_string import CosmicString

__all__ = [
    "GodelUniverse",
    "godel_ctc_radius",
    "GottPair",
    "gott_critical_velocity",
    "gott_critical_mu",
    "KerrSpacetime",
    "LineSingularity",
    "CosmicString",
]
