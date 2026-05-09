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

__all__ = ["GodelUniverse", "godel_ctc_radius"]
