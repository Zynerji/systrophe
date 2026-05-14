"""implosion-carving: Möbius-Z3 trapped-null-pocket carver for Systrophe.

Given a target radius r_target (and an underlying van Stockum exterior),
this tool engineers a Schwarzschild mass M such that the hybrid
spacetime supports a photon sphere at r_target, then wraps a Z_3
monodromy on the resulting null-circular orbit and reports closure
residuals.

The "carving" metaphor: the tool sculpts a closed null geodesic into
the geometry by tuning M, and reports how well-trapped the resulting
pocket is.
"""

from __future__ import annotations

from .carver import ImplosionCarver, ImplosionSummary
from .pocket import PocketGeometry, carve_photon_pocket, closure_residual
from .monodromy import Z3MonodromySignature, compute_z3_signature

__all__ = [
    "ImplosionCarver",
    "ImplosionSummary",
    "PocketGeometry",
    "carve_photon_pocket",
    "closure_residual",
    "Z3MonodromySignature",
    "compute_z3_signature",
]
