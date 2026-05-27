"""cascade-dsi-explorer: scan + diagnose multi-scale Tipler cascade-DSI fractals.

Wraps the existing `systrophe.geometry.tipler_fractal` machinery in an
LPAnalyser-style one-object-per-cascade explorer with:

  - zero-set extraction
  - box-counting fractal dimension
  - geometric-progression verification (single-level sanity)
  - 2D parameter-sweep phase-boundary scan via the address-space
    novelty catcher (Hamming-graph λ₂)

The cascade-DSI fractal is the natural multi-scale extension of a
single Tipler sinusoid. For levels=1 the zero set is a pure
geometric progression; for levels ≥ 2 with non-trivial scale_factor
and amp_decay the zero set densifies into a Cantor-like multi-scale
structure with a non-trivial box dimension.
"""

from __future__ import annotations

from .explorer import (
    CascadeDSIExplorer,
    CascadeSummary,
)
from .phase_boundary import (
    PhaseBoundaryReport,
    scan_phase_boundary,
)

__all__ = [
    "CascadeDSIExplorer",
    "CascadeSummary",
    "PhaseBoundaryReport",
    "scan_phase_boundary",
]
