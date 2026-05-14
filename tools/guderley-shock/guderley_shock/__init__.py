"""guderley-shock: self-similar converging-shock solver + QFTCS-Boulware comparator.

Priority-1 sibling of implosion-carving / beamforming-inverse. The
original framing was "carve a converging-shock implosion profile via
CTC-stress tuning"; on closer inspection, the equivalence between
hydrodynamic Guderley divergence and QFTCS Cauchy-horizon Boulware
divergence is *not* a physically-derived relationship — it is at best
an empirical power-law match. This tool is therefore a Guderley
solver + an honest comparison routine that reports the
|p_guderley - p_qft| residual.

What it does:
  - Solve for the Guderley self-similarity exponent β(γ, n).
  - Compute the post-shock density / pressure divergence power at the
    implosion focus.
  - Compare to the QFTCS Boulware-state stress-tensor power at a
    Cauchy horizon (which is universally -1.000 on the supercritical
    Tipler exterior — Phase 2a's headline result).

What it does NOT do:
  - Derive any physical correspondence between the two.
  - Engineer a spacetime that hydrodynamically realises the shock.
  - Claim CTC-stress can be "tuned" to match arbitrary shock profiles.
"""

from __future__ import annotations

from .shock import (
    GuderleyExponent,
    GuderleyProfile,
    compute_guderley_exponent,
    density_power_at_focus,
    integrate_post_shock_profile,
)
from .compare import (
    ShockHorizonComparison,
    compare_to_cauchy_horizon,
)

__all__ = [
    "GuderleyExponent",
    "GuderleyProfile",
    "compute_guderley_exponent",
    "density_power_at_focus",
    "integrate_post_shock_profile",
    "ShockHorizonComparison",
    "compare_to_cauchy_horizon",
]
