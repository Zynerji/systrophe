"""lp-analyser: analytic-CTC physics for the Lewis-Papapetrou exterior.

A unified public API on top of the Systrophe roadmap items shipped
through v0.21.0:

  * **classical-GR backbone** (`vanstockum.VanStockumInterior` +
    Bonnor analytic Case III closed forms, the time-machine harness,
    geodesic / orbit machinery).
  * **Phase 1a / 1b**:  energy-condition diagnostics + analytic
    spacetimes (Godel, Gott, Kerr).
  * **Phase 2a**:  renormalised stress-energy <T_mu_nu>_ren in three
    vacuum states (Boulware / Hartle-Hawking / Unruh) on the
    supercritical Tipler exterior, with the simple-pole divergence
    fit at every Cauchy horizon.
  * **Phase 2b**:  4D Hadamard biparametrix coefficients V_0, V_1
    + WKB mode-sum + reconciliation with the 4D Kretschmann result.
  * **Phase 3a**:  N-cylinder SystropheArray with beam-forming,
    extinction, and beam-steer to a target radius.
  * **Phase 3b**:  off-axis pair quantitative orbits + topology
    classifier + geodesic-completeness test.

The headline class is `LPAnalyser(omega, R)`, which presents every
phase's main result as a method on a single object so a user can
characterise a supercritical Tipler exterior in one call:

    >>> from lp_analyser import LPAnalyser
    >>> a = LPAnalyser(omega=2.0, R=1.0)
    >>> a.summary()
    {... }

This tool is a thin re-export layer over `systrophe.*`; it does not
implement any new physics. Its job is to make the analytic-CTC stack
easy to drive from one entry point and to lock down a canonical set
of return-value shapes so downstream consumers do not need to track
the internal Systrophe module layout.
"""

from .analyser import (
    LPAnalyser,
    LPSummary,
)
from .pair import (
    PairAnalyser,
)

__all__ = [
    "LPAnalyser",
    "LPSummary",
    "PairAnalyser",
]

__version__ = "0.1.0"
