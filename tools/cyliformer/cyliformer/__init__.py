"""Cyliformer: Resonant Cylinder Transformer.

A transformer architecture derived from Systrophe Phases 2a/2b/3a/3b:

  * **Phase 3a / 3b (N-cylinder beam-forming + off-axis topology)** -> each
    layer hosts N "virtual cylinders" -- phase-shifted parallel processing
    paths that share a single FFN's weights.
  * **Phase 2a (Polyakov stress-energy)** -> a differentiable lambda_2
    catcher per cylinder reports the spectral coherence of that
    cylinder's activations.
  * **Phase 2b (Hadamard biparametrix)** -> a back-reaction proxy turns
    high lambda_2 into "low back-reaction" (cylinder kept) and low
    lambda_2 into "high back-reaction" (cylinder soft-pruned).

The "cylinder" name is the same poetic device used in Systrophe's
classical-GR backbone: each cylinder rotates with a different phase
(`phasor`), the cylinders' outputs are summed (beam-formed), and the
catcher measures whether the beam is constructive (high lambda_2) or
incoherent (low lambda_2). Soft pruning via back-reaction makes the
architecture sparse on the fly.

This tool depends on the Systrophe core (`systrophe.novelty_catcher`)
for the address-space lambda_2 primitives. Built on top of the v0.21.0
release; see ../systroformer/ for a simpler precursor.
"""

from .catcher import LearnedAddressCatcher
from .block import CylinderBlock
from .model import Cyliformer
from .loss import (
    TorsionalResonanceLoss,
    cyliformer_loss,
)
from .kv_cache import SelectiveKVCache

__all__ = [
    "CylinderBlock",
    "Cyliformer",
    "LearnedAddressCatcher",
    "TorsionalResonanceLoss",
    "cyliformer_loss",
    "SelectiveKVCache",
]

__version__ = "0.1.0"
