"""Morphic-resonance falsification harness.

A statistical adjudicator for the *empirically testable* core of Sheldrake's
morphic-resonance hypothesis, built on the Systrophe address-space novelty
catcher and the ELF tool's surrogate-null protocol ("no verdict without a
null run").

What this is
------------
Sheldrake's morphic resonance, stripped to its falsifiable skeleton, is the
claim that the *acquisition cost* of a form for instance ``i`` declines as a
function of the **cumulative number of prior instantiations** of that form,
through no conventional (genetic / physical / communicative) channel.

The decisive methodological problem -- the one that sank the rat-maze data --
is that cumulative count is nearly collinear with **calendar time**, so a
"morphic count effect" is statistically confounded with an ordinary secular
time-trend (better methods, contamination, drift). The effect is only
*identifiable* when the instantiation rate varies enough that cumulative
count decouples from linear time.

This package operationalizes exactly that. It will return:

  * ``no_structure``              -- panel looks like independent learners,
  * ``conventional_trend``        -- structure present but it is a secular
                                     time-trend, not count-coupling,
  * ``unidentifiable``            -- count and time too collinear to separate
                                     (the honest verdict for most real data),
  * ``morphic_signature``         -- cost tracks cumulative count beyond time,
  * ``acausal_signature``         -- the distinctive CTC-resonance fingerprint:
                                     cost depends on *future* instantiations.

What this is NOT
----------------
A claim that morphic fields exist. The harness is built to *falsify*. Its
most valuable output is a clean negative: showing when a claimed morphic
effect is indistinguishable from conventional learning.
"""

from __future__ import annotations

from .generate import (
    Panel,
    independent_learners,
    secular_trend,
    morphic_field,
    local_diffusion,
    ctc_resonance,
    multiform_forms,
)
from .detect import (
    catcher_verdict,
    count_vs_time_identifiability,
    acausal_test,
    acausal_across_forms,
)
from .nulls import (
    order_shuffle_null,
    independent_learner_null,
)
from .harness import falsify, MorphicVerdict, falsify_acausal, AcausalVerdict

__all__ = [
    "Panel",
    "independent_learners",
    "secular_trend",
    "morphic_field",
    "local_diffusion",
    "ctc_resonance",
    "multiform_forms",
    "catcher_verdict",
    "count_vs_time_identifiability",
    "acausal_test",
    "acausal_across_forms",
    "order_shuffle_null",
    "independent_learner_null",
    "falsify",
    "MorphicVerdict",
    "falsify_acausal",
    "AcausalVerdict",
]
