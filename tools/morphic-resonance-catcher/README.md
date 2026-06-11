# morphic-resonance-catcher

**A falsification harness for the empirically testable core of Sheldrake's
morphic-resonance hypothesis**, built on the Systrophe address-space novelty
catcher and the ELF tool's surrogate-null protocol ("no verdict without a
null run").

This is a *derived tool* in the Systrophe `tools/` tree. It imports
`systrophe.catchers` and follows the repo's always-on novelty-catcher rule. It
is **not** a claim that morphic fields exist; it is built to *falsify*, and its
most valuable outputs are clean negatives.

## What it does

Sheldrake's morphic resonance, stripped to its falsifiable skeleton, claims
the **acquisition cost** of a form for instance *i* declines as a function of
the **cumulative count** of prior instantiations of that form — through no
conventional (genetic / physical / communicative) channel. The harness
adjudicates that claim against the confounds that have historically sunk it.

### Single-form adjudicator — `falsify(panel)`
Returns one of:

| verdict | meaning |
|---|---|
| `no_structure` | consistent with independent learners |
| `conventional_trend` | real structure, but a secular **calendar-time** trend, not count-coupling |
| `unidentifiable` | structure present, but count is too collinear with time to attribute (the rat-maze failure mode) |
| `morphic_signature` | cost tracks cumulative **count** beyond time, surviving the order-shuffle null |

### Across-forms adjudicator — `falsify_acausal(forms)`
Tests the distinctive **CTC-resonance** fingerprint: does a form's *early* cost
depend on its *eventual* total (information looped back from the future)?

| verdict | meaning |
|---|---|
| `no_acausality` | early cost explained by early count alone |
| `unidentifiable` | early count too collinear with eventual total |
| `acausal_signature` | early cost depends on eventual total, surviving the label-shuffle null |

## Quickstart

```bash
# from the Systrophe repo root, with `pip install -e .` already done:
cd tools/morphic-resonance-catcher
python experiments/run_all.py        # the four derived-concept probes (A-D)
python experiments/derived_extra.py  # power curve (E) + real-vs-fake (F)
python -m pytest tests/ -q           # 14 tests
```

```python
import morphic_catcher as mc

# A real count effect, under a realistically bursty instantiation schedule:
panel = mc.morphic_field(coupling=0.6, schedule="bursty", seed=0)
print(mc.falsify(panel).summary())          # -> morphic_signature

# The same true effect under uniform instantiation -> unidentifiable:
print(mc.falsify(mc.morphic_field(schedule="uniform")).verdict)  # unidentifiable

# The CTC-resonance fingerprint across forms:
forms = mc.multiform_forms(mechanism="ctc", acausal_fraction=0.6, seed=0)
print(mc.falsify_acausal(forms).summary())  # -> acausal_signature
```

## The CTC-resonance model (`morphic_catcher/ctc.py`)

The one rigorous physics analog of "the present constrained by a consistency
condition that loops through time" is the closed-timelike-curve / closed-time-
path formalism — the **Dinos** Möbius loop (Schwinger–Keldysh saddle, verified)
and the **Systrophe** Deutsch-CTC fixed point. This module ports that skeleton
to a classical "establishment field" solved as a self-consistent fixed point

```
f = (1 - γ)·s  +  γ·W·f          (Keldysh source + loop coupling)
  = (1 - γ)·(I − γW)⁻¹·s         (resolvent / Schwinger–Dyson form)
```

via the same Picard iteration Dinos verifies as Keldysh saddle-finding. The
kernel `W` mixes a causal (past) part with an acausal (future) seam weighted by
`acausal_fraction`. At `acausal_fraction = 0` it reduces to ordinary causal
cumulative reinforcement; above 0 it carries information from the future back
to the past. The iteration count tracks the Systrophe D-CTC spectral-gap oracle
(see `ctc_iteration_count`).

**Key honest finding:** at a *single form* the acausal lift is numerically
negligible — the causal ramp dominates and there is no future information to
loop. The CTC fingerprint is only identifiable **across forms** with varying
eventual totals. See `FINDINGS.md`.

## Scope / what this is NOT

- Not evidence for morphic fields. The generators are synthetic; the harness is
  a measuring instrument with characterized power and specificity.
- Counts alone cannot establish **non-locality** (Concept C): conventional local
  diffusion produces the same `morphic_signature`. Distinguishing them requires
  ruling out every conventional channel, which count data cannot do.
- The synthetic positive controls exist to prove the instrument has power; a
  real application would feed measured acquisition-cost panels (maze latencies,
  crystallization times, learning curves) into the same `falsify` / `falsify_acausal`.

## Files

```
morphic_catcher/
  generate.py   panel + multiform generators (independent / secular / morphic /
                diffusion / CTC) and the bursty instantiation schedule
  ctc.py        the CTC-resonance self-consistent fixed-point field
  detect.py     catcher verdict, count-vs-time identifiability, acausality tests
  nulls.py      order-shuffle and independent-learner surrogate nulls
  harness.py    falsify() + falsify_acausal() adjudicators
experiments/
  run_all.py        concepts A-D -> run_all_results.json
  derived_extra.py  concepts E (power curve) + F (real-vs-fake) -> derived_extra_results.json
tests/
  test_harness.py   14 tests (power, specificity, identifiability, CTC)
FINDINGS.md     honest verdict log (Dinos-style)
```
