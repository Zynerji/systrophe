# FINDINGS — morphic-resonance falsification harness

Honest verdict log, in the style of `Dinos/HYPOTHESIS.md`: every claim graded,
failures kept. All figures below are reproducible via
`python experiments/run_all.py` (8 seeds, 200 surrogates per null).

## The question

Can the Systrophe address-space catcher + surrogate-null protocol adjudicate
the *empirically testable* core of Sheldrake's morphic resonance — that a
form's acquisition cost falls with the cumulative count of prior instantiations,
through no conventional channel — and the CTC/Keldysh self-consistent
reformulation of it?

## Build verdicts

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | The harness has POWER: it fires on a true count-effect | **CONFIRMED** | `morphic_field` → `morphic_signature` 8/8 seeds, order-shuffle p ≈ 0.005 |
| 2 | The harness has SPECIFICITY: silent on independent learners | **CONFIRMED** | `independent_learners` → `no_structure` 8/8 |
| 3 | A pure time-trend is NOT misread as morphic | **CONFIRMED** | `secular_trend` → `conventional_trend` 8/8 (time_t dominates count_t) |
| 4 | A genuine count-effect is identifiable only under unequal instantiation | **CONFIRMED (sharp boundary)** | uniform schedule → `unidentifiable` (VIF ≈ 1e12); bursty → `morphic_signature` (VIF ≈ 1.8–5.2) |
| 5 | The CTC-resonance model reduces to causal reinforcement at `acausal_fraction=0` | **CONFIRMED** | field monotone, `f[0]=0`, statistically identical to `morphic_field` |
| 6 | The CTC acausal fingerprint is detectable ACROSS forms | **CONFIRMED** | `ctc` → `acausal_signature` 8/8 at a∈{0.3,0.6,0.9}; mean eventual_t = −9.7, −19.0, −28.4 |
| 7 | The CTC acausal fingerprint is detectable at a SINGLE form | **FALSIFIED** | field nearly invariant in `acausal_fraction` (f[0]=0, monotone for all a); single-panel acausal test leaks any nonlinear trend |
| 8 | Counts alone can establish NON-LOCALITY (morphic vs conventional diffusion) | **FALSIFIED (the valuable negative)** | `local_diffusion` → `morphic_signature` 8/8 — identical verdict to true morphic coupling |
| 9 | The harness has a characterized power curve and bounded false-positive rate | **CONFIRMED** | FPR ≤ 0.08 at zero effect for all n; min-detectable coupling 0.2–0.6 depending on n (F4) |
| 10 | A morphic real-vs-fake result must be ACAUSAL, not mere familiarity | **CONFIRMED (sharp criterion)** | causal → no early gap; only acausal/CTC makes real forms cheaper early (F5) |

## The three substantive findings

### F1 — The rat-maze confound, made precise (Concept D)
A real morphic count-effect is recoverable **only** when the instantiation rate
is unequal enough to drop the count↔time variance-inflation factor below ~10.
Under uniform instantiation the *identical* true effect collapses to
`unidentifiable`: cumulative count and calendar time are collinear (VIF ≈ 10¹²)
and no method can attribute the decline to count rather than to a secular
time-trend. This is exactly why McDougall's rats-learn-faster data could not
distinguish morphic resonance from ordinary secular improvement — the controls
improved too. The harness turns that historical objection into a quantitative
gate.

### F2 — Non-locality is not a count statistic (Concept C)
Conventional local diffusion (earlier learners *tell* later ones over a
network) produces the *same* `morphic_signature` as a non-local morphic field.
By counts alone they are indistinguishable (8/8 identical verdicts). Morphic
resonance's defining claim — *no conventional channel* — is therefore not
adjudicable from acquisition-count data; it requires independently ruling out
every communication pathway, which count panels do not contain. Any positive
"morphic" reading from count data is at best `coupling-present`, never
`coupling-is-non-local`.

### F3 — The CTC fingerprint is real but lives across forms, not within one
The CTC-resonance model's only empirically distinctive prediction is **acausal**:
a form's early-acquisition cost depends on its *eventual* total. Within a single
form this is structurally unidentifiable — `prior + future = const`, the field
is dominated by its causal ramp, and the acausal lift is numerically negligible
(finding 7). Across forms with varying eventual totals it becomes cleanly
identifiable and the harness detects it with the expected dose-response in
`acausal_fraction` (finding 6). **Crucially, an across-form acausal effect — early
cost predicted by eventual popularity — is exactly what no controlled morphic
experiment has ever demonstrated.** The harness gives that test a precise,
null-calibrated form.

## Extended concepts (Concepts E, F — `experiments/derived_extra.py`)

### F4 — Power / effect-size requirements (Concept E)
Detection rate `P(morphic_signature)` over the (panel-size n × coupling) grid,
12 seeds, 150 surrogates:

| n \ coupling | 0.0 | 0.1 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|
| 40  | 0.00 | 0.17 | 0.33 | 0.83 | 0.92 | 1.00 |
| 80  | 0.08 | 0.25 | 0.58 | 0.75 | 0.92 | 0.92 |
| 160 | 0.00 | 0.75 | 0.83 | 1.00 | 1.00 | 1.00 |
| 320 | 0.00 | 0.58 | 0.92 | 1.00 | 1.00 | 1.00 |

- **False-positive rate at zero effect stays within the 0.05–0.08 surrogate
  budget** at every n — the harness does not manufacture signal.
- **Minimum detectable coupling at 80% power**: n=40 → 0.4, n=80 → 0.6,
  n=160 → 0.2, n=320 → 0.2. Below ~80 instances even a large count-effect is
  routinely missed. **Consequence:** an underpowered morphic experiment that
  returns null is uninformative; effect size and N must be pre-registered, or
  "no effect" is indistinguishable from "not enough data."

### F5 — Real-vs-fake patterns: what a positive result must look like (Concept F)
Sheldrake's actual design (real foreign words / genuine patterns are claimed
to be learned faster than fakes because millions already learned them), recast
on the across-forms machinery with early counts matched across classes:

| mechanism | verdict | real − fake early cost |
|---|---|---|
| causal | `no_acausality` | +0.03 (no difference) |
| ctc/acausal | `acausal_signature` | −0.81 (real cheaper EARLY) |

A conventional (familiarity / cumulative-culture) mechanism produces **no**
early real-vs-fake gap once early practice is matched. Only an **acausal**
mechanism — early cost driven by a form's *eventual* establishment — makes real
forms cheaper early. So a genuine morphic result is not merely "real beats
fake"; it must be an early advantage that tracks eventual popularity beyond
early exposure. The harness makes that the explicit, null-calibrated bar.

## Honest scope statement

- These are synthetic generators. The positive controls prove the instrument
  has power and a characterized false-positive profile; they are not evidence
  for morphic fields.
- The CTC model is an explicit analogy to the Dinos Keldysh saddle and the
  Systrophe Deutsch-CTC fixed point, not a derivation. There is no claim it is
  the unique classical limit of Deutsch's condition.
- The harness's verdicts are deliberately conservative (it prefers
  `unidentifiable` to a false `morphic_signature`). The right real-world use is
  to feed measured panels (maze latencies, crystallization times, learning
  curves, real-vs-fake-word recall) into `falsify` / `falsify_acausal` and let
  it return the weakest claim the data support — which, on the historical
  evidence, will most often be `unidentifiable` or `conventional_trend`.

## Catcher coverage (Systrophe always-on rule)

The model-free gate `detect.catcher_verdict` runs `systrophe.catchers.
novelty_catcher.scan_novelty` over the instance-order axis on every panel, so a
catcher verdict + λ₂ surface accompanies each adjudication, per the project rule
that no "validated"/"null" verdict issues without a catcher run.
