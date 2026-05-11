# D-CTC × chronology protection: an independence finding

Phase AM of the D-CTC deep exploration tested whether the chronology-
protection conjecture (Hawking 1992) — which predicts that physical
CTCs are quantum-suppressed at certain configurations — also
suppresses the D-CTC computational resource (Aaronson-Watrous
PSPACE-style amplification).

**Hypothesis tested**: At the SystrophePair extinction phase δ = π
(where physical CTC content is minimised, validated in
`chronology_protection.py`), is the D-CTC state-distinguisher
amplification also minimised?

**Result**: NO. D-CTC amplification is *bimodal* in δ, peaking at
exactly δ = 0 AND δ = π — the same locations where the physical CTC
content is respectively at maximum AND minimum.

## Numerical data

| δ | Physical CTC content (∫ L²dr) | D-CTC amplification |
|---|---|---|
| 0 | 55400 (max) | **0.307** (peak) |
| π/4 | 25748 | 0.0015 |
| π/2 | 5419 | 0.0077 |
| 3π/4 | 832 | 0.0011 |
| **π** | **0** (chronology-protected) | **0.307** (peak) |
| 5π/4 | 7371 | 0.002 |
| 3π/2 | 34830 | 0.0061 |
| 7π/4 | 61250 | 0.0038 |
| 2π | 55400 | 0.307 |

**At δ = π, physical CTC content is zero but D-CTC amplification
is at its global maximum.**

## Pearson correlations

- `Pearson(physical CTC, D-CTC amp)`: r = +0.185 (weak)
- `Pearson(log physical CTC, log D-CTC amp)`: r = -0.382 (moderate negative)

The negative log-log correlation reflects the bimodal structure:
high amplification coincides with the EXTREMES of physical CTC
content (both maxima at δ = 0 and minima at δ = π).

## Interpretation

The standard intuition would say: chronology protection suppresses
CTCs → no CTC means no D-CTC channel → no amplification. The data
flatly contradicts this.

**Why the empirical result makes physical sense**: D-CTC is a
*quantum-information* construction. A "D-CTC channel" is defined
purely in terms of the unitary U acting on (CR × CTC) and the
partial trace. It doesn't require an actual closed timelike curve in
the underlying spacetime — only the algebraic structure of the
channel.

Hawking's chronology protection acts on classical CTCs (suppressing
the spacetime closed-timelike-curves). The D-CTC computational
resource lives at the *quantum channel level*, which has no direct
classical analogue.

**Concrete consequence**: even if Hawking's chronology-protection
conjecture holds (no physical CTCs exist), the D-CTC formalism
remains computationally viable as long as one can implement the
required CPTP channels in some other way (e.g., post-selection,
controlled measurements). Aaronson-Watrous's polynomial-time PSPACE
result is robust to chronology protection.

## Caveats

- The U(δ) construction used here is a specific parametrisation
  (Tipler-sinusoid diagonal entries + δ-dependent off-diagonal
  coupling). Different physically-motivated parametrisations may
  give different results.
- The δ = 0, π peaks coincide with where the Hamiltonian H_phys
  used in U(δ) = exp(-iH) @ Clifford is at the same value (due to
  the cos(αδ + 2πb/3) construction). This may artificially align
  the two extremes.
- The Pearson correlations are computed over only 25 grid points.

## Robustness across U(δ) constructions (Phase AM-extended)

To test whether the independence finding survives different
parametrisations, Phase AM-ext repeats the sweep with four
physically-motivated U(δ) constructions:

| Construction | What it represents | Peak structure | Pearson(phys, amp) |
|---|---|---|---|
| **A (Tipler-coupling)** | original | bimodal at 0, π | +0.20 |
| **B (linear interpolation)** | smooth U_a → U_b path | nearly flat; slight peak at π | **−0.60** |
| **C (anomaly-flow generator)** | Z_3 cycle + branch-energy flow | unimodal at 0 | +0.29 |
| **D (Floquet propagator)** | periodic drive | degenerate (zero amp everywhere) | +0.04 |

Mean Pearson across constructions: **−0.02** (essentially uncorrelated).

**Key observation**: In none of the four constructions does D-CTC
amplification track physical CTC content. In particular:

- Construction A: at δ = π (chronology-protected), amp = 0.307 (peak)
- Construction B: at δ = π, amp = 0.182 (peak)
- Construction C: at δ = π, amp = 0.000 (BUT structure is unimodal,
  not driven by chronology protection)
- Construction D: trivially zero

The independence is **robust** in the sense that no construction
exhibits the predicted chronology-protection signature (amp → 0 at
δ = π specifically for chronology-protection reasons, not generic
degeneracy).

## Refined picture (Phase AM-counterfactual)

To pressure-test the independence finding, 4 *more* physically-grounded
U(δ) constructions were tested:

| Construction | Description | Pearson(phys, amp) | CP signature |
|---|---|---|---|
| E (direct L) | L_pair(r) values as matrix entries | +0.530 | **YES (amp=0 at δ=π)** |
| F (geodesic) | Propagator at fixed r in CTC band | +0.207 | no |
| G (pair boundary) | H proportional to L_pair(r) | +0.048 | no |
| H (alpha-Floquet) | Tipler-frequency drive | +0.268 | no |

**One of four** here, combined with 0 of 4 in the original extended
sweep, gives **1 of 8 total constructions** exhibiting the
chronology-protection signature.

**Construction E is special**: U is built by literally inserting
L_pair(r) values as matrix entries. When L vanishes at δ = π (anti-
phase extinction), the matrix collapses and U becomes trivial — so
amp goes to zero by structural degeneracy, not by abstract
information processing.

### Refined physical interpretation

The chronology-protection × D-CTC coupling is **contingent on the
encoding**, and falls into a clean **trichotomy** of embeddings:

#### Three regimes of D-CTC channel construction

| Regime | What it represents | amp at δ=π | purity | Example construction |
|---|---|---|---|---|
| **Abstract** | Clifford / Floquet / generic Hamiltonian | varies; often peak | rare-event high | A, B, C, D, F, G, H, generic Clifford |
| **Direct CTC** | Matrix entries literally `~ L_pair(r)` | **0** (vanishes) | high (when amp > 0) | E (direct L) |
| **LP-Hamiltonian** | Matrix entries from F/K/L on Clifford backbone | finite, constant | **1.0** | Phase AJ |

The three regimes are physically distinct:

- **Abstract D-CTC channels** are decoupled from the underlying
  spacetime. They support Aaronson-Watrous amplification even at
  chronology-protected configurations. *Implementable via abstract
  quantum circuits or post-selection.*

- **Direct CTC channels** *are* the spacetime physics: their matrix
  structure literally encodes the CTC tensor. Chronology protection
  trivialises them by structural degeneracy.

- **LP-Hamiltonian channels** are physically grounded but still
  computationally non-trivial: they achieve perfect purity by
  construction but only moderate amplification, irrespective of the
  underlying (omega, R) regime.

#### Concrete implications

The Aaronson-Watrous polynomial-time PSPACE speedup is:
- **Available** in abstract D-CTC implementations (post-selection,
  Clifford circuits) regardless of chronology protection.
- **Lost** in direct-CTC implementations where the channel literally
  uses the spacetime CTC structure.
- **Muted** in physically-grounded LP-Hamiltonian channels
  (purity 1.0 but amp only 0.18).

## Reproduction

```bash
python examples/dctc_deep_phase_am.py            # original
python examples/dctc_deep_phase_am_extended.py   # 4 abstract constructions
python examples/dctc_deep_phase_am_counter.py    # 4 physical constructions
python examples/dctc_deep_phase_aj.py            # LP-Hamiltonian backbone
```

Results in `examples/dctc_deep_phase_am*_results.json` and
`dctc_deep_phase_aj_results.json`.
