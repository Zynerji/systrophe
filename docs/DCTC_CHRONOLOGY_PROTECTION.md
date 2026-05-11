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

## Reproduction

```bash
python examples/dctc_deep_phase_am.py
```

Writes results to `examples/dctc_deep_phase_am_results.json`.
