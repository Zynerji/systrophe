# Deep exploration of D-CTC convergence on Haar-random unitaries

Four-phase study of the empirical observation from `docs/STRESS_TESTS.md`
that Deutsch-CTC fixed-point iteration on Haar-random unitaries shows
heavy-tailed convergence and occasional near-pure fixed points.

Scripts:
- `examples/dctc_deep_phase_a.py` — scaling sweep over (dim_CR, dim_CTC)
- `examples/dctc_deep_phase_b.py` — spectral characterization
- `examples/dctc_deep_phase_c.py` — structure of high-purity samples
- `examples/dctc_deep_phase_d.py` — heavy-tail distribution fit

Result files (JSON) in `examples/dctc_deep_phase_*_results.json`.

---

## Phase A — Scaling laws

Swept `dim_CR ∈ {1, 2, 3, 4, 5, 6, 8}` × `dim_CTC ∈ {2, 3, 4, 5}` with
150 Haar samples per cell. Initial state: pure σ_CR = |0⟩⟨0| × random
pure ρ_init on CTC.

**Finding A.1: power-law convergence rate.**
For `dim_CR ≥ 2`, the median iteration count scales as
```
iter_median ~ C · dim_CR^(-α),   α ≈ 0.85 (for dim_CTC ∈ {3, 4, 5})
α ≈ 0.64 (for dim_CTC = 2).
```
The α exponent is essentially flat across dim_CTC ∈ {3, 4, 5}.

**Finding A.2: `dim_CR = 1` is degenerate.**
At `dim_CR = 1`, the "channel" reduces to a single unitary evolution
acting on a pure state. The iteration rotates within the pure-state
manifold without converging. **`dim_CR ≥ 2` is required** for the
D-CTC iteration to admit a non-trivial fixed point.

**Finding A.3: purity scales inversely with both dims.**
- (dim_CR=2, dim_CTC=2): max purity 0.989 (nearly rank-1!)
- (dim_CR=2, dim_CTC=3): max 0.932
- (dim_CR=8, dim_CTC=3): max 0.432 (close to floor 0.333)

Larger dim_CR or dim_CTC produces more thoroughly mixed fixed points.

---

## Phase B — Spectral oracle for iteration count

For each U we built the channel superoperator
`E: ρ_CTC → Tr_CR[U(σ_CR ⊗ ρ_CTC)U†]` as a (d²_CTC × d²_CTC) matrix,
then computed `|λ₂(E)|` (second-largest-magnitude eigenvalue; |λ₁|
is always 1 for the CPTP channel).

Theoretical mixing-time prediction:
```
iter_predicted = -log(tol) / log|λ₂|^(-1)
```

**Finding B.1: empirical iteration count is predicted by |λ₂(E)|.**

| Config | Pearson(iter, predicted) | log-log slope |
|--------|--------------------------|---------------|
| dim_CR=2, dim_CTC=3 | **0.989** | 0.91 |
| dim_CR=4, dim_CTC=3 | 0.953 | 0.86 |
| dim_CR=2, dim_CTC=4 | 0.980 | 0.89 |
| dim_CR=3, dim_CTC=4 | 0.974 | 0.88 |
| (Phase D, n=2000)   | **0.990** | 0.92 |

The slight under-1 log-log slope reflects sub-leading eigenvalue
contributions that accelerate the late-stage convergence beyond the
strict |λ₂| asymptote.

**Finding B.2: spectral algorithmic shortcut.**

Iterating to tol = 1e-10 costs **O(iter · d_total³)** work per sample.
Computing `|λ₂(E)|` once costs **O(d_total⁶)** per sample.
For `d_total ~ 10`, the spectral approach is cost-comparable to the
iterative approach for one sample, but for batched / parameterised
sweeps it offers practical advantages:
- No tol-dependent compute
- Returns the full spectrum, not just iteration count
- Enables theoretical bounds on convergence behaviour without simulation

This is a **clean algorithmic finding**: D-CTC convergence rate is a
spectral property of the channel, and `|λ₂(E)|` is the relevant
scalar invariant. The Pearson correlation 0.99 makes this an
empirically robust oracle.

---

## Phase C — Structure of high-purity unitaries

Sampled 1000 Haar-random U at (dim_CR=2, dim_CTC=3). 13 of them
(1.3%) produced fixed points with purity > 0.9. Max observed purity:
0.986.

**Question**: do high-purity samples share a structural signature?

**Diagnostics tested**:
- `|λ₂(E)|` — spectral gap of the channel
- Schmidt entropy of U as a bipartite CR⊗CTC tensor
- Iteration count
- Density-matrix rank of fixed point

**Finding C.1: Pearson correlations are all weak.**

| Pair | Pearson r |
|------|-----------|
| purity vs |λ₂| | -0.11 |
| purity vs Schmidt(U) | +0.09 |
| purity vs iter count | -0.03 |
| |λ₂| vs Schmidt(U) | -0.25 |

**High-purity fixed points are NOT explained by any of the obvious
structural features.** They emerge from generic Haar randomness
without correlating with channel mixing rate, U entanglement
content, or convergence time.

**Finding C.2: slow-mixing ≠ high-purity.**
- 7 samples (of 2000 in Phase D) have `|λ₂| > 0.95`. These take 200+
  iterations to converge, but their fixed points are typically
  **near-maximally-mixed** (purity ≈ 1/dim_CTC).
- The 15 high-purity samples have `|λ₂|` distributed near the
  population mean (~0.74); they converge in 30-150 iterations.

These are **two independent rare-event tails**:
1. Slow-mixing channels (|λ₂| → 1): low mean purity, long convergence
2. Pure-fixed-point channels (purity > 0.9): typical convergence,
   no obvious structural marker

---

## Phase D — Heavy-tail distribution fit

Generated 2000 fresh Haar samples at (dim_CR=2, dim_CTC=3).

**Finding D.1: iteration-count distribution is log-normal.**

| Distribution | AIC |
|--------------|-----|
| Log-normal (μ=4.33, σ=0.43) | **19648** |
| Exponential (λ=0.0118) | 21749 |
| Power-law tail (α=3.6, x_min=q50) | 9438 (tail-only) |
| Power-law tail (α=4.2, x_min=q90) | 2022 (tail-only) |

Log-normal beats exponential by ΔAIC = 2101 — overwhelming evidence.
The power-law tail fits suggest a heavy tail with exponent α ≈ 4,
giving finite mean and variance but slow tail decay.

This **log-normal signature is consistent with a multiplicative-
cascade process**: the iteration count is the product of many
sequentially-applied stochastic factors, which by the central limit
theorem makes the *logarithm* of iteration count normal-distributed.

**Finding D.2: |λ₂| distribution is unimodal around 0.74.**

Histogram of |λ₂(E)| across 2000 Haar samples:

```
[0.50, 0.55):  40  ####
[0.55, 0.60): 124  ##############
[0.60, 0.65): 222  ##########################
[0.65, 0.70): 322  ######################################
[0.70, 0.75): 392  ##############################################
[0.75, 0.80): 420  ################################################## (peak)
[0.80, 0.85): 248  #############################
[0.85, 0.90): 157  ##################
[0.90, 0.95):  55  ######
[0.95, 1.00):   7  
```

The slow-mixing tail (|λ₂| > 0.95) is ~0.3% of Haar samples.

**Finding D.3: high-purity and slow-mixing are nearly independent.**

| Joint count (n=2000) | High-purity (>0.9) | Low-purity |
|---------------------|--------------------|-----------|
| High `|λ₂|` (>0.85) | 1 | 218 |
| Low `|λ₂|` (≤0.85) | 14 | 1767 |

The joint distribution shows these two rare-event populations are
nearly independent. The single sample in both classes is a 0.05%
probability.

---

## Synthesis

Four substantive findings from the deep exploration:

1. **Power-law scaling** `iter ~ dim_CR^(-0.85)` for `dim_CR ≥ 2`,
   `dim_CTC ∈ {3, 4, 5}`. Provides a predictive rule for D-CTC
   convergence at any (dim_CR, dim_CTC).

2. **Spectral oracle**: `|λ₂(E)|` predicts iteration count with
   Pearson r = 0.99, log-log slope ≈ 0.92. Cheaper than full
   iteration for batched computations.

3. **Log-normal heavy tail**: iteration distribution is log-normal,
   indicative of a multiplicative-cascade structure in the iteration
   dynamics. Tail exponent α ≈ 4 (finite variance).

4. **Independent rare-event tails**: ~1% of Haar samples produce
   near-pure (purity > 0.9) fixed points, and these do NOT correlate
   with the ~0.3% slow-mixing (|λ₂| > 0.95) channels. The mechanism
   producing high-purity fixed points remains structurally opaque
   given the diagnostics tested.

## Implications and open questions

**Computational quantum information**:
The near-pure-fixed-point class (1% of Haar samples) is a candidate
for Aaronson-Watrous-style D-CTC computation. These channels
preserve quantum coherence through the CTC loop without obvious
structural design — they emerge from random U. **Open**: can the
channel-purity property be characterised by a cheaper invariant
than the full fixed-point computation?

**Mixing time theory**:
The 0.85 scaling exponent of iteration count with dim_CR appears
to be a new empirical regularity for the D-CTC iteration on Haar
inputs. **Open**: is there a theoretical derivation from random-
matrix-theory ensembles?

**Algorithmic deployment**:
The spectral oracle `|λ₂(E)|` is implementable as a public API on
`d_ctc.py` (next planned PR). It allows users to predict whether a
given channel U is "fast" or "slow" mixing without iterating.

## Reproduction

```bash
python examples/dctc_deep_phase_a.py  # 5-6 minutes
python examples/dctc_deep_phase_b.py  # <30 seconds
python examples/dctc_deep_phase_c.py  # ~3 minutes
python examples/dctc_deep_phase_d.py  # ~20 seconds
```

All four scripts write results to `examples/dctc_deep_phase_*_results.json`.
