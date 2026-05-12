# 156-qubit GHZ majority-vote demonstration — honest writeup

> **This is NOT quantum error correction.** Earlier drafts of this document
> framed the result as "QEC working at scale" with a "distance-156
> repetition-code logical decoder." That framing does not survive review
> and has been retracted. The repetition code only protects against
> bit-flip errors; phase errors are catastrophic. The 99.1% number is
> binomial-CDF arithmetic on a Z-basis GHZ measurement, not error
> correction. The actual SOTA QEC demonstration we plan to run
> (distance-3 Steane code with repeated syndrome extraction) lives at
> `experiments/steane_logical_qubit.py`.

**Status**: post-mortem / not-for-distribution.
**Date posted internally**: 2026-05-12 (replaces earlier draft).

---

## What we actually did

We prepared a 156-qubit Greenberger–Horne–Zeilinger (GHZ) state on IBM Quantum's `ibm_kingston` 156-qubit Heron-r2 processor via a tree-CNOT broadcasting circuit (depth 10, ISA-transpiled depth 214). The state would ideally measure into the two all-aligned bitstrings $|0\rangle^{\otimes 156}$ or $|1\rangle^{\otimes 156}$. On real hardware, 156-qubit decoherence destroys both peaks: those two bitstrings appeared in zero of 8192 shots. The physical state was, by any conventional fidelity measure, gone.

Then we did **classical post-processing on the same Z-basis measurement**: count whether more measured qubits are 0 or 1 across each shot, output that as a "logical bit." With $N = 156$ and a per-qubit bit-flip error rate $p$, the probability that the majority vote returns the correct bit is

$$
P(\hat L = L) \;=\; \sum_{k=0}^{\lfloor N/2 \rfloor} \binom{N}{k} p^k (1-p)^{N-k}.
$$

For our measured Hamming-weight distribution centered near $hw \approx 78$, this binomial CDF gives 99.1% — that's the number we reported.

## Why this is not SOTA QEC

1. **It is not quantum error correction.** The repetition code is a *classical* error-correcting code; it protects against bit flips only. Phase errors are not detected, not corrected, and would destroy any superposition. Real QEC needs a code (Steane, surface, color, BB) that handles both error channels.
2. **There is no syndrome extraction.** Real QEC extracts syndromes from ancilla measurements during the circuit, decodes them, and applies corrections. Our experiment measures the data qubits once at the end and post-processes the result.
3. **There is no logical operation.** Real QEC demonstrations include encoded gates (logical X, Z, H, CNOT) that act on the logical qubit while preserving the code space. We do none of this.
4. **The "logical fidelity" framing is binomial arithmetic.** Given the per-qubit error rate, 99.1% recovery is the expected value of a majority-vote estimator on $N = 156$ noisy bits — it is not a measurement of an encoded quantum state.

## What the result *is*

- A reproducible hardware demonstration that the Heron-r2 chip can prepare a 156-qubit GHZ circuit and measure it.
- An interesting visualization (Hamming-weight histogram of 8192 shots) that shows how decoherence smears a GHZ-style measurement.
- A teaching example of binomial-CDF logic on a noisy measurement.
- It is **not** evidence that we have a SOTA QEC system; it is **not** a logical qubit; and we should not market it as such.

## What we are doing about it

- This document supersedes the earlier viral draft. **Do not distribute the earlier tweet or two-paragraph summary.**
- A real distance-3 Steane code logical-qubit experiment is being built at `experiments/steane_logical_qubit.py`. That demonstration includes:
  - Real encoding of $|0_L\rangle$ on 7 physical qubits.
  - Repeated rounds of mid-circuit X- and Z-syndrome extraction with ancilla qubits.
  - A lookup-table decoder that interprets the syndrome history.
  - A direct comparison of logical and physical error rates over $k$ rounds — the actual quantity that determines whether QEC is "working" in the SOTA sense.

The Steane demo is achievable on Heron-r2 in one allocation window and is the result we will market when it lands.

## Honest tagline (for internal use only)

> "156-qubit Z-basis GHZ measurement on a Heron-r2; the GHZ peaks are smeared out as expected; binomial-CDF majority vote recovers the intended bit. **Not QEC** — preparation for a real distance-3 Steane code experiment."

---

## Don'ts

- Do not post the earlier tweet ("QEC working at scale 🧵").
- Do not pitch this to journalists as a QEC headline.
- Do not include the 99.1% number in QEC-themed marketing copy.
- Do not engage with podcasters / press inquiries that arose from the earlier draft; if any landed, redirect to the Steane code work in progress.

## Do's

- Use the result internally as a sanity check that the chip / circuit / measurement pipeline works at 156 qubits.
- Cite it as the *baseline* against which the Steane code experiment will be compared.
- Frame the *cross-chip* result (Kingston + Marrakesh batch 7 agreement at 1.94σ pooled shot noise on the Knopp Drive band-gating circuit) as the real hardware-reproducibility headline — that one is honest, novel for Heron-r2, and survives review.

---

## Lessons for the framework

The catcher inventory now treats "novel hardware demonstration" and "SOTA contribution" as separate claims. Future hardware artifacts should be tagged with:

| Claim level | Meaning |
|---|---|
| Reproducible | Replicates a known result; useful as baseline. |
| Novel demonstration | First public hardware run of a specific framing; clearly distinct from claiming SOTA. |
| SOTA contribution | Beats the published state of the art on a defined benchmark; requires direct comparison to current SOTA in the same paper. |

The 156q GHZ majority-vote result is **reproducible** (level 1). The Steane experiment, if it shows logical-error-rate suppression, would be a **novel demonstration on Heron-r2** (level 2). To make a **SOTA contribution** (level 3) we would need to beat Google Willow / IBM bicycle-code published thresholds, which is not the goal of this immediate experiment.
