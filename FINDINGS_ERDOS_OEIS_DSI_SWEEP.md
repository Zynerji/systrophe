# Erdős × OEIS DSI sweep — a validated negative

`examples/erdos/erdos_oeis_dsi_sweep.py`

The `erdos_dsi_sweep.py` net, validated on a hand-built battery, turned
loose on the **real** Erdős↔OEIS linkage (the `data/problems.yaml`
maintained at github.com/teorth/erdosproblems by Bloom & Tao).

## Method

1. Parse `problems.yaml`; collect OEIS A-numbers from problems with
   growth-flavoured tags (number theory, additive combinatorics, primes,
   …).
2. Fetch + cache each sequence's OEIS b-file (browser UA; OEIS 403s the
   default urllib UA).
3. Keep sequences suitable for log-periodicity detection: all-positive,
   ≥ 200 terms, ln-n span ≥ 3, and clean power-law-ish growth
   (`corr(ln n, log a) ≥ 0.97`).
4. Run the validated detector — Lomb–Scargle in `ln n`, AR(1) red-noise
   surrogate significance with the max-power (within-sequence
   look-elsewhere) statistic — plus two synthetic controls and prime
   `psi(x)-x` as anchors; Bonferroni across the real batch.

## Result

| | value |
|---|---|
| candidates fetched | 260 |
| suitable scanned | **77** |
| skipped | 183 (134 too short, 28 not growth-like, 21 non-positive) |
| Bonferroni threshold | p ≤ 0.05/77 = **0.00065** |
| **DSI hits among real OEIS sequences** | **0** |

**Anchors (validate the batch):**

| anchor | omega | power | p | verdict |
|--------|-------|-------|---|---------|
| synthetic log-periodic (true ω=8) | 7.980 | 0.926 | 0.0066 | DSI ✓ |
| prime psi(x)-x | 14.134 | 0.210 | 0.0066 | DSI ✓ (γ₁=14.13) |
| AR(1) red noise | 2.464 | 0.030 | 0.278 | null ✓ |

**Top real sequences (none significant):**

| OEIS | Erdős # | omega | power | p | note |
|------|---------|-------|-------|---|------|
| A388851 | 374 | 2.000 | 0.475 | 0.0132 | ω at search-band floor → residual-trend leakage, not a real peak |
| A002048 | 359 | 2.174 | 0.343 | 0.0662 | low-ω / trend |
| **A003002** | **3** | **5.948** | **0.533** | **0.119** | **r₃(n): largest subset of [1,n] with no 3-term AP (Erdős–Turán). The one genuine mid-band peak — suggestive, killed by multiple testing.** |
| A182237 | 849 | 2.871 | 0.416 | 0.119 | binomial coefficients |

## Interpretation

This is a **real negative, not a detector failure** — the anchors fire
correctly (including recovering the first zeta zero from the primes), so
the null verdict on the 77 Erdős-linked sequences is informative:

**Log-periodicity / discrete-scale-invariance is not a generic feature of
Erdős extremal sequences.** This is exactly consistent with the controlled
finding in `FINDINGS_ERDOS_DSI_SWEEP.md` — DSI tracks **Euler-product
(ζ-driven) structure** (`psi`, squarefree counts), which most combinatorial
/ additive extremal sequences simply do not have. The sweep correctly does
not manufacture a signal where the structure is residue-class, lattice, or
log-power-trend in nature.

The one item worth a deeper look is **A003002** (the no-3-term-AP / Behrend
sequence, Erdős problem 3, $5000): it is the only real sequence with a
genuine mid-band peak (ω≈5.95, geometric ratio λ≈2.88) and the highest
look-elsewhere power, but at p=0.119 it does not survive correction. It is
also a short, hard-to-compute sequence — a longer b-file (if it ever
exists) is the natural follow-up before reading anything into it.

## Limits

- The binding constraint is **sequence length**: 134/260 candidates were
  too short (< 200 terms). Most Erdős OEIS b-files are tiny because the
  underlying sequences are expensive to extend. DSI detection needs length
  *and* ln-n span; this caps what is testable today.
- Restricted to positive, growth-like sequences; signed oscillators
  (Mertens/Liouville-type) and 0/1 indicators are excluded.
- `omega ∈ [2, 60]`; the pile-up of null peaks at the ω=2 floor is
  residual-trend leakage (a higher detrend degree would move it but not
  change the null verdicts).
- A `sqrt(n)`-variable variant would catch additive/lattice structure that
  the `ln n` detector is (correctly) blind to.

## Reproduce

```
python examples/erdos/erdos_oeis_dsi_sweep.py --max-fetch 260
python examples/erdos/erdos_oeis_dsi_sweep.py --quick           # small smoke run
```

Writes `examples/erdos_oeis_dsi_sweep_results.json`. b-files cache under
`examples/oeis_cache/` (gitignored). Requires `numpy`, `scipy`, network
access to OEIS + raw.githubusercontent.com.
