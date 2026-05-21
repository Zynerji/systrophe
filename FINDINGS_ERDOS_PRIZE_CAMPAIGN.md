# Erdős monetary-prize campaign — where DSI lives, and why no prize is winnable this way

An autonomous pass pointing the validated Systrophe DSI toolkit (Lomb–Scargle
in the natural variable + AR(1) red-noise significance + Bonferroni; see
`FINDINGS_ERDOS_DSI_SWEEP.md`) at **every monetary-prize Erdős problem**.

## Honest premise

A spectral instrument detects *structure*; it does not *prove* theorems. So
it cannot collect a proof-prize. The goal of this campaign is therefore a
clean map: which prize problems are even in the toolkit's domain, and does
any testable one carry real discrete-scale-invariant structure (a lead)?

## The prize landscape (teorth/erdosproblems linkage)

Open prize problems: **22 with an OEIS link, 31 without.** Prize histogram
(all states): \$10000×2, \$5000×1, \$1000×10, \$500×32, \$250×19, \$100×27,
\$50×3, \$25×4, \$10×3. The two \$10,000 problems: **#4 (large prime gaps,
already proved 2014 — Maynard / Ford–Green–Konyagin–Tao)** and **#142 (the
Erdős AP / divergence conjecture, open)**.

## Breadth — prize battery (`examples/erdos_prize_battery.py`)

Pointed the DSI ln-n detector at all 28 OEIS sequences attached to open
prize problems. Anchors validated the run (synthetic log-periodic ω=7.98
p=0.003; prime ψ ω=14.13 p=0.003).

| outcome | count | notes |
|---|---|---|
| testable (≥150 terms, growth-like) | **6** | A003002, A002975, A263996, A387704, A006037, A143824 |
| untestable | 22 | mostly **too short** — Ramsey n=4–23, Golomb rulers n=27, geometry n=6–20 |
| **DSI leads** | **0** | closest A003002 (\$10k AP) p=0.10 — the known sub-threshold tickle |

**The binding wall is data length.** The prize problems are dominated by
sequences (Ramsey numbers, Golomb rulers, distinct-distance sets) with only
a handful of known terms — far too few for any spectral test. Of the six
testable, none shows significant log-periodicity.

## Depth — the structural map (where DSI *does* live)

The campaign's real value is a precise map of which number-theoretic objects
carry discrete-scale invariance, validated by the detector recovering known
analytic constants to <0.1%:

| object | variable | result |
|---|---|---|
| prime ψ(x)−x | ln x | DSI at ω=γ₁=14.13 (zeta comb) — `FINDINGS_ERDOS_DSI_SWEEP.md` |
| squarefree-count error | ln x | DSI at ω=γ₁/2 (zeros of ζ(2s)) |
| greedy 3-AP-free set (A005836, Cantor) | ln N | DSI at ω=2π/ln3=5.72, ratio=3 (the base) — `FINDINGS_ERDOS_AP_COMPLEX_DIMENSION.md` |
| Dirichlet divisor error Δ(x) | √x | comb at 4π√n to ~0.01% (Voronoi) — `FINDINGS_…` |
| Gauss circle error P(x) | √x | comb at 2π√n, sums-of-two-squares only (Hardy) |
| exact extremal r₃(n) (A003002) | ln n | null (ω≈5.95 but p≈0.10; 211 terms can't resolve it) |
| Collatz total stopping time (#1135, \$500) | ln n | **null** (bin-averaged σ over n≤2×10⁶: peak power 0.008, p=1.0 vs AR(1); growth slope 10.2·ln n) — no log-periodicity, the per-n noise averages to a smooth trend |

**The pattern:** DSI is a signature of *multiplicative / self-similar*
structure (Euler-product errors in ln x; the Cantor construction; lattice
errors in √x). It is correctly absent from extremal/combinatorial prize
sequences, which is why the battery is a clean negative.

## Verdict

- **No monetary prize is approachable by spectral methods.** The prizes
  require proofs; the in-domain sequences are either data-length-walled
  (Ramsey/Golomb/geometry) or structurally non-log-periodic (combinatorial
  extremal values). This is consistent and expected, not a failure.
- **The toolkit is a validated structural instrument**: it recovers γ₁,
  γ₁/2, 2π/ln3, 4π√n, 2π√n to <0.1% where the structure is real, and
  returns honest nulls where it isn't. That map — not a prize — is the
  deliverable.

## Files

`examples/erdos_prize_battery.py` (+ results), `erdos_sqrtn_lattice_dsi.py`,
`erdos_apfree_construction_dsi.py`, `erdos_a003002_recheck.py`,
`erdos_collatz_dsi.py`, and the companion FINDINGS_ERDOS_*.md.
