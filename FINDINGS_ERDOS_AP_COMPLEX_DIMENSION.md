# AP-free sets and the complex dimension — where "make it a sinusoid" is true

Closing out the A003002 thread from the Erdős×OEIS sweep (Erdős problems 3
and 142 — largest 3-term-AP-free subset of [1,n]). Two parts:
`examples/erdos_a003002_recheck.py` (b) and
`examples/erdos_apfree_construction_dsi.py` (a).

## (b) The exact extremal sequence is data-length-walled

A003002 has 211 usable terms (`n <= 211`, `ln n` span 5.35). Re-scanning
the sweep's `omega ~ 5.95` peak with 5000 AR(1) surrogates per detrend
degree:

| detrend deg | peak omega | ratio | power | p_value |
|-------------|-----------|-------|-------|---------|
| 2 | 5.950 | 2.875 | 0.528 | 0.0894 |
| 3 | 5.950 | 2.875 | 0.534 | 0.0822 |
| 4 | 5.908 | 2.897 | 0.517 | 0.0996 |

The peak is **robustly located** (≈5.95, power ≈0.52, stable across detrend
choice) but **not significant** (best p = 0.082 — fails even uncorrected
0.05, far from the sweep's Bonferroni 0.00065). With span 5.35 the Rayleigh
resolution is ~1.17, so 5.95 is **within one resolution element** of the
greedy-AP-free Cantor frequency `2*pi/ln3 = 5.72`: the exact sequence can
neither confirm a complex dimension nor even resolve its frequency. Verdict:
noise-level, consistent with the data-length wall. (And, as established, no
invertible transform can fix this — data-processing inequality.)

## (a) The complex dimension lives in the CONSTRUCTION — and the catcher nails it

The greedy 3-term-AP-free set (Stanley sequence A005836 = nonneg integers
with no digit 2 in base 3) is computable to arbitrarily large N. Its
counting function `C(N) = #{m <= N : base-3(m) has no 2}` is the textbook
Cantor / complex-dimension object:

    C(N) ~ N^{ln2/ln3} * P(ln N),   P log-periodic with period ln 3,
    omega_theory = 2*pi/ln 3 = 5.7192,  geometric ratio lambda = 3 (the base).

Running the DSI catcher over `N in [1e3, 3^20]` (span ln N = 15.07,
2500 points):

| quantity | recovered | theory | error |
|----------|-----------|--------|-------|
| power-law exponent | 0.6299 | ln2/ln3 = 0.6309 | −0.16% |
| **log-periodic omega** | **5.7159** | 2π/ln3 = 5.7192 | **−0.06%** |
| **geometric ratio** | **3.0019** | 3.0000 | +0.06% |
| LS power / p_value | 0.902 / **0.0020** | — | (bootstrap floor) |

The catcher recovers the **complex dimension** to better than 0.1% in both
the power-law exponent and the log-periodic frequency, identifies the
geometric ratio as exactly the base (3), and the signal is decisively
significant.

## Synthesis

- **Discrete-scale invariance in the AP problem is real and provable — in
  the greedy/Cantor construction**, where the self-similarity (base-3
  digit structure) generates an exact complex dimension at `omega = 2π/ln3`.
  This is the legitimate, true form of the "turn the descent into a
  sinusoid" idea: for the construction the sinusoid is genuinely present,
  and our spectroscopy reads it to high precision.
- **It is not detectable in the exact extremal sequence** r_3(n) from
  current data (211 terms): the tickle at ω≈5.95 is within one resolution
  element of the construction's 5.72 but is not statistically significant.
  Whether the extremal values inherit a faint echo of the construction's
  complex dimension is **open and unresolvable without far more terms** —
  and r_3(n) is exactly the sequence that cannot be cheaply extended (exact
  max-AP-free search is the reason A003002 stops at ~211).
- **None of this touches the $10,000 problem (#142).** The greedy set is a
  weak lower bound (Behrend beats it; the extremal growth exponent is the
  open question), and the complex dimension is a property of the
  construction's self-similarity, not a handle on the density bound. The
  honest result is a precise positive about *constructions* and a clean
  negative about the *extremal sequence* — not progress on the prize.

## Reproduce

```
python examples/erdos_a003002_recheck.py
python examples/erdos_apfree_construction_dsi.py
```

Writes the two `*_results.json`. Requires `numpy`, `scipy`; reuses
`dsi_scan` / `_detrend_logpower` from `erdos_dsi_sweep.py`.
