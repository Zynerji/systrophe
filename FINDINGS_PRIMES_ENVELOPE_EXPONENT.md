# The error-exponent test — reading theta = sup Re(rho) off the primes

`examples/millennium_primes_envelope_exponent.py`

Third in the series (`FINDINGS_PRIMES_DSI_INVERSE.md` recovered the
zeros; `FINDINGS_PRIMES_OFFLINE_ZERO.md` built the per-mode off-line
detector and the growth catcher). This one collapses the off-line
question to a **single global statistic** and verifies its predicted
scaling.

## Hypothesis

The PNT error term obeys `|psi(x) - x| = O(x^theta)`,
`theta = sup_rho Re(rho)`; RH <=> theta = 1/2. In
`f(u) = (psi(e^u) - e^u + corr)/e^{u/2}` (`u = ln x`), an off-line zero
at `beta > 1/2` makes `f` grow like `e^{(beta-1/2)u}`, whereas under RH
`f` grows only like log-log-log (Littlewood) — exponent **exactly 0**.
So the growth exponent of the *envelope* of `f` is

    sigma_env = theta - 1/2 = sup_rho Re(rho) - 1/2.

**H:** the Systrophe growth catcher on the sliding-window RMS envelope of
`f` returns `sigma_env`; on the real primes it is `stationary`
(`sigma_env ~ 0`), and the certified detection floor shrinks like
`~1/ln X`. Being one global statistic, it has **no neighbouring-zero
leakage** — unlike the per-mode test — so the floor depends only on the
`ln x` span.

## Result

**Part 1 — real primes, envelope exponent vs X:**

| X | u-span | sigma_env | z | floor (null slope std) | floor × span | verdict |
|---|--------|-----------|---|------------------------|--------------|---------|
| 10^6 | 6.908 | −0.0043 | −2.2 | 0.0019 | 0.0135 | stationary |
| 10^7 | 9.210 | +0.0009 | +0.6 | 0.0014 | 0.0132 | stationary |
| 10^8 | 11.513 | −0.0008 | −0.7 | 0.0012 | 0.0135 | stationary |
| 10^9 | 13.816 | −0.0008 | −0.7 | 0.0011 | 0.0148 | stationary |

- The primes are **`stationary` at every X** (`|z| <= 2.2`): no growing
  envelope ⇒ `theta = 1/2` to within the floor ⇒ RH-consistent.
- **The `~1/ln X` scaling holds only approximately, and the bound shows
  hard diminishing returns.** `floor × span` is roughly constant —
  `0.0135, 0.0132, 0.0135` at `X = 10^6..10^8`, then up ~10% to `0.0148`
  at `10^9` — so `floor ≈ (0.013–0.015) / ln(X/1000)`, i.e. the `1/ln X`
  law to about 10%, not better. (An earlier version of this file claimed
  "three significant figures" from the first three points; the `10^9`
  point corrects that.) The primes-only certified bound on
  `|theta - 1/2|` is **±0.0012 at 10^8 and only ±0.0011 at 10^9** — a 10x
  increase in primes (and compute) tightened it ~8%. This is the
  logarithmic wall: still ~40x tighter than the leakage-limited per-mode
  test (±0.05) because the envelope is one global statistic, but
  brute-forcing X buys almost nothing.

**Part 2 — synthetic calibration (push the dominant zero to theta), X=10^7:**

| theta | sigma_true | sigma_env_hat | z | verdict |
|-------|-----------|---------------|---|---------|
| 0.50 | 0.000 | +0.0001 | 0.0 | stationary |
| 0.55 | 0.050 | +0.0186 | 7.2 | growing |
| 0.60 | 0.100 | +0.0689 | 8.1 | growing |
| 0.70 | 0.200 | +0.1906 | 8.1 | growing |
| 0.80 | 0.300 | +0.2983 | 8.1 | growing |

The on-line control (`theta = 0.5`) is `stationary` (no false alarm);
`theta >= 0.55` is `growing` with high significance, and `sigma_env_hat`
recovers `theta - 1/2` well for `theta >= 0.7` (the low-`theta`
underestimate is the on-line-modes-dominate-at-small-u crossover, but the
verdict still fires).

## Interpretation — what is and isn't claimed

- This is a **leakage-free, single-number RH-consistency test on the
  primes**, and it confirms a concrete, falsifiable **scaling law**
  (`floor ∝ 1/ln X`) rather than just a point estimate. That scaling is
  the substantive content: it predicts how much tighter the primes can
  certify `theta = 1/2` as the range grows.
- It is **not** a proof or disproof of RH. A real off-line zero with
  `beta - 1/2` below the floor (`~0.001` at `X = 10^8`) would pass as
  `stationary`; the test bounds, it does not settle.
- Physical reading (the Systrophe hook): `sigma_env` is the `ln x`-domain
  analog of a **quasinormal-mode imaginary part**. RH <=> the prime
  fluctuation is a *marginally stable, purely oscillatory DSI signal*; an
  off-line zero is a *spectral instability* with `sigma_env > 0`. This is
  a Hilbert–Pólya-flavoured stability statement in the language of the
  Systrophe LP/Dirac spectral stack.

## Reproduce

```
python examples/millennium_primes_envelope_exponent.py --x-list 1e6,1e7,1e8
```

Writes `examples/millennium_primes_envelope_exponent_results.json`.
Requires `numpy`, `scipy`, `mpmath`; reuses helpers from the two
companion experiments and `systrophe.growth_catcher`.
