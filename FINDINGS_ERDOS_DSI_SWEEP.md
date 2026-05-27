# Erdős DSI discovery sweep — log-periodicity is an Euler-product signature

`examples/erdos/erdos_dsi_sweep.py`

A discovery net that scans integer-sequence error terms (Erdős and
Erdős-adjacent) for hidden **discrete-scale invariance** — oscillations
periodic in `ln x`, `f(x) ~ x^c [1 + A cos(omega ln x + phi)]`, geometric
ratio `lambda = exp(2 pi / omega)` — using the Systrophe DSI framing
(`systrophe.catchers.dsi_observables`, `tipler_fractal`) and the mandated-catcher
discipline (controls + significance, no "validated" without numbers).

Motivated by Bloom & Tao's observation that most Erdős problems are about
an integer sequence, now being linked to the OEIS.

## Method (built to NOT cry wolf)

Each sequence is reduced to an oscillation series `y(t)`, `t = ln x`
(signed error term normalised by its conjectured power, or a
log-power-detrended count). Then:

1. Lomb–Scargle periodogram in `t`; take the peak `omega`, power.
2. **Red-noise significance**: AR(1) surrogate null (preserves
   autocorrelation — naive shuffling over-reports), using the **max**
   periodogram power per surrogate (within-sequence look-elsewhere).
3. **Bonferroni** across the battery (6 non-control tests ⇒ `p <= 0.0083`).

Two synthetic controls gate trust in the net: one log-periodic (must be
caught and `omega` recovered), one AR(1) red-noise (must stay null).

## Result (n_points = 1500, 200 surrogates)

| sequence | omega | ratio | LS power | p_value | verdict |
|----------|-------|-------|----------|---------|---------|
| control_logperiodic | 7.998 | 2.194 | 0.929 | 0.0050 | **DSI** |
| psi_error (primes) | 14.142 | 1.559 | 0.210 | 0.0050 | **DSI** |
| squarefree_error | 7.031 | 2.444 | 0.320 | 0.0050 | **DSI** |
| divisor_error | 15.786 | 1.489 | 0.010 | 0.0945 | null |
| goldbach_count | 20.817 | 1.352 | 0.005 | 0.1592 | null |
| control_rednoise | 2.484 | 12.55 | 0.030 | 0.2935 | null |
| circle_error | 32.911 | 1.210 | 0.006 | 0.7562 | null |
| mult_table_erdos | 13.948 | 1.569 | 0.090 | 0.9403 | null |

(`p_value = 0.0050 = 1/201` is the bootstrap floor — no surrogate beat the
peak — so those three are *at least* this significant.)

## What the net found

**The controls validate it.** The synthetic log-periodic signal is caught
and its frequency recovered (`omega = 7.998` vs true `8.0`); the AR(1)
red-noise control stays `null` (`p = 0.29`) — the net does not manufacture
DSI from autocorrelation.

**Log-periodicity tracks the Euler product.** The two error terms with
multiplicative (Euler-product → Riemann `zeta`) origin light up:

- `psi(x) - x` peaks at `omega = 14.14` — the **first zeta zero**
  `gamma_1 = 14.1347` (the explicit-formula comb).
- the squarefree-count error `Q(x) - 6x/pi^2` peaks at `omega = 7.03` —
  **half the first zeta zero** (`gamma_1/2 = 7.067`), exactly as predicted
  from the zeros of `zeta(2s)` (squarefree Dirichlet series
  `zeta(s)/zeta(2s)`). Geometric ratio `lambda = 2.44`.

**Everything else is correctly null** — and for structural reasons, not
weak data:

- `divisor_error` (Dirichlet `Delta`) and `circle_error` (Gauss) oscillate
  in `sqrt(x)` (Voronoi / Hardy), **not** `ln x` — additive/lattice
  problems are not log-periodic. The net says null in the `ln x` variable
  (correct); their structure lives in a `sqrt(x)` periodogram instead.
- `mult_table_erdos` (Erdős distinct-products count) is null because its
  `(ln n)^{-delta}` correction (`delta ≈ 0.0860`, Ford) is a smooth
  log-power **trend**, not an oscillation — that is the `growth_catcher`'s
  job, the alternative attack mode.
- `goldbach_count` is null: its structure is residue-class / arithmetic
  (the mod-6 comet bands found earlier), not log-periodic.

## Takeaway

**DSI / log-periodicity in number-theoretic error terms is a signature of
multiplicative (Euler-product) origin** — it appears for `zeta`-driven
errors (`psi`, squarefree) at frequencies `gamma_k` (or `gamma_k/2`), and
is correctly absent for additive-lattice (`sqrt x`) and extremal-count
(log-power-trend) problems. The catcher discriminates these classes
automatically, with calibrated controls.

This is the net working as designed — known positives recovered, controls
clean, negatives explained. It is **not** a new theorem; it is a validated
instrument. The point of validating it here (on sequences whose answer we
can predict) is to turn it loose next on the **OEIS-linked Erdős sequences
that are genuine unknowns**, where a surviving DSI hit would be new.

## Limits

- p-values are floored at `1/(n_boot+1) = 0.0050`; raise `n_boot` for finer
  resolution on the hits (they are already at the floor).
- AR(1) is a simple red-noise null; a more structured background could
  shift marginal cases (e.g. `divisor_error` at `p = 0.09`).
- Restricted to positive / power-normalisable series; signed oscillators
  (Mertens, Liouville summatory) need a sign-aware variant.

## Reproduce

```
python examples/erdos/erdos_dsi_sweep.py            # full
python examples/erdos/erdos_dsi_sweep.py --quick    # fast smoke
```

Writes `examples/erdos_dsi_sweep_results.json`. Requires `numpy`, `scipy`;
reuses `prime_power_steps` from `millennium_primes_dsi_inverse.py`.
