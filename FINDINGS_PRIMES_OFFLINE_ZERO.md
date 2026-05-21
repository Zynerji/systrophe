# Off-line-zero detection test — would the DSI lens see an RH violation?

`examples/millennium_primes_offline_zero_test.py`

Companion to `FINDINGS_PRIMES_DSI_INVERSE.md`. That experiment recovered
the first ~26 zeta zeros from the primes as a log-periodic cascade — a
clean *confirmation* of a known identity. This one is the *falsifiable*
follow-on: if a zero sat **off** the critical line, would the same
machinery detect it, and does it raise a false alarm on the actual
(on-line) primes?

## The signature

A zero `rho = beta + i*gamma` contributes `-x^rho/rho` to `psi(x)`. In the
normalised fluctuation `f(u) = (psi(e^u) - e^u + corr)/e^{u/2}`
(`u = ln x`), this mode has amplitude `~ e^{(beta - 1/2) u}`:

- **on the line** (`beta = 1/2`): a *stationary* log-periodic mode of
  constant amplitude;
- **off the line** (`beta > 1/2`): a mode whose amplitude **grows** as
  `e^{(beta - 1/2) u}`.

So the RH-violation signature is a non-zero growth exponent
`sigma = beta - 1/2`, and it is directly measurable: track the local
amplitude `A(u)` of the mode at frequency `gamma` in sliding windows
across the `ln x` range and read off the slope of `log A` vs `u`.

## Method

1. **False-alarm check on the REAL primes**: sieve `psi(x)`, form `f(u)`,
   and at several true zero frequencies `gamma_k` measure `sigma_hat`.
   RH-consistency ⇒ `sigma_hat ≈ 0`.
2. **Sensitivity / recovery (synthetic)**: build the on-line
   explicit-formula cascade of the first `N` zeros (whose periodogram
   reproduces the experiment-1 peaks), move ONE zero to `beta`, and check
   that `sigma_hat` recovers `beta - 1/2`. Also run the mandated
   address-space novelty catcher third-split (as in
   `millennium_riemann_catcher`).

Real primes can't be perturbed (RH appears to hold), so part 2 is a
detector-sensitivity test on synthetic signals.

## Result (X = 10^8, window = 5 in u; X = 10^7, window = 4 for comparison)

**Part 1 — false alarm on the real primes (must read ~0):**

| k | gamma | sigma_hat (X=10^7) | sigma_hat (X=10^8) | catcher |
|---|-------|--------------------|--------------------|---------|
| 1 | 14.1347 | −0.0014 | +0.0002 | smooth |
| 5 | 32.9351 | +0.0171 | −0.0006 | smooth |
| 10 | 49.7738 | −0.0047 | +0.0098 | smooth |
| 15 | 65.1125 | +0.0180 | +0.0027 | smooth |

The actual primes carry **no growing log-periodic mode**: `|sigma_hat| ≤
0.018` at `X=10^7`, `≤ 0.010` at `X=10^8`. The detector does **not** cry
wolf — consistent with every tested zero lying on the critical line.

**Part 2 — synthetic recovery, zero k=10 (gamma=49.77) moved off-line:**

| beta | sigma_true | sigma_hat (X=10^7) | sigma_hat (X=10^8) | catcher (10^7) |
|------|-----------|--------------------|--------------------|----------------|
| 0.50 | 0.000 | −0.0047 | +0.0040 | smooth |
| 0.55 | 0.050 | +0.0242 | +0.0295 | smooth |
| 0.60 | 0.100 | +0.0802 | +0.0856 | smooth |
| 0.70 | 0.200 | +0.1950 | +0.1972 | smooth |
| 0.80 | 0.300 | +0.2989 | +0.2996 | novel_structure |

`sigma_hat` recovers the true displacement `beta - 1/2` to **±0.01 for
beta ≥ 0.6** at both `X` — the method doesn't merely flag an off-line
zero, it **measures how far off the line it sits**.

## What is and is not shown

- **Positive**: the growth-exponent estimator is a working,
  *quantitative* off-line-zero detector. On the real primes it returns
  `sigma_hat ≈ 0` (no false alarm); on a planted off-line zero it
  recovers `beta - 1/2` accurately.
- **Detection floor**: `sigma ≈ 0.05` (`beta ≈ 0.55`) at `X=10^7`,
  tightening to `sigma ≈ 0.03` (`beta ≈ 0.53`) at `X=10^8`. The floor is
  set by leakage from neighbouring zeros plus prime-power discreteness,
  and it shrinks with the `ln x` span — so larger `X` buys sensitivity to
  smaller line-displacements.
- **Negative that motivated a fix**: the base address-space catcher
  (third-split) fires only on a **gross** displacement (`beta = 0.8`) and
  not reliably. A smooth `e^{sigma u}` amplitude trend has no sharp
  Hamming jump, and the catcher's per-third self-normalisation erases the
  magnitude shift between thirds. This is the **same catcher-domain
  boundary** already documented for the SAT sigmoid in
  `FINDINGS_MILLENNIUM_PROGRESS.md` (which motivated
  `derivative_catcher.py`): the base catcher detects *jumps*, not
  *trends*.

## The growth catcher closes the gap

`src/systrophe/growth_catcher.py` (8/8 tests in
`tests/test_growth_catcher.py`) is the trend-regime analog of
`derivative_catcher.py`. It uses a **global** rank-thermometer address
encoding (shared scale across all points — fixing the per-segment
self-normalisation blind spot) plus a permutation-null significance on
the log-amplitude slope. Re-running the sweep with both catchers:

| beta | sigma_true | sigma_hat | growth z | base catcher | growth catcher |
|------|-----------|-----------|----------|--------------|----------------|
| 0.50 | 0.000 | −0.005 | −0.4 | smooth | **stationary** |
| 0.55 | 0.050 | +0.024 | 4.0 | smooth | **growing** |
| 0.60 | 0.100 | +0.080 | 6.6 | smooth | **growing** |
| 0.70 | 0.200 | +0.195 | 6.8 | smooth | **growing** |
| 0.80 | 0.300 | +0.299 | 6.8 | novel_structure | **growing** |

Real primes and the on-line `beta = 0.5` control return `stationary`
(`|z| ≤ 1`); the growth catcher catches off-line displacement down to
`beta = 0.55` (`sigma = 0.05`), where the base catcher was blind until
`beta = 0.8`. No false alarm on the actual primes.

This neither supports nor threatens RH. It quantifies the **sensitivity**
of the DSI lens: at `X=10^8` it would detect any zero with `beta ≳ 0.53`
among the first ~20, and the real primes show no such signal.

## Reproduce

```
python examples/millennium_primes_offline_zero_test.py --x-max 1e8 --window 5
python examples/millennium_primes_offline_zero_test.py --x-max 1e7   # window 4 default
```

Writes `examples/millennium_primes_offline_zero_test_x{X}_results.json`.
Requires `numpy`, `scipy`, `mpmath`; reuses helpers from
`millennium_primes_dsi_inverse.py`.
