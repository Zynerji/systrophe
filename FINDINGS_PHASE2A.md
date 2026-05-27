# Phase 2a — Quantitative chronology protection on the supercritical Tipler exterior

## One-line result

The renormalised stress-energy of a massless conformally-coupled scalar
field in the natural (Boulware) vacuum diverges as a clean simple pole
`<T_{tt}>_B ~ F'_H / (96 pi (r - r_H))` at every Cauchy horizon of the
supercritical Tipler exterior, with measured power `-1.000 +- 0.005` and
universality across alpha = 0.66 to 5.92. Hawking's chronology-protection
conjecture is **quantitatively consistent** in the cleanest analytically-
solvable CTC spacetime.

## What was computed

For a static Lewis-Papapetrou (LP) exterior `ds^2_2 = -F(r) dt^2 + h(r) dr^2`
(`h = 1` at LP leading order), the 2D Polyakov action gives the *exact*
renormalised stress tensor of a massless conformally-coupled scalar.
Following Davies-Fulling-Unruh (1976) and Christensen-Fulling (1977),
with `sigma(r) = (1/2) ln F` and tortoise `dr_* = sqrt(h/F) dr`:

```
<T_{tt}>_B        = (1/(24 pi))  (∂_{r_*} sigma)^2
<T_{r_*r_*}>_B    = (1/(24 pi)) [(∂_{r_*} sigma)^2 - 2 ∂_{r_*}^2 sigma]
<T_{t r_*}>_B     = 0                              (no flux: Boulware static)
trace             = R_{2D} / (24 pi)               (Polyakov identity)
```

Hartle-Hawking analog and Unruh analog add the canonical thermal offsets
`pi T_H^2 / 12` to `T_uu, T_vv` (symmetric for HH, one-sided for Unruh).

## Headline numerical results

### Single-spacetime, three horizons (`omega = 2, R = 1`, `alpha = sqrt(15)`):

| horizon  | r_H    | kappa | T_H    | <T_tt> power | <T_rr> power | fit rms |
|----------|--------|-------|--------|--------------|--------------|---------|
| 1        | 1.4054 | 2.000 | 0.3183 |  -1.007      |  -1.998      | 5.3e-3  |
| 2        | 3.1629 | 2.000 | 0.3183 |  -0.997      |   —          | 1.6e-3  |
| 3        | 7.1181 | 2.000 | 0.3183 |  -1.001      |   —          | 8.6e-4  |

All three horizons share `kappa = 2.0` and `T_H = 1/pi` — the LP log-
periodic discrete-scale-invariance signature. Powers match the analytic
predictions (`-1, -2`) to better than 1%.

Polyakov trace identity holds at **2.22e-16** (machine precision) across
a 20-point sweep — confirming the (2D-effective) Polyakov stress-energy
is internally consistent.

### Alpha universality (`a = omega R` swept 0.6 -> 3.0):

```
mean <T_tt> power = -1.0047 +/- 0.0012   (theory: -1)
mean <T_rr> power = -1.9987 +/- 0.0005   (theory: -2)
```

The Boulware divergence rate is **alpha-independent** — it is a geometric
fact about simple zeros of `F`, not a dynamical fact about LP log-frequency.

### 2D vs 4D cross-check at `omega = R = 1`:

| quantity                       | divergence power |
|--------------------------------|------------------|
| Kretschmann `K_4d`             | +0.019 (FINITE)  |
| 4D trace anomaly `K/(2880 pi^2)` | +0.019 (FINITE) |
| 2D `R_2d`                      | -1.995          |
| 2D Polyakov anomaly `R/(24 pi)` | -1.995         |
| Boulware `<T_{tt}>_B`          | -1.008          |

Reconciles with v0.10.0 `cauchy_horizon_finiteness_check.py`: the 4D
*local geometric* trace anomaly is bounded at `F = 0` (the Cauchy horizon
is a coordinate singularity, not a curvature singularity). What
**does** blow up is the 2D-effective (t, r)-sector Polyakov stress
tensor — the *state-dependent* quantum back-reaction. That is exactly
Hawking's chronology-protection claim: the natural vacuum's
stress-energy diverges at any developing Cauchy horizon.

## State-resolved <T_{mu nu}> at midpoint r = 1.203 (omega = 2, R = 1):

| state         | T_tt     | T_rr     | T_{t r_*} |
|---------------|----------|----------|-----------|
| Boulware      | +0.0345  | +0.3360  |  0.0000   |
| Hartle-Hawking| +0.0876  | +0.4113  |  0.0000   |
| Unruh         | +0.0610  | +0.3736  | +0.0265   |

- Boulware = HH up to the constant `pi T_H^2 / 12` Planck-thermal pressure.
- Unruh carries the canonical *radial energy flux* `<T_{t r_*}>` — the
  signature of one-sided thermal emission. Magnitude ~0.027, sets the
  scale of "Hawking emission" from this CTC-laden geometry.

## What this is *not*

- **Not** a dynamical back-reaction solve. We compute on a *fixed*
  vacuum LP background; Einstein's equations with the renormalised
  source on the RHS are not solved self-consistently. That is Phase 2b
  (still future work).
- **Not** a 4D mode-sum. The 4D off-trace `<T_{mu nu}>` requires a full
  Hadamard point-splitting subtraction on the LP exterior, which is
  Phase 2b. The geometric piece (`hadamard_offtrace.py`, v0.13) is
  available; the state-dependent mode-sum is still future work.
- **Not** a proof of chronology protection. The conjecture is about
  *dynamical* horizon formation, not static QFTCS on a fixed background.
  We confirm it is *consistent* with the static stress-tensor diagnostic,
  which is the standard pedagogical test.

## Files

- `src/systrophe/ctc/stress_energy_ctc.py` (430 lines)
- `tests/ctc/test_stress_energy_ctc.py` (18 tests, all pass)
- `examples/phase_modules/phase_2a_chronology_protection.py` (end-to-end report)
- `examples/phase_modules/phase_2a_alpha_sweep.py` (alpha universality)
- `examples/phase_modules/phase_2a_4d_cross_check.py` (2D vs 4D)
- `examples/phase_2a_chronology_protection_results.json`
- `examples/phase_2a_alpha_sweep_results.json`
- `examples/phase_2a_4d_cross_check_results.json`

## References

1. S.W. Hawking, *Chronology protection conjecture*, Phys. Rev. D 46
   (1992) 603.
2. P.C.W. Davies, S.A. Fulling, W.G. Unruh, Phys. Rev. D 13 (1976) 2720.
3. S.M. Christensen, S.A. Fulling, Phys. Rev. D 15 (1977) 2088.
4. N.D. Birrell, P.C.W. Davies, *Quantum Fields in Curved Space*, CUP
   1982, Ch. 8.

Emergent #26 in the Systrophe inventory.
