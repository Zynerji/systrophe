# guderley-shock

**Converging self-similar shock + empirical comparison to QFTCS
Cauchy-horizon divergence.** Priority-1 of the implosion-carving
family, but reshaped: the original framing assumed a physically
derivable correspondence between hydrodynamic Guderley divergence and
QFTCS Boulware-state Cauchy-horizon divergence. That correspondence
is NOT derived in the literature — both are power-law singularities,
but the physics is unrelated. This tool ships the honest pieces:

1. **Literature β lookup** for `(n, γ)` ∈ canonical table.
2. **Asymptotic density-divergence power** `ρ ~ r^{-2(1-β)/β}` as
   r → 0 at t = t_focus.
3. **QFTCS comparison**: report Guderley density power side-by-side
   with the Boulware `<T_tt>` Cauchy-horizon power (= -1.000
   universally per Phase 2a) and the absolute residual.

What is **not** included:
- Full Guderley post-shock profile integration. Naive
  forwards-from-shock LSODA integration hits the singular saddle and
  blows up. A correct integrator uses Lazarus 1981's
  backwards-from-sonic-point procedure, which I declined to ship in
  hand-derived form. `integrate_post_shock_profile` raises
  `NotImplementedError` with a documentation pointer.
- Eigenvalue determination of β for `(n, γ)` outside the literature
  table. Same reason. `compute_guderley_exponent` raises
  `NotImplementedError` for unsupported pairs.
- Any claim of physical equivalence between hydrodynamic and QFTCS
  divergence. The residual is just an empirical number.

## Headline empirical result

For γ=5/3 spherical (n=3):
* Guderley density-divergence power: -0.9054
* QFTCS Boulware `<T_tt>` power at supercritical Tipler Cauchy horizon: -1.000
* Absolute residual: 0.095

The fact that two unrelated physical phenomena both give nearly-(1/r)
divergences is curious; the tool reports the residual but does not
interpret it.

## API

```python
from guderley_shock import (
    compute_guderley_exponent, density_power_at_focus,
    compare_to_cauchy_horizon,
)
from systrophe.geometry.vanstockum import VanStockumInterior

e = compute_guderley_exponent(gamma=5/3, n=3)
print(e.beta)                            # 0.688376

p = density_power_at_focus(gamma=5/3, n=3)
print(p)                                 # -0.9054

vs = VanStockumInterior(omega=2.0, R=1.0)
cmp_ = compare_to_cauchy_horizon(vs, gamma=5/3, n=3)
print(cmp_.guderley_density_power,       # -0.9054
      cmp_.qftcs_T_tt_power,             # -1.0000
      cmp_.absolute_residual)            # 0.0946
```

## Tests

17 tests, all offline, fast (< 2 s):

```
PYTHONPATH=src:tools/guderley-shock python -m pytest \
    tools/guderley-shock/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
