# cascade-dsi-explorer

**Multi-scale Tipler cascade-DSI fractal explorer.** Wraps
`systrophe.geometry.tipler_fractal.CascadeDSI` in an LPAnalyser-style
one-object-per-cascade API. Adds the address-space novelty catcher
as a built-in 2D phase-boundary scanner.

## Why this exists

`systrophe.geometry.tipler_fractal` ships the cascade-DSI primitive (multi-
scale sum of log-periodic cosines) and its diagnostics, but each
diagnostic is its own function. This tool collects them under one
object and reports a flat `CascadeSummary` dataclass — matching the
pattern of `tools/lp-analyser/`, `tools/implosion-carving/`, etc.

The 2D phase-boundary scanner wraps the address-space λ₂ novelty
catcher (the protocol mandated for every Systrophe deliverable). It
hashes each cascade's zero set to a bit address and reports
algebraic-connectivity jumps across a (scale_factor, amp_decay) grid.

## What a cascade is

```
F_cascade(r) = sum_{k=0..L-1} A_k cos(alpha_k * ln(r/R) + delta_0)

    alpha_k = alpha_0 * scale_factor^k       (geometric in k)
    A_k     = A_0 * amp_decay^k              (geometric in k)
```

* **levels=1** → bare cosine; zero set is a geometric progression
  with ratio exp(π/alpha).
* **levels≥2** with non-trivial scale_factor and amp_decay → multi-
  scale Cantor-like zero set with non-trivial box dimension.

## API

```python
from cascade_dsi_explorer import CascadeDSIExplorer, scan_phase_boundary

exp = CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=4,
                          scale_factor=2.5, amp_decay=0.6)
zs = exp.zeros(r_min=1.05, r_max=1e5)
dim = exp.box_dimension(r_min=1.05, r_max=1e5)
summ = exp.summary(r_min=1.05, r_max=1e5)
print(summ.n_zeros, summ.box_dimension)

rep = scan_phase_boundary()                 # default grid, default cascade
print(rep.verdict, rep.max_lambda_2_jump)
```

## Tests

15 tests, all offline:

```
PYTHONPATH=src:tools/cascade-dsi-explorer python -m pytest \
    tools/cascade-dsi-explorer/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
