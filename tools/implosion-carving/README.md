# implosion-carving

**Möbius-Z3 trapped-null-pocket carver** for the Systrophe analytic-CTC
stack.

Given a van Stockum rotating-cylinder exterior and a target radius
`r_target`, this tool engineers a Schwarzschild mass `M` such that the
hybrid (Schwarzschild + LP cylinder) spacetime admits a **photon
sphere at r_target** — a closed null geodesic. It then wraps the
Möbius-Z₃ cover (3-fold monodromy) on the resulting orbit and reports
closure residuals.

The carver's value-add over `systrophe.geometry.photon_sphere` directly:
* one API object per spacetime (LPAnalyser pattern),
* records the Schwarzschild reference `M = r_target / 3` so the
  cylinder's distortion is visible at a glance,
* reports stability + symmetry closure residuals alongside the engineered
  `M`, so a downstream consumer can score the carving fidelity,
* decorates the pocket with a Z₃ monodromy signature drawn from the
  vendored Dinos Möbius-Z₃ cover (continuum triplet → {0, 1/9, 4/9}).

## What it is

This is the **priority-3** (photon-sphere / null-pocket) sub-tool in
the implosion-carving family. Two other sub-tools, both planned:

* priority-2: **N-cylinder beamforming inverse** — solve for cylinder
  amplitudes/phases that carve a prescribed field at radius r.
* priority-1: **Converging-shock CTC-stress profile** — carve a
  Guderley-style self-similar implosion via Polyakov stress-energy +
  pendulum mass-engineering.

## API

```python
from implosion_carving import ImplosionCarver

car = ImplosionCarver(omega=1.0, R=1.0)   # van Stockum exterior
pocket = car.carve(r_target=1.5)            # carve closed null pocket
print(pocket.M_engineered)                  # ~ 0.617
print(pocket.schwarzschild_limit_M)         # 0.5  (= r_target / 3)
print(pocket.is_stable)                     # True (trapped)
print(pocket.closure_residual_dbdr)         # ~ 1e-6

sig = car.z3_signature(N=256)
print(sig.triplet_eigenvalues)              # → [0, 1/9, 4/9] in continuum
print(sig.triplet_convergence_error)        # ~ 1e-5

summ = car.summary(r_target=1.5)            # everything in one dataclass
```

## Theoretical scope

This tool **does not** claim to crack the chronology-protection
conjecture or derive new physics. It uses two existing analytic
constructions side-by-side:

1. The hybrid photon-sphere of `systrophe.geometry.photon_sphere` — a known
   no-go for bare van Stockum + the known Schwarzschild-perturbation
   route to restore one.
2. The vendored Dinos Möbius-Z₃ cover — a topological device whose
   spectrum is independent of the spacetime; included here as a
   "label" of the carved pocket, not a load-bearing physics input.

Closure residuals are honest finite-difference scores of how flat
`db/dr` is at the engineered radius (Brent search tolerance bounds
them by ~1e-6). The Z₃ closure phase Σ ω^k = 0 is exact arithmetic
and used only as a cross-check on the vendored OMEGA constant.

## Tests

```
PYTHONPATH=src:tools/implosion-carving python -m pytest \
    tools/implosion-carving/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
