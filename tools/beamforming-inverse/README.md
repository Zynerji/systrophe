# beamforming-inverse

**N-cylinder SystropheArray inverse problem.** Given fixed cylinder
geometry (R_i, α_i) and a prescribed target complex phasor profile
z_target(r_m) at M sample radii, solve for the complex amplitudes
c_i = A_i e^{i δ_i} that best synthesise it.

Priority-2 sibling of `tools/implosion-carving/` (priority-3 was the
photon-pocket carver; priority-1 is the converging-shock CTC-stress
profile).

## Forward model (from `systrophe.geometry.array.SystropheArray.phasor_field`)

```
z(r) = sum_i  c_i * exp(i α_i ln(r / R_i)),    c_i = A_i e^{i δ_i}.
```

Linear in c. Inverse is a complex least-squares problem:

```
G c ≈ z_target,    G[m, i] = exp(i α_i ln(r_m / R_i)),    G ∈ C^{M×N}.
```

* **Overdetermined** (M > N): least-squares via `np.linalg.lstsq`.
* **Underdetermined** (M < N): min-norm via the same call.

## API

```python
from beamforming_inverse import (
    BeamformingDesign, solve_beamforming_inverse, synthesised_array,
)
import numpy as np

a = np.array([0.6, 0.75, 0.9])           # distinct -> independent basis cols
alpha = np.sqrt(4 * a * a - 1)
design = BeamformingDesign(R=np.ones(3), alpha=alpha, a=a, p=np.ones(3))

rs = np.linspace(1.05, 8.0, 100)
z_target = ...  # your target complex array, shape (100,)

result = solve_beamforming_inverse(design, rs, z_target)
print(result.A, result.delta)             # solved amplitudes & phases
print(result.relative_residual)           # how well it fits

synth = synthesised_array(design, result) # rebuild the SystropheArray
```

## When the inverse is well-posed

* **Distinct α_i** across cylinders → columns of G are linearly
  independent → full-rank (rank = min(M, N)).
* **Matched cylinders** (same a, R) → columns are identical → rank-1
  → the inverse is degenerate, only the SUM of c_i is determined. In
  this regime the forward solver's `to_single_sinusoid()` is the
  natural representation; the inverse here will still return *a*
  c-vector but it lives in a 2-dim subspace.

## Tests

12 tests, all offline:

```
PYTHONPATH=src:tools/beamforming-inverse python -m pytest \
    tools/beamforming-inverse/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
