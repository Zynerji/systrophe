# Photon spheres in the Systrophe LP exterior

A **photon sphere** is a locus where photons travel in closed
circular orbits. For Schwarzschild, the photon sphere sits at
`r = 3M`. For Kerr, prograde/retrograde photon spheres appear at
different radii. For the Tipler / van Stockum rotating-cylinder
spacetime, **photon spheres do not exist** in the bare LP exterior —
the impact-parameter function b(r) is monotonically increasing
across all rotation regimes.

## The no-go result

The photon impact parameter on the LP exterior is

```
b(r) = (K(r) ± r) / F(r)
```

(prograde/retrograde branches). A photon sphere requires `db/dr = 0`.
Numerical scans across subcritical, critical, and supercritical
regimes (and across r ∈ [1.01R, 100R]) find:

- `db/dr > 0` everywhere
- No sign changes in 5000-point grids
- The function diverges at chronology horizons (F = 0) but does not
  have local extrema between them

**Reason**: in canonical Weyl coordinates with K² + LF = r², the
impact parameter b = (K ± r)/F has a geometric pole at F → 0 that
dominates the log-periodic oscillation. Between consecutive
chronology horizons, b stays on one sign and grows monotonically.

The Tipler exterior therefore differs from Schwarzschild/Kerr in
the absence of any closed photon orbit at finite b.

## Creating a photon sphere: hybrid Schwarzschild + cylinder

To engineer a photon sphere, perturb the LP exterior with a
Schwarzschild mass M at the cylinder axis. The combined effective
metric becomes (leading-order)

```
F_hybrid(r) = F_LP(r) · (1 − 2M/r)
K_hybrid(r) = K_LP(r)
```

valid for `r > 2M`. The hybrid impact parameter
`b_hybrid(r) = (K_LP ± r) / F_hybrid` admits photon spheres at
positions controlled by the relative weight of `M/R` and `ωR`.

In the weak-cylinder limit (`ωR → 0`), the hybrid photon sphere
approaches the Schwarzschild value `r = 3M`. With non-trivial
cylinder rotation, the photon sphere is shifted by the LP
log-periodic structure.

## Engineering a photon sphere at a target r

The function `engineer_photon_sphere_via_mass(vs, r_target)` solves
for the Schwarzschild mass `M` such that

```
db_hybrid/dr (r_target) = 0
```

via Brent's method on a 30-point M grid. For a given target radius,
the function returns the matched mass, or None if no solution
exists in the search range.

Example:
- Cylinder ω = 1, R = 1 (a = 1, supercritical)
- Target r_target = 2.5
- Result: M ≈ 0.7 (depends on numerics)
- Verify by checking that the hybrid metric at r_target indeed has
  db/dr = 0

## Implications

1. **Black-hole-like shadows are not native to the cylinder**:
   a distant observer of a rotating-cylinder spacetime would not see
   a Schwarzschild-like black silhouette. The cylinder produces
   different optical signatures (e.g., chronology-horizon caustics).

2. **To get a shadow, you must add a mass**: only a hybrid
   Schwarzschild-cylinder configuration produces a finite-b photon
   sphere and hence a visible shadow.

3. **Tipler-pair extinction at δ = π persists in the hybrid case**:
   the L_pair extinction proxy doesn't affect F_hybrid directly, so
   the chronology-protection mechanism (Section I.3 of
   `docs/INTERPRETATIONS.md`) is independent of the photon-sphere
   structure.

## Module

`src/systrophe/photon_sphere.py` provides:

- `bare_lp_has_photon_sphere(vs)` — no-go check (returns False for
  any standard van Stockum cylinder)
- `impact_parameter_bare(vs, r, branch)` — bare LP impact parameter
- `effective_b_hybrid(vs, r, M, branch)` — hybrid impact parameter
- `hybrid_photon_sphere_radii(vs, M, ...)` — all hybrid photon
  spheres in a range
- `hybrid_photon_sphere_stability(vs, r_ph, M)` — stable (True) or
  unstable saddle (False)
- `hybrid_photon_sphere_omega(vs, r_ph, M)` — orbital frequency
- `engineer_photon_sphere_via_mass(vs, r_target, M_range)` — solve
  for M
- `photon_sphere_descriptor(vs, r_ph, M, branch)` — full descriptor

16 tests in `tests/test_photon_sphere.py`, all passing.
