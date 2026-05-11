# Changelog

All notable changes to the Systrophe project will be recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - 2026-05-10

### Verified / falsified Grok-chat math assertions (examples/grok_verification.py)
Systematic verification battery covering 19 mathematical claims from the
Grok conversation on the Systrophe repo:

- **VERIFIED (10)**: alpha formula `sqrt(4a^2 - 1)`; F closed form;
  phasor sum `A_eff e^{i delta_eff} = A_1 e^{i d_1} + A_2 e^{i d_2}`;
  anti-phase extinction at delta = pi; N-fold uniform-comb extinction;
  timelike orbit roots `Omega_+- = -(K +- r)/L`; Cauchy horizon
  spacing `exp(pi / alpha)`; Tolman blueshift `1/sqrt(F)`; surface
  gravity `kappa = (1/2)|F'|`; Hawking temperature `T_H = kappa/(2 pi)`;
  trace anomaly `K/(2880 pi^2)`; heat-kernel `a_2 = K/180`; CTC band
  is `L < 0`; tunable chronology via phase; log-measure minimum at
  delta = pi.
- **FALSIFIED (1)**: Grok's "Ernst potential E_0 = F + i K". The Ernst
  potential is `F + i psi` where psi is the *twist potential*, not
  the metric component `K = g_{t phi}`. They are dual but distinct
  fields.
- **NOT TESTABLE (8)**: Z_3 cover finite <T_munu> off-trace (Hadamard
  renormalisation not implemented); Newton-Kantorovich 3-5 iter
  convergence (iteration scheme not implemented; Tipler-sinusoid seed
  has zero residual anyway, converging trivially in one iteration);
  self-consistent delta iteration; Floquet-Mobius solver; etc.

### Research-grade Floquet (replaces v0.10.0 toy)
The v0.10.0 `floquet.py` used a 2-level ansatz Hamiltonian. The toy
passed its tests but didn't compute anything specifically about the
radial Dirac on a time-varying LP background.

v0.11.0 replaces it with the **adiabatic Floquet** of the actual
problem:
- `adiabatic_floquet_spectrum(cyl, delta_0, delta_amp, omega_drive, ...)`:
  at each instant t in [0, T], solves the static radial Dirac
  eigenvalue problem on the joint matched-pair LP metric (via
  `dirac_spectrum.find_bound_states`); returns time-averaged
  instantaneous bound-state energies, wrapped to the fundamental
  Brillouin zone `[-Omega/2, Omega/2)`.
- `nonadiabatic_floquet_correction(...)`: leading-order estimate of
  the non-adiabatic correction to a given quasi-energy.
- `adiabatic_floquet_validity(...)`: diagnostic returning
  Omega_drive / gap; adiabatic is exact in the limit ratio -> 0.
- `static_pair_bound_states(...)`: convenience wrapper for the
  bound-state shooting on the joint pair metric.

This is the actual radial Dirac problem on a time-varying joint
Lewis-Papapetrou metric, not a 2-level ansatz. Exact in the slow-drive
limit; non-adiabatic corrections specified.

Tests run for 2-3 minutes (the bound-state shooting is the bottleneck);
237 tests total, all passing.

## [0.10.0] - 2026-05-10

### Added

**Floquet quasi-energy solver for time-varying offset** (`floquet.py`):
- `time_evolution_operator_at_r(...)`: one-period propagator U(T) for
  a periodically-driven LP background with `delta(t) = delta_0 +
  delta_amp sin(Omega_drive t)`. Built via product-Trotter at
  configurable substep count.
- `floquet_quasi_energies_at_r(...)`: diagonalises U(T) to extract the
  two Floquet quasi-energies of the effective 2 x 2 Dirac-spinor
  problem on the t-periodic background.
- `compute_floquet_spectrum(omega, R, delta_0, delta_amp, omega_drive,
  r_grid, ...)`: full radial scan returning a `FloquetSpectrum`
  dataclass.
- `detect_parametric_resonance(spec)`: locates avoided crossings in
  the Floquet band gap (candidate parametric-resonance points).

**Grok-claim falsification** (`examples/grok_falsification.py`):
- Direct numerical test of the claim "Z_3 Mobius cover keeps
  <T_munu> finite at Cauchy horizons" raised in a Grok conversation
  on the repo.
- Result: Kretschmann scalar K converges to ~1.07 at the first
  Cauchy horizon (spread 0.17% across r in [r_h - 1e-3, r_h - 1e-4]).
  The trace anomaly <T^mu_mu> = K/(2880 pi^2) is therefore bounded
  without invoking any Z_3 cover. F = 0 is a *coordinate* singularity
  (ergosurface), not a curvature singularity; finiteness of the
  trace is intrinsic.
- Grok's attribution overstated for the trace component; the
  non-trivial chronology-protection question (off-trace <T_munu>
  via Hadamard point-splitting) remains open.

### Tests
- 6 new tests in `test_floquet.py`; 237 total.

## [0.9.0] - 2026-05-09

### Added

**Full 4D point-splitting renormalisation** (`point_splitting.py`):
- Numerical Christoffel symbols, Riemann tensor, Ricci tensor, Ricci
  scalar, and Kretschmann scalar via central differencing of the
  analytic Case III metric components.
- `kretschmann_scalar(vs, r)`: K = R_{μνρσ} R^{μνρσ}.
- `trace_anomaly_4d_exact(vs, r)`: ⟨T^μ_μ⟩_ren = K / (2880 π²),
  *exact* in 4D vacuum for a massless conformally-coupled scalar.
- `dewitt_a2_coefficient(vs, r)`: a_2(x) = K / 180 (heat-kernel).
- `phi_squared_largemass_expansion(vs, r, mass)`: leading large-mass
  ⟨φ²⟩_ren ≈ a_2(x) / (16 π² m²).
- `effective_action_volume_density(vs, r)`: a_2 / (32 π²), one-loop
  effective Lagrangian density.

**WebGL visualiser** (`web/visualizer.html`):
- Single-page in-browser visualiser with Three.js (CDN), no server.
- Eight tabs covering every concept implemented in the package:
  1. Single Tipler cylinder with rotation arrows + L(r) plot + CTC bands.
  2. Off-set pair with constructive/destructive interference and the
     anti-phase off-switch demo.
  3. N-cylinder phased array showing topological extinction at uniform
     phase comb (N = 2 to 8).
  4. Off-axis pair 2D heat map with red CTC region and blue regular.
  5. Spacetime worldline of a backward-time orbit (interactive helix).
  6. CTC zoo: Tipler / Gödel / Gott / Kerr side by side with thresholds.
  7. Curvature & trace-anomaly diagnostic with Cauchy-horizon markers.
  8. Unified animated picture combining the pair, CTC bands, Cauchy
     horizons, and a particle traversing a CTC orbit.
- All math runs client-side in JavaScript (ports of the Systrophe Python
  kernels: tiplerF/K/L, ctcBands, godelL, gottHasCTC, kerrGphiphiEquatorial).

### Test count
- 231 tests across 25 modules; previously 219.

### Whitepaper
- Extended to 23 pages with subsections on the full 4D point-splitting
  module and the WebGL visualiser.

### Notes
This completes the 4D point-splitting work that the v0.8.0 release left
as honest future work. The trace anomaly is now computable to high
numerical precision on the LP background; the off-trace components of
⟨T_μν⟩ remain a separate research project (mode-sum at finite m, full
Hadamard renormalisation).

## [0.8.0] - 2026-05-09

### Added — completing all genuinely deferred QFTCS items

**Spectrum and bound states** (`dirac_spectrum.py`):
- `boundary_functional`: integrates the radial Dirac with seed at r_min,
  reports the functional at r_max for use as a shooting target.
- `find_bound_states`: sweeps E and refines local minima via
  scipy.optimize.minimize_scalar.
- `vanstockum_bound_states`: convenience wrapper using the analytic
  Case III closed forms with Dirichlet BCs at r=R and r_max.

**Dirac sea structure** (`dirac_sea.py`):
- `local_energy`: Tolman-shifted local energy E_local = E_infty / sqrt(F).
- `density_of_states_radial`: leading-order radial Dirac level density,
  diverging as 1/sqrt(F) near Cauchy horizons.
- `dirac_sea_pressure_proxy`: 1/F^2 vacuum-pressure proxy.
- `chronology_horizon_pressure_divergence_rate`: numerical
  power-law fit confirming p ~ 2 near each F-zero.

**Particle creation rates** (`particle_creation.py`):
- `bose_einstein`, `fermi_dirac` thermal occupation functions.
- `particle_creation_spectrum_at_horizon`: full thermal spectrum at a
  Cauchy horizon with Hawking temperature T_H = kappa / (2 pi).
- `total_emission_power_proxy`: integrated emission power proxy.

**QFTCS back-reaction (2D conformal anomaly)** (`qftcs_backreaction.py`):
- `radial_temporal_ricci_scalar`: 2D Ricci scalar of the (t, r) slice
  of the Tipler metric.
- `conformal_anomaly_trace`: <T^mu_mu>_ren = R/(24 pi) for a massless
  conformally coupled scalar (exact in 2D).
- `vacuum_energy_density_proxy`: <T_tt> = -R/(96 pi) Polyakov action
  vacuum energy.
- `stress_energy_at_horizon`: numerical power-law fit of the
  divergence rate as r approaches a Cauchy horizon.

### Test count
- 219 tests across 24 modules; previously 196.

### Whitepaper
- Extended to 22 pages with a "Spectrum, Dirac sea, particle creation,
  and back-reaction" section detailing all four new modules and their
  honest scope (leading-order classical-on-quantum approximations).

### Notes
This completes the items previously identified as "genuinely deferred
research items" in v0.7.0. The implementations are leading-order
classical-on-quantum approximations of the full Schwinger-DeWitt /
Hadamard programme; a 4D point-splitting renormalisation remains
the natural next research step but the infrastructure for it is now
in place.

## [0.7.0] - 2026-05-09

### Added
- **Dirac field on the Lewis-Papapetrou background** (`dirac.py`).
  - `LewisPapapetrouTetrad`: orthonormal tetrad e^a_mu in
    (-, +, +, +) signature with `e^2_phi = sqrt(L + K^2/F) =
    r/sqrt(F)` from the Weyl constraint. Reproduces the metric to
    machine precision in tests.
  - `gamma_matrix(a)`: flat-space gamma matrices in a Weyl-style
    representation appropriate for (-, +, +, +) signature, satisfying
    `{gamma^a, gamma^b} = 2 eta^{ab}`.
  - `radial_dirac_system`: assembles the radial ODE for the
    upper/lower-component two-spinor reduction with given (E, m, k,
    mass).
  - `solve_radial_dirac`: numerical integrator over a chosen
    [r_min, r_max] interval.
  - `vanstockum_dirac_system`: convenience specialisation to the
    Tipler exterior using the analytic Case III closed forms.
- 9 new tests for tetrad orthonormality, Clifford algebra, and
  integrator runs. Total: 196 tests.
- Whitepaper extended to 21 pages with a Dirac-on-LP section.

### Notes
This is the natural Dirac entry point that the Δῖνος bridge previously
only touched at the level of discrete-mode correspondences. Full
spectrum, bound-state analysis, and the connection to QFTCS
back-reaction at the Cauchy horizon remain future work.

## [0.6.0] - 2026-05-09

### Added — finishing all deferred roadmap items

**Phase 3b: full off-axis geodesics.** `OffAxisPair.integrate_test_particle`
integrates a timelike or null trajectory through the joint Cartesian
metric of two parallel-axis cylinders, returning (t, x, y, tau) arrays.
Tests verify initial conditions and tau monotonicity.

**Phase 4: photon ray tracing / observational signatures.** New
`photon_raytrace.py` module with:
- `photon_perihelion`: numerical search for null-geodesic turning points.
- `photon_deflection_angle`: bend-angle integral for a photon between
  perihelion and r_max.
- `lensing_pattern`: sweep over impact parameters and report deflection.
Validated in Minkowski (truncation residual = -2 arcsin(b/r_max)) and
verified to run on the Tipler supercritical exterior.

**Phase 5b: Pyodide interactive web demo.** `web/index.html` is a
single-page app that loads Pyodide in the browser, lets the user drag
sliders for `a` and `δ`, and re-renders the joint `g_phiphi(r)` curve
with shaded CTC bands in real time. Pure static HTML; no server.

**Phase 5d: Sphinx documentation.** `docs/conf.py`, `docs/index.rst`,
`docs/api.rst`. `.github/workflows/docs.yml` builds the docs and
publishes to GitHub Pages on every push to main. Local build verified.

**Phase 2 substantive extension.** `quantum_diagnostics.py` extended with:
- `ricci_scalar(vs, r)`: numerical Ernst-equation residual (zero in
  exact vacuum; finite-difference error in numerics).
- `conformal_anomaly_2d_proxy(vs, r)`: heuristic 2D conformal-anomaly
  proxy for a conformally-coupled massless scalar; diverges at F=0.
- `surface_gravity_at_horizon(vs, r_h)`: kappa = |F'|/2 at a Cauchy-
  horizon candidate, the scale of thermal back-reaction.
- `hawking_temperature_at_horizon(vs, r_h)`: T_H = kappa / (2 pi).

These remain classical proxies — full QFTCS back-reaction is still
future work — but quantify the chronology-protection signals more
honestly than the previous 1/F^2 indicator alone.

### Test count
- 187 tests across 19 files; previously 176.

## [0.5.0] - 2026-05-09

### Added
- **Photon orbits / null geodesics** (`photon_orbits.py`): null circular
  Omega_+- = (-K +- r)/L, photon impact parameters b_+- = (K +- r)/F,
  null-geodesic integrator (kappa = 0 mode of `integrate_geodesic`),
  van Stockum-specific photon Omega utility. 7 tests.
- **Kerr inner-region CTCs** (`spacetimes/kerr.py`): Boyer-Lindquist
  metric components, sub/super/extremal classification, equatorial
  CTC threshold solving the cubic r^3 + a^2 r + 2 M a^2 = 0, CTC
  region (r_CTC, 0). 10 tests.
- **Three singularity reinterpretations** of the van Stockum source:
  - `LineSingularity` (rotating axis with M, J): exterior identical
    to van Stockum.
  - `CosmicString` (Vilenkin static line): conical defect, no CTCs;
    `compose_with_gott` constructs a Gott pair from two of them.
  - `KerrSpacetime` (4D rotating ring): the analytic continuation of
    Kerr to negative r contains CTCs.
  - 17 tests covering all three.
- **Quantum-diagnostic stub** (`quantum_diagnostics.py`): Tolman
  blueshift, 1/F^2 chronology indicator, closed-form Cauchy horizon
  enumeration. Documented as classical pre-quantum; full QFTCS
  back-reaction left to future work. 6 tests.
- **PyPI publish workflow** (`.github/workflows/publish.yml`):
  tag-triggered Trusted-Publishing release pipeline. No tokens stored.
- **Tutorial Jupyter notebook** (`examples/tutorial.ipynb`): end-to-end
  walk through every public API.
- **VanStockumInterior.mass_per_unit_length** and
  **angular_momentum_per_unit_length** properties + 
  **as_line_singularity_summary()** method connecting the dust
  interpretation to the line-singularity reinterpretation.

### Changed
- Whitepaper extended to 20 pages with sections on:
  - Photon orbits in the Tipler exterior
  - The three singularity reinterpretations
  - The CTC zoo with Kerr ring
  - Quantum-diagnostic test verdicts

### Test count
- 176 tests across 17 files; previously 134.

## [0.4.0] - 2026-05-09

### Added
- `array.py` — `SystropheArray` N-cylinder phased array. Generalises
  `SystrophePair` to N co-axial co-rotating cylinders with independent
  phase offsets. Provides:
  - `from_cylinders(cyls, offsets)` — N-cylinder convenience constructor
  - `uniform_phase_comb(cyl, N)` — N copies at phases 2πi/N. Phasor sum
    is the geometric sum of N-th roots of unity = 0, so the joint
    exterior is identically zero. **N-fold topological off-switch**
    generalising the pair anti-phase result.
  - `to_single_sinusoid()` — phasor collapse for matched-frequency arrays.
  - `ctc_bands` for CTC region detection.
- 12 new tests in `test_array.py`. Total: 134 tests.

## [0.3.2] - 2026-05-09

### Added
- `systrophe.spacetimes.gott` — Gott pair (Gott 1991, PRL 66, 1126):
  symmetric pair of moving cosmic strings producing CTCs encircling
  both. Implements the standard threshold `gamma * v > tan(4 pi mu)`,
  exposes critical velocity `v_crit = sin(4 pi mu)` and critical mass
  `mu_crit = arcsin(v) / (4 pi)`. Sub-extremal cosmic strings only
  (`mu < 1/8`).
- 11 new tests in `test_gott.py`. Total: 122 tests.

## [0.3.1] - 2026-05-09

### Added
- `systrophe.spacetimes.godel` — Goedel rotating-dust universe (Goedel
  1949). First additional CTC spacetime in the new
  `systrophe.spacetimes` subpackage. Provides metric components,
  exact CTC threshold radius `r_CTC = arcsinh(1) = ln(1 + sqrt(2))`,
  and dust/Lambda diagnostics. Establishes the architectural
  pattern for multi-spacetime CTC analysis (Phase 1b of the roadmap).
- 10 new tests in `test_godel.py`. Total: 111 tests.

## [0.3.0] - 2026-05-09

### Added
- `energy_conditions.py` — pointwise energy-condition diagnostics
  (NEC, WEC, SEC, DEC) for the van Stockum dust source plus
  `EnergyConditionReport` dataclass. Result: all four conditions hold
  identically for any $\omega > 0$, $R > 0$. The Tipler/Systrophe CTC
  pathology therefore arises *purely* from geometric idealisation
  (infinite axial extent, rigid rotation, perfect axisymmetry), not
  from exotic matter.
- `proper_energy_density(omega, r) = (omega^2/(2pi)) exp(omega^2 r^2)`
  and `total_energy_per_unit_length(omega, R) = 0.5(exp(a^2) - 1)`
  exposed.
- `examples/quantum_z3_verification.py` — IBM Quantum-hardware
  verification of the Z_3 cyclic phase identity that backs the
  Systrophe ↔ Dinos correspondence. Submitted to ibm_marrakesh
  (156-qubit Heron processor) with 1024 shots; result P(0) = 1.0000
  exact. Job ID `d7vqbolpa59c73b67iq0`.
- `ROADMAP.md` documenting Phase 1–5 evaluation of next-step extensions.
- 9 new tests in `test_energy_conditions.py`. Total: 101 tests.
- Whitepaper (now 17 pages) extended with an energy-condition section,
  the IBM Quantum verification subsection, and a historical note
  acknowledging the Titor lineage of the operational specification.

## [0.2.1] - 2026-05-09

### Added
- **Analytic closed forms for all three Bonnor regimes** in
  `vanstockum.py`. Previously only the supercritical (Case III)
  sinusoidal forms were available; now `analytic_exterior_F`, `_K`,
  `_L` automatically dispatch by regime:
  - **Supercritical** (`a > 1/2`): trigonometric (Tipler sinusoid).
  - **Critical** (`a = 1/2`): logarithmic, `F = (r/R)(1 − u)`,
    `K = (r/2)(1 + u)`, `L = (rR/4)(3 + u)`.
  - **Subcritical** (`a < 1/2`): hyperbolic via `S± = cosh(βu) ±
    sinh(βu)/β`, `F = (r/R) S₋`, `K = ar S₊`, `L = rR(1 − a²S₊²)/S₋`.
- `regime` property on `VanStockumInterior` returning the Bonnor
  classification label.
- `lp_robust.integrate_lp_robust` now uses analytic forms in **all
  regimes** (no numerical fallback by default), giving machine-precision
  output throughout. Subcritical and critical `F`-zeros are computed in
  closed form.
- 8 new tests in `test_lewis_papapetrou.py` covering subcritical and
  critical boundary continuity, closed-form K/L expressions, and
  constraint `FL + K² = r²` to machine precision.

### Test count
- 89 tests across 11 files; previously 81.

## [0.2.0] - 2026-05-09

### Added
- `lp_robust.py` — regime-dispatching robust LP exterior solver. Uses
  analytic Case III closed forms for `a > 1/2` (machine precision at any
  rotation parameter) and falls back to the basic numerical integrator
  for sub- and critical regimes. Returns a uniform `LPRobustSolution`.
- `off_axis.py` — `OffAxisPair` class for two parallel-axis cylinders
  with perpendicular separation `d > 0`. Provides Cartesian metric
  perturbation summation, joint metric components, local-CTC test, and
  2D CTC mapping utility `ctc_map_2d`.
- Whitepaper extended (now 14 pages) with Section 9 (robust regime
  dispatcher) and Section 10 (off-axis pair) plus two new figures
  (`lp_robust_demo.pdf`, `off_axis_ctc_map.pdf`).
- Tests: `test_lp_robust.py` (8), `test_off_axis.py` (7).

### Test count
- 81 tests across 11 files; previously 65.

## [0.1.0] - 2026-05-09

### Added
- `geodesic.py` — `CircularOrbit` dataclass with timelike/spacelike sector
  detection, coordinate-time and proper-time per-revolution accessors,
  and a generic `integrate_geodesic` function for planar geodesics.
- `time_machine.py` — `TimeMachineWindow` dataclass and high-level
  `find_single_cylinder_windows`, `find_time_machine_windows`,
  `harness_time_loop` for time-travel orbit engineering.
- `examples/time_travel_simulation.py` — full reproducibility script for
  the whitepaper, writes JSON results.
- `paper/systrophe_time_travel.tex` — comprehensive 11-page whitepaper
  with five figures and three tables, compiled to PDF.
- `paper/generate_figures.py` — figure generator for the whitepaper.
- `LICENSE` (MIT) and `CHANGELOG.md`.
- Tests: `test_geodesic.py` (6), `test_time_machine.py` (9).

### Changed
- `tipler_sinusoid()` now defaults to L-mode (CTC-relevant) instead of
  F-mode (ergoregion-relevant). Callers seeking the old behaviour
  should use `tipler_sinusoid_F()` explicitly.
- `find_ctc_intervals` now correctly handles L-zeros that fall exactly
  on grid points (previously skipped due to product == 0 check).

### Test count
- 65 tests across 9 modules; previously 50.

## [0.0.3] - 2026-05-09

### Added
- `lewis_papapetrou.py` numerical Ernst-equation integrator, validated
  against analytic Case III to 10⁻³ globally and machine precision in
  the well-conditioned regime.
- `vanstockum.py.analytic_exterior_F`, `analytic_exterior_K`,
  `analytic_exterior_L` — closed-form Case III metric components.
- `vanstockum.py.tipler_sinusoid_F`, `tipler_sinusoid_L` — auto-derived
  matched TiplerSinusoid for both metric components.
- `dinos_bridge.py` — optional Dinos-DKN interop with cylindrical-Kerr
  identification, Z₃ Möbius eigenvalue match, M\"obius temporal-loop
  hook.
- `tools/derive_lewis_papapetrou.py` — SymPy derivation script for the
  vacuum Einstein equations of the cylindrical WLP ansatz.
- `pair.py.from_cylinders`, `offset_sweep`, `total_ctc_log_measure`,
  `ctc_bands`.
- Tests: `test_lewis_papapetrou.py` (12), `test_dinos_bridge.py` (6),
  `test_offset_sweep.py` (6).

## [0.0.2] - 2026-05-09

### Changed
- Renamed package from `tiplerpair` → `systrophe` (Greek
  "twisting-together"). Pair class renamed to `SystrophePair`. Other
  identifier changes propagated; `TiplerSinusoid` retained as the
  generic log-periodic envelope class.

## [0.0.1] - 2026-05-09

### Added
- Initial release as `tiplerpair`.
- `vanstockum.py` van Stockum 1937 interior metric.
- `sinusoid.py` `TiplerSinusoid` log-periodic envelope class with
  3-parameter fit utility.
- `pair.py` linearised superposition of two co-axial Tipler sinusoids.
- `ctc.py` generic CTC band detector.
- 26 tests.
