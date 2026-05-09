# Changelog

All notable changes to the Systrophe project will be recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
