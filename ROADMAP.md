# Systrophe roadmap

This document tracks planned extensions, in rough priority order. Items
move into `CHANGELOG.md` as they ship.

## Phase 1 — From "computable" to "interpretable"

### 1a. Energy-condition diagnostics — `energy_conditions.py` (in progress)
Compute the proper energy density `ρ(r)`, pressure components, and check
the standard energy conditions (null, weak, strong, dominant) for the
van Stockum dust source. Frames every CTC result with a clear
"realizability" verdict: which conditions the source violates and where.
Spoiler: van Stockum dust satisfies all four, so the Tipler/Systrophe
CTCs come purely from idealised geometry (infinite, rigid, perfectly
axisymmetric), not from exotic matter.

### 1b. Other analytically-tractable CTC spacetimes
Each becomes a `systrophe.spacetimes.<name>` module sharing the existing
`CircularOrbit`, `find_ctc_intervals`, and time-machine harness.

- **Gödel universe (1949)**: rotating dust universe, every point
  globally has CTCs through it. Canonical CTC reference. Closed form,
  small.
- **Gott pair (1991)**: two cosmic strings passing each other
  relativistically. Closest conceptual analog to Systrophe's pair
  construction; 2+1-dimensional, fully analytic.
- **Kerr inner region**: extends the implementation to realistic
  compact-object physics. Past the inner horizon, near the ring
  singularity, CTCs exist. Connects to LIGO/EHT-relevant black holes.
- **Tomimatsu–Sato δ = 2**: distorted Kerr generalisation with richer
  CTC structure.

The shared infrastructure means each module is small (≈ 200 lines)
once the metric is known.

## Phase 2 — From "classical" to "semi-classical"

### 2a. Stress-energy on a CTC background — **SHIPPED in v0.20.0 (2026-05-13)**
2D Polyakov renormalised `<T_{mu nu}>` of a massless conformally-coupled
scalar in three canonical vacuum states (Boulware, Hartle-Hawking
analog, Unruh analog) on the supercritical Tipler exterior. Boulware
`<T_{tt}>` fits a clean simple pole `~ 1/(r - r_H)` at three
independent Cauchy horizons (powers -1.007, -0.997, -1.001); Boulware
`<T_{rr}>` fits the inverse square (power -1.998). Polyakov trace
identity holds at machine precision. Verdict:
`chronology_protection_consistent`. See `stress_energy_ctc.py`,
`tests/test_stress_energy_ctc.py`, and
`examples/phase_2a_chronology_protection.py`.

### 2b. Hadamard / point-splitting for cylindrical Tipler — **SHIPPED in v0.21.0 (2026-05-13)**
4D Hadamard biparametrix module `hadamard_modesum.py`: V_0(x) = 0
(vacuum + massless + conformal), V_1(x) = K_Kretsch / 720, 2 V_1 /
(8 pi^2) = K / (2880 pi^2) cross-validates `point_splitting`. WKB
radial mode-sum + Hadamard UV subtraction. 4D chronology-protection
scan: V_1 BOUNDED at every Cauchy horizon (powers ~ 0), confirming
v0.10.0 -- the divergence is state-dependent (Phase 2a), not local.
14 tests, example in `examples/phase_2b_hadamard_modesum.py`.

## Phase 3 — From "pair" to "array"

### 3a. N-cylinder phased array — **SHIPPED in v0.21.0 (2026-05-13)**
`SystropheArray` (v0.4.0) extended with beam-forming + extinction
diagnostics: `phasor_field`, `array_factor`, `extinction_check`
(N=2..8 uniform-phase comb verified to machine precision),
`dirichlet_pattern`, `beam_steer` (analytic placement of L-node at
chosen r_target to machine zero), `beam_pattern`, side-lobe-level
diagnostic. 9 new tests, example in `examples/phase_3a_beam_steering.py`.

### 3b. Off-axis pair quantitative orbits — **SHIPPED in v0.21.0 (2026-05-13)**
`OffAxisPair` (v0.2.0) extended with topology + quantitative
diagnostics: `ergosurface_2d`, `ctc_region_topology` (NEW FINDING:
canonical separation-3 pair has CTC = 1 component + 2 holes,
resolution-stable), `trace_anomaly_2d_sector`,
`geodesic_completeness_test`. 6 new tests, example in
`examples/phase_3b_off_axis_topology.py`.

## Phase 4 — From "implementation" to "observation"

### 4a. Observational signatures
- Photon orbits in a Tipler exterior — what does a Tipler-like
  spacetime look like to a distant observer? Lensing patterns,
  redshift gradients, Faraday rotation analogues.
- Cross-check against near-extremal Kerr black-hole observations
  (M87*, Sgr A*) for any Tipler-like signature.

### 4b. Pair geodesic visualisation
3D worldline plots of N-revolution CTC orbits with proper-time vs
coordinate-time animation.

## Phase 5 — From "code" to "tool"

### 5a. PyPI release
Make `pip install systrophe` work directly, no editable install
required.

### 5b. Pyodide / interactive web demo
A single-page app where users drag sliders for `ω`, `R`, `δ`, and see
CTC bands shift in real time. Lowest-friction outreach.

### 5c. Tutorial notebook
Step-by-step Jupyter notebook walking through the math + code with
runnable cells. Replaces the current README quickstarts as the primary
onboarding path.

### 5d. Sphinx / MkDocs documentation site
API reference auto-generated from docstrings. Hosted on GitHub Pages.

## Out of scope

- Engineering/manufacturing studies — the source idealisation makes
  this premature.
- ADM 3+1 numerical relativity for fully nonlinear two-cylinder
  evolution — would require importing or wrapping a NR code (Einstein
  Toolkit, etc.); out of scope for a package this size.
- Field theory in curved spacetime beyond free scalar — hand off to
  specialised QFTCS packages.

---

## Long-horizon north star

Make Systrophe the canonical reference for *analytical CTC physics*: a
single library covering every analytically-solvable CTC spacetime, with
unified interfaces for orbit tuning, energy-condition checks, and
quantum back-reaction diagnostics. Every claim verified by tests to
machine precision. Every result reproducible from the source.
