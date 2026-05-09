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

### 2a. Stress-energy on a CTC background
Compute the renormalised quantum stress-energy ⟨T_μν⟩ of a free scalar
field on the supercritical Tipler exterior. The chronology-protection
conjecture (Hawking 1992) predicts ⟨T_μν⟩ diverges as the Cauchy horizon
is approached; verifying this in the cleanest cylindrical setting is a
quantitative test of CP.

### 2b. Hadamard / point-splitting for cylindrical Tipler
Standard mode-sum technique adapted to the log-periodic exterior.
Scientifically the most interesting extension; adds a few hundred
lines and a real physics result.

## Phase 3 — From "pair" to "array"

### 3a. N-cylinder phased array
Generalise `SystrophePair` to `SystropheArray(cylinders, offsets)`.
Phasor superposition of N log-periodic sinusoids; CTC band structure
becomes the result of multi-source interference. Beam-forming analogues
become natural.

### 3b. Off-axis pair quantitative orbits
Current `OffAxisPair` is a leading-order CTC-region detector; extend
with full geodesic integration in the joint metric (numerical).

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
