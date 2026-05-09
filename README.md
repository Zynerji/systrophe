<div align="center">

# Συστροφή — Systrophe

**A co-rotating Tipler-cylinder pair as a tunable time-travel harness.**

[![Tests](https://github.com/Zynerji/systrophe/actions/workflows/tests.yml/badge.svg)](https://github.com/Zynerji/systrophe/actions/workflows/tests.yml)
[![Python ≥ 3.10](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 134 passing](https://img.shields.io/badge/tests-134%20passing-brightgreen.svg)](#tests)
[![Whitepaper](https://img.shields.io/badge/whitepaper-PDF-informational.svg)](paper/systrophe_time_travel.pdf)
[![Version 0.2.0](https://img.shields.io/badge/version-0.2.0-blue.svg)](pyproject.toml)

*Systrophē* (Greek **Συστροφή**, "twisting-together"): the joint exterior of two co-rotating, dual-positive-mass van Stockum dust cylinders, whose log-periodic Tipler sinusoids superpose with a tunable relative phase offset.

</div>

---

## What this is

A complete numerical and analytic implementation of:

1. **The single rotating cylinder** — exact van Stockum 1937 interior + analytic Bonnor Case III exterior (closed forms for `F`, `K`, `L`).
2. **The Lewis–Papapetrou ODE integrator** — Ernst-equation numerical exterior for any rotation parameter, validated to machine precision against the closed forms.
3. **The Systrophe pair** — two co-axial supercritical cylinders linearly superposed, producing an *off-set Tipler sinusoid* whose CTC band positions are tunable by the relative phase `δ₂ − δ₁`.
4. **A time-machine harness** — locate CTC bands, derive timelike-orbit angular-velocity sectors, tune coordinate-time-per-revolution to a target value, simulate forward and backward time-travel orbits.
5. **A bridge to the Δῖνος (Dinos) Dirac–Kerr–Newman framework** — exact identification of the Tipler log-grid fundamental eigenvalue with the Z₃ Möbius cover branch=0 fundamental; non-trivial branches realise the off-set sectors of the systrophic pair.

The package ships with a **comprehensive whitepaper** ([`paper/systrophe_time_travel.pdf`](paper/systrophe_time_travel.pdf)) deriving every formula, all five figures, and reproducing the simulation results end-to-end.

---

## Installation

```bash
pip install -e ".[dev]"        # editable install with dev tools
pytest                         # 65 tests, ~5 seconds
```

Optional extras:
- `.[symbolic]` — SymPy for the one-shot derivation script (`tools/derive_lewis_papapetrou.py`).
- `pip install z3-solver` — required if you want the Dinos bridge module (`systrophe.dinos_bridge`).

---

## Quickstart: build a single time machine

```python
import numpy as np
from systrophe import VanStockumInterior, find_single_cylinder_windows, harness_time_loop

# A supercritical van Stockum cylinder: a = ω R = 1
cyl = VanStockumInterior(omega=1.0, R=1.0)

# Locate CTC bands in r ∈ [1.001, 200]
windows = find_single_cylinder_windows(cyl, r_min=1.001, r_max=200.0)
for w in windows:
    print(f"CTC band:  r ∈ [{w.r_inner:.3f}, {w.r_outer:.3f}]   "
          f"deepest L = {w.L_min:.3f}")

# Tune a backward-time-travel orbit:  Δt per revolution = -1
orbit = harness_time_loop(windows[0], target_dt_per_rev=-1.0, n_revolutions=10)
print(f"Ω = {orbit['Omega']:+.4f}")
print(f"per rev: Δt = {orbit['dt_per_revolution']:+.3f},  Δτ = {orbit['dtau_per_revolution']:.3f}")
print(f"after 10 revs: Δt_total = {orbit['total_coord_time_advance']:+.3f},  "
      f"Δτ_total = {orbit['total_proper_time_advance']:.3f}")
```

Expected output:
```
CTC band:  r ∈ [1.001, 6.134]   deepest L = -3.351
CTC band:  r ∈ [37.622, 200.000] deepest L = -126.065
Ω = -6.2832
per rev: Δt = -1.000,  Δτ = 11.354
after 10 revs: Δt_total = -10.000,  Δτ_total = 113.537
```

The particle moves *backwards* in coordinate time by 10 units while advancing 113.5 units of its own proper time.

---

## Quickstart: tune the harness via a co-rotating pair

```python
from systrophe import SystrophePair, VanStockumInterior

cyl = VanStockumInterior(omega=1.5, R=1.0)
pair = SystrophePair.from_cylinders(cyl, cyl, delta_offset=0.7853981633974483)  # π/4

print(f"phase offset = {pair.phase_offset:.4f} rad")
bands = pair.ctc_bands(r_min=1.05, r_max=20.0)
print(f"{len(bands)} CTC bands; first at r ∈ [{bands[0][0]:.3f}, {bands[0][1]:.3f}]")
```

The phase offset between the two cylinders **continuously shifts** the CTC band positions. At exact anti-phase (`δ = π`), all CTC bands extinguish — a topological off-switch.

---

## End-to-end demonstration

```bash
python examples/time_travel_simulation.py
```

Runs the full numerical experiment from the whitepaper: identifies CTC bands, sweeps offset, computes time-travel orbits, writes machine-readable JSON results to `examples/time_travel_simulation_results.json`.

---

## Mathematical highlights

**Tipler log-frequency.** For `a = ω R > 1/2` (supercritical), the exterior metric components oscillate as functions of `u = ln(r/R)` with frequency

```
α = √(4 a² − 1).
```

**Closed forms** (all three Bonnor regimes).

*Supercritical* (`a > 1/2`): with `γ = π − arctan α`,
```
F(r) = (r/R) · sin(α u + γ) / sin γ
K(r) = (r/α)  · [ ((α² − 1)/2) sin(α u + γ) − α cos(α u + γ) ]
L(r) = (r R sin γ / α²) · [ Q sin(α u + γ) + α(α² − 1) cos(α u + γ) ]
```
with `Q = α² − (α² − 1)² / 4`.

*Critical* (`a = 1/2`):
```
F(r) = (r/R)(1 − u)        K(r) = (r/2)(1 + u)        L(r) = (rR/4)(3 + u)
```

*Subcritical* (`a < 1/2`): with `β = √(1 − 4a²)` and `S± = cosh(βu) ± sinh(βu)/β`,
```
F(r) = (r/R) S₋(u)         K(r) = a r S₊(u)           L(r) = rR(1 − a²S₊²)/S₋
```

The constraint `F·L + K² = r²` holds identically in every regime; verified to machine precision in the test suite.

**Pair superposition.** For matched `α`, the joint envelope is a single sinusoid whose amplitude and phase come from the phasor sum

```
A_eff · exp(i δ_eff) = A₁ exp(i δ₁) + A₂ exp(i δ₂).
```

**Time-travel orbit.** A circular orbit at fixed `r` in a CTC band has

```
Δt    = 2 π / Ω                              (coordinate time per rev)
Δτ    = √(F − 2 K Ω − L Ω²) · |Δt|           (proper time per rev)
```

The full derivation, with Lewis–Papapetrou Ernst-equation reduction and Bonnor's Case classification, is in the whitepaper.

---

## Architecture

```
src/systrophe/
  vanstockum.py        — Interior metric + analytic Case III exterior closed forms
  lewis_papapetrou.py  — Basic numerical Ernst-equation integrator (Radau IRK)
  lp_robust.py         — Regime-dispatching robust solver (machine precision at any a)
  sinusoid.py          — Generic TiplerSinusoid log-periodic envelope class
  pair.py              — SystrophePair co-axial linearised superposition
  off_axis.py          — OffAxisPair parallel-axis Cartesian superposition + CTC map
  ctc.py               — Generic CTC band detector (sign-change bisection)
  geodesic.py          — CircularOrbit + integrate_geodesic
  time_machine.py      — TimeMachineWindow + harness_time_loop
  dinos_bridge.py      — Optional Dinos-DKN interop (Z₃ Möbius eigenvalue match)

tests/                 — 81 tests across 11 modules
paper/                 — LaTeX whitepaper + figure generator + compiled PDF (14 pp)
examples/              — Reproduction scripts + JSON simulation outputs
tools/                 — One-shot SymPy derivations
```

---

## The Δῖνος bridge

The package optionally interoperates with [Dinos-DKN](https://github.com/Zynerji/dinos-DKN), exposing a striking structural correspondence:

> **Theorem (numerical, exact to machine precision).**
> Sample the supercritical Tipler exterior at `N` nodes per log-period `2π/α`. The fundamental discrete-Laplacian eigenvalue equals exactly the `branch = 0` mode-1 eigenvalue of the Dinos Z₃ Möbius cover at the same `N`. The non-trivial branches `b ∈ {1, 2}` (complex conjugate pair, `2π/3` phase advance per node) sit at strictly lower eigenvalue and are the discrete signature of the off-set sectors of the systrophic pair (`δ = ±2π/3`).

```python
from systrophe import VanStockumInterior
from systrophe.dinos_bridge import z3_branch_match_to_tipler_alpha

vs = VanStockumInterior(omega=1.0, R=1.0)
out = z3_branch_match_to_tipler_alpha(vs, N=24)
# {'tipler_eigenvalue': 0.06814834..., 'z3_eigenvalues': (0.06815, 0.00761, 0.00761),
#  'best_branch_match': 0, 'relative_residual': 0.0}
```

Requires `pip install z3-solver` and Dinos-DKN on `PYTHONPATH`. The test suite skips silently if either is unavailable.

---

## Tests

```bash
pytest                 # 65 tests, ~5 seconds
pytest -v              # verbose
pytest --cov=systrophe # with coverage (requires pytest-cov)
```

Test suite breakdown:

| Module | Tests | Coverage |
|---|---:|---|
| `test_vanstockum.py` | 7 | Minkowski limit, invariants, threshold, exterior rejection |
| `test_sinusoid.py` | 6 | α formula, log-periodicity, fit recovery |
| `test_pair.py` | 7 | Phasor collapse, anti-phase cancellation, principal-range wrap |
| `test_ctc.py` | 6 | Sign-band detection, destructive interference, modulated bands |
| `test_lewis_papapetrou.py` | 12 | Continuity, supercritical numerical-vs-analytic, constraint |
| `test_lp_robust.py` | 8 | Regime dispatch, machine-precision supercritical, F-zero formula |
| `test_off_axis.py` | 7 | Construction guards, reflection symmetries, 2D CTC map |
| `test_dinos_bridge.py` | 6 | Kerr mapping, Z₃ branch correspondence (skipped if z3 absent) |
| `test_offset_sweep.py` | 6 | Convenience constructor, offset-π minimum, type checks |
| `test_geodesic.py` | 6 | Minkowski circular orbit, Ω-bounds, target-time inverse |
| `test_time_machine.py` | 9 | Band detection, target-Δt matching, spacelike rejection |

---

## Whitepaper

[`paper/systrophe_time_travel.pdf`](paper/systrophe_time_travel.pdf) (11 pages, includes 5 figures and 3 tables) covers:

1. Mathematical framework — van Stockum, Lewis–Papapetrou, Ernst, Bonnor cases.
2. Co-rotating cylinder pair — linearised superposition, off-set Tipler sinusoid.
3. Closed timelike curves and time-travel orbits — geodesic structure, coordinate vs proper time.
4. Time-machine harness — single-cylinder and pair-tuned numerical results.
5. Connection to Dinos via Z₃ Möbius eigenvalue correspondence.
6. Implementation, tests, evaluation.
7. Limitations and open questions (chronology protection, asymptotic non-flatness, idealised source).

To regenerate the PDF:
```bash
python paper/generate_figures.py     # produces paper/figures/*.pdf
cd paper && pdflatex systrophe_time_travel.tex && pdflatex systrophe_time_travel.tex
```

---

## Limitations

- **Linearised pair.** Two-cylinder Einstein vacuum has no closed form; both `SystrophePair` (co-axial) and `OffAxisPair` (parallel-axis) treat the second source as a linearised perturbation. Cross-terms `h^(1) · h^(2)` are formally O(G²) and not modelled.
- **Off-axis quantitative limits.** In the off-axis case, the Case III exterior is not asymptotically flat, so both single-cylinder perturbations are simultaneously "large" at most points; the linearised superposition is best read as a qualitative tool for identifying CTC regions rather than as a quantitative orbital framework.
- **Idealised source.** Infinite, rigid, perfectly axisymmetric dust column. No known matter form realises this; Tipler-cylinder time-travel scenarios are theoretical exercises in the structure of GR vacuum solutions.
- **Asymptotic non-flatness.** The Case III exterior oscillates indefinitely; there is no privileged "observer at infinity" against whom coordinate time can be synchronised.
- **No chronology protection.** This is a pre-quantum, classical-GR construction. The chronology-protection conjecture (Hawking 1992) is not addressed.

---

## Citation

```bibtex
@misc{Knopp2026Systrophe,
  author = {Knopp, Christopher},
  title  = {{Systrophē}: A co-rotating Tipler-cylinder pair as a tunable
            time-travel harness},
  year   = {2026},
  note   = {Python implementation of van Stockum interior, Lewis--Papapetrou
            exterior, off-set Tipler sinusoid, and Z\_3 M\"obius cover
            correspondence; 65 passing tests.},
  url    = {https://github.com/Zynerji/systrophe}
}
```

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Contact

`cknopp@gmail.com`

---

## References

- W. J. van Stockum, *The gravitational field of a distribution of particles rotating about an axis of symmetry*, Proc. Roy. Soc. Edin. **57** (1937) 135.
- T. Lewis, *Some special solutions of the equations of axially symmetric gravitational fields*, Proc. Roy. Soc. London A **136** (1932) 176.
- F. J. Tipler, *Rotating cylinders and the possibility of global causality violation*, Phys. Rev. D **9** (1974) 2203.
- W. B. Bonnor, *The exterior gravitational field of a rotating cylinder of dust*, J. Phys. A **13** (1980) 2121.
- S. W. Hawking, *The chronology protection conjecture*, Phys. Rev. D **46** (1992) 603.
