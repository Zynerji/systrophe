# Lorenz-class rotation for a Tipler cylinder

Can a Tipler cylinder's rotation be made "complex, like a Lorenz attractor"? This folder
answers it from first principles. Full write-up: [`../../FINDINGS_LORENZ_ROTATION.md`](../../FINDINGS_LORENZ_ROTATION.md).

**Short answer:** Not in the *geodesic* sector (Hamiltonian ⇒ Liouville ⇒ no attractor),
but **yes** in the *matter* sector — the dust column treated as a differentially-rotating
dissipative fluid reduces to the Lorenz system, reproduces the canonical strange attractor
to ≤1%, and drives a chaotically flickering CTC band structure.

## Files

| file | role |
|---|---|
| `rotating_dust_lorenz.py` | **H3** — Saltzman–Lorenz reduction of a dissipative rotating dust column; Lyapunov spectrum (QR), Kaplan–Yorke dim; `a(t)` mapping. |
| `geodesic_rotation_chaos.py` | **H0/H2** — geodesics under rigid vs. time-dependent rotation; FTLE and phase-space divergence (the Liouville check). |
| `chaotic_ctc.py` | feeds `a(t)` into the Bonnor Case III CTC harness → flickering time-machine bands. |
| `run_experiment.py` | orchestrator; writes `lorenz_rotation_results.json` + mandatory novelty-catcher verdict. |
| `test_lorenz_rotation.py` | 11 tests (run from this directory). |

## Run

```bash
python run_experiment.py                       # ~140 s
python -m pytest test_lorenz_rotation.py -q     # ~25 s, 11 passed
```

The core rigid-rotation modules in `src/systrophe/` are **not modified**; this is an
additive dynamical-rotation layer.
