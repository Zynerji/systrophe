# reservoir-catcher

**An honest physical-reservoir-computing test bench.** The falsifiable core of
the question "can a self-organizing nonlinear medium integrate information /
compute?" — built so the answer is allowed to come back null.

It decides two questions with matched reservoirs (identical size, spectral
radius, leak, input scaling, ridge readout, seed, and task data — only the
variable under test changes):

- **Q1 — does *nonlinearity* compute?** linear vs nonlinear (tanh) medium.
- **Q2 — does helical/Möbius *structure* beat plain-random connectivity?**

Tasks: **NARMA-10** (nonlinear system ID), **parity-3 / parity-5** (temporal
XOR — linearly inseparable, the decisive nonlinearity probe), and **linear
memory capacity** (the memory↔nonlinearity tradeoff). An independent readout
reuses the canonical Systrophe address-space catcher
(`systrophe.catchers.novelty_catcher.lambda_2_of_hamming_graph`): λ₂ of the
Hamming graph over visited reservoir states — an emergence signal that does not
look at task error.

## Run

```bash
python run_ab.py            # full 5-seed run
python run_ab.py --fast     # quick smoke run
pytest tests/               # pins both findings + mechanics
```

## Result (see FINDINGS.md)

| arm | NARMA10 NRMSE ↓ | parity-3 | parity-5 | lin MC | addr λ₂ |
|---|---|---|---|---|---|
| linear-random | 0.529 | 0.498 (chance) | 0.495 (chance) | 9.28 | 0.34 |
| **nonlinear-random** | **0.500** | **0.996 (solved)** | **0.739** | 7.28 | 0.24 |
| nonlinear-helical | 0.779 | 0.561 | 0.502 | 0.96 | 0.01 |
| nonlinear-mobius | 0.593 | 0.734 | 0.504 | 3.69 | 0.01 |

**Q1: nonlinearity computes** (parity solved only with tanh).
**Q2: designed helical/Möbius topology does *not* help — it hurts**, because
smooth `cos(ω·Δθ)` matrices are near-low-rank and collapse the medium's
dynamical richness (helical memory capacity craters to 0.96 vs 7.28). The
catcher agrees: random media explore a richer state space.

This is the same verdict the rest of the Systrophe/local corpus keeps
returning (HeliSpec-X, Kanon, Toron, the helical-Fiedler A/B): **the topology
is decoration; the nonlinearity is the engine.** Bounded honestly — this
falsifies *these specific* helical/Möbius constructions, not all structure. A
rank-preserving structured medium could in principle help, and this bench is
exactly where to test one.
