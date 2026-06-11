# FINDINGS — reservoir-catcher

**Date:** 2026-06-11
**Status:** validated (5 tests green; 5-seed A/B reproducible via `run_ab.py`)

## Question

The only falsifiable core of the "sentient plasmoid AI" question is:
*can a self-organizing nonlinear medium integrate information and compute, and
does a special (helical/Möbius) geometry confer special computational power?*
"Sentience" has no test or construction on any substrate and is out of scope;
we test the buildable part — physical reservoir computing.

## Method

Leaky echo-state reservoir (standard PRC model). Four arms, everything matched
except the variable under test (size n=300, spectral radius 0.95, leak 0.3,
input scaling 0.5, ridge reg 1e-6, 5 seeds, identical task data per seed):

- `linear-random` (f=identity, random W) — Q1 baseline
- `nonlinear-random` (f=tanh, random W) — Q1 treatment / Q2 baseline
- `nonlinear-helical` (f=tanh, `cos(ω·Δθ)` ring) — Q2
- `nonlinear-mobius` (f=tanh, helical × antiperiodic twist) — Q2

Tasks: NARMA-10 (NRMSE), parity-3/5 (accuracy; temporal XOR is linearly
inseparable), linear memory capacity. Independent readout: address-space λ₂
via `systrophe.catchers.novelty_catcher`.

## Results (5 seeds, mean ± std)

| arm | NARMA10 NRMSE | parity-3 | parity-5 | lin MC | addr λ₂ |
|---|---|---|---|---|---|
| linear-random | 0.529 ± 0.021 | 0.498 ± 0.009 | 0.495 ± 0.005 | 9.28 | 0.337 |
| nonlinear-random | 0.500 ± 0.018 | 0.996 ± 0.003 | 0.739 ± 0.031 | 7.28 | 0.238 |
| nonlinear-helical | 0.779 ± 0.032 | 0.561 ± 0.039 | 0.502 ± 0.007 | 0.96 | 0.008 |
| nonlinear-mobius | 0.593 ± 0.021 | 0.734 ± 0.017 | 0.504 ± 0.010 | 3.69 | 0.013 |

## Verdicts

**Q1 — Nonlinearity is what computes. CONFIRMED.** A linear medium is pinned at
chance on parity (input→state→readout stays linear, and XOR is not linearly
separable). Adding tanh solves parity-3 (0.996). Positive reproduction of
Dianoia's null at the substrate level.

**Q2 — Designed helical/Möbius topology does not help; it hurts. CONFIRMED.**
Plain-random connectivity beats both designed topologies on every task.
Mechanism (honest and explainable): smooth `cos(ω·Δθ)` matrices are near
low-rank, so they collapse reservoir richness — helical memory capacity falls
to 0.96 vs 7.28 for random. The independent λ₂ catcher concurs: random media
reach a richer, better-connected state space.

## Boundary of the claim

This falsifies *these specific* helical/Möbius constructions, not all
structure. The failure mode is **rank collapse**, which points directly at the
fix: a *rank-preserving* structured medium (e.g. incommensurate /
quasicrystalline frequencies — the one non-numerological idea surfaced in the
local-repo scan, `quasicrystal-llm`'s `Q(√5)⊥Q(√2)` independence — or an
orthogonal reservoir). Whether such a medium beats random is the open question,
and this bench is where to settle it.

## Negative-space note

Consistent with the wider corpus pattern — HeliSpec-X (shuffled-phase wins),
Kanon/Overtone (SVD dominates), Toron (dual-branch falsified), the
helical-Fiedler warm-start A/B (topology term inert): across many independent
tests, helical/Möbius decoration is inert or harmful once matched against a
fair control. The nonlinearity, not the geometry, does the work.
