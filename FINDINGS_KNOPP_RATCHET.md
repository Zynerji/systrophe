# Knopp-Drive reversible-ratcheting pendulum — directional bias controller

A speculative warp-engineering controller that gives a *direction-blind* Knopp
Drive a stable, tunable, reversible forward bias by porting the proven
TriCameral `ParetoRatchet` (rising-floor pawl) onto the horn-toroidal steering
control, supplies the negative energy as amplified (dynamical) Casimir vacuum
rather than an exotic-matter reservoir, and books the whole journey against the
Ford–Roman quantum-interest bound and an SI feasibility floor.

**Status: speculative controller, self-consistent, NOT a validated result.**
Address-space novelty catcher returns `smooth` throughout (continuously
tunable bias, no emergent transition). All numbers below are measured from the
code, not asserted.

## Deliverables

| Artifact | What |
|---|---|
| `src/systrophe/knopp_ratchet.py` | `WarpParetoRatchet`, `ratchet_traversal`, `reverse_asymmetry`, `casimir_pump_accounting`, `capacitor_accounting`, `bias_energy_ledger`, `ledger_Q_sweep`, `feasibility_report`, `novelty_scan` |
| `src/systrophe/knopp_drive.py` | `KnoppDrive.bias / asymmetry / pump_accounting / capacitor_accounting / energy_ledger / feasibility_report` |
| `tests/test_knopp_ratchet.py` | 26 tests, all passing |
| `examples/knopp_ratchet_walkthrough.py` | end-to-end walkthrough + JSON results |

## The chain of results

### 1. The bare drive is direction-blind

`composite_E_neg` (`knopp_drive.py:140`) depends on `epsilon_horn` and
`r_orbit` but **not** on the horn-twist axis `theta_0_horn`: only the steering
*vector* rotates, the *cost* does not.

```
E_advance(heading = 0)  = 5.07836e-03
E_advance(heading = pi) = 5.07836e-03   (identical)
```

A symmetric pendulum that swings the axis nets zero displacement — the
warp-bubble analogue of Purcell's scallop theorem. The drive has no built-in
preferred direction; any bias must be *manufactured*.

### 2. The rising-floor pawl manufactures a tunable asymmetry

Port of `TriCameral.ai/tricameral/jdhart_v51/proven.py:154` — the floor only
ever moves up, so forward progress locks in and regression rolls back. The
asymmetry comes from (a) forfeiting the free Tipler-band recovery on the
reverse stroke, and (b) the pawl stiffness `floor_pct/(1-floor_pct)`.

| `floor_pct` | E_advance | E_reverse | reverse / forward |
|---|---|---|---|
| 0.00 (no pawl) | 5.08e-3 | 8.41e-3 | **1.66×** |
| 0.50 | 5.08e-3 | 1.35e-2 | 2.66× |
| 0.85 (proven default) | 5.08e-3 | 3.72e-2 | **7.32×** |
| 0.95 | 5.08e-3 | 1.05e-1 | 20.66× |
| 0.99 | 5.08e-3 | 5.11e-1 | 100.66× |

`floor_pct = 0` → reciprocal pendulum (scallop theorem, ratio from the gate
asymmetry alone). `floor_pct → 1` → near-perfect one-way valve. Reversible:
re-anchor with `heading + pi` to lock the opposite direction.

### 3. Biased traversal: net directed transport, stable

64 cycles, heading 0: **net displacement +14.36, exotic budget 0.325, PF-OK
every stroke, 0 rollbacks.** Ramping the pendulum past the horn pinch
(`eps → 1.2`) provokes rollbacks that snap the bubble back to its last good
locked state — closed-loop stability, not open-loop runaway.

### 4. The negative energy is amplified Casimir vacuum, not an exotic reservoir

Two distinct "no exotic matter" claims, kept separate:

- **Geometric (Claim A):** the LP/Knopp throat is vacuum (`R_μν = 0`), sourced
  remotely by frame-dragging — needs no *local* exotic matter
  (`exotic_matter_accounting.py:9-21`). Not a Casimir argument.
- **Casimir (Claim B):** where the *shell* wants negative energy, it is the
  parametrically squeezed Casimir vacuum — degenerate parametric amplification
  at `2·f_0`, i.e. the dynamical Casimir effect (`dynamical_casimir.py`; Wilson
  et al. 2011 observed it in a superconducting circuit).

**What we did NOT find:** static parallel-plate Casimir is far too weak
(`exotic_matter_accounting.py:50-56` — needs sub-Planckian plate separation).
Only the *amplified/dynamical* route is viable, and it **respects** the
Pfenning–Ford / Ford–Roman quantum inequality — trading amplitude (`1/Q`) for
duration (`Q`) so the bounded product is invariant. Measured:

```
Q=8     P_drive=1.06e-4   PF_ok   E*tau/bound=7207.5x
Q=1000  P_drive=6.77e-9   PF_ok   E*tau/bound=7207.5x   (same ratio = the trade)
```

So: no reservoir of exotic *matter* — a real, positive-energy *pump*
(`P_drive ∝ 1/Q²`) sustaining a squeezed vacuum, within the QI.

### 4b. The shell is a surface "capacitor": peak charge vs total throughput

The high-Q standing wave is an energy accumulator. It rings up
`E_shell(t) = E_shell(0)·exp(g·t)` and at saturation holds only
`E_peak = |E_neg_bare|/Q` over the cavity lifetime `τ = Q/f_0`.

```
Q=1      peak/bare = 1.00     throughput = 10.90
Q=1e6    peak/bare = 1.0e-6   throughput = 10.90
Q=1e12   peak/bare = 1.0e-12  throughput = 10.90
```

**The capacitor reduces the instantaneous negative-energy charge on the
surface by `1/Q`** — you never marshal the whole requirement at once; you
trickle-charge a small standing charge and let it act over `τ`. It also drops
the pump *power* by `1/Q²`. **What it cannot do:** lower the time-integrated
throughput (the Ford–Roman principal), which is Q-independent (`10.90` above).
The capacitor solves PEAK and POWER, never TOTAL.

### 5. Bias energy ledger with Ford–Roman quantum interest

`bias_energy_ledger` integrates the journey and weighs it against the Ford–
Roman 1999 quantum-interest conjecture (negative energy borrowed for `τ_hold`
must be repaid by a larger positive pulse). 64 cycles, Q=10, α=1:

```
E_pump   (real input)       = 1.090
E_borrow (neg held)         = 1.090
interest rate r             = 9.00         (r = Q^alpha - 1)
E_repay  (principal+int)    = 10.90
-------------------------------------
E_ledger TOTAL              = 11.99
irreducible floor (Q->inf)  = 10.90
reverse-undo energy         = 87.8         (~7.3x forward — the ratchet)
```

**At α = 1 the per-stroke debt is conserved:** `E_repay == floor`,
Q-independent. Amplification only buys down the *pump overhead*, never the
quantum-interest *principal*:

```
Q=1     +100.0% over floor
Q=10    + 10.0%
Q=100   +  1.0%
Q=1000  +  0.1%
```

For superlinear interest (α > 1) the cost-per-displacement Q-sweep has an
interior optimum at `Q* = (1/(α-1))^(1/α)` (α=1.1 → Q*≈8), beyond which the
interest premium outweighs the pump savings.

### 6. SI feasibility — the irreducible principal is Jupiter-mass scale

`feasibility_report` converts the floor to joules via the textbook geometrized
estimate `E_geom = κ·(v/c)²·R²/σ` (κ≈1/12) × `c⁴/G = 1.21×10⁴⁴ J/m`. It
deliberately does **not** route `alcubierre_total_negative_energy` through
`c⁴/G`: that integrator scales as `R²σ` (a relative comparator), not as
geometrized energy.

| bubble R | wall σ | floor [J] | Jupiter mass-energies | shortfall vs lab |
|---|---|---|---|---|
| 1 m | 1 m | 6.46e44 | 3.8× | 10^60 |
| 10 m | 1 m | 6.46e46 | 380× | 10^62 |
| 100 m | 1 mm | 6.46e51 | 3.8e7× | 10^67 |

**A 1-metre luminal warp bubble's irreducible negative-energy principal ≈ 3.8
Jupiter mass-energies, ~60 orders of magnitude beyond a lab squeezed-vacuum
budget.** Scaling confirms `E ∝ R²/σ`. The capacitor reduces the *peak charge*
held at once and the pump *power*, but this *total* is the same wall every warp
metric hits.

## Honest caveats / domain limits

- "Energy" = exotic-matter budget / pump energy; "displacement" = the gated
  steering-dipole impulse. Both are framework heuristics, not first-principles
  GR stress-energy integrals.
- Whether the horn dipole produces genuine center-of-mass translation (vs a
  wobble) in full GR is the real open question; the geometric-phase / gauge-
  kinematics argument is *why it could*, not proof that it does.
- The quantum-interest law `r = Q^α − 1` is a **documented parametrization** of
  Ford–Roman 1999 (which fixes existence + monotonicity of the premium, not
  this closed form). α is the dial; α=1 is the conservative conserved-debt case.
- `κ` (Alcubierre coefficient) and `lab_source_J` (default 1e-15 J) are
  documented, overridable inputs; the `8π` from `G_μν = 8πG/c⁴ T_μν` is dropped
  as O(1). The ~60-orders conclusion is robust to all of these.
- Catcher verdict `smooth` everywhere: continuously tunable control, no
  emergent phase transition. Per the standing project rule, no result here is
  called "validated".

### 7. Band-coverage optimization (relocating the paid fraction)

The principal is paid only on the **out-of-band fraction** of a route — where
the worldline lies outside the source's CTC bands. Nesting concentric
supercritical cylinders tiles more of the route with bands, but the bands
*interfere* (the uniform-phase comb extinguishes them entirely), so the phase
offsets must be optimized. `band_coverage_optimizer.optimize_offsets` does this
with an amplitude-modulation cull (method ported from the ResonantQ optimizer:
reweight candidates by `exp(-α·E/E_max)`, `E` = out-of-band fraction; elite +
shrinking perturbation per pass). Measured on route `r ∈ [1.05, 60]`:

| configuration | out-of-band (paid) fraction |
|---|---|
| single cylinder | 0.534 |
| naive aligned nesting (R = 1,3,9,27) | 0.329 |
| **optimized phase offsets** | **0.139** |
| uniform-phase comb (extinction control) | 0.858 |

**74% reduction vs a single cylinder, 58% vs naive nesting** — the paid portion
of the trip drops to ~1/7 of the route. Catcher: `novel_structure` on the
descent curve. **Honest scope:** this shrinks the *out-of-band fraction* (an
engineering-reducible term that relocates cost toward reusable
nested-cylinder/singularity infrastructure). It does **not** touch the
Ford–Roman principal per metre of band traversed — that remains the same
conservation-bounded wall. No optimizer beats a quantum inequality.

### 8. Geometry — shrinking the principal itself (Van den Broeck scaling)

Everything above *relocates* the principal; the one lever that shrinks it is
geometry, because `E_principal ∝ R_shell²/σ`. Van den Broeck (1999) decouples
the energy-driving shell radius from the passenger pocket via a topological
neck, so the dragged wall can be microscopic while the crew volume stays
macroscopic. `warp_geometry.optimize_geometry` minimizes the shell principal
over QI-allowed `(R_shell, σ)`:

| quantity | value |
|---|---|
| baseline principal (R=σ=1 m) | 1.01e43 J |
| optimized (R_shell=σ=100·ℓ_Planck=1.6e-33 m) | **1.63e10 J (~16 GJ)** |
| reduction | **−33 orders of magnitude** |
| × band-coverage paid fraction (0.139) | 2.3e9 J (~2.3 GJ) paid |
| residual gap to a lab squeezed-vacuum source (1e-15 J) | **still ~24 OOM** |

Catcher: `smooth` (power-law, no emergent transition).

**Two load-bearing caveats (why this is NOT feasibility):**
- `wall_is_planck_scale = True`: the optimum sits at a ~Planck-scale wall where
  the semiclassical / QFTCS treatment is no longer trustworthy.
- `blowup_energy_modeled = False`: this is the Alcubierre *shell* energy only.
  Van den Broeck's neck blow-up region carries additional negative energy that
  Pfenning–Ford (1997) showed re-inflates the total in a model-dependent,
  contested way. The 16 GJ is an **optimistic lower bound**, not a validated
  total.

So geometry closes ~33 of the ~60-order gap on paper — a real, scaling-grounded
reduction — but lands at theory-breaking scales, omits the blow-up term, and
still leaves ~24 orders of magnitude. The principal-per-metre remains
conservation-bounded; geometry only tunes the `R_shell²/σ` prefactor within the
QI-allowed region.

## Bottom line

The reversible-ratcheting pendulum is a clean, stable, reversible *control*
layer for steering a warp bubble, and the amplified-Casimir capacitor makes the
*peak charge* and *pump power* arbitrarily small with Q. Band-coverage
optimization cuts the paid out-of-band fraction by ~74%, and Van den Broeck
geometry shrinks the principal itself by ~33 orders of magnitude (optimistically,
at Planck-scale walls, blow-up term omitted) — yet ~24 orders still remain. But the Ford–Roman
*principal* — the time-integrated negative energy that must flow and be repaid —
is invariant under all of it, and lands at Jupiter-mass scale for a 1-metre
bubble. Elegant control over an irreducible energy wall.

## References

- Ford, Roman, *The quantum interest conjecture*, Phys. Rev. D 60 (1999) 104018.
- Pfenning, Ford, *The unphysical nature of warp drive*, Class. Quantum Grav. 14 (1997) 1743.
- Wilson et al., *Observation of the dynamical Casimir effect in a superconducting circuit*, Nature 479 (2011) 376.
- Alcubierre, *The warp drive*, Class. Quantum Grav. 11 (1994) L73.
- TriCameral `ParetoRatchet`: `TriCameral.ai/tricameral/jdhart_v51/proven.py:154`.
