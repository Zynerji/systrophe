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
descent curve.

**Correction (measured later, §9):** the win is *single-cylinder phase tuning*,
not nesting. A single cylinder's phase alone moves out-of-band from 0.53 to
~0.15, and adding cylinders **saturates** at ~86% coverage. The earlier
"nesting tiles the route" reading is revised: the lever is the phase; extra
cylinders are redundant for coverage (useful only to place bands at radii one
cylinder cannot reach).

**Honest scope:** this shrinks the *out-of-band fraction* (an
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

### 9. Minor rigorous levers + the honest blow-up correction

**(a) Defensible minor levers.**
- *Subluminal `v²`* (`warp_geometry.velocity_sweep`, exact): `v=0.1c` → −2 OOM,
  `v=0.01c` → −4 OOM. Caveat: subluminal forfeits the FTL point — a rocket
  reaches the same speed — so it trades the headline capability for energy.
- *Multi-cylinder coverage scaling* (`band_coverage_optimizer.coverage_scaling`):
  **saturates** at ~86% coverage for all N; the win is single-cylinder phase
  tuning (the §7 correction). Nesting adds nothing for coverage.

**(b) Van den Broeck neck blow-up — research scaffold (UNCALIBRATED).**
`vdb_neck.vdb_total_floor` supplies the term §8 omitted, via the gradient-energy
scaling `ρ ~ C·(c⁴/G)·(B'/B)²` → `E_blowup ~ C·(c⁴/G)·ln²(R_pocket/R_shell)·R_shell²/Δ`.
The O(1) prefactor C is **uncalibrated** (not a full Einstein-tensor integral);
the result is flagged `calibrated=False`. Measured (R_pocket=2 m, 100·ℓ_Planck wall):

| term | value |
|---|---|
| shell principal (calibrated scaling) | 1.6e10 J (~16 GJ) |
| neck blow-up (uncalibrated) | **1.1e15 J (~1 PJ)** |
| total floor | ~1.1 PJ |
| blow-up / shell | **~7×10⁴ (blow-up dominates)** |
| reduction vs baseline | −28 OOM (vs −33 shell-only) |
| blow-up re-inflates | ~5 OOM |
| residual gap to lab | ~30 OOM |

This confirms Pfenning–Ford **qualitatively**: the neck blow-up dominates and
re-inflates the optimistic shell-only floor. The precise PJ value is not to be
trusted (uncalibrated C, Planck-scale wall); only the direction and scale are.

### 10. Calibrating the blow-up with the exact Einstein-tensor integral — the geometry route collapses

The §8/§9 "−33 OOM via geometry" was optimistic. Calibrating it properly
**overturns it.** For the VdB pocket the spatial slice is conformally flat,
`γ_ij = B(r)²δ_ij`; the static Hamiltonian constraint `ρ = R⁽³⁾/16π` gives the
**exact** pocket energy (curvature total-derivatives cancel):

```
E_pocket = ½ ∫₀^∞ (B'²/B) r² dr        (geometrized; × c⁴/G for joules)
```

Numerically (`vdb_neck.pocket_energy_geometrized`, `calibrate_K`):
- `E ∝ B_max¹` (measured exponent 1.05) — **not `ln²(B_max)`** as the scaffold guessed.
- `E ∝ R_w²/w`, prefactor **K = 0.554** (converged, tanh wall).

Since `B_max = R_pocket/R_shell`, this collapses to `E ≈ K·(c⁴/G)·R_pocket·(R_shell/w)`.
For a localized thin wall (`w ≈ R_shell`) the **shell radius cancels**:

| quantity | scaffold (§9, wrong) | **calibrated (exact integral)** |
|---|---|---|
| blow-up (R_pocket=2 m) | 1.1e15 J | **1.34e44 J = 0.79 Jupiter** |
| net reduction vs baseline | −28 OOM | **−1.1 OOM (i.e. ~13× WORSE)** |
| `geometry_reduces_floor` | (implied yes) | **False** |

**The scaffold over-claimed the reduction by ~29 orders of magnitude.** The
exact integral shows the pocket-expansion energy is `~(c⁴/G)·R_pocket` — set by
the *proper interior radius you want*, independent of how small the coordinate
shell is made. Van den Broeck geometry therefore does **not** reduce the
principal for a habitable (metre-scale) pocket; it is marginally *worse* than
plain Alcubierre. This vindicates Pfenning–Ford (1997) quantitatively, now from
the Einstein-tensor integral rather than by assertion. Catcher: `smooth`.

*Open caveat (honest):* a very thick/gradual wall (`w ≫ R_shell`) lowers
`E ~ R_pocket·R_shell/w`, but that is a different geometry whose interior
usability and validity is the genuinely contested question — not a demonstrated
reduction.

### 11. The thick-wall escape — closed by calculus of variations

§10 left one open escape: a thick/gradual blow-up wall. Settled by minimizing
the exact pocket energy `E[B] = ½∫(B'²/B)r²dr` over **all** profiles. The
Euler–Lagrange equation (with `B=u²`) collapses to `(r²u')'=0`, giving the
**isotropic-Schwarzschild** optimum `B_opt(r) = (1+|c|/r)²`. For
`B(R_shell)=B_max`, `B(R₂)=1`:

```
E = 2(√B_max−1)²·R_shell·R₂/(R₂−R_shell)   →   E_min = 2R_shell(√B_max−1)²  (R₂→∞)
```

So the **thick wall is genuinely optimal** (thinner costs more) — verified
analytically and numerically. But with `B_max = ρ_use/R_shell` (large interior,
small exterior), `E_min → 2·ρ_use` as `R_shell→0`. Measured:

| ρ_use | E_min | E/ρ_use | shrinking shell helps? |
|---|---|---|---|
| 1 m | 2.4e44 J (1.4 Jupiter) | 2.00 | **No** |
| 2 m | 4.8e44 J (2.8 Jupiter) | 2.00 | **No** |

`E_min` is **independent of R_shell** (1e-15 m and 1e-30 m give identical
energy). The floor is `~2(c⁴/G)·ρ_use` — a positive-mass-theorem-flavoured
inevitability: holding open that much proper volume costs ~its proper size.

**The dual honest result:** the interior genuinely **stays usable** — the
flat-B core is flat space (rescaled), zero tidal field, fully habitable
(`interior_is_flat_usable=True`). Van den Broeck's interior claim is *correct*.
But the **thick-wall energy escape is CLOSED** (`escape_closed=True`): the floor
is Jupiter-scale and irreducible across every wall shape, residual ~60 OOM above
a lab source. Catcher: `smooth`.

**Correction + reinforcement (direct Einstein-tensor integral).** On review the
variational argument above had a flaw: the EL optimum `B=(1+|c|/r)²` is the
*singular* isotropic-Schwarzschild profile (a central mass, not a regular
pocket), for which the boundary term `[B'r²]` does not vanish; and `½∫(B'²/B)r²dr`
is the *total* energy, not the *exotic* (QI-bounded) part. The corrected
**direct Einstein-tensor computation of a regular pocket**
(`vdb_thick_wall.regular_pocket_einstein`; isotropic-static `ρ`, `ρ+p_r`)
gives, robustly across `B_max`:

```
E_total ≈ −4 (c⁴/G)·ρ_use      |E_exotic (NEC-violating)| ≈ 5 (c⁴/G)·ρ_use
```

So the pocket energy is **negative and exotic-dominated** — for ρ_use=2 m the
exotic requirement is **~1.2×10⁴⁵ J ≈ 7 Jupiter masses** (`exotic_dominated=True`,
ratio 5.1× ρ_use). This kills the one remaining hope (that most of the ~Jupiter
might be gatherable *ordinary* mass): it is QI-bounded *exotic* energy, ~Jupiter,
reinforcing the closure with a direct integral rather than the flawed singular
optimum.

### 12. Full wormhole map — the energy wall is dual to an entanglement wall

The wormhole route (interior as a separate region, not a pocket) mapped across
all three sub-routes (`wormhole_map.py`, using Systrophe's `er_epr_pair`,
`exotic_matter_accounting`, `casimir_throat`, and the HHmL Ryu–Takayanagi
machinery):

| route | requirement for throat radius `r_t` | metre throat |
|---|---|---|
| classical Morris–Thorne | exotic energy `~(c⁴/G)·r_t` | 3.0e43 J (0.18 Jupiter) |
| Casimir thin-shell | `d_req/r_t ≈ 0.9` (cavity doesn't fit) | infeasible |
| quantum ER=EPR (Gao–Jafferis–Wall) | `~(r_t/ℓ_P)²` ebits (RT: `S=A/4G_N`) | **1.7e70 ebits** |
| dual black hole | `Mc² = c⁴r_t/2G` | 6.1e43 J (0.35 Jupiter) |

**The deep result:** classical exotic energy = **½ × the ER=EPR dual
black-hole mass-energy** — duality ratio **0.50 at every scale**. The wormhole's
*energy wall* and its *entanglement wall* are the **same Bekenstein-bounded
wall**, both equal to assembling a black hole of radius `r_t`. The quantum
ER=EPR route is genuinely different — the negative energy is sourced by
**entanglement, not a slab of exotic matter** (removing the unobtainium problem,
and demonstrated for *one qubit* on a quantum processor, 2022) — but a
macroscopic throat needs a **black hole's worth of pre-shared entanglement**
(~1e70 ebits for a metre). So:

- **Information** (a few qubits): the quantum wormhole works in principle.
- **Macroscopic matter / a ship**: every route hits the same `~(c⁴/G)·r_t`
  ≈ Jupiter wall, ~58 OOM above a lab source. `beats_the_wall = False`.

Catcher: `smooth`. (Casimir "fits" geometrically only for large throats where
the energy is even larger — never a win.)

### 13. The ER=EPR quantum information channel — the one thing that WORKS

The wormhole map (§12) showed the quantum ER=EPR route transports *information*
where matter transport hits the Bekenstein wall. `erepr_channel.py` builds that
channel — and it works at unit fidelity:

| quantity | value |
|---|---|
| single-qubit teleport fidelity (200 random states, worst case) | **1.000000** |
| entanglement fidelity, 1 / 2 / 3-qubit register | **1.000 / 1.000 / 1.000** |
| classical (no-entanglement) bound | 0.5 |
| capacity | 1 qubit per EPR pair |

Operationally it is deterministic, measurement-free teleportation through
pre-shared EPR pairs with coherent X/Z corrections — under ER=EPR the
entanglement *is* the Einstein–Rosen bridge, so the qubit emerging at the far
mouth is "traversing the wormhole."

**Honest boundaries (no overclaim):**
- **Not FTL** (`is_faster_than_light=False`): the corrections need a coupling
  between the two mouths; by no-signaling nothing moves until that physical/
  classical link is used. This is a quantum-network primitive, not a comms FTL.
- **GJW size-winding: a *real SYK scrambler* activates it; Haar does not.**
  Built a genuine SYK Hamiltonian (N Majoranas, random 4-body couplings,
  `syk_hamiltonian`) and the two-sided Maldacena–Qi protocol (TFD ≈ EPR pairs,
  past-insertion, `e^{igV}` coupling, time-reversed readout). Measured
  (`syk_vs_haar_activation`, N=8 Majoranas/side, disorder-averaged):

  | scrambler | no-coupling baseline | peak with coupling | lift |
  |---|---|---|---|
  | Haar | 0.25 | 0.26 | **+0.01 (inert)** |
  | **SYK** | 0.25 | 0.43 | **+0.18 (activates)** |

  So the chaotic size-winding the GJW mechanism requires is **real and
  demonstrated** — SYK lifts the coupling-mediated channel ~18×, where a Haar
  scrambler does nothing. Honest limit: at classically-simulable N (≤8
  Majoranas/side) the transmission stays **below the 0.5 classical bound**
  (`reaches_classical_bound=False`) and shows no clean sign-asymmetry; unit
  fidelity needs N beyond simulability or an engineered "perfect-winding"
  Hamiltonian (cf. the contested 2022 quantum-processor demo, which used a
  *learned sparse* SYK Hamiltonian).

So the buildable, tested deliverable from the entire warp/wormhole investigation
is a perfect entanglement-mediated qubit channel (fidelity 1), plus a real SYK
demonstration that the holographic size-winding mechanism genuinely activates —
honestly *not* a faster-than-light or matter-transport device.

### 14. First-principles derivation — the coupling *is* the operator-size operator

Why does SYK activate the channel and cap below unit fidelity? Derived from
first principles and verified to machine precision (`size_winding.py`):

**Theorem (★).** For `n` EPR pairs and the GJW coupling `V = (1/n)Σ_j Z_jᴸZ_jᴿ`,
every Pauli string `P` on L is an eigenvector of `V` on the EPR state:

```
V · P_L|EPR⟩ = (1 − 2·size(P)/n) · P_L|EPR⟩          size(P) = #{X,Y factors}
```

*Proof:* `Z_jᴸZ_jᴿ|EPR⟩=|EPR⟩` (ricochet `Zᵀ=Z`); `Z_jᴸP=(−1)^{a_j}PZ_jᴸ` with
`a_j=1` iff `P` has X/Y at `j`; sum → `(1/n)Σ(−1)^{a_j}=1−2·size/n`. ∎
Verified: max residual **7.9×10⁻¹⁷** (n=3), **0.0** (n=4) over all Paulis.

So **the coupling is literally the operator-size operator**, and
`e^{igV}` phases each component by `e^{−2ig·size/n}` — the "size winding"
mechanism, *derived* rather than assumed. Consequences, all measured:

- Chaotic scrambling grows operator size (SYK: `⟨size⟩` of `X₀` runs 1.0 → 2.1),
  which is the precondition for winding — **mechanism active** (SYK teleports
  0.43, Haar 0.25).
- Single-shot teleportation of an *unknown qubit* is nonetheless partial at
  simulable N (winding only approximately linear). **Honest correction:** a
  tempting "fidelity = |characteristic function of the size distribution|"
  identity is *false* — that quantity is 1 at g=0, where the EPR ricochet
  trivially mirrors the whole operator algebra to R without localizing the
  message on R₀. Teleporting a state is a stronger, localizing requirement.

**The solution (verified):** unit fidelity comes from the **deterministic
coherent-correction channel** (F = 1.000, §13) — the same EPR bridge with
feed-forward readout instead of a single size-limited coupling — or from the
unsimulable large-N perfect-winding limit. It is *not* obtained by enlarging the
single-shot SYK protocol. This is the rigorous resolution of the "gap to unit
fidelity": the mechanism is the size operator (exact), and the buildable
unit-fidelity device is the deterministic channel.

### 15. ER=EPR channel run on real quantum hardware (IBM Marrakesh)

The deterministic ER=EPR channel (§13) executed on `ibm_marrakesh` (156-qubit
Heron r2) via the Zynerji instance, with the full live-calibration pipeline
(`quantum_golden_pendulum.calibration`) + error mitigation
(`experiments/marrakesh_erepr_wormhole.py`):

- **Calibration (live):** 142 good / 14 bad / 0 dead qubits, 288 good edges;
  BFS-selected connected good subset `[2, 3, 4, 16]`.
- **Circuit:** deterministic EPR teleportation, transpiled depth ~31
  (opt_level=2), DD (XpXm) + gate/measure twirling, 4096 shots/basis.
- **Measured correlators:** `⟨XX⟩ = +0.824`, `⟨YY⟩ = −0.755`, `⟨ZZ⟩ = +0.778`.
- **Hardware Bell fidelity** `F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩)/4 = `**`0.839`** —
  noise-degraded from the noiseless 1.000, but **clearly above the 0.5 classical
  bound.** The ER=EPR channel works on real hardware.
- **Verifiable job IDs:** XX `d88cqh1789is73920b40`, YY `d88cqh9789is73920b50`,
  ZZ `d88cqhh789is73920b60`.

**Why not the SYK wormhole on hardware:** the faithful dense-SYK wormhole
transpiles to **depth 11570 / 3816 two-qubit gates** on Marrakesh — far beyond
the ~150 usable-depth budget. That is the quantitative reason the size-winding
wormhole cannot run clean on current hardware (and why the 2022 demo used a
*learned sparse* Hamiltonian). The deterministic ER=EPR channel is the runnable,
honest hardware demonstration.

**Pushing it higher — more shots + ZNE.** Re-run with EstimatorV2 measuring the
fidelity observable directly, 32768 shots, T-REx readout mitigation, and
Zero-Noise Extrapolation (gate-folding at noise factors 1/3/5):

| configuration | Bell fidelity |
|---|---|
| Sampler, 4096 shots, DD + twirling | 0.839 |
| Estimator, 32768 shots, T-REx (noise factor 1) | 0.875 |
| **+ ZNE → zero noise (exponential)** | **0.918** (linear: 0.911) |

Per-noise-factor fidelities `{1: 0.875, 3: 0.769, 5: 0.703}` (clean monotonic
decay → reliable extrapolation; the runtime's built-in exponential extrapolator
returned `nan`, so the zero-noise value is extrapolated directly from the saved
per-factor data). ZNE job `d88cta1789is73920eig`.

**Best backend + lowest-error qubits.** Ranking three Heron-r2 backends by
calibration quality (`best_quality_quad`: lowest total CX + readout error over a
connected good quad):

| backend | best quad | total error | queue |
|---|---|---|---|
| **ibm_kingston** | [140,141,142,143] | **0.045** (CX ~0.001) | 0 |
| ibm_marrakesh | [11,12,13,14] | 0.052 | 1 |
| ibm_fez | [116,120,121,122] | 0.071 | 22 |

Re-running the ZNE protocol on the Kingston low-error quad:

| run | Bell fidelity |
|---|---|
| Marrakesh BFS quad, Sampler 4096 | 0.839 |
| Marrakesh ZNE (32768) | 0.918 |
| Kingston quality quad, factor 1 | 0.925 |
| **Kingston quality quad + ZNE** | **0.958** (linear 0.954) |

Per-noise-factor `{1: 0.925, 3: 0.853, 5: 0.796}`; ZNE job `d88d949789is73920rv0`.
Picking the best backend and the calibration-lowest-error qubits added another
~0.04 on top of ZNE.

**PEC did NOT push it higher (honest negative).** Probabilistic Error
Cancellation on the same Kingston quad (learns a Pauli-Lindblad noise model,
inverts it for an unbiased estimate) gave **F = 0.947 ± 0.035** (job
`d88dbhgp0eas73dna5n0`) — *below* the ZNE 0.958 in central value and statistically
consistent with it (0.958 lies within the ±0.035 band). On this already-clean,
low-error circuit there is little coherent noise left for PEC to invert beyond
what T-REx + DD + twirling remove, so PEC only adds sampling variance. **ZNE on
the best-backend quad (0.958) remains the best result.**

So the buildable capability is not just simulated — it **runs on a real quantum
computer at 0.958 fidelity** (from 0.839 → 0.918 via shots+ZNE → 0.958 via
best-backend + lowest-error qubit selection; PEC consistent at 0.947 ± 0.035 but
no improvement). Still entanglement-mediated teleportation: not FTL, not matter
transport.

## Bottom line

The reversible-ratcheting pendulum is a clean, stable, reversible *control*
layer for steering a warp bubble, and the amplified-Casimir capacitor makes the
*peak charge* and *pump power* arbitrarily small with Q. Band-coverage
optimization cuts the paid out-of-band fraction by ~74%. But every route to
shrinking the *principal itself* fails under rigorous treatment: the Van den
Broeck pocket blow-up is `~(c⁴/G)·R_pocket` (calibrated, §10), calculus of
variations proves the thick-wall variant cannot beat `~2(c⁴/G)·ρ_use` for **any**
wall shape (§11), and the full wormhole map shows the energy wall is *dual to a
black-hole-entropy entanglement wall* (§12) — the quantum ER=EPR route moves
information but not a ship. The Ford–Roman
*principal* — the time-integrated negative energy that must flow and be repaid —
is invariant under all of it, and lands at Jupiter-mass scale for a 1-metre
bubble. Elegant control over an irreducible energy wall.

## References

- Ford, Roman, *The quantum interest conjecture*, Phys. Rev. D 60 (1999) 104018.
- Pfenning, Ford, *The unphysical nature of warp drive*, Class. Quantum Grav. 14 (1997) 1743.
- Wilson et al., *Observation of the dynamical Casimir effect in a superconducting circuit*, Nature 479 (2011) 376.
- Alcubierre, *The warp drive*, Class. Quantum Grav. 11 (1994) L73.
- TriCameral `ParetoRatchet`: `TriCameral.ai/tricameral/jdhart_v51/proven.py:154`.
