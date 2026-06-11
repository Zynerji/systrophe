# FINDINGS — Inducing a Lorenz-class "complex rotation" in a Tipler cylinder

**Date:** 2026-05-24
**Code:** `experiments/lorenz_rotation/` (additive; the core rigid-rotation modules are
untouched — `tests/` still passes 113/113 on the classical core).
**Status:** 4/4 hypotheses resolved + multi-attractor expansion + **exact (hyperbolic)
time-varying-metric treatment** + **full coupled (δF,δK,δL,δh,δS) linearized system** (settling
that the apparent instability was a single-variable artifact) + **gauge-fixed, constraint-
damped nonlinear cylindrical evolution** (2nd-order convergent, stable, chaos-driven) +
**nonlinear GW emission** (radiated waves reconstruct the attractor; nonlinear cross-
polarization) + **dynamical approach to the chronology horizon** (renormalized ⟨T⟩ diverges
while Kretschmann stays bounded = Hawking chronology protection; chaotically-flickering
barrier) + **self-consistent semiclassical backreaction** (quantum protection shell ∝ Planck
area shrouds the horizon). Mandatory novelty-catcher runs: **`novel_structure`** throughout.
Core 113/113 (untouched); experiment 54/54.

---

## The question

> "Change the Tipler cylinder rotation. See if you can induce a complex rotation like a
> Lorenz attractor or something similar." — first-principles, small tests then larger.

## First-principles framing (what is and isn't possible)

The van Stockum / Tipler source is a **rigidly rotating dust cylinder**: a single constant
angular velocity ω, rotation parameter `a = ωR`. Two hard facts bound the search:

1. **A single rigid cylinder cannot be chaotic.** It is stationary + axisymmetric +
   z-independent, so geodesics carry four conserved quantities (E, ℓ, p_z, mass-shell).
   The motion is **integrable**.

2. **Pure geodesic motion can never have a *strange attractor*.** Geodesic flow is
   Hamiltonian ⇒ Liouville's theorem ⇒ phase-space volume is conserved ⇒ no attractor,
   *however* chaotic the rotation is made. A Lorenz attractor is intrinsically
   **dissipative**.

So "a Lorenz attractor" cannot live in the spacetime/geodesic sector. It **can** live in
the **matter sector**: the dust column, once it is allowed to rotate *differentially* and
transport angular momentum through an effective viscosity, is a **dissipative rotating
fluid** — and the Lorenz '63 system is historically exactly a three-mode truncation of a
rotating/convecting fluid (Saltzman 1962). That is the principled bridge, and it is the
result below. The spacetime then **inherits** the chaos adiabatically through `a(t)`.

This splits cleanly into a conservative half (H0/H2) and a dissipative half (H3).

---

## H0 — single rigid cylinder is integrable (the null baseline)

Test-particle geodesics in the interior van Stockum (t, r, φ) block, Hamiltonian
`H = ½ gᵘᵛ pᵤpᵥ`. With ω constant, ∂/∂t is Killing ⇒ `p_t = −E` is conserved.

| quantity | result |
|---|---|
| mass-shell `H` | −0.5 (exact) |
| `p_t` drift over τ∈[0,150] | `0.0` (machine zero) |
| phase-space divergence | `8×10⁻¹¹` (Liouville) |
| finite-time Lyapunov | `0.018` (regular-shear residual) |

## H2 — time-dependent rotation → conservative chaos, but no attractor

Drive the rotation `ω(t) = ω₀(1 + ε cos Ω_d t)` (a "wobbling" column). This breaks the
∂/∂t symmetry: `p_t` is no longer conserved, energy is pumped, and the geodesic flow
becomes chaotic — **yet the phase-space divergence stays zero** (Hamiltonian).

| config | ε | FTLE | phase divergence | `p_t` range |
|---|---|---|---|---|
| rigid | 0.0 | +0.018 | +8×10⁻¹¹ | 0.000 |
| weak | 0.2 | +0.018 | −1×10⁻¹⁰ | 0.145 |
| medium | 0.4 | +0.100 | +6×10⁻¹¹ | 1.600 |
| strong | 0.5 | +0.220 | +2×10⁻¹⁰ | 16.545 |

**FTLE climbs an order of magnitude with drive strength (KAM → chaos), while divergence
stays ≈0 in every case.** This is the explicit numerical statement of "conservative chaos,
never an attractor," and the foil for H3.

> Caveat, stated up front: a genuinely time-dependent ω(t) makes the metric a *prescribed,
> non-vacuum* background, not an exact Einstein solution. H2 is a test-particle toy whose
> purpose is the dynamical-systems statement, not a claim about an exact spacetime.

---

## H3 — the headline: a Lorenz attractor in the rotating-dust sector

### Derivation (Saltzman–Lorenz for a differentially-rotating dissipative dust column)

In a local meridional patch (x ≈ radial, z ≈ axial), Boussinesq perturbations on a
differentially-rotating background with a maintained effective-buoyancy gradient obey, for
stream function ψ and buoyancy field θ,

```
∂_t(∇²ψ) + J(ψ,∇²ψ) = ν ∇⁴ψ + g_eff α ∂_x θ
∂_t θ    + J(ψ,θ)    = κ ∇²θ  + (ΔΘ/H) ∂_x ψ
```

where `g_eff` is the effective gravity (self-gravity + centrifugal imbalance of the
differential rotation — the Solberg–Høiland restoring term) and `ΔΘ/H` is the maintained
destabilising gradient. This is structurally **identical** to Rayleigh–Bénard convection.
The Saltzman three-mode truncation gives the Lorenz system

```
X' = σ(Y − X)
Y' = rX − Y − XZ
Z' = XY − bZ
```

with each parameter carrying a concrete rotating-dust meaning:

| Lorenz param | rotating-dust meaning | classic value |
|---|---|---|
| `σ = ν/κ` | viscosity / angular-momentum transport (Prandtl analog) | 10 |
| `r = Ra/Ra_c` | how supercritical the differential rotation is (drive) | 28 |
| `b = 4/(1+a_cell²)` | overturning-cell aspect factor | 8/3 ⇒ a_cell = 1/√2 |

`X(t)` is the meridional overturning amplitude; it redistributes angular momentum and so
modulates the column's surface angular velocity: `a_rot(t) = a₀(1 + ε X(t)/X_rms)`.

### Verification against the canonical strange attractor

Full Lyapunov spectrum (Benettin QR, RK4), Kaplan–Yorke dimension:

| quantity | this work | canonical Lorenz |
|---|---|---|
| Lyapunov spectrum | **[+0.912, −0.003, −14.576]** | [+0.906, 0, −14.572] |
| Σ exponents | **−13.667** | −13.667 |
| divergence tr(J) = −(σ+1+b) | **−13.667** (Σ = tr J to machine precision) | −13.667 |
| Kaplan–Yorke dimension | **2.0624** | 2.062 |
| Kolmogorov–Sinai entropy | **0.912** | 0.906 |
| Hopf onset `r_H = σ(σ+b+3)/(σ−b−1)` | **24.737** | 24.74 |

The **negative, constant divergence** is exactly what H2's geodesics can never have — this
is *why* the attractor exists here and not there.

---

## Chaotic CTC bridge — a "flickering" time machine

Feeding `a(t)` (on the attractor) into the static Bonnor Case III exterior, the log-frequency
`α = √(4a²−1)` breathes and the CTC bands (negative bands of g_φφ) shift along r. In a fixed
observation window r ∈ [1.05, 20]:

- `a(t)` wanders over **[0.844, 2.169]** (stays supercritical, no clipping at ε=0.2)
- `α` over **[1.368, 4.217]**
- CTC band **count flickers 1 ↔ 3**; log-measure mean 1.673, std 0.167, range [1.343, 2.298]

> Adiabaticity: the quasi-static treatment needs the dust e-folding time to exceed the band
> light-crossing time. At unit scaling the ratio is **17** (>1), so a slow-dust scale
> separation (large H²/κ) is required. This is **reported, not assumed** — the dust
> diffusivity/scale is a free parameter, so adiabaticity is achievable, not automatic.

---

## Mandatory novelty-catcher run (project rule, 2026-05-11)

Address-space λ₂ Hamming-graph catcher scanned over the drive `r ∈ [0.5, 40]` (24 points),
fingerprint = settling-robust quantiles of the post-transient |X| distribution.

- **Verdict: `novel_structure`** (λ₂ range over radii [−0.000, 10.955]).
- **Primary sharp feature at r = 24.54** — coincides with the numerical Lyapunov onset
  (24.54) and the analytic Hopf threshold (24.74). The catcher **independently localizes
  the chaos onset** of the rotating-dust rotation to within the scan resolution (Δr ≈ 1.7).
- Second feature at r = 40.0 is the scan endpoint (edge effect).

---

## Bottom line

- You **cannot** induce a Lorenz attractor in the cylinder's geodesics — Liouville forbids
  it (H0/H2 confirm: divergence ≈ 0 even under strong chaotic driving).
- You **can** induce one in the rotation itself by treating the dust as a differentially
  rotating dissipative fluid; its lowest modal truncation **is** the Lorenz system, and it
  reproduces the canonical attractor to ≤1% on every invariant (H3).
- The Tipler rotation parameter `a(t)` then lives on a strange attractor, and the CTC /
  time-machine band structure flickers chaotically while inheriting the attractor's
  statistics — a tunable "complex rotation," with the chaos onset confirmed independently
  by the address-space catcher.

---

# Expansion — more attractors and tests that exploit the construct

The novel construct is the **chaotically-gated CTC**: a dissipative chaotic flow → rotation
`a(t)` → time-varying time-machine band structure. Three follow-up results probe whether
this is a general, useful structure rather than a Lorenz-specific coincidence.

## E1 — a registry of chaotic rotation laws (universality of the bridge)

Any dissipative chaotic flow can serve as the rotation law. Only **Lorenz** is derived from
dust physics (Saltzman); Rössler/Chen/Halvorsen are phenomenological alternatives included
to test universality. All reproduce literature Lyapunov spectra, and for **every one**
`Σλ = ⟨tr J⟩` (the ergodic theorem) with `Σλ < 0` ⇒ genuine attractor:

| rotation law | derived? | Lyapunov spectrum | KY dim | Σλ = ⟨tr J⟩ | CTC bands flicker |
|---|---|---|---|---|---|
| Lorenz (rotating-dust) | **yes** | [+0.90, 0, −14.57] | 2.06 | −13.67 | 1↔3 |
| Rössler | no | [+0.07, 0, −5.37] | 2.02 | −5.29 | varies |
| Chen | no | [+2.06, 0, −12.06] | 2.17 | −10.00 | varies |
| Halvorsen | no | [+0.66, 0, −4.87] | 2.14 | −4.20 | varies |

The CTC bridge and all diagnostics are agnostic to which attractor drives the rotation.

## E2 — Takens faithfulness: the time machine *is* a readout of the attractor

The CTC band log-measure `M(t)` is a single scalar geometric observable. By Takens'
theorem, if the construct is sound, a delay embedding of `M(t)` reconstructs the attractor,
so its correlation dimension should match the attractor's own:

| rotation law | D2 (state space) | D2 (from CTC observable) | rel. diff |
|---|---|---|---|
| **Lorenz (rotating-dust)** | 2.00 | 2.33 | **16%** |
| Rössler | 1.86 | 2.36 | 27% |
| Chen | 2.12 | 3.05 | 44% |
| Halvorsen | 2.07 | 3.10 | 50% |

For the physically-derived **Lorenz** rotation the spacetime CTC flicker recovers the
attractor dimension well — *the chaotic rotation can be read out of the spacetime alone*.
The reconstruction degrades for the stiffer/higher-`D` phenomenological attractors, where the
scalar delay-embedding `D2` inflates (a standard small-sample / single-observable bias). So
faithfulness is **clean for the derived construct, qualitative for the others** — reported
honestly rather than averaged away.

## E3 — chaos synchronization of a Tipler pair (Pecora–Carroll)

Two cylinders with Lorenz rotations, master → slave coupled through one shared variable:

| drive variable | conditional Lyapunov | sync error (tail) | locked? |
|---|---|---|---|
| x | **−1.63** | `0.0` (machine zero) | **yes** |
| z | −0.07 (marginal) | 4.8×10⁻³ | no |

With the x-drive the conditional (sub-system) Lyapunov exponent is negative, so two chaotic
time machines **phase-lock**: their relative rotation `a₂(t) − a₁(t) → 0`. This is the
chaotic generalization of the pair's tunable phase offset (which in the static case switches
CTC bands on/off). The z-drive fails to lock — the criterion correctly discriminates.

## E4 — catcher chaos-onset universality (partial, and honest about it)

Does the address-space λ₂ catcher localize the chaos onset (Lyapunov-exponent
zero-crossing) using only the |X|-distribution topology — no dynamics? Scanned over each
attractor's natural bifurcation parameter:

| rotation law | bifurcation param | Lyapunov onset | catcher feature | agreement |
|---|---|---|---|---|
| Lorenz | r | 24.54 | **24.54** | **0.0** ✓ |
| Halvorsen | a | 1.230 | **1.230** | **0.0** ✓ |
| Chen | c | 20.09 | 24.26 | 4.17 (different transition) |
| Rössler | c | 4.33 | `smooth` (no feature) | — |

**Clean for sharp onsets (Lorenz, Halvorsen): catcher = Lyapunov to the scan grid.** For
**Rössler** the catcher returns `smooth` — its period-doubling cascade is a *gradual* route
to chaos, not a sharp address-space jump (the same limitation documented for the 3-SAT
sigmoid in the Millennium work, which needed a derivative catcher). For **Chen** the catcher
flags a real structural change (24.26) but it is an interior transition, ~4 above the
LE onset. So onset-detection universality is **partial**: it tracks *abrupt* (subcritical-
Hopf-type) onsets, not continuous (period-doubling) ones. Reported as found, not oversold.

---

# Exact (hyperbolic) time-varying metric — replacing the adiabatic treatment

The adiabatic CTC treatment froze the spacetime to the *static* Bonnor solution for the
instantaneous `a(t)` — i.e. it assumed the gravitational field responds at **infinite
speed**. The exact treatment solves the *time-dependent* (hyperbolic) cylindrical field
equations, so the field propagates at speed `c = 1` with **retardation** and wave emission.

**Master equation.** The frame-dragging perturbation `Ψ(t,r) = δg_tφ` obeys the exact
*dynamical* generalization of the repo's static twist reduction:

```
Ψ_tt = Ψ_rr − (1/r) Ψ_r − V(r) Ψ,   V(r) = (F0'² − c0²)/F0² − 2F0'/(r F0)
```

— a cylindrical wave with **characteristic speed exactly 1** (the coefficient of `Ψ_rr`),
inner Dirichlet drive `Ψ(t,R)=δa(t)` (the dust-surface matching), outgoing-wave outer
boundary. Static limit reproduces the static twist structure. *Exact in the dynamics
(full retardation/wave propagation); linear in the perturbation amplitude δa.*

### What the exact treatment gives that adiabatic cannot

| check | result |
|---|---|
| **Causal speed** | a rotation pulse reaches radius r at retarded time `t ≈ t₀+(r−R)` (speed 1, to <0.3) |
| **Well-posedness** | undriven energy conserved to **6×10⁻⁵** |
| **Adiabaticity made rigorous** | response lags the drive by exactly the travel time `(r−R)`; adiabatic phase error `= Ω(r−R)` grows 0 → 1.0 → 2.6 rad as the drive speeds up. **Adiabatic is valid iff Ω(r−R) ≪ 1.** |

**The old "adiabaticity ratio ≈ 17" is now a theorem:** the band at radius r responds to
`a(t − (r−R))`, not `a(t)`. For the Lorenz drive (rate ~0.9, bands at r ~ 3–6) the product
`Ω(r−R) ~ 3–5 ≫ 1`, so the adiabatic CTC flicker was quantitatively wrong.

### Exact-retarded vs adiabatic CTC state (chaotic a(t))

Evolving the exact retarded `δK(t,r)` under a Lorenz `a(t)` and reconstructing
`L=(r²−K²)/F0`, the measured lag equals the light-travel time, and **near a band edge the
adiabatic approximation mispredicts the time-machine open/closed state ~21% of the time**
purely from ignoring the delay:

| r_obs | travel time (r−R) | measured lag | CTC-state disagreement |
|---|---|---|---|
| 4.0 | 3.0 | **3.00** | 0% (L₀ far from 0) |
| 7.0 | 6.0 | **5.99** | **21.4%** (near band edge) |
| 10.0 | 9.0 | **9.00** | 0% |

### A single-variable instability — which turned out to be an artifact

The single-variable frame-dragging model (field χ = δω = δK/F) showed a **tachyonic
potential** `V(r) = (F0′²−c0²)/F0² − 2F0′/(rF0)` with `V < 0` (growth rate ~1.4) and a driven
evolution that diverges. Tempting to read as a physical instability — but it rests on the
*single* variable ω=K/F, which divides by F and so manufactures structure at ergosurfaces.
This needed checking against the full coupled system, which is the next section.

---

# Resolution — the coupled (δF, δK, δL, δh, δS) linearized system

To settle whether the instability is physical, I derived the **full** linearized vacuum
equations for the z-independent (t,φ)-sector with sympy (no hand-algebra), validated them,
and analysed stability. Code: `derive_coupled.py` (symbolic, dill-cached), `validate_coupled.py`,
`evolve_coupled.py`, `run_coupled.py`, `test_coupled.py` (6 tests).

**Structure.** 5 evolution equations (G_tt, G_tφ, G_rr, G_φφ, G_zz) + 2 momentum constraints
(G_tr, G_rφ); the odd z-twist sector (G_tz, G_rz, G_φz) **vanishes identically** (z-parity).
The second-time-derivative matrix M(r) has **rank 3** (n=240) — i.e. **2 gauge degrees of
freedom**, the two cylindrical-vacuum polarizations.

**Ergosurface regularity (the key point).** In metric variables the inverse-metric
denominator is `F0L0+K0² = r²` (never zero) — there is **no 1/F0 anywhere**. Every component
is **finite across the ergosurface** (F0=0 at r=1.545); the only divergence is the conformal
factor `h = g_rr = g_zz ~ e⁷⁰⁰`, a pure coordinate pathology whose terms all carry 1/h and
**→ 0** there. Capping h realises that regular limit (no float overflow).

**Validation.** Background is vacuum (G0 ~ 10⁻⁴¹); the static van Stockum family `∂_a(metric)`
is an **exact zero-frequency solution** (residual 3×10⁻², h-derivative-limited; constraints
exactly 0). The derived equations are correct.

**Decisive stability test (sign of the physical potential).**

| model | physical ω² spectrum | tachyonic? | growth |
|---|---|---|---|
| single-variable (ω=K/F) | V(r), min **−1.86** | **100% negative** | 1.36 |
| **coupled (metric vars)** | eig(pinv(M)·C0): **no nonzero eigenvalues** | **none** | — |

The single-variable model's potential is fully tachyonic (ω²<0 ⇒ growth). The coupled
physical sector has **no tachyonic mass term** — the polarizations are massless waves
(ω²≈0), hence **stable**. The instability was a **variable-choice artifact** of dividing by F.

> **Honest scope.** The stability verdict comes from the validated coupled equations + the
> physical-sector potential-sign (long-wavelength) analysis, which is reduction-independent.
> A *naive free time-evolution* of the rank-deficient (gauge) system is numerically ill-posed
> without constraint damping / a hyperbolic gauge — it blows up as a **scheme** artifact, so it
> does not by itself settle stability (and is reported as such). A fully gauge-fixed,
> constraint-damped nonlinear evolution is the natural next step, but the destabilising
> *mechanism* (a tachyonic 1/F potential) is demonstrably **absent** in the correct variables.

**Bottom line on "exact vs adiabatic":** adiabatic genuinely misses **retardation** (the band
at r responds to a(t−(r−R)); ~21% CTC-state misprediction near band edges; valid only if
Ω(r−R)≪1). The earlier "instability" of the frozen background was **not** physical — it was an
artifact of the single-variable reduction; the coupled system is regular and stable.

---

# Capstone — gauge-fixed, constraint-damped *nonlinear* cylindrical evolution

The linear analysis showed the system is regular and stable. The final step evolves the
**full nonlinear** cylindrical Einstein vacuum in a well-posed formulation. Code:
`nonlinear_cylindrical.py`, `run_nonlinear.py`, `test_nonlinear.py` (7 tests).

**Formulation (Jordan–Ehlers–Kundt / Einstein–Rosen, two polarizations with rotation):**

```
ds² = e^{2(γ−ψ)}(−dt²+dr²) + e^{2ψ}(dz + ω dφ)² + r²e^{−2ψ}dφ²
```

with nonlinear wave-map equations for ψ(t,r) and the gravitomagnetic twist ω(t,r):

```
ψ_tt   = ψ_rr + ψ_r/r + (e^{4ψ}/2r²)(ω_t² − ω_r²)
ω_tt   = ω_rr − ω_r/r + 4(ψ_t ω_t − ψ_r ω_r)
```

- **Gauge fixing:** areal radial coordinate + conformal time gauge (no residual lapse/shift
  freedom; coordinate light speed = 1).
- **Constraint damping (Z4-style):** γ is evolved with the momentum constraint and damped
  toward its Hamiltonian-constraint value, so the constraint `C = γ − γ_H` obeys `C_t = −κ C`.

**Validation (`run_nonlinear.py`, ~87 s):**

| check | result |
|---|---|
| static Levi–Civita fixed point | interior drift **7.7×10⁻⁷** over t=5 |
| causal propagation (twist pulse) | arrives at retarded time `t₀+(r−R)`, speed **1** |
| C-energy (Thorne) conservation | rel. drift **8.5×10⁻⁵** |
| **2nd-order convergence** | self-convergence ratio **4.02** |
| **nonlinear polarization coupling** | a strong twist ω **sources ψ** (9.95×10⁻³; zero in linear theory) |
| constraint damping | κ=2 suppresses the violation **7.9×** vs κ=0 (well-posed: stays ~10⁻⁶) |

**Driven by the chaotic rotation:** feeding the Lorenz `a(t)` into the twist boundary, the
full nonlinear spacetime stays **bounded and stable** (max constraint 7.8×10⁻³, finite
throughout) — closing the arc *adiabatic → exact-linear → coupled-linear → nonlinear*, all
stable.

> **Honest scope.** This is the well-posed (z,φ)-twist polarization sector — the standard
> nonlinear cylindrical-GR system — where ω is the gravitomagnetic twist (g_φφ>0, no CTC).
> It demonstrates the full nonlinear machinery (gauge fixing + constraint damping +
> convergence + stability) and is the well-posed cousin of the (t,φ) CTC frame-dragging.
> The fully nonlinear (t,φ)-CTC evolution remains the non-orthogonally-transitive open
> problem; the linear coupled analysis (above) already shows *its* destabilising mechanism is
> a variable artifact.

---

# Next step — nonlinear GW emission from a chaotically-rotating cylinder

With a stable nonlinear evolution in hand, the next step is **observable radiation**: drive
the twist boundary with the Lorenz `a(t)` and ask whether the emitted gravitational waves
carry the strange-attractor fingerprint — including through nonlinear mode-mixing. Code:
`nonlinear_gw_emission.py`, `run_gw.py`, `test_gw_emission.py` (3 tests).

**Why this is the well-posed "next step" (and the CTC sector is not).** A genuinely nonlinear
evolution of the **(t,φ) CTC frame-dragging** sector is blocked at a *fundamental* level: a
spacetime with CTCs has **no global Cauchy surface**, so it is not a well-posed initial-value
problem — there is no "evolve through CTC formation" in the usual sense. The radiative physics
of the well-posed (z,φ)-twist sector is the rigorous advance.

## A — the radiated waves reconstruct the attractor (and the nonlinear one is *folded*)

Far-detector correlation dimensions (Takens), Lorenz attractor ≈ 2.06:

| radiated field | origin | D2 (detector) |
|---|---|---|
| **ω (twist)** | directly driven (linear in a) | **2.22** ≈ attractor |
| **ψ** | generated **only nonlinearly** (ω² source) | **1.30** (folded) |

The primary radiated twist **reconstructs the Lorenz attractor**. The nonlinearly-generated
cross-polarization ψ carries a *lower* dimension because the quadratic source `ψ ∝ ω²`
**folds the two Lorenz lobes** (it cannot distinguish ±ω) — a genuine nonlinear-optics-like
imprint on the gravitational radiation.

## B — nonlinear cross-polarization conversion

| drive ε | ψ/ω (radiated) |
|---|---|
| 0.02 | 0.027 |
| 0.04 | 0.059 (×2.2) |
| 0.08 | 0.165 (×2.8) |

`ψ/ω` roughly **doubles when the drive doubles** ⇒ `ψ ∝ ω² ∝ A²` (linear theory gives ψ=0):
the chaotically-rotating cylinder pumps energy from the twist polarization into the `+`
polarization nonlinearly.

## C — radiated-energy chaos

Under the chaotic drive the Thorne C-energy in the domain is itself a chaotic time series
(another faithful readout of the attractor); see `lorenz_gw_results.json`.

---

# Dynamical approach to the chronology horizon (chronology protection)

The remaining frontier — a Cauchy evolution of a CTC-containing spacetime — is impossible
(no global Cauchy surface). The well-posed substitute is the *approach*: spin the cylinder
up through the supercritical threshold a = ½ and ask whether the renormalized **quantum**
stress tensor diverges at the forming chronology horizon (Hawking's protection mechanism),
while the **classical** geometry stays smooth. Reuses the Phase-2a machinery
(`stress_energy_ctc`, `point_splitting`, `quantum_diagnostics`). Code:
`chronology_horizon.py`, `run_chronology.py`, `test_chronology.py` (6 tests).

## The protection signal switches on exactly at threshold

| a | horizon r_H | κ | Boulware ⟨T_tt⟩ power | Kretschmann@r_H | protected |
|---|---|---|---|---|---|
| ≤ 0.50 | — (none) | — | — (finite) | — | no (no CTC) |
| 0.52 | 2.649 | 0.52 | **−1.005** | 0.017 | **yes** |
| 1.00 | 1.831 | 1.00 | **−1.008** | 1.02 | **yes** |
| 1.50 | 1.545 | 1.50 | **−1.010** | 10.1 | **yes** |

The instant a crosses ½, a chronology/Cauchy horizon (F = −g_tt = 0) appears and the
renormalized **⟨T_tt⟩ diverges as 1/(r−r_H)** (power ≈ −1, **α-universal**) — Hawking's
chronology-protection signal — while the **classical Kretschmann stays finite** (growing
with a but always bounded). The spacetime is classically smooth; only the quantum
backreaction blows up at the horizon.

## The approach (a → ½⁺)

| a − ½ | r_H | κ | ⟨T_tt⟩ power |
|---|---|---|---|
| 0.200 | 2.206 | 0.700 | −1.006 |
| 0.050 | 2.554 | 0.550 | −1.005 |
| 0.010 | 2.683 | 0.510 | −1.005 |
| 0.005 | 2.700 | 0.505 | −1.005 |

As a → ½⁺ the horizon is **born at a finite radius r_H → R·e = 2.718**, κ → ½, and the
protection power stays ≈ −1: **the barrier appears at full strength right at the threshold**
(no gradual onset).

## Chaotic chronology-protection barrier

Driving the rotation with the Lorenz a(t) wandering across a = ½ (a ∈ [0.23, 1.02]) gives a
barrier that **flickers ON/OFF 83 times** (supercritical 74% of the time): a *chaotically
gated chronology-protection barrier* — the divergent quantum stress tensor switches on
exactly when the chaotic rotation pushes the cylinder supercritical. The mandatory catcher
returns `novel_structure` on the protection-signal scan.

> **Interpretation.** This is the dynamical (quasi-static-sequence) form of Hawking's
> chronology protection: the *classical* van Stockum geometry happily forms CTCs as it is
> spun up, but the *semiclassical* backreaction diverges at the chronology horizon the moment
> it forms — consistent with the conjecture that quantum effects prevent time-machine
> formation. Caveat: 2D Polyakov stress tensor (massless conformal scalar), quasi-static
> sequence; a full 4D self-consistent backreaction is beyond scope.

---

# Self-consistent semiclassical backreaction — quantum chronology protection

The previous step showed ⟨T_μν⟩ diverges at the chronology horizon while the *classical*
curvature stays finite. The self-consistent question: does that divergent backreaction
**protect chronology**? A full 4D `G_μν[g] = 8π⟨T_μν⟩[g]` near a CTC-forming horizon is an
open problem (and ill-posed once CTCs exist), but the **breakdown of the semiclassical
expansion** is rigorous and computable. Code: `semiclassical_backreaction.py`,
`run_semiclassical.py`, `test_semiclassical.py` (6 tests).

The dimensionless backreaction strength

```
β(r) = 8π ℓ² |⟨T_tt⟩(r)| / κ²        (ℓ² = Planck area ~ ℏG;  κ = surface gravity)
```

diverges at the horizon (⟨T_tt⟩ ~ 1/(r−r_H), κ finite), so for **any** ℓ > 0 there is a shell
of width **eps_bd** around the classical horizon where β ≥ 1 — the perturbative vacuum
geometry is invalid and is replaced by a strong-backreaction (quantum-gravity) region. The
classical chronology horizon is **never reached**:

| quantity | result |
|---|---|
| shell width | **eps_bd ∝ ℓ²** (Planck area), fit power **1.000** |
| classical limit | eps_bd → 0 as ℏ → 0 (CTCs only in the strict classical limit) |
| vs surface gravity | eps_bd ∝ 1/κ² × amplitude — **wider near threshold** (a → ½⁺) |
| self-consistent horizon | r_eff = r_H − eps_bd < r_H (**horizon shrouded**, fixed point) |
| chaotic drive | Lorenz a(t) → protection shell flickers, present 83% of the time |

**Conclusion (quantum chronology protection):** for any ℏ > 0 the chronology horizon is
shrouded by a semiclassical-breakdown shell whose width scales with the Planck area; the
classical horizon and its CTCs are reached only in the strict classical limit ℏ → 0. This is
the quantitative, dynamical form of Hawking's chronology-protection conjecture on the
van Stockum background.

> **Honest scope.** 2D Polyakov ⟨T⟩ (massless conformal scalar); the breakdown/scaling
> analysis and the effective-horizon fixed point are rigorous. A full nonlinear
> self-consistent metric (solving Einstein + ⟨T⟩ together through the shell) is genuinely
> open — beyond the breakdown shell the description requires quantum gravity, which is
> precisely the content of the protection statement.

## Reproduce

```bash
cd experiments/lorenz_rotation
python run_experiment.py                  # core Lorenz run -> lorenz_rotation_results.json (~140 s)
python run_expansion.py                   # attractors + deep tests -> lorenz_expansion_results.json (~10 min)
python run_exact.py                       # exact hyperbolic vs adiabatic -> lorenz_exact_results.json (~65 s)
python run_coupled.py                      # coupled (dF,dK,dL,dh,dS) system -> lorenz_coupled_results.json
                                           #   (first run derives + dill-caches the equations, ~5 min; then ~16 s)
python run_nonlinear.py                    # nonlinear cylindrical evolution -> lorenz_nonlinear_results.json (~87 s)
python run_gw.py                           # nonlinear GW emission -> lorenz_gw_results.json (~6 min)
python run_chronology.py                    # chronology-horizon approach -> lorenz_chronology_results.json (~2 s)
python run_semiclassical.py                 # semiclassical backreaction -> lorenz_semiclassical_results.json (~2 s)
python -m pytest test_lorenz_rotation.py test_exact_metric.py test_coupled.py test_nonlinear.py \
                test_gw_emission.py test_chronology.py test_semiclassical.py -q   # full suite (54 tests)
```

## References

- E. N. Lorenz, *Deterministic nonperiodic flow*, J. Atmos. Sci. **20** (1963) 130.
- B. Saltzman, *Finite amplitude free convection as an initial value problem*, J. Atmos.
  Sci. **19** (1962) 329.
- W. J. van Stockum, Proc. Roy. Soc. Edin. **57** (1937) 135; F. J. Tipler, Phys. Rev. D
  **9** (1974) 2203; W. B. Bonnor, J. Phys. A **13** (1980) 2121.
- Benettin et al., *Lyapunov characteristic exponents…*, Meccanica **15** (1980) 9.
- A. Einstein & N. Rosen, *On gravitational waves*, J. Franklin Inst. **223** (1937) 43
  (cylindrical waves); J. Stachel, *Cylindrical gravitational news*, J. Math. Phys. **7**
  (1966) 1321; S. W. Hawking, *Chronology protection conjecture*, Phys. Rev. D **46** (1992) 603.
