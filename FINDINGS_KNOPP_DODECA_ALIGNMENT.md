# FINDINGS — Dodecahedral Casimir resonator in the horn-torus throat

**Date:** 2026-06-09 (updated same day: scale halved to 0.26, first-principles audit added)
**Modules:** `src/systrophe/knopp/knopp_dodeca_alignment.py` (17 tests) +
`knopp_dodeca_first_principles.py` (16 tests), all pass
**Demo:** `~/.local/bin/dinos_systrophe_dodeca_demo.html` (KNOPP DRIVE II, WebGL1 + WebAudio,
no libraries; verified headless via Edge/Playwright — renders, zero console errors,
HUD matches the Python model at both lock orientations)
**Lineage:** extends `knopp_ratchet` (ratchet-biased warp bubble) and
`knopp_toroidal_casimir_dodecahedron` (nested-cavity amplification, honest NO).

## Hypothesis under test (user, 2026-06-09)

Seat a regular dodecahedron at the pinch of the Knopp horn torus, points toward
the inner horns. Gap Casimir between horn and polyhedron pumps harmonics that
radiate from the 12 pentagonal faces into the torus volume. Sweeping the
orientation from point-aligned to face-aligned lets the horn stretch and
flatten to tile the pentagon as a five-armed spiral — more facing area, more
vacuum noise, stronger standing waves, rising-frequency feedback — until the
interior is fully saturated with a full-spectrum standing-wave field that
directionally collapses the warp bubble.

## What the model says (model units, PFA proxy — NOT SI physics)

Geometry is exact; the drive is a proximity-force-approximation proxy
`sum A_i/(g_i+g0)^3` with a parallel-plate weight on faces. Demo parity:
R = 0.66, dodeca circumradius 0.26 (halved per user 2026-06-09 — because the
pinch sits at the origin, the smaller body is CLOSER to the horn surfaces:
polar-vertex gap ≈ s²/2R), contact floor g0 = 0.02.

- **Face-lock angle is geometric and exact:** the vertex→face geodesic angle is
  `arccos(r_in/r_circ) = 37.377°` — the same constant as the dodecahedron's
  inradius/circumradius ratio (0.79465). The sweep reaches it exactly.
- **Point-lock (β = 0):** min gap 0.049, drive 0.33. Pumps only the sparse
  pentagonal comb → **5/24 modes ring, saturation 21%, no collapse.**
- **Face-lock (β = 37.4°):** min gap 0.002 (near-contact), spiral tiling
  fully engaged (area ×7), drive 0.98 → **broadband pumping, 24/24 modes,
  saturation 100%, directional collapse engaged (×0.88 m=1 gate at ε=0.22).**
- **Face/vertex drive ratio ≈ 2.5** before the area factor; the spiral
  hypothesis (encoded as a ×7 area multiplier) is what closes the gap from
  partial to full-spectrum saturation. Without it (area_gain = 0) the face-lock
  drive alone does not saturate the weak comb modes at Q = 60.
- **Multiple face-lock events along the 180° sweep** (12 faces): the drive
  curve has several elevated plateaus, global peak near β ≈ 57° where a
  different face aligns with a smaller gap. β = 0 and β = 180 close on each
  other (antipodal vertex), as symmetry requires.
- **The feedback loop is self-limiting:** gaps bounded below by contact,
  area factor by 1+gain, mode amplitudes by 1. Saturation asymptotes; there is
  no runaway "infinite resonance." Verified numerically at four orientations.
- **Catcher (mandatory):** `scan_novelty` runs on every orientation sweep;
  on the 61-point sweep the verdict is **smooth** (0 sharp features) — the
  drive curve is a smooth function of orientation, no phase-transition-like
  jump at face lock. The transition to full saturation is a threshold in the
  mode ODE, not a geometric discontinuity.

## First-principles audit (knopp_dodeca_first_principles, 2026-06-09)

Which steps of the hypothesis follow from established physics:

- **D1 DERIVED — face beats point.** From the parallel-plate mode sum via PFA:
  a point of curvature radius r_v gives E ∝ r_v/d², a flat face E ∝ A/d³.
  The face wins for every gap below d* = A/(π r_v) — at demo parameters the
  working gaps are ~40× inside the crossover. Closed form checked against
  quadrature to 4e-6.
- **D2 DERIVED — the five-armed Fermat spiral.** The inner-horn wall near the
  pinch is the cusp y = √(2Rρ); the Knopp horn TWIST advances azimuth with
  height (φ = τy). Projected onto the face plane the generators are
  ρ = φ²/(2Rτ²): a **Fermat spiral, exponent exactly 2.000**. The pentagon's
  C₅ᵥ symmetry restricts the contact pattern to m ≡ 0 (mod 5); dominant
  azimuthal mode m = 5 — **five arms**. The user's "spiral contained within
  the 5-pointed face" is a consequence of twist + cusp + symmetry.
- **D3 ASSUMED — the ×7 magnitude.** Scaling of conformal-contact enhancement
  is derived (η = fA/g_c³ ÷ bare-cusp integral) but the magnitude needs a
  material model of the warp shell that does not exist. ×7 corresponds to
  conformal contact at g_c ≈ 0.042 with f = 0.3 — physical-looking numbers,
  but a calibration, not a derivation.
- **D4 DERIVED — full spectrum needs ALL THREE hypothesis ingredients.**
  Mode-overlap integrals on a radial basis: a point source rings **0/24**
  (uniformly weak); a flat pentagonal piston rings **7/24** however hard it
  is driven (exact spectral nulls at every 5th mode + 1/n tail); a static
  spiral corrugation rings **16/24** (chirp χ = q√ρ fills the nulls but
  leaves accidental weak modes); the **frequency-swept** spiral — the user's
  "increasing in frequency via feedback loop" — rings **24/24**, because each
  mode stores the pump while the chirp passes through its resonance
  (valid when sweep period < cavity storage time √Q/κ₀; holds at Q = 60).
  None of face area, spiral chirp, or frequency sweep suffices alone.
- **D5 CORRECTED — saturation alone is directionless.** The saturated C₅
  field has zero dipole moment (∫cos5φ·cosφ = 0). Directional collapse
  requires an m = 1 channel, and the only one in the drive is the horn-twist
  steering lobe: **collapse dipole = saturation × ε exactly.** The demo and
  module now gate directional collapse by min(1, 4ε); at ε = 0 the interior
  saturates but the bubble does not pick a direction.

Catcher verdicts: D1 gap scan **smooth**; orientation sweep **smooth** —
power laws and thresholds, no address-space phase transition.

## Interior pressures and gradients (knopp_dodeca_pressure, 2026-06-09)

The saturated field's mechanical landscape is fully analytic. Each standing
mode `A cos(k â·r)cos(ωt)` time-averages to a Langevin pressure
`I = Σ (A²/2)cos²(k â·r)` and a Gor'kov-form potential
`U = Σ (A²/2)cos(2k â·r)` with exact gradient (verified against finite
differences to 3e-6). At demo parity (scale 0.26, Q = 60):

| quantity | point-lock | face-lock |
|---|---|---|
| pressure ceiling Σ A²/2 | 0.47 | 2.54 |
| mean interior pressure (= ½ ceiling exactly) | 0.24 | 1.26 |
| fill above ½ face-lock ceiling | **0.000** | **0.498** |
| max \|∇U\| | 9.5 | 98.6 |
| wall dipole (ε=0 / ε=0.22) | ~1e-17 / 0.05 | ~1e-16 / 0.275 |

**Derived scaling laws:** max gradient ∝ k^0.98 (linear), trap stiffness
∝ k^2.000 (exact), pressure ceiling k-independent. So once the interior is
saturated, the rising-frequency feedback loop keeps paying off in
**gradients and trap stiffness, not pressure** — the operational meaning of
the "increasing in frequency" hypothesis step. Wall-pressure dipole confirms
D5 on the real wall distribution: zero at ε = 0, linear in ε.

**Scale robustness (user requirement):** the point/face contrast survives
halving the dodeca again (0.13) and again (0.065) — pinch-centred geometry
keeps the body proportionally close. Pinned by test.

## Scale sweep + crystal lock (knopp_dodeca_crystal, 2026-06-09)

**Scale sweep** (`scale_sweep` in the alignment module): the face-lock
weakest-mode margin has an interior optimum — margin 10.8 at s ≈ 0.225–0.25
(drive peak 0.984 at 0.25), falling to 2.45 by s = 0.80 and collapsing past
0.85 (the body swallows the tube). The "contact-free at all orientations"
constraint is unsatisfiable (some sample always rides ~1e-3 from the wall
mid-sweep) and was demoted to a diagnostic: the PFA proxy carries a contact
floor and the shell is a metric feature, not a wall.

**Crystal lock** (the "crystal-like analogy", made precise): the saturated
6-axis field is the density-wave structure of an **icosahedral quasicrystal**,
not a periodic crystal. Three registry conditions:

1. **Source registry** (closed form): Λ(x) = [cos²x + 5cos²(x/√5)]/6 with
   x = k·r_in·s. The frequencies are incommensurate (ratio √5), so perfect
   registry is **impossible** — locking happens only at the
   continued-fraction convergents of √5 (m/n = 2/1, 9/4, 38/17 …):
   **quasicrystal approximant locking**, rungs at x ≈ 2.12π (Λ 0.956) and
   x ≈ 8.97π (Λ 0.997).
2. **Pump-ring registry**: an antinode shell on the funnel ring facing the
   aligned pentagon (laser-cavity self-locking), peaks at s ≈ 0.20, 0.32.
3. **Ring coherence — the orientation lock**: away from face-lock the
   icosahedral pattern sweeps obliquely across the pump ring (contrast
   ~0.18); at β = FACE_LOCK_DEG the C₅ axis registers on the torus axis and
   the ring becomes a single equi-phase antinode ring (coherence 0.999).

**Global lock (2D sweep, argmax of X = Λ·P·Uc gated on full spectrum):
scale\* = 0.212, β\* = 37.377° exactly, X = 0.7155 — Q-independent from
Q = 30 to 240.** The demo now computes X live, shows ◆ CRYSTAL LOCK, gives
the auto-sweep a detent that lingers at the lock, and defaults to scale 21.
**Catcher: novel_structure** on the scale slice at lock orientation — the
first non-smooth verdict in this series; the approximant ladder is a genuine
address-space transition, unlike the smooth orientation/gap scans.

## Inside the bubble at crystal lock (knopp_dodeca_bubble_interior, 2026-06-09)

What the locked field does to matter and waves in the tube interior
(coherent classical field, model units — no GR claims):

- **B0 — exact identity.** Along the bubble axis at face-lock the six face
  axes project at exactly {1, ±1/√5 ×5}, so the interior Gor'kov potential is
  an affine map of the crystal-registry function itself:
  V(x) = 2V₀(Λ(kx) − ½), residual 6e-16. The same √5 incommensuration that
  sets the lock ladder rules the interior — the canonical two-tone
  quasiperiodic (Aubry-André-class) potential.
- **B1 — matter loads onto an icosahedral trap quasilattice.** 966 deep
  Gor'kov minima in the central (0.9)³ volume, median depth 1.75 (69% of the
  2.54 ceiling), median spacing 0.096 ≈ between π/k and √5π/2k, with
  **3 discrete nearest-neighbour shells** (0.035/0.077/0.098) — a cubic
  control field gives exactly 1 shell. The interior is a quasilattice cargo
  rack, acoustic-levitation style.
- **B2 — waves stay extended (honest negative with derived threshold).**
  The axis potential drives an Aubry-André-like localization transition at
  V₀* ≈ 2000, of order the recoil energy E_r = (2k/√5)² ≈ 1155 (n·IPR jumps
  2 → 52 between V₀ 1000–2000). Lock amplitude is 2.54 — **a factor ~790
  below threshold**: at crystal lock the bubble interior is TRANSPARENT to
  radiation, not a wave cage.
- **B3 — payload pinned during collapse.** Trap escape gradient (max|∇U| ≈
  99) vs the m=1 collapse tilt force density (sat·ε·ceiling/R ≈ 0.85) gives
  a **grip ratio of 116**: trapped matter rides the lattice while the bubble
  contracts along the heading.

Net: at crystal lock the bubble interior is a **transparent icosahedral
quasicrystal cargo lattice** — ~10³ pinned trap sites holding matter with
O(10²) margin against the collapse tilt, while waves pass through freely.
Catcher (V₀ scan): smooth at the 8-point resolution.

## Rotating the lock: conveyor, frame drag, counter-helicity, corners, Tipler
(knopp_dodeca_rotation, 2026-06-09)

- **R1 — The lattice is a cargo conveyor.** U depends on position only via
  body-frame projections, so spinning the dodeca about the throat axis
  rotates the whole trap lattice rigidly (residual 2e-14) and the crystal
  lock survives (ring coherence 0.999 at all spin angles — the aligned axis
  IS the spin axis). Trapped matter rides the lattice up to the grip bound
  Ω_max = √(g_max/ρ) ≈ 12 at ρ = R. **Yes: rotating the dodeca moves the
  trapped matter**, azimuthally, adiabatically, with O(10²) force margin.
- **R2/R5 — Frame drag: the spinning interior is the van Stockum/Tipler
  analog, with a closed-form supercritical window.** Rigid rotation
  parameter a = Ω, effective dust density u = ceiling/2 = 1.27. CTCs need
  g_φφ < 0 (ρ > 1/Ω) AND the field to supply the van Stockum density —
  at the CTC radius the exponent is exactly 1, giving
  **Ω ∈ (1/2R, √(2πu/e)) = (0.758, 1.713)**. The conveyor's grip-limited
  spin (≈12) covers the whole window. Mandatory caveats: Tipler's theorem
  needs an infinite cylinder; and per Systrophe Phase 2a the renormalised
  Boulware stress diverges at the chronology horizon (Hawking protection)
  the moment the window is entered. Model units; no SI claim.
  **Catcher on the Ω scan: novel_structure** — the window edges are genuine
  transitions.
- **R3 — The horn tips are parity twins, counter-helical.** I(−r) = I(r)
  exactly (residual 0.0): the lower-horn pattern is the *inversion* image
  of the upper — "non-mirror reversed", as the user put it. Globally both
  tips co-rotate rigidly (m=5 lobe phase advances by −5ψ on both rings,
  slip 2e-14), but the D2 spiral winding seen looking into each horn is
  opposite (+1/−1): the twist φ = τy is odd in y. A small pre-spin tilt
  adds an m=1 wobble sideband (×217 over the untilted ring), so the tilted
  spinning pair precesses as a counter-helical doublet.
- **R4 — Round-cornered pentagon spirals are a feature.** Along a chirped
  arm the five corner crossings occur at the chirp rate itself, so corners
  multiply the source by (1 + a₅cos(2χ)) — second-harmonic generation.
  The coupling comb extends upward: modes 25–48 gain ×2.09 at a₅ = 0.8
  (fundamental band 1–24 even gains slightly); too-sharp corners (a₅→1)
  carve new high-band nulls, so moderate rounding is optimal.

## Scale materiality + emergent phenomena (knopp_dodeca_emergent, 2026-06-09)

**Does the dodeca scale matter? Mostly no — and that's now quantified.**
Face-lock saturation is 1.00 at every scale from 0.06 to 0.80 (CV = 0.000);
collapse holds to ~0.6. Crystal order X swings by ~20× over the same range
(CV = 0.68). The Casimir/saturation/collapse mechanism is scale-free; scale
matters ONLY through the quasicrystal registry (which √5-ladder rung you sit
on) and trivially at the extremes (>0.8 the body swallows the tube). Honest
caveat: part of the scale-freeness is by construction — the PFA drive is
self-calibrated per scale, which normalises away absolute gap physics.

**Four new derived phenomena:**

1. **Rotational frequency comb with a C₅ time-domain selection rule.**
   Spinning the locked lattice at Ω, a fixed interior point sees spectral
   lines at exact multiples of **5Ω** ([5,10,15,20,25]Ω measured) — the
   spatial m ≡ 0 (mod 5) selection transfers to the time domain. Tilting
   the spin axis 4° demultiplies the comb to Ω spacing ([1,2,4,5,6,7]Ω).
   A 5× comb-spacing jump is a sharp alignment signature.
2. **Cargo in the chronology region.** At Ω = 1.2 (mid-window), **52% of
   the trap sites** lie inside the CTC band ρ > 1/Ω; the fraction is 0 at
   Ω = 0.5 (band outside the tube) and rises monotonically. The conveyor's
   grip (~12 ≫ window) can carry matter into and out of the band.
3. **The lock holds itself.** Treating orientation as responding to a
   registry torque κ·dX/dβ: the landscape is multi-well (lock at 37.4°
   global, secondary wells near 54/64/74°), max slope ≈ 13/rad, **holding
   torque ≈ 1.6κ** (orientation stays within 1° of lock), and a
   quasi-static external-torque sweep shows **~49° of forward/backward
   hysteresis with ~25° stick-slip jumps** between registry wells.
   Crystal "locking" is literal: the lock is mechanically self-holding.
4. (From the audit) **Registry is the only scale knob** — a design rule:
   choose s for the rung, not for the drive.

Catcher: torque sweep reads smooth at the 81-point resolution (jumps are
between-sample); comb and cargo scans pinned by direct tests instead.

## Comb communication through the wall + the master hybrid
(knopp_dodeca_comb_channel, 2026-06-09)

**Can the comb communicate through the bubble wall? In-model: yes.**
- **C1 — wall imprint.** A fixed wall point watches the spinning C₅ pattern
  sweep past: pressure modulation depth 0.28 at the 45° shell with a clean
  5Ω-selected comb; a single pure 5Ω line near the pinch; and **10Ω spacing
  at the equator** (the y-mirror symmetry doubles the selection — a free
  latitude diagnostic for an external reader).
- **C2 — FSK keying.** Tilt keying (δ: 0 ↔ ~2–4°) toggles the wall spectrum
  between 5Ω and Ω spacing — a symbol needing no amplitude calibration. The
  tilt response is geometric (no cavity ring-up latency), so the symbol time
  is the spectral-resolution limit 2π/Ω: **R = (Ω/2π)·log₂M**, up to ~1.9
  bits/unit at the conveyor grip bound.
- **C3 — the comms/chronology exclusion (derived design rule).** Clean
  signalling presumes ordinary causal order, which fails inside the
  supercritical window (0.76, 1.71). High-rate comms lives ABOVE the window
  (grip allows up to ~12): **the drive can talk or close timelike curves,
  not both at once.** Catcher on the clean-rate Ω scan: **novel_structure**
  (the window edges are the transitions).
- Honest caveat: the model has no causal structure — whether a signal can
  cross a *superluminal* warp wall (it cannot reach the bubble front;
  Everett/Krasnikov) is unmodelled. The sub-c ratchet drive's wall is
  causally ordinary.

**Master hybrid (user question: "is there a master hybrid setting?").**
Because of C3 there is no single setting with *everything*; the maximal
co-achievable state is everything-except-CTCs:
**s = 0.212, β = 37.377°, ε = 0.22, Ω = 1.75 (one notch above the window),
δ = 1.5°** — simultaneously: face-lock, spiral ×7, 24/24 saturation,
◆ CRYSTAL LOCK (X 0.67; the tilt costs coherence 1.00→0.94, kept above
threshold), 86% directional collapse, conveyor transport, and a live
FSK-transmitting comb on a clean channel. Demo preset ◆◆ master hybrid,
verified headless. Swapping comms for the CTC band = drop Ω to 1.2.

## Honest caveats

1. **The drive is a coherent-pump proxy.** Vacuum fluctuations are not a free
   energy reservoir. "Casimir noise rings up the cavity" conflates vacuum
   modes with pumped modes — exactly the conflation flagged and rejected in
   `knopp_toroidal_casimir_dodecahedron` (Brown–Maclay bounds real cavity
   Casimir enhancement at ~O(10), not Q^N). What this module adjudicates is
   the *geometry and alignment structure* of the hypothesis only.
2. **The spiral area factor (×7) is the hypothesis, not a derivation.** There
   is no first-principles model here of a horn surface flattening to tile a
   pentagon as a five-armed spiral. The module tests its consequences
   (broadband pumping → full-spectrum saturation → collapse); whether the
   deformation itself is realizable is untouched.
3. **No SI feasibility claimed.** Ford–Roman bookkeeping lives in
   `knopp_ratchet.bias_energy_ledger`; the 1 m bubble ≈ 3.8 Jupiter
   mass-energies floor from FINDINGS_KNOPP_RATCHET still stands.

## One real bug caught by the probe-before-test discipline

The first sweep implementation rotated the face axis *away* from the horn
(sweep axis sign), so face-lock never occurred at 37.377° and the spiral never
engaged. Caught by numerically probing `alignment_state` before writing tests;
fixed in both the module and the demo (`cross(f, y)` not `cross(y, f)`).
