# Interpretations: from ansatz to validation modules

In earlier releases this document catalogued six **ansatz-level
claims** about Systrophe physics that were not directly testable
without external input. As of v0.15.0, each claim has a corresponding
**validation module** in the codebase that concretises the missing
input and produces a quantitative answer.

This document tracks the resolution of each item.

For the *fully verified* mathematical claims that landed earlier,
see:

- `examples/external_claim_verification.py` (10 verified, 1 falsified, 8 not-testable),
- `examples/external_claim_verification_extended.py` (4 verified, 4 falsified, 6 not-testable),
- The "Bucket C" modules (`src/systrophe/tipler_fractal.py`,
  `anomaly_inflow.py`, `horned_torus.py`),
- The "Bucket A" modules (`src/systrophe/hadamard_offtrace.py`,
  `newton_kantorovich.py`, `floquet_mobius.py`, `acoustic_metric.py`,
  `casimir_throat.py`).

---

## Resolution table

| # | Claim | Module | Verdict |
|---|------|--------|---------|
| I.1 | Z_3 cover physically realises a wormhole throat | `wormhole_throat.py` | **NOT a Morris-Thorne wormhole**: the LP L=0 locus is vacuum, single connected manifold, no junction. Kinematic shape-function mapping is well-defined but does not carry MT topology |
| I.2 | Casimir replaces exotic matter | `exotic_matter_accounting.py` | **vacuous question**: LP throat is vacuum, needs no exotic matter. Conditional answer (if interpreted as MT throat) is still quantitative NO |
| I.3 | Self-consistent delta selects chronology-protecting phase | `chronology_protection.py` | consistent for matched pair |
| I.4 | Mobius monodromy is a U(1) Wilson loop | `wilson_loop.py` | concrete connection constructed |
| I.5 | DCE photon flux at off-resonance | `dynamical_casimir.py` | testable for any (Q, eps, detuning) |
| I.6 | Throat cavity is the natural QFT vacuum | `vacuum_states.py` | NO; Hartle-Hawking analog does not exist |

---

## I.1 The Z_3 cover physically realises a wormhole throat

**Earlier status.** Asserted as a natural identification but no
gluing map specified.

**Validation module.** `src/systrophe/wormhole_throat.py`.

**Concrete construction.** The Lewis-Papapetrou angular metric

    g_phi phi(r) = L(r)

admits an effective Morris-Thorne shape function

    b_eff(r) = r - L(r) / r

and effective redshift Phi_eff(r) = (1/2) ln F(r) (where F > 0).

**Throat candidate.** b_eff(r_t) = r_t is exactly L(r_t) = 0, i.e.
the CTC band boundary of the supercritical LP exterior. The
flaring-out condition b_eff'(r_t) < 1 is satisfied at the *outer*
boundary of each CTC band.

**Topology check.** A genuine Morris-Thorne wormhole connects two
asymptotic regions and has a *junction* at the throat. The LP
exterior is a **single connected** spacetime; the L = 0 locus is a
*coordinate* feature where the angular metric component vanishes,
not a junction between two regions. The kinematic mapping is
mathematically well-defined but does NOT promote the LP exterior
to a Morris-Thorne wormhole geometry.

**Stress-energy check.** Direct evaluation of the Ricci tensor at
L = 0 in the supercritical LP exterior gives R_{mu nu} = 0 (within
finite-difference noise; verified by `point_splitting.vacuum_residual`).
The "throat" sits in **Einstein vacuum**: no local stress-energy is
required to maintain the geometry. The CTC structure is sourced by
the rotating *cylinder interior* (positive-energy van Stockum
dust), not by anything at the L = 0 surface.

**Verdict.** The L = 0 locus is a vacuum-supported CTC band boundary
with a well-defined kinematic Morris-Thorne mapping --- but it is
NOT a Morris-Thorne wormhole throat. It is single-connected,
junction-free, and locally vacuum. The throat stays open *without*
exotic matter, because no Morris-Thorne wormhole is actually being
constructed.

---

## I.2 Casimir negative energy replaces exotic matter

**Earlier status.** Asserted as a Casimir-replaces-exotic-matter
route to traversable wormholes; the assumption was that the LP
"throat" is a Morris-Thorne wormhole junction.

**Validation module.** `src/systrophe/exotic_matter_accounting.py`.

**The actual answer is two layers deep:**

### Layer 1: Does the LP throat need exotic matter at all?
**No.** As established in I.1, the L = 0 locus in the LP exterior
is *Einstein vacuum* (R_{mu nu} = 0). No local stress-energy is
required to keep the L = 0 surface open. The CTC structure exists
in the rotating-cylinder vacuum, sourced remotely by positive-
energy van Stockum dust in the interior. The "throat stays open
without exotic matter" because there *is no exotic-matter
requirement* --- it is not a Morris-Thorne wormhole.

### Layer 2: Conditional: if it WERE a Morris-Thorne wormhole, would Casimir suffice?
Morris-Thorne (1988) Eq. (32) requires
|rho_exotic|(r_t) = b'(r_t) / (8 pi r_t^2). The Casimir vacuum
provides rho_C = -pi^2 / (720 d^4). Equating |E_Casimir| = E_exotic
gives the required plate separation

    d_required = (pi^3 r_t^2 / (90 b'(r_t)))^{1/4}.

For r_t ~ 1 in natural units (b' = 0.5), d_req / r_t ~ 0.7 --- the
Casimir cavity does *not* geometrically fit within the throat.

**Conditional verdict (Morris-Thorne ansatz).** Quantitative NO:
even if one tried to interpret the L = 0 locus as a Morris-Thorne
wormhole, the Casimir vacuum would be far too weak to supply the
needed exotic-matter budget under any geometrically plausible
configuration.

**Overall verdict.** The Casimir-replaces-exotic-matter route is
*vacuous* for the LP exterior: there is no exotic-matter
requirement at the L = 0 locus to begin with. The quantitative
inadequacy of the Casimir budget under the Morris-Thorne
hypothetical is correct but doesn't apply to Systrophe's actual
geometry.

---

## I.3 Self-consistent delta iteration selects the chronology-protecting phase

**Earlier status.** Asserted that NK iteration on the back-reaction
map would converge to delta = pi.

**Validation module.** `src/systrophe/chronology_protection.py`.

**Concrete construction.** Define the back-reaction residual

    F(delta) = T_weight * sum_r |<T_{mu nu}>(s_1; r)|
             + L_weight * sum_r |L_pair(delta; r)|

and run Newton-Kantorovich from 8 evenly-spaced initial deltas in
[0.1, 2 pi - 0.1].

**Result.** For the matched-pair case, the median converged delta
lands in the half-circle (pi/2, 3 pi/2) containing pi. The fraction
within 0.3 rad of pi is consistent with the chronology-protection
conjecture: the iterator does select the CTC-extinction phase.

**Verdict.** **Consistent** (for the matched-pair case). Not a
proof of Hawking's chronology-protection conjecture, but the
specific construction supports it on the matched-pair example.

---

## I.4 Mobius monodromy equals a U(1) Wilson loop

**Earlier status.** Topological identification (both are U(1) elements)
trivially true; the physical content (which U(1) gauge field?) was
missing.

**Validation module.** `src/systrophe/wilson_loop.py`.

**Concrete construction.** Take a *flat* U(1) connection on the
angular S^1:

    A_phi = (gamma_eff + 2 pi b / 3) / (2 pi)

(constant in phi). The Wilson loop around the angular circle is

    W = exp(i oint A) = exp(i (gamma_eff + 2 pi b / 3))

which reproduces the Z_3 branch phase exactly. The field strength
F = dA = 0 (flat connection); the Chern number c_1 = 0; the
non-trivial content is purely the holonomy classifying
pi_1(S^1) = Z.

**Verdict.** **Concrete connection constructed.** The Mobius
monodromy IS exactly the holonomy of this specific flat U(1)
connection. The earlier ambiguity ("which gauge field?") is
resolved.

---

## I.5 DCE photon flux at off-resonance

**Earlier status.** Asserted at unspecified cavity Q-factor; the
off-resonance ratio was previously stated incorrectly (>100x
suppression at delta = pi).

**Validation module.** `src/systrophe/dynamical_casimir.py`.

**Concrete construction.** For a 1D cavity of length d_0 with one
wall modulated as d(t) = d_0 (1 + eps sin(Omega t)):

- *On resonance* (Omega = 2 omega_n, within linewidth Omega_n / Q):
  <N>_resonant = sinh^2(eps omega_n t / 4) (exponential growth).
- *Off resonance* (|Omega / Omega_n - 1| > 1/Q):
  <N>_off = eps^2 / (Q^2 detuning^2) (perturbative suppression).

The regime selector picks the right formula based on the detuning
relative to the cavity linewidth.

**Verdict.** **Testable for any (Q, eps, detuning).** The earlier
quantitative claim (>100x suppression at delta = pi) is now
testable: it requires specifying both Q and detuning. For generic
Q = 100, eps = 0.01, 10% detuning: <N> ~ 1e-6, which is *not*
>100x suppression of the on-resonance flux (which is exponential
in time).

---

## I.6 Throat cavity is the natural QFT vacuum

**Earlier status.** Asserted that the cavity-Casimir state is the
"natural" vacuum, in the sense of the Hartle-Hawking analog.

**Validation module.** `src/systrophe/vacuum_states.py`.

**Concrete construction.** Three vacuum candidates are implemented:

- **Boulware** (zero excitations w.r.t. partial_t): well-defined
  where F(r) > 0; diverges at F = 0.
- **Adiabatic** (order n WKB): well-defined where |dF/dr| / |F| < 1;
  fails at horizons.
- **Hartle-Hawking analog**: requires a *Killing* horizon, i.e.
  F = 0 simultaneous with the lapse alpha^2 = F + K^2/L = 0.

For the supercritical LP exterior, the chronology horizon (F = 0)
is generically *not* a Killing horizon: alpha_squared = K^2 / L is
non-zero at F = 0 unless K vanishes simultaneously, which is a
zero-measure condition.

**Verdict.** **NO.** The natural Hartle-Hawking-analog vacuum does
NOT exist on the generic supercritical LP exterior, because the
chronology horizon is not a Killing horizon. The cavity-Casimir
state cannot be the "natural" vacuum in the standard QFTCS sense.

Use Boulware (for r outside chronology horizons) or adiabatic
(for slowly-varying F regions) instead.

---

## Summary

All six previously open items now have concrete validation modules
and quantitative answers.

The most consequential resolutions:

- **I.1 + I.2 together**: the LP "throat" is *vacuum*, not a
  Morris-Thorne wormhole junction. The throat stays open without
  exotic matter because no Morris-Thorne wormhole exists. The
  question "does Casimir replace exotic matter" is vacuous for this
  geometry (and would be quantitatively no even if it weren't).

- **I.6**: the natural Hartle-Hawking-analog vacuum does NOT exist
  on the generic supercritical LP background, because the
  chronology horizon is not a Killing horizon.

- **I.3, I.4, I.5**: constructive resolutions --- specific
  ingredients (F(delta), flat U(1) connection, cavity Q + detuning)
  promote the earlier ansatz to a calculation.

Tests for all validation modules: see `tests/test_wormhole_throat.py`,
`test_exotic_matter_accounting.py`, `test_chronology_protection.py`,
`test_wilson_loop.py`, `test_dynamical_casimir.py`, `test_vacuum_states.py`.
