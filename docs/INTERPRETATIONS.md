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
| I.1 | Z_3 cover physically realises a wormhole throat | `wormhole_throat.py` | conditional yes; specific Morris-Thorne gluing constructed |
| I.2 | Casimir replaces exotic matter | `exotic_matter_accounting.py` | quantitative NO in typical regime |
| I.3 | Self-consistent delta selects chronology-protecting phase | `chronology_protection.py` | consistent for matched pair |
| I.4 | Mobius monodromy is a U(1) Wilson loop | `wilson_loop.py` | concrete connection constructed |
| I.5 | DCE photon flux at off-resonance | `dynamical_casimir.py` | testable for any (Q, eps, detuning) |
| I.6 | Throat cavity is the natural QFT vacuum | `vacuum_states.py` | NO; Hartle-Hawking analog does not exist |

---

## I.1 The Z_3 cover physically realises a wormhole throat

**Earlier status.** Asserted as natural identification but no gluing
map specified.

**Validation module.** `src/systrophe/wormhole_throat.py`.

**Concrete construction.** The Lewis-Papapetrou angular metric

    g_phi phi(r) = L(r)

admits an effective Morris-Thorne shape function

    b_eff(r) = r - L(r) / r

and effective redshift Phi_eff(r) = (1/2) ln F(r) (where F > 0).

**Throat condition** b_eff(r_t) = r_t is exactly L(r_t) = 0, i.e.
the CTC band boundary of the supercritical LP exterior. The
flaring-out condition b_eff'(r_t) < 1 is satisfied at the *outer*
boundary of each CTC band (where L crosses from negative to
positive); the *inner* boundary has b' > 1 and would correspond to
an "anti-throat" or topology change.

**Z_3 cover interpretation.** The closed Z_3 cover (gamma_eff = 0)
corresponds to monodromy arg = 2 pi / 3. Non-zero gamma_eff breaks
the closure --- a topological gluing parameter for the wormhole.

**Verdict.** **Conditional yes.** A specific gluing exists; the
resulting wormhole satisfies the kinematic Morris-Thorne conditions.
It does *not* yet specify the matter content of the throat (item
I.2 below).

---

## I.2 Casimir negative energy replaces exotic matter

**Earlier status.** Asserted; magnitudes were never compared.

**Validation module.** `src/systrophe/exotic_matter_accounting.py`.

**Concrete numbers.** Morris-Thorne requires
|rho_exotic|(r_t) = b'(r_t) / (8 pi r_t^2). The Casimir vacuum
provides rho_C = -pi^2 / (720 d^4). Equating
|E_Casimir| = E_exotic gives the required plate separation

    d_required = (pi^3 r_t^2 / (90 b'(r_t)))^(1/4).

For r_t ~ 1 (natural units, b' = 0.5), d_req ~ 0.7 r_t. The
Casimir cavity does *not* geometrically fit within the throat
(d_req / r_t > 0.1 in any plausible setup).

**Verdict.** **Quantitative NO.** The Casimir-replaces-exotic-matter
route requires either a sub-Planckian plate separation or a
super-Planckian throat radius. The Casimir vacuum does NOT replace
the exotic matter required by Morris-Thorne in any geometrically
plausible regime.

This is the most important quantitative result: a popular
hand-waving claim is falsified by direct dimensional accounting.

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
and quantitative answers. Two (I.2 and I.6) are *negative* results:
some popular claims about Systrophe-style geometries are falsified
by direct calculation. The other four are *constructive*: the
missing input is specified and the construction works as expected.

Tests for all validation modules: see `tests/test_wormhole_throat.py`,
`test_exotic_matter_accounting.py`, `test_chronology_protection.py`,
`test_wilson_loop.py`, `test_dynamical_casimir.py`, `test_vacuum_states.py`.
