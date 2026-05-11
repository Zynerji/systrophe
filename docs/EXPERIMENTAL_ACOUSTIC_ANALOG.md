# BEC-vortex experimental design for a rotating-cylinder acoustic analog

This document is an experimental design proposal: how to realise the
Systrophe rotating-cylinder analog in a Bose-Einstein-condensate (BEC)
or rotating-water apparatus, and what observables predict positive
detection.

The mathematical foundation is in `src/systrophe/acoustic_metric.py`
(Unruh 1981 acoustic-metric mapping):

    ds^2_acoustic = (rho / c) [ -(c^2 - v^2) dt^2 - 2 v . dx dt + dx . dx ]

with the LP exterior identification

    rho_acoustic = sqrt(L),    v_phi = K / sqrt(L),    c^2 - v^2 = F.

The acoustic horizon at c^2 = v^2 coincides exactly with the
chronology horizon F = 0, and the *supersonic* region (c^2 < v^2)
exactly with the *CTC* region.

---

## 1. Target apparatus

Two routes are currently feasible:

### 1A. BEC sonic-horizon apparatus (Steinhauer-style)
- Atomic species: Rb-87 or Na-23.
- Trap: cylindrical or elongated cigar trap with axial step in
  scattering length (the Steinhauer 2016/2019 step-scaling).
- Sonic horizon: created by a moving step or a sweeping potential.
- Adaptation: introduce *rotation* via a Laguerre-Gauss stirring beam
  carrying angular momentum L = hbar. A single vortex gives the
  v_phi(r) = hbar / (m r) profile of a quantised vortex.

### 1B. Rotating water tank (Weinfurtner/Faccio-style)
- Bathtub-vortex flow over a draining hole.
- Sonic horizon: where azimuthal flow speed equals the surface-wave
  speed c = sqrt(g h).
- Adaptation: control draining rate to set v_phi(r); the analog of the
  cylinder radius R is the draining-hole radius.

We focus on 1A because it has the cleanest Bogoliubov-mode spectrum
and the closest match to the Tipler exterior structure.

---

## 2. Mapping LP -> BEC parameters

Let the BEC have:
- atomic mass: m
- average density: rho_0 (atoms / m^3)
- s-wave scattering length: a_s
- effective transverse size: sigma_perp

Then in the local-density approximation the sound speed is

    c = sqrt(4 pi hbar^2 a_s rho_0 / m^2).

A single-quantum vortex in this BEC has azimuthal flow

    v_phi(r) = hbar / (m r).

The acoustic-horizon radius r_H satisfies v_phi(r_H) = c, giving

    r_H = hbar / (m c) = healing_length.

This is order ~ 0.2-1 micron in typical Rb-87 BECs (xi_BEC ~ 0.5 um at
n = 10^14 cm^-3).

### 2.1 Tipler-cylinder analog from the vortex

Identify the LP cylinder radius R with the *vortex core radius*
(~ a few healing lengths). The Tipler critical condition omega R > 1/2
becomes, in BEC variables:

    (hbar / (m R^2)) * R > c / 2

i.e. v_phi(R) > c / 2, or equivalently R < 2 hbar / (m c) = 2 xi_BEC.

So *every quantised vortex in a BEC is automatically supercritical*
in the Systrophe sense (the vortex core is smaller than 2 xi_BEC).
The supersonic region (v_phi > c) is the *CTC analog*; the region
v_phi < c is the chronology-protected analog.

### 2.2 Z_3 triple-vortex configuration

For a Z_3-cover analog, place three identical co-rotating vortices
at the vertices of an equilateral triangle of side d. Far from the
triple, the combined flow is

    v_total(r) = sum_{j=1}^{3} v_phi^{(j)}(r_j)

and the corresponding acoustic metric exhibits Z_3 symmetry. The
relative phase between vortices is the Z_3 phase parameter
gamma_eff / (2 pi) ; vortices with identical winding give
gamma_eff = 0.

This realises the Z_3 Mobius cover construction of
`src/systrophe/dinos_bridge.py` (specifically the gamma_eff=0,
3-branch case) as a physical lab geometry.

### 2.3 Linear-pair (SystrophePair) variant

For a 2-vortex configuration with adjustable relative *phase* (one
vortex with winding +1, the other +1 at a different topological
charge separation), one can engineer the *anti-phase extinction*
documented in `paper/systrophe_time_travel.tex`. Specifically: a
counter-rotating vortex pair (winding +1 and -1) has v_phi which
cancels on the line bisecting the two cores. This is the BEC
realisation of delta = pi extinction.

---

## 3. Predicted observables

### 3.1 Analog Hawking temperature
The acoustic-Hawking temperature at the sonic horizon is

    T_H_acoustic = kappa / (2 pi)

with surface gravity kappa = (1/2) |dF/dr|_{r_H}. For our LP
exterior (Systrophe quantum_diagnostics module), this is computable
in closed form once (omega, R) are fixed. In BEC units, T_H is on
the order of nano-kelvin for typical (xi_BEC, c) values --- within
the temperature resolution of modern BECs.

### 3.2 Phonon pair-correlation signature
The hallmark Steinhauer-style observable: the density-density
correlation function

    g_2(z, z') = <delta_n(z) delta_n(z')>

exhibits a characteristic "tongue" of correlations *outside* the
sonic horizon paired with anti-correlations *inside*. For our
rotating analog, this becomes a *radial* correlation function
g_2(r, r') with the same structure across r = r_H, plus an
additional *angular* anti-correlation between the three Z_3
branches at separation 2 pi / 3 in phi.

The Z_3 cross-correlation is the smoking-gun signature distinguishing
a triple-vortex from three independent single vortices.

### 3.3 Acoustic-CTC analog: closed null-acoustic curves
The supersonic region (v_phi > c) admits closed *acoustic* null
characteristics. These are not closed *timelike* curves in the
underlying lab spacetime --- the lab is Minkowski. They are closed
sonic rays in the *acoustic* metric.

A phonon wavepacket launched into the supersonic region will, in
the eikonal limit, follow these closed rays. Observationally: a
density perturbation injected at (r > r_H, phi = 0) at time t = 0
will reappear at (r > r_H, phi = 2 pi) at the *same* lab time t = 0
(the acoustic time-loop closes), but *de-phased* by the eikonal
phase integral.

This is the analog of the rotating-cylinder closed-timelike-curve
revolution. The de-phasing is calculable from Systrophe's
geodesic.omega_for_target_coord_time module.

### 3.4 DCE-like phonon flux at the horizon
A periodic modulation of the vortex angular momentum (driven by a
modulated stirring beam) produces phonon pairs in the supersonic
region by the acoustic analog of the dynamical Casimir effect (DCE).
The on-resonance flux is calculable from the Floquet quasi-energies
of `src/systrophe/floquet_mobius.py`.

The relevant resonance condition is Omega_drive = e_1 - e_0 where
e_b are the static phonon energies on the three Z_3 branches.

---

## 4. Concrete experimental parameters (numerical example)

For Rb-87, n = 10^14 cm^-3, a_s = 5.3 nm:
- m = 1.45e-25 kg
- xi_BEC = hbar / sqrt(2 m mu) = 4 * 10^-7 m = 0.4 um
- c (sound speed at this density) = 4 * 10^-3 m/s
- T_H_acoustic = hbar c / (2 pi xi_BEC k_B) = ~ 0.5 nK

For T_BEC = 50 pK (state of art at Stanford / NIST), T_H = 500 pK
is *measurable* with O(10) signal-to-noise per shot, requiring ~ 30
shots to resolve the spectrum. This is a few hours of data acquisition.

For Z_3 triple-vortex (Steinhauer-equipment-style): three
Laguerre-Gauss beams at 120-degree azimuthal phases, intensity
matched, focus separation d ~ 5 * xi_BEC ~ 2 um.

---

## 5. Risk register

- **Damping**: phonon damping (Beliaev/Landau processes) can
  obscure Hawking signal. Steinhauer mitigates by averaging over
  ~ 1000 shots and using narrow Bragg windows.
- **Vortex pinning**: three vortices at non-symmetric positions
  drift. Mitigation: anchor to a 3-fold symmetric optical lattice
  (Hadzibabic-style).
- **Eikonal vs full-field**: the closed-null-curve prediction is
  eikonal. Full BdG simulation should be done before lab attempt.

---

## 6. Code reference

The full mathematical machinery is in:

- `src/systrophe/acoustic_metric.py`: c^2 - v^2 = F identification
- `src/systrophe/quantum_diagnostics.py`: T_H_acoustic from kappa
- `src/systrophe/floquet_mobius.py`: DCE Floquet resonance
  condition
- `src/systrophe/horned_torus.py`: regular + inverted horn modes
  for vortex-core profiles
- `examples/grok_verification.py`: closed-form Tipler verification
- `paper/systrophe_qft_on_ctc.tex`: detailed derivation (Whitepaper
  II, forthcoming v0.13)

---

## 7. Next steps

1. Numerically simulate the Z_3 triple-vortex BdG modes (open
   project; would benefit from a BEC theory collaborator).
2. Identify the smallest signal that distinguishes the analog
   Hawking signal from background thermal phonons in a 3-vortex vs
   1-vortex configuration.
3. Send this document to one or more groups in `CONTACTS.md`
   (Tier 1) for assessment of experimental feasibility.

---

## References

- W. G. Unruh, "Experimental black-hole evaporation?",
  Phys. Rev. Lett. 46 (1981) 1351.
- J. Steinhauer, "Observation of self-amplifying Hawking radiation
  in an analogue black-hole laser", Nature Physics 10 (2014) 864.
- J. Steinhauer, "Observation of thermal Hawking radiation and its
  temperature in an analogue black hole", Nature 569 (2019) 688.
- S. Weinfurtner et al., "Measurement of stimulated Hawking emission
  in an analogue system", Phys. Rev. Lett. 106 (2011) 021302.
- T. Torres et al., "Rotational superradiance from a laboratory
  vortex", Nature Physics 13 (2017) 833.
