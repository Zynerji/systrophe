# arXiv submission plan for Systrophe Whitepaper II

Status of `paper/systrophe_qft_on_ctc.tex` and v0.14.0 codebase as of
2026-05-10; plan to submit Whitepaper II to arXiv.

---

## Categories

**Primary**: `gr-qc` (General Relativity and Quantum Cosmology)
- The core subject is renormalised QFT on a CTC-containing GR background.

**Cross-list**:
- `quant-ph` (Quantum Physics) - because of the D-CTC module and the
  anomaly-inflow + Floquet-Mobius work that has direct quantum-info
  content.
- `hep-th` (High Energy Physics - Theory) - for the Hadamard
  renormalisation, Brown-Maclay Casimir, anomaly inflow. Optional;
  decide based on referee community.
- `cond-mat.quant-gas` (Quantum Gases) - if the BEC-vortex acoustic
  analog design is in the paper. Recommend skipping cond-mat for the
  initial submission and breaking out the BEC design as a separate
  short note for that audience.

---

## Author endorsement

Christian Knopp is not currently an arXiv-endorsed author in `gr-qc`.
Steps to obtain endorsement (per arXiv policy):

1. Identify a `gr-qc`-endorsed colleague willing to endorse this
   submission. Required: someone with at least 2 submissions to
   `gr-qc` in the past 5 years.
   
   **Candidates** (from `CONTACTS.md`):
   - Matt Visser (Wellington) - actively publishes in `gr-qc`,
     textbook on wormholes makes this his wheelhouse.
   - Francisco Lobo (Lisbon) - publishes regularly in `gr-qc` and
     `Phys. Rev. D` on wormhole physics.
   - Silke Weinfurtner (Nottingham) - publishes in `gr-qc` /
     `cond-mat` cross-listed analog-gravity work.

2. Before contacting, finalise the abstract (see below) and double-
   check that the paper is in arXiv-acceptable LaTeX form (single
   `.tex` file, all figures embedded or absent).

3. Send a brief email to the chosen endorser linking to the GitHub
   repo and the PDF, including the abstract and the request for
   endorsement.

4. arXiv generates an endorsement code; the endorser submits it via
   their arXiv account.

---

## Abstract (final version, for arXiv form)

```
We extend the open-source Systrophe package
(https://github.com/Zynerji/systrophe) from a classical-GR Tipler-pair
simulator to a quantum-field-theoretic laboratory. New infrastructure
supports (i) renormalised stress-energy tensor on the Lewis-Papapetrou
exterior with exact recovery of the conformal anomaly K/(2880 pi^2) in
vacuum; (ii) closed-form Atiyah-Patodi-Singer eta-invariant on the Z_3
Mobius cover with verified eta_0 + eta_1 + eta_2 = 0 anomaly closure
and Callan-Harvey inflow balance; (iii) joint Floquet quasi-energies
on the (t, b) Brillouin product torus with Z_3 cyclic-permutation
symmetry; (iv) Unruh acoustic-metric mapping identifying the chronology
horizon as an acoustic horizon, with the gravitational and acoustic
Hawking temperatures coinciding to machine precision; (v) Brown-Maclay
<T_munu> at a Casimir cavity with Lewis-Papapetrou curvature correction;
(vi) Newton-Kantorovich back-reaction solver. The package contains 406
passing tests and ships the math via a clean module-per-construction
layout. Open speculative items (wormhole gluing, vacuum-selection
criterion, gauge field for Mobius=Wilson loop) are documented as ansatz-
level interpretations needing external input. A BEC-vortex experimental
design proposal is included as a supplementary document.
```

(193 words; arXiv limit is 1920 characters, this is well within.)

---

## Title

**Systrophe II: Quantum field theory on a Tipler-pair background**

70 characters, clean.

---

## MSC / PACS

- **MSC 2020**: 83C57 (Black holes), 81T20 (Quantum field theory on
  curved space), 83C50 (Electromagnetic fields).
- **PACS** (legacy): 04.62.+v (QFT in curved spacetime), 04.20.Gz
  (Spacetime topology and structure), 03.65.Vf (Phases: geometric;
  Berry phase; etc.).

---

## License

The arXiv submission license should match the repo license. Repo is
MIT. For arXiv, prefer:

- **arXiv non-exclusive license** (default) - allows free
  redistribution from arXiv.

Do not submit under "arXiv perpetual non-exclusive license" because
that prevents some downstream uses inconsistent with MIT.

---

## Pre-submission checklist

- [ ] Author endorsement obtained
- [ ] `pdflatex` clean compile (no overfull boxes, no missing refs)
- [ ] Final test count in abstract matches `pytest` output
- [ ] BibTeX entries verified against published references
- [ ] Companion repo tagged at the corresponding version (v0.14.0)
- [ ] Whitepaper I cited in Whitepaper II
- [ ] Acknowledgements section accurate (Anthropic Claude attribution)
- [ ] No use of unreviewed Grok-chat material in the body
- [ ] Source `.tex` is self-contained (no external `\include`)
- [ ] License section in source matches MIT

---

## After acceptance

1. arXiv ID will be in form `2605.NNNNN` (May 2026).
2. Add the arXiv ID + DOI to README, CITATION.cff, and CHANGELOG.
3. Tag a release v0.14.0 on GitHub with the arXiv link in the release
   notes.
4. Send a brief notification to each `CONTACTS.md` Tier 1 recipient
   linking to the arXiv version.

---

## Companion deposit: Whitepaper I

Whitepaper I (`systrophe_time_travel.tex`) was written first and is
the classical-GR ground for Whitepaper II. Decision: deposit only
Whitepaper II to arXiv initially; reference the repo for Whitepaper
I. If a referee insists on the classical paper for completeness, it
can be deposited as a follow-up with cross-citation to II.

Rationale: II contains all of I's results in the references section
plus the quantum extensions that constitute the novel material.
Depositing both would double the submission overhead with low
additional return.
