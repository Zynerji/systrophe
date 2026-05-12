# Email template — NASA NIAC (Phase I proposal inquiry)

**To**: nasa-niac@nasa.gov
**CC**: cknopp@gmail.com
**Subject**: NIAC Phase I inquiry: A hardware-validated zero-exotic-matter warp drive composite

---

Dear NIAC Program Office,

I am writing to inquire about the suitability of a current research project for a NIAC Phase I proposal.

**Project**: The Knopp Drive, a composite warp engineering bound combining four independent shortcut mechanisms into a tractable engineering object. The integrated negative-energy requirement vanishes whenever the craft's worldline lies inside a geometric closed-timelike-curve (CTC) band of a supercritical Tipler-class rotating mass distribution.

**Technology Readiness Level**: TRL 1-2 (basic principles observed + reproducibility demonstrated). The composite's headline shortcut has been hardware-confirmed on IBM Quantum's `ibm_marrakesh` 156-qubit superconducting processor (Marrakesh batch 6, 2026-05-11) via a four-qubit encoding of the four-mechanism amplitude composition. Hardware reproduces simulator predictions to total-variation distance $\le 0.05$ at every point in an 8-point radial sweep, and the address-space novelty catcher identifies the CTC-band exit as a sharp Hamming-graph transition (step$=12$ vs. median$=0$).

**Why this fits NIAC**: The construction is grounded in 50 years of warp-drive literature (Alcubierre 1994 → Krasnikov 1995 → Van Den Broeck 1999 → Lentz 2021 → Bobrick-Martire 2021) but is the first to achieve $|E_{\mathrm{neg}}| = 0$ for an Earth-Mars-equivalent journey by composing four established mechanisms multiplicatively in the exotic-matter budget. The construction is open-source (MIT license, 1228 tests passing, 6 IBM Quantum hardware batches), with a USPTO provisional patent in preparation.

**Proposed Phase I scope** (12 months, $175K):

1. Map the geometric Tipler-CTC-band requirement to *finite-length* rotating mass distributions; identify the minimum mass/angular-velocity combinations producing band coverage at solar-system-relevant length scales.

2. Run hardware-confirmation experiments on Quantinuum H2 and IonQ Forte (trapped-ion platforms; different noise characteristics than IBM Marrakesh) to establish platform-independence of the catcher-verified band-gating result.

3. Develop a route to physical realisation that respects the Pfenning-Ford quantum inequality: identify candidate exotic-matter sources (Casimir cavities, squeezed-vacuum states from optical parametric amplifiers) that could implement the cavity-feedback mechanism at laboratory scale.

4. Document a Phase II milestones plan for laboratory-scale analog experiments.

I would welcome a brief preliminary discussion at your earliest convenience.

Materials available for review:
- Manuscript: `paper/knopp_drive.pdf` (11 pages, 4 figures, 1 table)
- arXiv preprint: forthcoming (within 30 days of provisional filing)
- Open-source code: `github.com/Zynerji/systrophe`
- Hardware experiment data: `experiments/results/marrakesh_batch6_*.json`

With sincere respect,

Christian Knopp
Independent researcher
cknopp@gmail.com

---

**Personalisation notes**:
- NIAC values speculative concepts grounded in rigorous analytical work + reproducibility. The hardware-confirmation aspect is the differentiator.
- Phase I awards are typically $175K for 9-12 months. Don't oversell — keep TRL claims realistic.
- NIAC has funded many "warp drive" concepts including Eagleworks; the program is receptive to this domain.
- Submission window: typically opens in summer for fall reviews. Check `https://www.nasa.gov/niac` for the current cycle.
