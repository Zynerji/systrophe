# arXiv submission packet

This document is the working checklist for submitting the Knopp Drive paper to arXiv. Once the user is ready to submit, they perform the steps below themselves from their own arXiv account at `https://arxiv.org/submit`.

---

## Files to upload

Primary source archive (tarball or single .tex):

```
knopp_drive_arxiv.tex                      # main manuscript
figures/knopp_tipler_gate.pdf              # Figure 1
figures/knopp_E_neg_vs_r.pdf               # Figure 2
figures/knopp_marrakesh_batch6.pdf         # Figure 3
figures/knopp_warp_comparison.pdf          # Figure 4
```

All four figures are auto-regenerable via `paper/generate_knopp_drive_figures.py` from the publicly-available JSON results in the GitHub repository.

The latex compiles cleanly with `pdflatex -interaction=nonstopmode knopp_drive_arxiv.tex` (twice for cross-references).

## Metadata fields

| Field | Value |
|---|---|
| **Title** | The Knopp Drive: A Four-Mechanism Composite Warp Engineering Bound with Hardware Confirmation on IBM Quantum `ibm_marrakesh` |
| **Authors** | Christian Knopp |
| **Affiliation** | Independent researcher |
| **Email** | `cknopp@gmail.com` |
| **Abstract** | (copy from paper, ~10 lines / 250 words) |
| **Primary category** | `gr-qc` |
| **Cross-list** | `quant-ph`, `hep-th` |
| **License** | arXiv non-exclusive license (default) |
| **Comments** | 9 pages, 4 figures, 1 table. Open-source code at https://github.com/Zynerji/systrophe |
| **MSC class** | 83C57 (Black holes), 83C75 (Topology, causal structure) |
| **ACM-class** | (none) |
| **Report-no** | (none) |
| **DOI** | (none yet; will be assigned by arXiv) |

## Abstract for arXiv (verbatim)

> We introduce the Knopp Drive, a composite warp-drive engineering object that combines four independent shortcut mechanisms each validated against an address-space novelty catcher. The mechanisms are: (i) Tipler CTC-band gating, in which the supercritical Lewis–Papapetrou exterior of a rotating dust cylinder supplies the cone-tilt for a Krasnikov-style causal corridor, reducing the exotic-matter requirement to exactly zero inside each CTC band; (ii) Krasnikov tube embedding in the bubble shell, providing a directed causal corridor; (iii) Q-cavity feedback amplification, in which a parametric resonator in the shell trades instantaneous-impulse infinite power for sustained drive power scaling as 1/Q^2; and (iv) horn-toroidal twist, in which a small theta-dependent ADM-mass asymmetry yields a continuous steering dipole p ~ R^2 epsilon |m_ADM|. The four reductions compose multiplicatively in the exotic-matter budget. The composite respects the Pfenning–Ford quantum inequality by construction. We confirm the Knopp Drive's headline shortcut on the IBM Quantum 156-qubit Heron-r2 processor ibm_marrakesh: a four-qubit encoding of the composite reproduces the predicted CTC-band gating at total-variation distance <= 0.05 across an 8-point radial sweep, with the band exit identified by the catcher as a sharp Hamming transition (step 12 vs median 0). The Knopp Drive converts the canonical "infinite negative energy" warp-drive requirement into a finite, geometry-bounded engineering budget that vanishes whenever the craft's worldline lies inside a geometric CTC band. At Earth–Mars distance (L=0.52 AU equivalent in geometric units), the journey lies entirely inside the first CTC band of a unit supercritical cylinder, so the composite exotic-matter requirement is exactly zero.

## Submission steps (manual)

1. Go to `https://arxiv.org/submit` (logged in as `cknopp@gmail.com`).
2. Click "Start new submission".
3. Select **primary category** `gr-qc` (General Relativity and Quantum Cosmology).
4. Cross-list categories: `quant-ph`, `hep-th`.
5. License: arXiv non-exclusive license (the default).
6. Upload the source archive containing `knopp_drive_arxiv.tex` and the `figures/` directory.
7. Paste the metadata fields above.
8. Preview the compiled PDF (arXiv compiles on submission).
9. Verify the figures render correctly.
10. Submit. (You will receive an arXiv ID within 24 hours.)

## Post-submission

- Add the arXiv ID and DOI to the GitHub README badge.
- Update the bibtex entry in `paper/knopp_drive.tex` to cite the arXiv preprint.
- Tweet the arXiv link (optional).
- Email the abstract to relevant researchers (Bobrick, Martire, Lentz, IBM Quantum team).

## Journal submission strategy (post-arXiv)

After the preprint is live and has accumulated some citations/discussion (recommended waiting period: 1-3 months):

1. **Primary target: Classical and Quantum Gravity** (IOPscience).
   - Submission portal: `https://mc04.manuscriptcentral.com/cqg-iop`
   - Strong fit with the warp-drive literature (CQG has published Alcubierre, Krasnikov, Pfenning–Ford, Van Den Broeck, Lentz, Bobrick–Martire).
   - Expected review time: 3-6 months.

2. **Backup target: Physical Review D**.
   - Submission portal: `https://prd.aps.org`
   - Particularly strong fit if reviewer feedback steers toward the analytic GR / Pfenning–Ford direction.

3. **Companion paper option**: split the hardware-confirmation aspect into a separate manuscript for **Quantum Science and Technology** (IOPscience), highlighting the catcher methodology and the quantum-circuit encoding. This is recommended if the main paper feels too long after the first round of revisions.

## Anticipated metrics

If accepted in CQG and given the topic's appeal (warp drives + IBM Quantum) plus open-source code:

- Citations in year 1: ~5-15 (warp-drive papers typically accumulate slowly).
- arXiv views in week 1: 100-500.
- Press coverage potential: high (any warp-drive paper with hardware confirmation has popular-press appeal).

The arXiv preprint should be cited in the patent provisional (`patents/knopp_drive_provisional.tex`) once the arXiv ID is assigned.
