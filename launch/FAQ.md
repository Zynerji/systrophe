# Frequently asked questions

Top 30 expected questions for the Knopp Drive public launch. Curated from anticipated reviewer concerns (`arxiv/responses_to_anticipated_reviews.md`), podcast-brief preparation (`launch/podcast_brief.md`), and predicted social-media engagement (`launch/twitter_thread.md`).

---

## What it is

### Q1. What is the Knopp Drive?

A composite warp-drive engineering object combining four independent mechanisms multiplicatively in the exotic-matter budget: (1) Tipler closed-timelike-curve (CTC) band gating, (2) Krasnikov tube embedding, (3) Q-cavity feedback amplification, and (4) horn-toroidal twist (steering). Inside a Tipler CTC band, the integrated exotic-matter requirement is exactly zero.

### Q2. Who built it?

Christian Knopp, an independent researcher, designed the composite and implemented the open-source framework. The hardware experiment was run on IBM Quantum's `ibm_marrakesh` 156-qubit Heron-r2 processor under the Zynerji IBM Quantum instance.

### Q3. What "warp drive" means here?

A localized modification of Minkowski spacetime that allows a craft to traverse macroscopic distances in arbitrarily short proper time. This usage follows Alcubierre (1994) and the subsequent literature; it is NOT science-fiction "warp drive" with humans on board.

### Q4. Why is it called the "Knopp Drive"?

Named for myself, the sole inventor and open-source maintainer.

---

## How it works

### Q5. How does it achieve zero exotic matter?

Inside a Tipler closed-timelike-curve band of a supercritical rotating mass distribution, the local frame-dragging in the Lewis-Papapetrou exterior already provides the spacetime cone-tilt that a Krasnikov tubular spacetime engineering would otherwise need to produce with exotic matter. The geometric tilt cancels the engineered tilt. Composite exotic-matter requirement = 0.

### Q6. What is a Tipler CTC band?

A radial range in the supercritical Lewis-Papapetrou exterior of a fast-rotating mass cylinder where the metric component $L(r) < 0$, indicating closed timelike curves (cycles in spacetime that close back on themselves in time). Tipler 1974 identified these as a consequence of GR + rotation.

### Q7. What is a Krasnikov tube?

A 1+1D spacetime modification along a worldline that allows backward causal travel inside the tube only. Introduced by Sergei Krasnikov in 1995. Requires negative energy in the tube wall.

### Q8. What does "Q-cavity feedback amplification" mean?

The bubble shell is built as a high-quality-factor parametric resonator. A pump at twice the shell's natural frequency drives the wall into a squeezed-vacuum state where $\langle T_{tt} \rangle$ is negative. The cavity factor $Q$ trades amplitude for sustained duration along the Pfenning-Ford quantum-inequality bound. Sustained drive power scales as $1/Q^2$.

### Q9. What is "horn-toroidal twist"?

A small azimuthal asymmetry $\epsilon$ in the bubble shell's ADM-mass distribution. The twist axis $\theta_0$ sets the steering direction; the magnitude $\epsilon \in [0, 1)$ sets the steering vector strength. Produces a continuous steering dipole on the exotic-matter shell.

### Q10. Why does the composition work multiplicatively?

In the linearised regime where each mechanism's metric perturbation is small, the four contributions to the exotic-matter density compose as a product (cross-terms are formally $O(G^2)$ and not modelled). The product structure is what allows ZERO total when any one factor is zero.

---

## Is this physically real?

### Q11. Is this a working warp drive?

**No**. The construction is at Technology Readiness Level 1-2 (basic principles observed + reproducibility demonstrated). What has been built is a mathematical framework with an open-source implementation. Physical realisation is not on the table.

### Q12. What was confirmed on IBM Quantum?

A 4-qubit circuit that encodes the *amplitude composition* of the four-mechanism construction. The hardware reproduces the predicted band-gated extinction (i.e., the data qubit's bias drops to ~50/50 inside the CTC band and recovers to ~60/40 outside). This is amplitude-level confirmation, not a literal realisation of exotic matter on the chip.

### Q13. Does this require infinite energy?

**No**. The Q-cavity feedback respects the Pfenning-Ford inequality. The $|E_{shell}| \cdot \tau$ product is bounded below; we trade vertical for horizontal axis along the bound, never crossing it. Sustained drive power can be made small by making $Q$ large.

### Q14. Does this violate causality?

**Not addressed in this paper**. The construction is pre-quantum. The chronology-protection conjecture (Hawking 1992) and Deutsch-CTC self-consistency (Deutsch 1991) are not engaged. Closed timelike curves are present in the underlying Tipler exterior; what they mean for causality is the same open question Tipler raised in 1974.

### Q15. Could a Tipler cylinder be built?

**Not with any known matter**. The Tipler cylinder is an idealised mathematical object — infinite length, rigid, perfectly axisymmetric, supercritical rotation. Finite-mass approximations may preserve the band structure approximately, but this has not been worked out in the current paper.

---

## The hardware experiment

### Q16. What did IBM Marrakesh actually do?

Ran an 8-circuit suite encoding the Knopp Drive's amplitude composition at 8 different "orbit radii" sampling across the first CTC band exit. Each circuit: 1 data qubit + 3 path qubits, controlled-RZ gates encoding the four-mechanism phase composition, dynamical decoupling, gate twirling, measurement twirling, 8192 shots. Total runtime ~5 minutes.

### Q17. Were the results clean?

Yes. Total-variation distance between hardware and simulator at every point: $\le 0.05$. The hardware catcher verdict matches the simulator verdict exactly (3 sharp Hamming transitions, primary at the band exit `r3 -> r4` with step=12).

### Q18. Can I reproduce the experiment?

Yes. Anyone with IBM Quantum access can run `experiments/marrakesh_batch_6_knopp_drive.py --hardware` from the open-source repository. The full circuit definition, transpilation parameters, and analysis pipeline are public.

### Q19. What about other quantum platforms?

The same 4-qubit circuit is designed to run unchanged on Quantinuum (H1, H2), IonQ (Aria, Forte), and Pasqal (Pulser-Cloud). Cross-platform reproduction is queued as Phase 4 / NIAC Phase I work.

---

## Comparison to prior work

### Q20. Does this contradict Alcubierre 1994?

No. The Alcubierre metric still requires infinite exotic matter in the strict-thin-wall limit; that's a structural fact. The Knopp Drive operates in a different geometric regime where the Tipler frame-dragging provides the cone-tilt for free.

### Q21. How does this compare to Lentz 2021?

Lentz's subluminal soliton claim is positive-energy (NEC $\ge 0$) but does not provide FTL capability. The Knopp Drive is FTL-capable AND zero-exotic-matter inside the band — the unique entry in the comparison achieving both.

### Q22. How does this compare to Bobrick-Martire 2021?

Bobrick-Martire's $m_{ADM}$-parameterised family is the small-amplitude shell description used in the Knopp Drive (mechanism 4's underlying machinery). The Knopp Drive composes B-M with three other mechanisms to achieve the band-gating shortcut.

### Q23. Is this a Van Den Broeck-style topological trick?

No. Van Den Broeck 1999 uses topological compactification to reduce the total exotic-matter integral by orders of magnitude but not to zero. The Knopp Drive's band-gating is a geometric subtraction, not a topological trick.

---

## The novelty catcher

### Q24. What is the "address-space novelty catcher"?

A pure-classical methodology that hashes numeric outputs to bit-occupancy addresses, constructs a Hamming-distance graph, and flags successive Hamming steps that exceed a MAD-thresholded median. Used to validate that each mechanism produces a sharp transition where the theory predicts.

### Q25. Why is the catcher reliable?

It's deterministic, reproducible, calibrated against known smooth functions (which it reports as smooth) and known sharp transitions (which it reports as novel). 20 catcher-verified emergents across the Systrophē framework, each with independent physical interpretation.

---

## Practical and legal

### Q26. What's the license?

MIT for the open-source code. CC-BY-4.0 for the hardware-experiment data. A USPTO provisional patent (12 claims) covers the method and apparatus; commercial use requires a separate patent license (see `commercial/licensing_model.md`).

### Q27. Is the patent filed?

A USPTO provisional patent application has been drafted and is in the process of being filed. Priority date: 2026-05-11. PCT international filing planned within 12 months.

### Q28. How can I cite this work?

Cite the GitHub repository (`https://github.com/Zynerji/systrophe`) and the arXiv preprint (forthcoming). Once the journal version is published, cite Classical and Quantum Gravity (or whichever journal accepts it).

### Q29. Can I use this commercially?

The MIT license permits commercial use of the open-source code. Commercial use of the patented method or apparatus requires a separate license; see `commercial/licensing_model.md` for the tier structure.

### Q30. Can I work on this with you?

Yes — engagement is welcome at every level. Email `cknopp@gmail.com` for collaboration inquiries. For pull requests against the open-source code, just open one on GitHub. For peer-review-grade critique, please cite specific equation numbers and provide reproducible counterexamples.

---

## Where to learn more

- **Whitepaper**: `paper/knopp_drive.pdf` (11 pages, 4 figures).
- **arXiv preprint**: `arxiv/knopp_drive_arxiv.pdf` (9 pages); arXiv ID forthcoming.
- **Hardware experiment**: `experiments/marrakesh_batch_6_knopp_drive.py` + JSON results in `experiments/results/`.
- **Tutorial notebook**: `examples/knopp_drive_tutorial.ipynb`.
- **API reference**: `docs/knopp_drive_api.md`.
- **Source repository**: `https://github.com/Zynerji/systrophe`.
- **Author contact**: `cknopp@gmail.com`.
