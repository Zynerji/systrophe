# Email template — Quantinuum Research Partner Program

**To**: research@quantinuum.com
**CC**: cknopp@gmail.com
**Subject**: Research-access request: cross-platform reproduction of a warp-drive composite hardware result

---

Dear Quantinuum Research team,

I am writing to request access to Quantinuum's H1 or H2 trapped-ion processors for a cross-platform reproduction of a recently-completed quantum-circuit physics experiment.

**Background**: I have published an open-source warp-drive composite engineering framework (the Knopp Drive, `github.com/Zynerji/systrophe`) and have confirmed its headline result on IBM Quantum's `ibm_marrakesh` 156-qubit superconducting processor (batch 6, 2026-05-11): a 4-qubit circuit reproduces the predicted band-gated extinction at TV $\le 0.05$ across an 8-point radial sweep.

**Proposed Quantinuum experiment**: reproduce the same 8-circuit suite on H1 or H2 (whichever is available within the research program) to establish *platform-independence* of the catcher-verified band-gating result. Trapped-ion qubits have substantially different noise characteristics than superconducting qubits; reproduction across platforms is a strong validation step.

**Circuit specifications**:
- 4 qubits required (1 data + 3 path)
- Native gate set: H, X, CRZ. All Quantinuum-supported.
- Circuit depth: $\le 60$ after compilation. Well within H1/H2 fidelity envelopes.
- 8 circuits, 8192 shots each.
- Expected total runtime: $\le 60$ min on H1.

**Resource ask**: ~$5K-$15K of compute credits (estimated).

**Publication / acknowledgement**:
- Quantinuum will be acknowledged as a co-platform in the resulting paper (target: Classical and Quantum Gravity).
- Data will be openly published in the GitHub repository under CC-BY-4.0.
- Quantinuum H1/H2's platform will be featured in any media coverage.

**Reproducibility kit**:
- Complete source code is at `github.com/Zynerji/systrophe`
- The Marrakesh batch 6 circuit definition is at `experiments/marrakesh_batch_6_knopp_drive.py`
- Adaptation to Quantinuum's Pulser-compatible Python API is estimated at 2-4 hours of work.

I would be grateful for any pathway to Quantinuum H1/H2 access, including via the standard research-access application, paid commercial access, or a co-developed case study.

Materials available for review:
- Manuscript: `paper/knopp_drive.pdf` (11 pages, 4 figures, 1 table)
- IBM Quantum result: `experiments/results/marrakesh_batch6_hw_analysis.json`
- Open-source code (MIT license): `github.com/Zynerji/systrophe`

With sincere respect,

Christian Knopp
cknopp@gmail.com
`github.com/Zynerji/systrophe`

---

**Personalisation notes**:
- Quantinuum runs a Research Partner Program; the standard application form is at `https://www.quantinuum.com/research`.
- Their H-series machines have high-fidelity all-to-all connectivity; the 4-qubit circuit will run well.
- The "cross-platform reproduction" framing positions them as the validation partner, not the secondary platform — this matters for their marketing.
- If they accept, the same template adapts for IonQ (Aria/Forte) and Pasqal (Pulser-Cloud).
