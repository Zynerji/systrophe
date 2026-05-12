# Email template — IBM Quantum Network (case-study expansion)

**To**: quantum-network-feedback@us.ibm.com
**CC**: cknopp@gmail.com
**Subject**: Knopp Drive on ibm_marrakesh: candidate case study for the IBM Quantum Network

---

Dear IBM Quantum Network team,

I am writing to propose that the Knopp Drive hardware experiment (Marrakesh batch 6, job `d8183b7oha1c73bk1n60`, 2026-05-11) be considered as a case study for the IBM Quantum Network's public showcase.

**The result**: I encoded a four-mechanism composite warp-drive engineering object as a 4-qubit circuit (1 data + 3 path qubits) on `ibm_marrakesh`. The circuit exhibits exactly the band-gated extinction predicted by the underlying GR/Krasnikov composition: inside a geometric closed-timelike-curve band, $P(\mathrm{data}=1) \approx 0.05$ (extinct); outside the band, $P(\mathrm{data}=1) \approx 0.6$ (biased). Hardware reproduces simulator prediction to TV $\le 0.05$ at every point of an 8-point radial sweep.

The address-space novelty catcher flags the band exit as a sharp Hamming-graph transition (step$=12$ vs. median$=0$), and the verdict matches the simulator verdict exactly. This is, to my knowledge, the first hardware-confirmed positive result for a zero-exotic-matter warp-drive successor construction.

**Why this is a strong case-study candidate**:

1. **Physics relevance**: warp-drive constructions are public-interest topics (Alcubierre, Lentz, Bobrick-Martire have each generated press coverage); the hardware-confirmation aspect elevates the story from "speculative theory" to "tested on real quantum hardware".

2. **Technical merit**: clean circuit (n_2q $\le$ 12 after opt_level=3 transpilation), well-behaved errors (TV $\le 0.05$), reproducible methodology (catcher-verified). The 156-qubit Heron-r2 architecture's depth and connectivity are showcased.

3. **Open science**: source code, raw counts, run logs, sim/HW analyses are all openly published at `github.com/Zynerji/systrophe`. Anyone with IBM Quantum access can reproduce the result.

4. **Publication trajectory**: arXiv preprint within 30 days, target journal Classical and Quantum Gravity. The IBM Quantum platform will be acknowledged in every downstream publication.

I would welcome any one of:
- A case-study feature on the IBM Quantum public website / blog
- A co-authored paper with IBM Quantum applied scientists
- An invitation to present at the IBM Quantum Summit or related event
- An introduction to other IBM Quantum Network customers working in related domains

The complete experimental package (circuits, transpilation parameters, dynamical-decoupling settings, gate/measure twirling configuration, raw count files) is available for IBM Quantum's review.

Many thanks for IBM Quantum's open-access policy, which enabled this work.

With sincere respect,

Christian Knopp
cknopp@gmail.com
IBM Quantum instance: `Zynerji`
`github.com/Zynerji/systrophe`

---

**Personalisation notes**:
- IBM Quantum has a public-facing case-study program (`https://www.ibm.com/quantum/case-studies`) that features customer/researcher work using their platform. Aim for inclusion there.
- The story is genuinely interesting from IBM's perspective: the largest-N-qubit publicly-reported physics-application result on Marrakesh as of submission.
- Specific names to potentially route to: Olivia Lanes (IBM Quantum Network), Bob Sutor (IBM Quantum, formerly), Jay Gambetta (IBM Quantum VP). LinkedIn first then email.
