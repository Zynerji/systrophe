# Email template — Erik W. Lentz (MPI for Gravitational Physics, Potsdam)

**To**: erik.lentz@aei.mpg.de
**CC**: cknopp@gmail.com
**Subject**: A composite warp construction citing your subluminal soliton work + hardware confirmation

---

Dear Dr. Lentz,

I want to share a paper draft (`paper/knopp_drive.pdf`) and supporting open-source framework that places your 2021 subluminal soliton construction in direct comparison with a four-mechanism composite I've been developing, called the Knopp Drive.

In Section 7 (Figure 4) of the manuscript, I compare integrated exotic-matter requirements at Earth--Mars equivalent distance across the five canonical warp-drive families, including your subluminal soliton's claim of NEC $\ge 0$ everywhere. The Knopp Drive is the unique entry in the comparison that combines FTL capability with $|E_{\mathrm{neg}}| = 0$ inside the geometric CTC band of a supercritical Tipler exterior.

The novel structural element is the gate factor $(1 - c\,T(r))_+$ that emerges from the Bonnor Case III metric components — when the local Tipler tilt $T(r) = |K|/|F|$ exceeds unity (which it does throughout the first CTC band of a unit cylinder), the gate factor is exactly zero and the Krasnikov-wall NEC contribution is geometrically cancelled. The resulting composite is hardware-confirmed on IBM Quantum's `ibm_marrakesh` (batch 6, 4-qubit encoding, TV $\le 0.05$ across 8 radial points).

Two questions I would value your perspective on:

1. Does the subluminal-soliton approach allow for an analogous "geometric subtraction" of a portion of the NEC requirement via embedding in a pre-existing tilted background? (i.e., is there a Lentz analogue of the Tipler-Krasnikov hybrid?)

2. Section 3.3 of the manuscript discusses a Q-cavity parametric feedback that reduces sustained drive power as $1/Q^2$. Does your 2021 construction admit a similar cavity engineering, or does the soliton's intrinsic standing-wave structure already saturate the equivalent constraint?

Source code is open under MIT at `github.com/Zynerji/systrophe`; the `knopp_drive` module is the canonical entry point. An arXiv preprint will follow shortly after a USPTO provisional patent filing.

I would be honoured to discuss further, and would also welcome any sharp critique — even a strong negative reaction would be valuable.

With great respect for your 2021 work,

Christian Knopp
cknopp@gmail.com
`github.com/Zynerji/systrophe`

---

**Personalisation notes**:
- Lentz has been criticised in the literature (Santiago et al. 2022) for over-strong NEC claims; my framing here lets him push back constructively rather than feel attacked.
- He is at MPI Potsdam, a serious GR institute. Tone: technical, peer-level, no popularisation.
- If he engages, the natural next step is an in-person discussion at GRG22 or the next APS April Meeting.
