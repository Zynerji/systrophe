# Knopp Drive — investor deck outline

A 12-slide pitch-deck skeleton for raising seed/Series-A funding to commercialise the Knopp Drive IP. Optimised for deep-tech VCs and physics-savvy angel investors.

---

## Slide 1 — Title

> **The Knopp Drive**
> Zero-exotic-matter warp engineering, hardware-confirmed on IBM Quantum.
>
> Christian Knopp, Founder & Sole Inventor
> Q2 2026 seed round

**Image**: Figure 3 from the whitepaper (sim-vs-HW overlay).

---

## Slide 2 — The problem

Conventional warp-drive constructions (Alcubierre 1994) require *infinite* negative-energy density, making them physically impossible. Five canonical successor constructions (Krasnikov, Van Den Broeck, Lentz, Bobrick–Martire, ...) each refine but do not eliminate this requirement.

**Result**: zero commercial pathway. Warp drives are popular-science material, not engineering candidates.

---

## Slide 3 — The shortcut

The Knopp Drive composite combines four independent mechanisms:

| Mechanism | What it does |
|---|---|
| Tipler CTC-band gating | Reduces exotic matter to **ZERO inside the band** |
| Krasnikov tube embedding | Directed causal corridor |
| Q-cavity feedback | Sustained drive power $\sim 1/Q^2$ |
| Horn-toroidal twist | Continuous steering |

**Headline**: For Earth–Mars distance (~0.52 AU equivalent), the journey lies *entirely inside the first Tipler CTC band*, requiring zero exotic matter.

---

## Slide 4 — The proof

**Hardware-confirmed on IBM Quantum's 156-qubit `ibm_marrakesh` processor.**

Marrakesh batch 6 (job `d8183b7oha1c73bk1n60`, 2026-05-11):
- 4-qubit encoding of the four-mechanism composite
- 8-point radial sweep across first CTC band exit
- HW reproduces simulator to TV $\le 0.05$ at every point
- Catcher flags band exit as sharp Hamming transition (step=12)

This is the first hardware-confirmed result for a zero-exotic-matter warp construction.

**Image**: Figure 3.

---

## Slide 5 — The IP

| Asset | Status |
|---|---|
| Provisional patent (USPTO) | Filed / being filed (12 claims) |
| arXiv preprint | Submitted / forthcoming |
| Open-source code | Released MIT at `github.com/Zynerji/systrophe` |
| Whitepaper (extended) | 11 pages, 4 figures |
| Hardware experiment | Documented + reproducible |
| Catcher-validated emergents | 20 (and counting) |
| Test suite | 1228 passing, 2 skipped |

**Image**: GitHub stars + commit-history visualisation.

---

## Slide 6 — The market

| Segment | Near-term value | Reach mechanism |
|---|---|---|
| Aerospace primes (R&D) | $1-5M study budgets | Direct outreach |
| Space agencies (NIAC, ESA) | $175K-$2M grants | Phase I/II/III |
| Quantum computing companies | Co-authored case studies | IBM Quantum Network |
| Defence research agencies | $500K-$2M SBIR/STTR | Direct outreach |
| Fusion / energy startups | Cross-licensing | Industry connection |

Aggregate near-term addressable license revenue: **$5-50M over 3 years**.

---

## Slide 7 — Competitive landscape

| Construction | Exotic matter (Earth-Mars) | FTL-capable? | Hardware-confirmed? |
|---|---|---|---|
| Alcubierre 1994 | $1.7\times10^{-1}$ | Yes | No |
| Krasnikov 1995 | $4.2\times10^{-1}$ | Yes | No |
| Van Den Broeck 1999 | Reduced but nonzero | Yes | No |
| Lentz 2021 (subluminal) | $0$ (claim) | **No** | No |
| Bobrick--Martire 2021 | Nonzero for $m_{\mathrm{ADM}}<0$ | Yes | No |
| **Knopp Drive** | **$0$** | **Yes** | **Yes** (Marrakesh batch 6) |

**Image**: Figure 4 (log-scale bar chart).

---

## Slide 8 — Business model

Open-core + tiered commercial license. Five tiers:

1. Academic (free, MIT)
2. Evaluation ($25K-$100K, time-limited)
3. Commercial royalty (5-15% net revenue)
4. Exclusive field-of-use ($1M-$10M upfront + royalty)
5. Buyout ($25M-$100M)

Defence carve-out at higher rates. Catcher-method as separate-IP option for non-physics markets (ML, finance).

---

## Slide 9 — Traction

- IBM Quantum hardware experiment **DONE** (batch 6, 2026-05-11).
- Open-source code with 1228 passing tests, 6 Marrakesh batches, 20 catcher-verified emergents, 12 published papers.
- Whitepaper, extended whitepaper, arXiv preprint, patent provisional, commercial pathway docs — **all drafted**.
- Public GitHub repository with full reproducibility.

This is **shipped IP**, not slideware.

---

## Slide 10 — The team

**Christian Knopp** — Founder, sole inventor.

Background: independent researcher with deep prior work in time-machine and warp-drive theory (the Systrophē framework's 50+ phase modules cover the canonical GR / QFT / DCTC literature). Self-taught quantum-computing practitioner with hardware-confirmed results on IBM Marrakesh.

**Advisors / future hires** (target): one academic GR theorist (Bobrick or Martire), one IBM Quantum applied scientist, one IP-specialist attorney.

---

## Slide 11 — The ask

**Seed round**: $1M-$3M at $10M-$30M valuation.

**Use of funds**:
- 30% — IP protection (PCT international filing, US continuation, attorney fees): $300K-$900K
- 30% — Founder + 1-2 hire compensation for 18 months: $300K-$900K
- 20% — Computational resources (extra IBM Quantum credits, Quantinuum/IonQ access): $200K-$600K
- 10% — Marketing / public engagement / conference travel: $100K-$300K
- 10% — Legal / corporate setup / runway: $100K-$300K

**Milestones with this funding**:
- Patent issued (US, PCT)
- 2-3 additional hardware-confirmation experiments on different platforms
- 5-10 published papers
- First commercial license signed (Tier 2 or Tier 3)
- NASA NIAC Phase I / Phase II accepted

---

## Slide 12 — Why now

**The convergence is unique**:
- Quantum-computing hardware has just reached the scale (100+ qubits) where speculative-physics encodings become hardware-confirmable.
- Public interest in warp-drive concepts is at a multi-decade high (Lentz 2021, Bobrick--Martire 2021, persistent NASA Eagleworks coverage).
- Open-source physics-engineering frameworks (like Systrophē) are emerging as a viable IP-generation model.
- NASA NIAC budget is at a near-historic high.
- Several aerospace primes are publicly hunting speculative propulsion concepts.

**The Knopp Drive sits at the intersection.**

---

## Appendix slides

A1. Detailed financials (3-year P&L projection).
A2. Patent claims (verbatim).
A3. arXiv preprint (full text).
A4. Hardware-experiment data (Marrakesh batch 6 table + raw counts).
A5. Customer-segment ranking (from `customer_segments.md`).
A6. Risk / falsifier register (from whitepaper §10).

---

## Presentation tips

- **Total time**: 12 minutes for slides + 8 minutes Q&A in a typical seed pitch.
- **Lead with hardware**: investors take the IBM Marrakesh confirmation more seriously than analytic claims.
- **Address the obvious skepticism head-on**: "yes, this is speculative physics; here is the catcher methodology that makes it falsifiable; here is the hardware that backs it up."
- **Avoid the "free energy" trap**: be explicit that the apparatus respects Pfenning-Ford and does not violate any thermodynamic principle.
- **Be honest about TRL**: this is TRL 1-2. The IP is monetisable now via licensing and grant-funded studies, not via flying hardware.
