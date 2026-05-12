# Knopp Drive — licensing model

A tiered licensing structure for the Knopp Drive IP that balances open-source academic adoption against commercial monetisation.

---

## Background and rationale

The Knopp Drive lives at the intersection of three legal regimes:

1. **Open source** — the Systrophē Python package is released under the MIT license and is publicly available at `github.com/Zynerji/systrophe`. The MIT license permits commercial use without royalty.
2. **Provisional patent** — the four-mechanism composite has been filed (or is in the process of being filed) with the USPTO as a provisional application. The patent rights attach to the *method* and *apparatus* claims, not the software code itself.
3. **Trade secret** — internal optimisations, customer-specific parameter tuning, and follow-up unpublished extensions can be retained as trade secrets.

This combination is the standard structure for an open-core + commercial-license model (e.g., Red Hat / MongoDB / Elastic).

---

## Tier structure

### Tier 1 — Academic / open-source (free)

**Recipient**: any non-commercial researcher, student, educational institution.

**Granted rights**:
- Full use of the Systrophē Python package under MIT license.
- Right to cite the patent for academic-research purposes.
- Right to publish derivative academic papers.

**Required attribution**:
- Cite the Knopp Drive arXiv preprint (forthcoming) and the GitHub repository.

**Cost**: $0.

### Tier 2 — Evaluation / pre-commercial

**Recipient**: company evaluating commercial use (aerospace prime, defence contractor, fusion company).

**Granted rights**:
- 12-month time-limited license to use the Knopp Drive method in feasibility studies, business cases, and internal R&D.
- Right to incorporate findings into the recipient's own white papers and SBIR/grant proposals (subject to attribution).

**Restrictions**:
- No right to manufacture or sell apparatus implementing the patent claims.
- Time-limited; conversion to Tier 3 or Tier 4 required for productisation.

**Cost**: $25K-$100K depending on company size.

### Tier 3 — Commercial / per-product royalty

**Recipient**: company manufacturing apparatus or providing a service that practices the Knopp Drive method claims.

**Granted rights**:
- Non-exclusive right to manufacture and sell apparatus or services implementing the patent claims.
- Royalty-bearing license: $X per apparatus / $Y per service-event.

**Royalty structure**:
- Royalty rate: 5-15% of net revenue from the licensed product, negotiable.
- Minimum annual royalty: $50K.
- Sublicensing: permitted with consent.

**Term**:
- 10-year initial term, renewable in 5-year increments.
- Co-terminus with the eventual issued patent (typically 20 years from provisional filing).

### Tier 4 — Exclusive field-of-use

**Recipient**: single company seeking exclusive rights within a defined field (e.g., "civilian space propulsion", "defence applications").

**Granted rights**:
- Exclusive license within the specified field of use.

**Cost**:
- Up-front payment: $1M-$10M depending on field breadth.
- Ongoing royalty: 3-10% of net revenue.
- Minimum annual royalty: $250K-$1M.

**Term**:
- Co-terminus with the patent.

### Tier 5 — Buyout / assignment

**Recipient**: single buyer of the entire IP portfolio.

**Granted rights**:
- Full assignment of the patent and any associated trade secrets.

**Cost**:
- Negotiable; expected range $25M-$100M assuming patent issues and a credible commercial pathway emerges.

---

## Dual-license + special clauses

### Defence carve-out

Tier 2/3/4 contracts include a "no offensive military application" clause as default; a separate defence license (Tier 4D) is available subject to ITAR/EAR compliance and direct negotiation. Defence-license royalty rates are typically 2-3x higher.

### Catcher-method license

The address-space novelty catcher is a separate-but-related IP. We license it under MIT (open) for academic use; a separate trade-secret + patent-protected commercial license applies for novelty-detection applications in machine learning / financial modelling / quality assurance (a much larger commercial market than warp drives).

### Hardware-experiment data clause

Hardware-experiment results (e.g., Marrakesh batch 6 counts data) are explicitly licensed under CC-BY-4.0 (Creative Commons Attribution). This allows the data to be reused by any third party with attribution, but the underlying *method* of the experiment remains patent-protected.

---

## Recommended initial pricing

For the first 12 months of commercial outreach:

| Tier | Recipient profile | Indicative price |
|---|---|---|
| Tier 2 (Eval) | Aerospace prime R&D division | $50K |
| Tier 2 (Eval) | Fusion startup | $25K |
| Tier 2 (Eval) | Defence contractor | $100K |
| Tier 3 (Commercial) | Royalty per apparatus | 10% of net revenue, $100K min |
| Tier 4 (Exclusive, civilian) | Single aerospace company | $5M upfront, 5% royalty |
| Tier 4 (Exclusive, defence) | Single defence prime | $10M upfront, 8% royalty |
| Tier 5 (Buyout) | Strategic acquirer | $50M (placeholder) |

These prices are calibrated against comparable speculative-propulsion IP licenses (e.g., NASA Lewis Center's Hall-effect thruster licenses to L3 and Aerojet Rocketdyne in the 1990s). They will be adjusted up or down based on:
- Patent issuance status (currently provisional only)
- Hardware-confirmation maturity (currently 4 qubits; higher-fidelity validation would raise prices)
- Public-domain prior art (currently mostly absent)
- Defence interest level (currently unknown)

---

## Negotiation playbook

For each potential licensee, the negotiation should follow this template:

1. **Discovery** (1-2 weeks): identify the licensee's intended use, expected scale, and budget tier.
2. **NDA + sandbox** (2-4 weeks): provide a Tier 2 evaluation license + technical-support engagement.
3. **Commercial term sheet** (1-2 months): negotiate Tier 3 or Tier 4 terms.
4. **Definitive agreement** (2-4 months): legal counsel-drafted patent license agreement.

Standard contract length: 30-60 pages, drafted by IP-specialised counsel (recommend Cooley LLP, Wilson Sonsini, or Foley & Lardner for US; Bird & Bird or Marks & Clerk for UK/EU).

---

## Open issues / pending decisions

- [ ] Whether to file PCT (international) within 12 months of US provisional filing — recommended if commercial interest emerges in Europe or Asia.
- [ ] Whether to maintain trade-secret protection for any optimisation found in follow-up research, or to publish everything openly.
- [ ] Whether to spin out a dedicated holding company (recommended for any commercial engagement > $1M).
- [ ] Whether to seek pro-bono representation from a university tech-transfer office (some IP attorneys take speculative cases on a deferred-fee basis).
