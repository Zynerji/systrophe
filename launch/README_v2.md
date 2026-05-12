# Polishing notes for the GitHub README

This file is the working notepad for first-time-visitor README polish. It is NOT itself a new README. Once the polish points below have been applied to `README.md`, this file can be archived.

The current `README.md` is a thorough technical document but is calibrated for an audience that already knows the warp-drive literature. For Phase 4 launch, we want the top-of-README to do one job: *convince a first-time visitor in 30 seconds that this is real, hardware-confirmed, and worth investigating further*.

---

## Polish checklist for the top of README.md

### Above-the-fold (first screen-height) priorities

1. **Title and one-sentence pitch** — currently buried mid-document. Move to top.

   Replace:
   > # Συστροφή — Systrophe
   > **A co-rotating Tipler-cylinder pair as a tunable time-travel harness — and the framework it now anchors.**

   With (proposed):
   > # Συστροφή — Systrophe
   > **A four-mechanism composite warp engineering bound, hardware-confirmed on IBM Quantum. The Knopp Drive requires ZERO exotic matter inside a Tipler CTC band.**

2. **Headline figure as the first visual** — currently no image above the badges. Add Figure 3 (sim-vs-HW overlay) immediately under the title.

   ```markdown
   ![Hardware-confirmed Knopp Drive on IBM ibm_marrakesh](paper/figures/knopp_marrakesh_batch6.pdf.png)
   ```

   (Note: GitHub README does NOT render PDF; need a PNG export. Run `paper/generate_knopp_drive_figures.py` with a PNG variant.)

3. **Three-bullet hook** — currently no quick-summary. Add after the figure:
   > - **Open-source warp engineering**. 4 mechanisms compose multiplicatively: Tipler CTC-band gating + Krasnikov tube + Q-cavity feedback + horn-toroidal steering. Zero exotic matter inside the band.
   > - **Hardware-confirmed**. IBM `ibm_marrakesh` 156-qubit Heron-r2 processor reproduces the composite's headline extinction at TV $\le$ 0.05.
   > - **Production-grade**. 1228 tests, 12 papers, 20 catcher-verified emergents, USPTO provisional patent filed, MIT license.

4. **Quick-start in the first 5 lines of code** — currently many lines until the first code block. Add immediately after the hook:

   ````markdown
   ```python
   from systrophe.knopp_drive import KnoppDrive

   drive = KnoppDrive(Q=100.0, epsilon_horn=0.2)
   report = drive.journey(distance=0.52)  # Earth-Mars equivalent

   assert report.exotic_matter_total == 0.0  # zero exotic matter!
   assert report.inside_band_fraction == 1.0
   ```
   ````

5. **Star button conspicuous** — GitHub auto-generates this but verify it's visible above the fold.

### Below-the-fold improvements

6. **Reorganise sections** in the order:
   - What is this?
   - Headline result
   - Installation
   - Quickstart
   - The Knopp Drive (formerly mid-document, promote up)
   - Marrakesh hardware
   - Whitepaper
   - Architecture
   - Tests
   - Falsifiers
   - Citation
   - Contact

7. **Remove the existing TODO-tagged sections** at the bottom (if any).

8. **Add a "What's new in v0.18" callout** near the top:
   > **v0.18.0 (May 2026)**: Headline IP release — the Knopp Drive. Hardware-confirmed on `ibm_marrakesh`. Extended whitepaper (11 pages, 4 figures). USPTO provisional filed. PDF whitepaper at [`paper/knopp_drive.pdf`](paper/knopp_drive.pdf).

9. **GitHub topics** (settable from the repo settings page):
   - `warp-drive`
   - `general-relativity`
   - `quantum-computing`
   - `ibm-quantum`
   - `closed-timelike-curves`
   - `physics-engineering`
   - `python`

10. **Repository description** (one-line, settable from repo settings):
    > "Συστροφή / Knopp Drive: a four-mechanism composite warp engineering bound with hardware confirmation on IBM Quantum."

### Discoverability / SEO

11. **README first paragraph keywords**: "warp drive," "Alcubierre," "Krasnikov," "Pfenning-Ford," "IBM Quantum," "exotic matter." All naturally insertable.

12. **Pinned issue or discussion**: open a pinned discussion thread "Welcome / FAQ" linking to `launch/FAQ.md`.

### Visual polish

13. **Replace text-only "Architecture" section** with a graphical module-dependency diagram (auto-generated from imports, or hand-drawn). Optional.

14. **Add `docs/figures/` PNG exports** of all four whitepaper figures (PDF → PNG via ImageMagick or `pdf2png`) so they render in GitHub README.

15. **Animated GIF** of a Knopp Drive journey traversal (catcher running across r-sweep) — high-effort, high-impact for visitor engagement. Optional Phase 4b extension.

### Accessibility

16. **Alt text** on every image.
17. **Plain-text equivalents** in the architecture diagram if a graphical one is added.

### Comments to remove

18. The existing test-breakdown table at "Test suite breakdown" is detailed but lengthy; collapse into a single number with a `<details>` block for the full table.

19. The historical "v0.2.0 baseline" content can be moved to a `CHANGELOG.md` (already exists) and removed from README.

20. The Δῖνος bridge section is interesting but specialist; demote below the warp-drive content.

---

## Per-language README versions (optional, lower priority)

- `README.zh-cn.md` (simplified Chinese) — quantum computing community in China is large.
- `README.de.md` (German) — MPI Potsdam audience.
- `README.es.md` (Spanish) — Alcubierre's UNAM audience.

These can wait until Phase 5 if there's bandwidth.

---

## Implementation order

If the user wants this polish applied to the actual `README.md`, recommended order:

1. Apply the title + figure + hook (highest impact, lowest effort).
2. Add the quickstart code block.
3. Add v0.18 callout.
4. Reorganise sections.
5. Set GitHub topics + description.
6. Generate PNG figures.
7. Open the "Welcome / FAQ" discussion.

This polish should be done as a single PR or commit to make the change reviewable.
