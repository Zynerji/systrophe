# Systrophe derived tools

This folder holds tools built **on top of** the Systrophe core
framework. Each tool lives in its own subdirectory and imports from
`systrophe` to reuse the address-space novelty catcher, the Lewis-
Papapetrou exterior solver, the QEC decoder stack, or any other
framework primitive.

## Status overview

| tool                  | status            | what it is                                                   |
|-----------------------|-------------------|--------------------------------------------------------------|
| `catcher-monitor/`    | active            | the Systrophe address-space λ₂ catcher exposed as a generic detector (phase transitions, training instability, OOD) |
| `dijkstra-mwpm/`      | active            | the Heron-r2 QEC decoder (Dijkstra-shortest-path MWPM on syndrome-difference history; +5-25pp over naive matching) |
| `lp-analyser/`        | active            | analytic-CTC physics: LP exterior + CTC bands + Phase 2a/2b/3a/3b stress-energy / Hadamard / arrays / off-axis |
| `gw-burst-catcher/`   | active            | unmodeled GW-burst detector: PSD-whiten → Q-transform → catcher; synthetic-injection mode for offline tests, optional GWOSC fetcher for real events |
| `brain-wallet-auditor/` | active          | Bitcoin brain-wallet passphrase strength auditor (SHA256 / WarpWallet / BIP-39); catcher diagnostic on the candidate pool; personal-recovery + custodial-audit + Vasek-paper reproduction |
| `morphic-resonance-catcher/` | active (falsification harness) | adjudicates the testable core of Sheldrake's morphic resonance: catcher + count-vs-time identifiability + surrogate nulls + a CTC-resonance self-consistent-field model. Honest negatives: morphic ≡ diffusion by counts; single-form acausality unidentifiable. See `FINDINGS.md`. |
| `cyliformer/`         | **research artifact (falsified)** | catcher-as-compute-gate in transformer FFN; 5 iterations, didn't beat matched-MLP control. See `cyliformer/STATUS.md` + `FINDINGS_QWEN_7B.md`. |
| `systroformer/`       | **research artifact (superseded)** | earlier prototype of the same idea; falsified by Cyliformer's matched-MLP A/B. See `systroformer/STATUS.md`. |

The three **active** tools all derive from claims that have been
*directly validated by Systrophe* (catcher detects 26+ emergents; the
Dijkstra-MWPM gave +5-25pp on Heron-r2 hardware; the LP exterior +
Phase 2a-3b stack is shipped and tested). The two **research artifact**
tools are kept in-tree (Dianoia tradition) so the negative results are
reproducible and the engineering pieces remain usable.

## Layout

```
tools/
├── README.md
├── catcher-monitor/         <- generic anomaly / phase-transition detector
├── dijkstra-mwpm/           <- standalone QEC decoder
├── lp-analyser/             <- analytic-CTC physics public API
├── gw-burst-catcher/        <- unmodeled GW-burst detection on whitened strain
├── brain-wallet-auditor/    <- Bitcoin brain-wallet passphrase strength auditor
├── morphic-resonance-catcher/ <- falsification harness for Sheldrake's morphic resonance
├── cyliformer/              <- ARTIFACT: catcher as LLM compute-gate
└── systroformer/            <- ARTIFACT: simpler earlier sibling of cyliformer
```

## Adding a new derived tool

1. Create `tools/<name>/` with a `README.md`, a `<name>/` python
   package, a `tests/` directory, and one or more reproducibility
   scripts under `experiments/` or `examples/`.
2. Import from `systrophe.*` — do not re-implement framework
   primitives.
3. Document what Systrophe concept the tool reuses and why. If the
   tool tests a research conjecture rather than packaging an
   already-validated primitive, plan the falsification protocol up
   front and write a matching `STATUS.md` when the data arrives
   (positive or negative).
4. Tests live in `tools/<name>/tests/`; the top-level `tests/`
   directory is reserved for the Systrophe framework.

## Why split?

The Systrophe core (`src/systrophe/`) is research-grade and small
(`pip install systrophe` pulls only numpy/scipy). Derived tools may
have heavier dependencies (torch, transformers, scipy.sparse,
matplotlib) that we don't want to force on framework-only users.
Splitting keeps the core minimal and lets each tool declare its own
`requirements.txt`.
