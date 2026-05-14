# STATUS: research artifact (hypothesis falsified)

**Verdict:** the Cyliformer architectural concept — use the Systrophe
address-space λ₂ catcher as a compute-gating signal inside transformer
layers — does **not** beat a matched-parameter dense MLP control at
7B fine-tune scale. Five iterations (v1 wave-basis cylinders → v2
diversity-pushed → v3 longer-trained → v4 Dianoia-pattern adapter +
λ₂ gate → v5 Mamba S6 selective SSM) all land within ±1% perplexity
on a vanilla `baseline + LoRA` baseline on WikiText-2, and the matched-
MLP control in v4 captures all of the perplexity improvement that
v1-v3 attributed to the architecture.

This is the Dianoia outcome (`C:/Users/cknop/.local/bin/Dianoia/`):
hypothesis-driven build, paired-A/B test, honest negative, repo
kept as research-artifact state with tests green and findings
documented.

## What is salvageable

* **Engineering patterns**: the Dianoia-derived `augment_with_adapter`
  pattern (additive residual after each transformer layer, near-
  identity init) is reusable for any future adapter that wants to
  preserve the pretrained model at construction.
* **Matched-parameter MLP-adapter control** (`matched_mlp_adapter`):
  the critical test that revealed v1-v3 were giving Cyliformer credit
  that belonged to LoRA on the eval distribution.
* **`LearnedAddressCatcher`**: a fully differentiable PyTorch λ₂
  estimator on a Hamming graph of learned binary addresses. Useful
  outside this tool if someone wants a differentiable spectral
  diagnostic signal.
* **`SelectiveSSMAdapter`**: minimal Mamba S6 form with the three
  numerical-stability fixes documented in commits ffd3af4, 76a45e1,
  e0ed244 (small init / fp32 scan / log_A in dedicated low-LR group).
  Drop-in-able into another insertion pattern if anyone wants to test
  it again.
* The 5-iteration writeup in `FINDINGS_QWEN_7B.md` is the load-bearing
  document: numbers, reproduction commands, what failed and why.

## What is not salvageable

* The Cyliformer.txt-era conjectures (+5-15% quality, 30-37% VRAM
  saving, neutral or +20% inference speed). Not supported by the 7B
  A/B at this scale and budget.
* The framing of the catcher as a *compute-gating signal*. The
  address-space catcher does fine as a *diagnostic / detector*
  (that's what gets the Systrophe project its 26+ emergents), and
  has been promoted to its own tool `tools/catcher-monitor/`. The
  gating use is the falsified part.

## Provenance lineage

This is iteration #4-5 of a wave-basis / spectral lineage that has
been falsified each time it was pushed harder:

1. **qGPT-Infinity** — wave basis as from-scratch LLM. KL wall at
   K=64 / K=96.
2. **Overtone** — wave basis as weight compressor. SVD-dominated at
   matched parameter budget.
3. **Dianoia** — wave basis as latent-reasoning between-layer block.
   No widening-gap signature on parity / add-chain.
4. **Cyliformer v1-v3** — wave basis as channel rotation through a
   shared FFN. Same family, same outcome at 7B.
5. **Cyliformer v4-v5** — DROPPED the wave basis; tested λ₂ gating
   (v4) and Mamba S6 (v5) inside the Dianoia insertion pattern. Both
   tied the matched-MLP control within single-seed noise.

The Systrophe address-space catcher remains *unfalsified as a
diagnostic tool* and *falsified as a useful LLM compute-gating signal*
at the scales tested. Those two statements are compatible.

## Do not iterate further on this concept

Five iterations on the same theme have produced the same finding.
Additional rounds (different base model, different fine-tune corpus,
different seeds) are unlikely to flip the verdict. The current
recommendation is to leave this tool in the repo as a fully tested,
fully documented research artifact and direct any future Systrophe
→ applied work into:

* `tools/catcher-monitor/` — use the catcher as a detector (its
  proven role).
* `tools/dijkstra-mwpm/` — the Heron-r2 QEC win.
* `tools/lp-analyser/` — the analytic-CTC physics stack from the
  shipped Phase 2a-3b roadmap items.
