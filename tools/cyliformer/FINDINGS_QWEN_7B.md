# Cyliformer × Qwen2.5-7B A/B test: honest findings (v1 → v2 → v3 iterations)

**Date**: 2026-05-13
**Hardware**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB)
**Base model**: Qwen/Qwen2.5-7B-Instruct (BF16)
**Conversion**: FFN-only swap (every `Qwen2MLP` → `CylinderFFN`,
                  attention/norms/embeddings untouched)
**Fine-tune**: LoRA r=8, α=16, gate/up/down_proj target modules,
               WikiText-2-raw-v1 train, lr=2e-4
**Eval**: WikiText-2-raw-v1 test, 4 blocks × 512 tokens = 2048 tokens
**Methodology**: train + benchmark in one process (PEFT only saves LoRA
                 deltas, not the new cylinder params).

## Multi-iteration summary table

| run                                | steps | zero-shot ppl | +LoRA ppl | tok/s | adapter | notes |
|------------------------------------|-------|---------------|-----------|-------|---------|-------|
| baseline (no LoRA, no adapter)     | 0     | 8.948         | —         | 88    | —       | reference |
| baseline + LoRA                    | 150   | 8.948         | **7.051** | 80    | —       | strong control |
| baseline + LoRA                    | 300   | 8.948         | 8.036     | 80    | —       | overfit at 300 |
| Cyliformer v1 (FFN-rewrite, n=2)   | 150   | 8.948         | 7.148     | 35    | none    | wave-basis, -57% tok/s |
| Cyliformer v2 (FFN-rewrite, n=4)   | 200   | 28.997        | 9.904     | 22    | none    | wave-basis broken |
| Cyliformer v3 (FFN-rewrite, n=2)   | 300   | 8.948         | 7.943     | 37    | none    | regularises @ 300 step |
| **v4 baseline (no adapter)**       | 150   | 8.948         | **7.064** | 80    | —       | repro of control |
| **v4 mlp_adapter (matched 6.5M)**  | 150   | 8.965         | 7.078     | 78    | dense MLP | -0.20% vs baseline |
| **v4 resonance_adapter (matched)** | 150   | 8.983         | 7.092     | 57    | λ₂-gated | -0.40% vs baseline; **=** MLP within 0.20%; -29% tok/s |

## Verdict (post-v4 with matched-parameter MLP control)

At 7B scale, 150 LoRA steps, WikiText-2 in-distribution eval:

- **Quality**: baseline + LoRA wins (ppl 7.064). The matched 6.5M-param
  dense MLP adapter is essentially tied (7.078, -0.20%). The
  ResonanceAdapter (λ₂-gated bottleneck) is also essentially tied
  (7.092, -0.40%, **=** matched-MLP within 0.2%). **No architectural
  contribution from the λ₂ catcher above what a same-size dense
  adapter provides.**
- **VRAM**: identical across all three arms (14.54-14.57 GB).
- **Throughput**: matched-MLP costs 2% vs baseline; ResonanceAdapter
  costs 29% (the catcher's per-layer power-iteration on a 96-node
  Hamming graph). The earlier v1-v3 wave-basis Cyliformer cost 57-75%
  on top of being intrusive.

The original Cyliformer.txt conjectures (+5-15% quality, 30-37% VRAM
saving, +20% inference speed) are **not supported** by this A/B at
7B scale.

**The matched-MLP control rules out**: "additive residual adapters
help WikiText-2 perplexity at this step budget." (They don't; both
hurt by ~0.20-0.40%.) So the question "does the λ₂ catcher add value
on top of an adapter?" has a clean answer **here**: no, within noise.

## v1 → v2 → v3 design iteration log

### v1 (initial)
- Phasors: `linspace[-π/2 + ε, π/2 - ε]`
- `backreaction_scale_init=0.10`
- Per-cylinder catcher (2 catcher calls per FFN)
- Catcher active in eval

Failure modes (revealed by the A/B):
1. `n_cylinders=2` with `linspace[-π/2, π/2]` produces exact anti-
   rotation cancellation: beam_gain = 0.0005, FFN output annihilated,
   zero-shot perplexity 260,081 (28,000× worse).
2. `backreaction_scale=0.10` shrinks every FFN output by ~0.92 at init
   (since catcher λ₂ = 0, sigmoid(-(0-0.18)·8) = 0.81, so FFN ×
   (1 - 0.10·0.81) = ×0.92). Caused the 1.4% regression even after
   the phasor fix.
3. Catcher λ₂ stays at exactly 0.0 throughout training -- the random
   address-projection produces degenerate addresses, and the LoRA
   penalty wasn't strong enough to drive learning.

### v2 (over-corrected)
- Phasors: `linspace[-π/4, π/4]` with jitter
- `n_cylinders=4` (richer diversity)
- `backreaction_scale_init=0.0` (no FFN scaling at init)
- Single shared catcher per FFN (cuts catcher cost 50%)
- Catcher skipped in eval (`compute_catcher_in_eval=False`)
- `phasor_diversity_weight=0.10` in loss (push apart)

Failure: zero-shot perplexity jumped to 28.997. The pretrained Qwen
FFN was trained to map identity-rotated inputs; even modest channel
rotations (±π/4 = 45°) wreck the non-linearity. The FFN gives up on
rotated inputs. 200 LoRA steps couldn't recover. The diversity
regulariser actively fought LoRA's attempt to revert phasors to near
zero.

This was the architectural revelation: **shared FFN + non-identity
input rotation is incompatible with a pretrained model**. The FFN
expects identity. v1 was right to keep phasors near zero — but then
the cylinders have no architectural diversity (all produce the same
FFN output up to ε rotation), so there's no architectural benefit
either.

### v3 (consistent v1)
- Phasors: `randn * 0.05` (very small, near zero, broken symmetry)
- `n_cylinders=2`
- `backreaction_scale_init=0.0` (kept from v2 — fixed v1 issue)
- Single shared catcher per FFN (kept from v2)
- Catcher skipped in eval (kept from v2)
- **No** lambda_2 floor penalty, **no** diversity regulariser
- 300 LoRA steps

Result: zero-shot exactly matches baseline (cylinders identical →
mean is just FFN(x)). +LoRA at 7.943 vs baseline+LoRA@300 at 8.036 →
1.2% improvement, possibly noise. Throughput 37 vs baseline+LoRA 80 →
−54%.

The 300-step baseline+LoRA regressed from 7.051 (@150) to 8.036
(@300) — this is overfitting WikiText-2 train. Cyliformer v3
avoided the regression. Whether this generalises to other corpora is
untested.

### v4 (Dianoia-pattern pivot — pull DOWN the wave basis)

The pivot was forced by reading `C:/Users/cknop/.local/bin/Dianoia/`
and learning that the wave-basis-as-compute-primitive lineage has been
falsified three times:

  * **qGPT-Infinity** (from-scratch LLM on wave basis) — KL wall at
    K=64 / K=96; basis too narrow to span teacher weights.
  * **Overtone** (wave basis as post-training weight compressor) —
    falsified by least-squares capacity test; strictly dominated by
    truncated SVD at matched parameter budget.
  * **Dianoia** (wave-basis inserted between-layer adapter for latent
    reasoning) — falsified by paired experiment; no widening-gap on
    parity / add-chain. The mathematical reason is given in
    Dianoia's FINDINGS: parity requires *input-dependent state
    dynamics* (Mamba's S6), and linear wave-basis evolution cannot
    supply that.

Cyliformer v1-v3 were iteration #4 of the same falsified pattern --
phasor channel-rotation is wave basis applied to channels.

v4 keeps two things and drops one:

  * **Engineering**: Dianoia's `augment_model` insertion pattern -- a
    new `ResonanceAdapter` is inserted as an *additive residual* after
    each transformer layer, with `down_proj` initialised at gain
    `0.02 / sqrt(r)` so the residual is near-zero at construction and
    the pretrained model is preserved exactly. No FFN rewrite.
  * **Honest control**: a matched-parameter dense MLP adapter
    (`matched_mlp_adapter`) inserted via the same pattern, so the
    catcher's contribution is measured against a same-size same-shape
    dense bottleneck rather than against no adapter.
  * **Wave-basis dropped**: no cos/sin phasor rotation, no oscillator
    bank. Only the address-space `LearnedAddressCatcher` (Systrophe's
    one unfalsified LLM primitive) is kept, and it feeds a learnable
    scalar gate `sigmoid(α (λ₂ - target))` that multiplies the
    adapter's contribution.

ResonanceAdapter at r=32 has ~ 230k params per block, 6.5M over 28
blocks -- matched to a dense MLP adapter of the same parameter count.

Result: at 150 LoRA steps, the 3-arm paired A/B gives

  baseline + LoRA       :  ppl 7.064  vram 14.54 GB  tok/s 80
  mlp_adapter + LoRA    :  ppl 7.078  vram 14.56 GB  tok/s 78  (-0.20%)
  resonance_adapter+LoRA:  ppl 7.092  vram 14.57 GB  tok/s 57  (-0.40%)

The matched-MLP and the ResonanceAdapter are within 0.20% perplexity
of each other -- single-seed noise. **The address-space lambda_2
catcher provides no measurable quality benefit over a same-size dense
adapter at this scale and budget.** The catcher does cost 27%
throughput.

## What this does not falsify

## Headline numbers

| metric         | baseline | baseline+LoRA | Cyliformer 0-shot | Cyliformer+LoRA |
|----------------|----------|---------------|-------------------|-----------------|
| perplexity     | 8.948    | **7.051**     | 8.948 (+0.00%)    | 7.148 (-20.12%) |
| peak VRAM (GB) | 14.39    | 14.80         | 14.40 (+0.04%)    | 14.83 (+3.01%)  |
| tokens/sec     | 88.43    | **80.14**     | 38.47 (-56.50%)   | 34.83 (-60.61%) |

**Verdict**: at this configuration and scale, Cyliformer adds zero
quality benefit, zero memory benefit, and costs ~57% throughput.

## What we expected vs what happened

`Cyliformer.txt` claimed:
1. Quality: +5–15% on long-context reasoning, coherence, factual benchmarks.
2. VRAM: 30–37% reduction vs equivalent dense 7B.
3. Inference speed: neutral to +20%.

What we measured:
1. Quality: **−1.4%** vs the apples-to-apples control (baseline+LoRA).
2. VRAM: **+0.2%** (essentially identical; the 3.2M new params are noise at 7B).
3. Inference speed: **−57%** vs baseline+LoRA.

The Cyliformer.txt numbers were research conjectures based on the
*shared-FFN-across-N-independent-views* assumption. In our actual
implementation:

* The shared FFN gives **no parameter savings vs the vanilla single-FFN
  Qwen2.5-7B** — they have the *same* per-layer FFN. The conjectured
  savings are only against a hypothetical "N-independent-FFNs-per-layer"
  baseline that does not exist as a deployed architecture.
* The 2 cylinder forward passes per FFN call double the FFN compute. The
  catcher (power-iteration λ₂ on a 64-node Hamming graph per cylinder,
  per layer) adds further overhead. The combined cost at 28 layers × 2
  cylinders is the −57% throughput we measured.
* Quality is determined by the FFN's actual computation. With phasors
  near zero (small random init), each cylinder is a near-identity
  rotation of the input through the same SwiGLU FFN. Averaging them
  gives ~vanilla output, modulated by a constant ~0.92 backreaction
  scale. The backreaction term doesn't carry information at this point
  because the catcher's λ₂ stays at exactly 0.

## Why the catcher failed to learn (λ₂ = 0.0 throughout training)

The `LearnedAddressCatcher` projects activations to 32-bit binary
addresses via a `Linear(d_model, n_bits)` whose weights are randomly
initialised. With random projection weights, all activation tokens hash
to the *same* address (the projection is too uniform), producing a
fully-connected Hamming graph with degenerate λ₂ = 0.

For the catcher to develop a meaningful signal, it would need:
* Sufficient training signal to differentiate addresses (the
  `lambda_target` floor penalty was set to 0.10 weight which proved
  too weak).
* Wider initial address dispersion (the random Gaussian init at
  `bias=True` does not guarantee anti-correlated bit channels).
* Possibly a contrastive/orthogonality regulariser on the projection
  to keep addresses informative across tokens.

We did not pursue these fixes because the throughput cost makes the
overall architecture uncompetitive even if quality were preserved.

## Why baseline+LoRA was the better choice on this eval

WikiText-2-test is in-distribution with the WikiText-2-train data we
fine-tuned on. Vanilla LoRA on `gate/up/down_proj` was sufficient to
capture the eval-distribution adaptation. The cylinder structure added
no additional capacity for that adaptation.

The architectural premise — "rotate the input N times, run the same
FFN, beam-sum coherent paths" — needs:
1. A diverse enough rotation that the N runs produce non-redundant
   outputs (failed: with init near 0, runs are near-identical).
2. A coherence signal that can distinguish good vs bad rotations
   (failed: catcher λ₂ never escaped 0).
3. A task where coherence-gated FFN diversity helps (not tested: the
   eval is fitting an in-distribution corpus; the LoRA dominates).

## Reproducibility

All scripts pushed to `tools/cyliformer/experiments/` and run from
this repo. Total wall-time on the Blackwell VM (including HF download
via hf-mirror.com):

* Qwen2.5-7B download (one-time): ~4 minutes
* Baseline A/B (perplexity + VRAM + tok/s): ~30 seconds
* LoRA fine-tune 150 steps: ~45 seconds (Cyliformer) / ~16 seconds (baseline)
* Total session: ~10 minutes after download

Result JSON files:
* `experiments/results_train_bench.json` — baseline → zero-shot → +LoRA
* `experiments/results_baseline_lora.json` — control (baseline + same LoRA)
* `experiments/results_ab_step2.json` — first benchmark after init fix

## What this does not falsify

This A/B test was deliberately the *minimum-viable PoC*: 2 cylinders,
small random phasor init, 150 LoRA steps, WikiText-2 perplexity. It
cannot rule out the following alternative configurations:

1. **More cylinders + active phase comb**: `n_cylinders ≥ 4` with
   forced phase diversity (e.g. phasor regulariser) could give
   genuine architectural diversity that LoRA + catcher can exploit.
2. **Catcher pre-training**: warm-start the catcher's projection with
   a contrastive objective before LoRA so λ₂ is informative from step 1.
3. **Different conversion depth**: full block replacement (attention
   too) instead of FFN-only could give the cylinder structure room
   for genuine sparsity benefits.
4. **Longer training + diverse data**: 150 LoRA steps on WikiText-2 is
   tiny. A multi-corpus instruct fine-tune over thousands of steps
   could give the catcher time to develop a signal.
5. **Real downstream eval**: WikiText-2 perplexity is a noisy proxy for
   coherence/reasoning. The Cyliformer's claimed strength is
   long-context coherence, which requires lm-eval-harness or similar.

Any of these could change the verdict. None were tested here.

## Honest summary

Four iterations (v1 wave-basis cylinders, v2 with diversity push, v3
near-identity cylinders + longer training, v4 Dianoia-pattern adapter
with matched-MLP control) all converge on the same finding:

  * The **engineering** of Cyliformer-style insertion is sound. v4's
    additive residual is near-identity at init (pre-LoRA ppl 8.98 vs
    baseline 8.95) -- the pretrained model is preserved.
  * The **matched-parameter MLP control** (introduced in v4 by porting
    Dianoia's harness) reveals that **no additive adapter helps at
    150 LoRA steps on WikiText-2** -- all adapter arms regress 0.20-
    0.40% vs the no-adapter baseline.
  * The **lambda_2 catcher specifically** provides no benefit beyond
    what a same-size dense MLP adapter gives. The 27% throughput cost
    of the per-layer power-iteration catcher buys nothing.
  * Per Dianoia's post-mortem, the **wave-basis lineage** that
    Cyliformer v1-v3 inherited (phasor channel-rotation) has been
    falsified three times before this; this is iteration #4 of the
    same falsified pattern. v4 dropped the wave basis and the
    architecture-only contribution still does not appear.

The full Systrophe lineage is about *verifying claims*, not preserving
them. The Cyliformer + Systroformer concept of "use the Systrophe
lambda_2 catcher to modulate transformer compute" is shipped honestly
as a negative result at 7B scale with LoRA fine-tune budgets.

Future directions the data points toward (but does NOT validate):

  * **Input-dependent state dynamics** (Mamba S6 style) inside the
    adapter -- the Dianoia FINDINGS already point here.
  * **Much longer training and out-of-distribution eval** -- the
    150-step / WikiText-2 budget is too small to differentiate
    architectures that converge to the same ~7.0 ppl floor.
  * **Different downstream task** -- WikiText-2 perplexity is a noisy
    proxy. Real reasoning / coherence benchmarks (lm-eval-harness)
    might reveal an effect the perplexity test cannot.

Any of these could change the verdict. None were tested here.

The Systrophe address-space catcher remains *unfalsified as a
diagnostic tool* (its original purpose: catching emergents in physics
modules and HW runs). It is *falsified as a useful gating signal for
LLM compute at this scale and budget*. Those two statements are
compatible.
