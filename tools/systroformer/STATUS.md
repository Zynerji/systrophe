# STATUS: research artifact (superseded by Cyliformer A/B verdict)

Systroformer was the first attempt at applying the Systrophe address-
space λ₂ catcher inside a transformer block, in the form
`ffn_out *= 1 + lambda_scale * λ_2`. It was validated only on a
synthetic copy task at toy scale (`d_model=32`, `seq_len=8`,
`vocab=10`, 4-layer mini-LM) and never matched against a real
baseline on a real eval.

Its successor, **Cyliformer** (`tools/cyliformer/`), ran the
matched-parameter MLP-adapter A/B test that Systroformer should have
run. Cyliformer's v4 result is the verdict for this whole line of
work:

> At Qwen2.5-7B + 150 LoRA steps + WikiText-2 test, the
> ResonanceAdapter (λ₂-gated, same shape as Systroformer's idea
> generalised to a Dianoia-style additive insertion) does **not**
> beat a matched-parameter dense MLP control. Catcher gating costs
> ~27% throughput and contributes no measurable quality benefit above
> what the same-size dense block already provides.

That verdict makes Systroformer's central claim (that λ₂ modulation
of an FFN gives a real quality signal) **falsified at scale** by the
sibling tool's A/B. Systroformer's own toy training run did converge
(loss 2.0 -> 0.05 on a copy task) but that result is not informative
about LLM quality — the matched-parameter MLP would have done the
same thing.

## What is salvageable

* The 10 unit tests in `tests/test_systroformer.py` continue to
  exercise the catcher / block / model / utils code paths.
* The `power-iteration λ₂` and `LearnedAddressNet` utilities are
  reusable building blocks for any pytorch implementation of a
  Hamming-graph λ₂. (Note: `tools/cyliformer/cyliformer/catcher.py`
  has the more polished version, fully differentiable in pytorch
  with no numpy roundtrip.)
* The scalability benchmark in `experiments/scalability_test.py`
  remains useful for measuring catcher overhead on toy models.

## What is not salvageable

* The hypothesis that `ffn_out *= 1 + lambda_scale * λ_2` is a useful
  primitive for transformer compute. Cyliformer's matched-MLP control
  ruled that in (and ruled it neutral-to-negative).
* Anyone landing on this tool should read `tools/cyliformer/STATUS.md`
  and `tools/cyliformer/FINDINGS_QWEN_7B.md` for the full A/B picture
  before treating Systroformer as a serious LLM primitive.

## Do not iterate further

Same recommendation as Cyliformer. The catcher's productive use is
as a detector, exposed via `tools/catcher-monitor/`. Compute-gating
applications of the catcher have run their course.
