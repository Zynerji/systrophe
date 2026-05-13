"""Paired A/B for ResonanceAdapter (Cyliformer v4) on Qwen2.5-7B-Instruct.

Three arms, same data + same optimizer + same step count + same seed:

  * baseline               : no adapter, LoRA only (= prior baseline_lora_control).
  * mlp_adapter            : matched-parameter dense bottleneck MLP, LoRA on top.
  * resonance_adapter      : v4 -- λ₂-gated bottleneck adapter (Dianoia pattern,
                              Systrophe catcher, NO wave basis), LoRA on top.

Per arm we measure perplexity (WikiText-2 test), peak VRAM, tokens/sec.
The signature of a real benefit is **resonance_adapter beating
mlp_adapter at matched parameter count**, not just beating no-adapter
baseline.

The previous Cyliformer v1/v2/v3 used the wave-basis cylinder approach
that was already falsified three times in the Systrophe lineage
(qGPT-Infinity, Overtone, Dianoia). v4 drops the wave basis entirely
and keeps only the address-space catcher.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ab_benchmark import measure_perplexity, measure_throughput, measure_vram_forward
from qwen_lora_finetune import chunks, get_text_corpus, setup_lora
from cyliformer.resonance_adapter import (
    ResonanceAdapter,
    augment_with_adapter,
    matched_mlp_adapter,
)
from cyliformer.ssm_adapter import SelectiveSSMAdapter, is_ssm_sensitive_param


def count_adapter_params(adapter: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in adapter.parameters())


def make_arm_model(arm: str, base_id: str, dtype, device_map,
                     r: int = 32, every: int = 1) -> tuple:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype, device_map=device_map,
        trust_remote_code=True,
    )

    if arm == "baseline":
        summary = {"adapter": "none", "added_params": 0}
    elif arm == "resonance_adapter":
        d_model = model.config.hidden_size
        def factory(_i):
            return ResonanceAdapter(d_model=d_model, r=r,
                                       compute_catcher_in_eval=True)
        summary = augment_with_adapter(model, factory, every=every)
        summary["adapter"] = "resonance"
    elif arm == "mlp_adapter":
        d_model = model.config.hidden_size
        # Match parameter count of one ResonanceAdapter:
        probe = ResonanceAdapter(d_model=d_model, r=r)
        target = count_adapter_params(probe)
        del probe
        def factory(_i, _target=target):
            return matched_mlp_adapter(d_model, _target)
        summary = augment_with_adapter(model, factory, every=every)
        summary["adapter"] = "mlp"
        summary["target_params_per_block"] = int(target)
    elif arm == "ssm_adapter":
        d_model = model.config.hidden_size
        def factory(_i):
            return SelectiveSSMAdapter(d_model=d_model, state_dim=r)
        summary = augment_with_adapter(model, factory, every=every)
        summary["adapter"] = "ssm"
    else:
        raise ValueError(f"unknown arm: {arm}")

    return model, tok, summary


def train_arm(model, tok, args, log_prefix: str = "") -> dict:
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # All-trainable: adapter params + LoRA over gate/up/down
    # (PEFT injects LoRA in place; we mark the adapter modules as
    # trainable via a name-pattern after PEFT freezes everything.)
    #
    # Snapshot which params were trainable BEFORE PEFT freezes them so we
    # can respect any module-level requires_grad=False decisions (e.g.
    # the SSM adapter freezes log_A and D explicitly to avoid the
    # cumulative-product gradient explosion observed at 7B + LoRA).
    pre_lora_trainable = {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    model = setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    n_set = 0
    for name, p in model.named_parameters():
        if any(token in name for token in (
            ".adapter.",
            "gate_alpha", "gate_target",  # ResonanceAdapter gate scalars
        )):
            # Strip PEFT name prefix (peft prepends "base_model.model.")
            stripped = name.replace("base_model.model.", "")
            if stripped in pre_lora_trainable:
                p.requires_grad = True
                n_set += 1
    print(f"{log_prefix} marked {n_set} adapter tensors trainable")
    model.print_trainable_parameters()

    text = get_text_corpus("wikitext_2_train", max_chars=args.corpus_chars)
    blocks = list(chunks(text, tok, block_size=args.seq_len))
    print(f"{log_prefix} corpus: {len(blocks)} blocks of {args.seq_len + 1}")

    device = next(model.parameters()).device
    # Split trainable params into a sensitive group (small LR) and the
    # rest. The sensitive group currently captures log_A from the SSM
    # adapter (Mamba's gradient-amplifying parameter); other adapter
    # types have no entries here so this becomes a no-op for them.
    sensitive = []
    standard = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_ssm_sensitive_param(name):
            sensitive.append(p)
        else:
            standard.append(p)
    trainable = standard + sensitive   # used below for clip_grad_norm
    n_train = sum(p.numel() for p in standard) + sum(p.numel() for p in sensitive)
    print(f"{log_prefix} trainable params: {n_train:,}  "
          f"(standard {sum(p.numel() for p in standard):,}, "
          f"sensitive {sum(p.numel() for p in sensitive):,})")
    param_groups = [{"params": standard, "lr": args.lr}]
    if sensitive:
        param_groups.append({"params": sensitive, "lr": args.lr_sensitive})
        print(f"{log_prefix} sensitive param group (e.g. log_A) lr = {args.lr_sensitive}")
    optim = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.0)

    model.train()
    log = []
    step = 0
    t0 = time.time()
    while step < args.max_steps:
        for block in blocks:
            block = block.to(device)
            x = block[:-1].unsqueeze(0)
            y = block[1:].unsqueeze(0)
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
            log.append({"step": step, "loss": float(loss.item())})
            if step % 25 == 0:
                print(f"{log_prefix} step {step:4d}  loss={float(loss.item()):.4f}")
            step += 1
            if step >= args.max_steps:
                break
    t_train = time.time() - t0
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    return {
        "elapsed_seconds": t_train,
        "final_loss": log[-1]["loss"] if log else None,
        "n_steps": len(log),
        "n_trainable": n_train,
    }


def measure_arm(model, tok, eval_text, args) -> dict:
    ppl = measure_perplexity(model, tok, eval_text, block_size=args.seq_len,
                                max_blocks=args.max_eval_blocks)
    vram = measure_vram_forward(model, tok, args.prompt * 50, seq_len=args.seq_len)
    thr = measure_throughput(model, tok, args.prompt, max_new_tokens=args.gen_new)
    return {"perplexity": ppl, "vram": vram, "throughput": thr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--gen_new", type=int, default=32)
    parser.add_argument("--max_eval_blocks", type=int, default=4)
    parser.add_argument("--prompt", type=str,
                          default="Explain the second law of thermodynamics in three sentences.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_sensitive", type=float, default=1e-5,
                          help="LR for gradient-sensitive params (SSM log_A). "
                          "10-50x smaller than --lr is Mamba's recipe.")
    parser.add_argument("--corpus_chars", type=int, default=200_000)
    parser.add_argument("--r", type=int, default=32,
                          help="ResonanceAdapter bottleneck width")
    parser.add_argument("--every", type=int, default=1,
                          help="Insert adapter every Nth layer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", type=str,
                          default="baseline,mlp_adapter,resonance_adapter,ssm_adapter",
                          help="Comma-separated arms to run")
    parser.add_argument("--out", type=str, default="/tmp/paired_resonance_ab.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        eval_text = "\n".join(s for s in ds["text"] if s.strip())[:200_000]
    except Exception:
        eval_text = ("Thermodynamics is the branch of physics that studies. " * 2000)[:200_000]
    print(f"Eval text: {len(eval_text)} chars")

    results = {"args": vars(args), "arms": {}}

    for arm in arms:
        print()
        print("=" * 70)
        print(f"ARM: {arm}")
        print("=" * 70)
        torch.manual_seed(args.seed)   # same init for fair compare
        model, tok, summary = make_arm_model(
            arm, args.base, dtype, args.device_map, r=args.r, every=args.every,
        )
        print(f"  adapter summary: {summary}")
        pre = measure_arm(model, tok, eval_text, args)
        print(f"  pre-LoRA ppl: {pre['perplexity']['perplexity']:.3f}  "
              f"vram: {pre['vram']['peak_vram_gb']:.2f} GB  "
              f"tok/s: {pre['throughput']['tokens_per_second']:.2f}")
        train_log = train_arm(model, tok, args, log_prefix=f"  [{arm}]")
        print(f"  training done in {train_log['elapsed_seconds']:.1f} s, "
              f"final loss {train_log['final_loss']:.4f}")
        post = measure_arm(model, tok, eval_text, args)
        print(f"  post-LoRA ppl: {post['perplexity']['perplexity']:.3f}  "
              f"vram: {post['vram']['peak_vram_gb']:.2f} GB  "
              f"tok/s: {post['throughput']['tokens_per_second']:.2f}")

        results["arms"][arm] = {
            "summary": summary,
            "pre": pre,
            "post": post,
            "training": train_log,
        }
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    # Print A/B summary
    print()
    print("=" * 70)
    print("PAIRED A/B SUMMARY (Qwen2.5-7B-Instruct, WikiText-2 test ppl)")
    print("=" * 70)
    print(f"  {'arm':<22s} {'pre-ppl':>10s} {'post-ppl':>10s} {'vram (GB)':>10s} {'tok/s':>10s}")
    for arm in arms:
        r = results["arms"][arm]
        print(f"  {arm:<22s} "
              f"{r['pre']['perplexity']['perplexity']:>10.3f} "
              f"{r['post']['perplexity']['perplexity']:>10.3f} "
              f"{r['post']['vram']['peak_vram_gb']:>10.2f} "
              f"{r['post']['throughput']['tokens_per_second']:>10.2f}")

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults: {args.out}")


if __name__ == "__main__":
    main()
