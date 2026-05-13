"""Honest control: baseline Qwen2.5-7B + same LoRA fine-tune, NO Cyliformer.

Trains a vanilla LoRA on gate_proj/up_proj/down_proj of the unmodified
Qwen2.5-7B, on the same WikiText-2 train chunks as the Cyliformer LoRA
run, then measures perplexity on the same WikiText-2 test chunks.

If baseline+LoRA reaches the same perplexity as Cyliformer+LoRA, the
Cyliformer architecture is *not* the source of the improvement -- the
improvement is just LoRA fine-tuning to the eval distribution. If
Cyliformer+LoRA is meaningfully better, that's evidence for the
architectural contribution.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ab_benchmark import (
    measure_perplexity,
    measure_throughput,
    measure_vram_forward,
)
from qwen_lora_finetune import chunks, get_text_corpus, setup_lora


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
    parser.add_argument("--corpus_chars", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="/tmp/baseline_lora_control.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=" * 64)
    print("BASELINE + LoRA control (no Cyliformer)")
    print("=" * 64)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=args.device_map,
        trust_remote_code=True,
    )

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        eval_text = "\n".join(s for s in ds["text"] if s.strip())[:200_000]
    except Exception:
        eval_text = ("Thermodynamics is the branch of physics that studies. " * 2000)[:200_000]
    print(f"Eval text: {len(eval_text)} chars")

    # Pre-train perplexity (sanity)
    print("Pre-LoRA perplexity:")
    pre = measure_perplexity(model, tok, eval_text, block_size=args.seq_len,
                                max_blocks=args.max_eval_blocks)
    print(f"  baseline ppl: {pre['perplexity']:.3f}")

    # LoRA setup
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    print()
    model.print_trainable_parameters()

    # Train on same WikiText-2 train chunks
    text = get_text_corpus("wikitext_2_train", max_chars=args.corpus_chars)
    blocks = list(chunks(text, tok, block_size=args.seq_len))
    print(f"Corpus: {len(blocks)} blocks of {args.seq_len + 1}")

    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

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
            if step % 10 == 0:
                print(f"  step {step:4d}  loss={float(loss.item()):.4f}")
            step += 1
            if step >= args.max_steps:
                break
    t_train = time.time() - t0
    print(f"Training done in {t_train:.1f} s")

    # Re-measure
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    print("\nPost-LoRA perplexity:")
    post = measure_perplexity(model, tok, eval_text, block_size=args.seq_len,
                                 max_blocks=args.max_eval_blocks)
    print(f"  baseline+LoRA ppl: {post['perplexity']:.3f}")
    vram = measure_vram_forward(model, tok, args.prompt * 50, seq_len=args.seq_len)
    print(f"  peak VRAM: {vram['peak_vram_gb']:.2f} GB")
    thr = measure_throughput(model, tok, args.prompt, max_new_tokens=args.gen_new)
    print(f"  tokens/s: {thr['tokens_per_second']:.2f}")

    out = {
        "args": vars(args),
        "pre_lora_perplexity": pre,
        "post_lora_perplexity": post,
        "vram": vram,
        "throughput": thr,
        "training": {
            "elapsed_seconds": t_train,
            "final_loss": log[-1]["loss"] if log else None,
            "n_steps": len(log),
        },
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nResults: {args.out}")
    print()
    print(f"DELTA: baseline {pre['perplexity']:.3f} -> baseline+LoRA "
          f"{post['perplexity']:.3f} ({100*(post['perplexity']-pre['perplexity'])/pre['perplexity']:+.2f}%)")


if __name__ == "__main__":
    main()
