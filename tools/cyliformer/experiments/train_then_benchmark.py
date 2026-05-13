"""End-to-end: load Qwen + convert + LoRA fine-tune + benchmark, in ONE process.

Why in one process: PEFT's `save_pretrained` only persists the LoRA
deltas, NOT the new cylinder parameters (phasors, catcher projection,
backreaction_scale). Saving + reloading in two processes loses the
trained cylinder state. To get an honest LoRA-tuned A/B number we
have to keep everything in memory.

This also doubles as the canonical "fast PoC" entry point.
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

from ab_benchmark import (
    collect_cyliformer_diagnostics,
    measure_perplexity,
    measure_throughput,
    measure_vram_forward,
)
from qwen_convert import convert_qwen_to_cyliformer
from qwen_lora_finetune import (
    chunks,
    get_text_corpus,
    mark_cylinder_params_trainable,
    setup_lora,
)


def measure_all(model, tok, eval_text, args, label: str) -> dict:
    """Run the three benchmarks + diagnostics."""
    print(f"\n--- {label} ---")
    ppl = measure_perplexity(model, tok, eval_text, block_size=args.seq_len,
                                max_blocks=args.max_eval_blocks)
    print(f"  perplexity: {ppl['perplexity']:.3f} "
          f"(over {ppl['n_blocks']} blocks, {ppl['n_tokens']} tokens)")
    vram = measure_vram_forward(model, tok, args.prompt * 50, seq_len=args.seq_len)
    print(f"  peak VRAM:  {vram['peak_vram_gb']:.2f} GB")
    thr = measure_throughput(model, tok, args.prompt, max_new_tokens=args.gen_new)
    print(f"  tokens/s:   {thr['tokens_per_second']:.2f}")
    diag = collect_cyliformer_diagnostics(model)
    if diag.get("available"):
        print(f"  diagnostics: lambda_2={diag['lambda_2_mean']:.4f}+-{diag['lambda_2_std']:.4f}  "
              f"backreact={diag['backreaction_mean']:.4f}  "
              f"beam_gain={diag['beam_gain_mean']:.4f}")
    return {
        "label": label,
        "perplexity": ppl,
        "vram": vram,
        "throughput": thr,
        "diagnostics": diag,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_cylinders", type=int, default=2)
    parser.add_argument("--lambda_target", type=float, default=0.18)
    parser.add_argument("--lambda_weight", type=float, default=0.10)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--gen_new", type=int, default=32)
    parser.add_argument("--max_eval_blocks", type=int, default=4)
    parser.add_argument("--catcher_max_nodes", type=int, default=64)
    parser.add_argument("--prompt", type=str,
                          default="Explain the second law of thermodynamics in three sentences.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--corpus_chars", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="/tmp/train_then_benchmark.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # ----- 1. Baseline measurement -----
    print("=" * 64)
    print(f"BASELINE: {args.base}")
    print("=" * 64)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=args.device_map,
        trust_remote_code=True,
    )

    # Eval text
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        eval_text = "\n".join(s for s in ds["text"] if s.strip())[:200_000]
    except Exception:
        eval_text = ("Thermodynamics is the branch of physics that studies. " * 2000)[:200_000]
    print(f"Eval text: {len(eval_text)} chars")

    baseline = measure_all(model, tok, eval_text, args, "BASELINE Qwen2.5-7B")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ----- 2. Cyliformer (zero-shot) -----
    print()
    print("=" * 64)
    print(f"CYLIFORMER (zero-shot)  n_cyl={args.n_cylinders}")
    print("=" * 64)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=args.device_map,
        trust_remote_code=True,
    )
    conv_summary = convert_qwen_to_cyliformer(
        model, n_cylinders=args.n_cylinders,
        lambda_target=args.lambda_target,
        catcher_max_nodes=args.catcher_max_nodes,
    )
    print(f"Conversion: {conv_summary['n_layers_converted']} layers, "
          f"+{conv_summary['new_params_added']:,} new params")

    zero_shot = measure_all(model, tok, eval_text, args,
                              f"CYLIFORMER zero-shot (n={args.n_cylinders})")

    # ----- 3. LoRA fine-tune in place -----
    print()
    print("=" * 64)
    print(f"LoRA FINE-TUNE   r={args.lora_r}  alpha={args.lora_alpha}  "
          f"steps={args.max_steps}  lr={args.lr}")
    print("=" * 64)
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    n_set = mark_cylinder_params_trainable(model)
    print(f"Marked {n_set} cylinder-side param tensors trainable")
    model.print_trainable_parameters()

    text = get_text_corpus("wikitext_2_train", max_chars=args.corpus_chars)
    blocks = list(chunks(text, tok, block_size=args.seq_len))
    print(f"Corpus: {len(text)} chars, {len(blocks)} blocks of {args.seq_len + 1}")
    if len(blocks) == 0:
        raise SystemExit("corpus too small")

    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_train:,}")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    model.train()
    log = []
    step = 0
    t_train_start = time.time()
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
            if args.lambda_weight > 0:
                lam_list = []
                # Find decoder layers
                cur = model
                for _ in range(5):
                    if hasattr(cur, "layers"):
                        break
                    for a in ("base_model", "model", "transformer"):
                        if hasattr(cur, a):
                            cur = getattr(cur, a)
                            break
                    else:
                        break
                for layer in getattr(cur, "layers", []):
                    mlp = getattr(layer, "mlp", None)
                    if mlp is not None and hasattr(mlp, "last_lambda2_per_cylinder"):
                        lam_list.extend(mlp.last_lambda2_per_cylinder)
                if lam_list:
                    lam_t = torch.tensor(lam_list, dtype=loss.dtype, device=loss.device)
                    floor = F.relu(args.lambda_target - lam_t).pow(2).mean()
                    loss = loss + args.lambda_weight * floor
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

    t_train = time.time() - t_train_start
    print(f"Training done in {t_train:.1f} s, final loss {log[-1]['loss']:.4f}")

    # ----- 4. Cyliformer + LoRA measurement -----
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    print()
    print("=" * 64)
    print("CYLIFORMER + LoRA")
    print("=" * 64)
    with_lora = measure_all(model, tok, eval_text, args,
                              f"CYLIFORMER + LoRA r={args.lora_r}")

    # ----- 5. Summary -----
    print()
    print("=" * 64)
    print("FINAL A/B SUMMARY")
    print("=" * 64)
    ppl_b = baseline["perplexity"]["perplexity"]
    ppl_z = zero_shot["perplexity"]["perplexity"]
    ppl_l = with_lora["perplexity"]["perplexity"]
    print(f"  perplexity:   baseline={ppl_b:.3f}   "
          f"zero-shot={ppl_z:.3f} ({100*(ppl_z-ppl_b)/ppl_b:+.2f}%)   "
          f"+LoRA={ppl_l:.3f} ({100*(ppl_l-ppl_b)/ppl_b:+.2f}%)")
    v_b = baseline["vram"]["peak_vram_gb"]
    v_z = zero_shot["vram"]["peak_vram_gb"]
    v_l = with_lora["vram"]["peak_vram_gb"]
    print(f"  peak VRAM:    baseline={v_b:.2f}   "
          f"zero-shot={v_z:.2f} ({100*(v_z-v_b)/v_b:+.2f}%)   "
          f"+LoRA={v_l:.2f} ({100*(v_l-v_b)/v_b:+.2f}%)")
    t_b = baseline["throughput"]["tokens_per_second"]
    t_z = zero_shot["throughput"]["tokens_per_second"]
    t_l = with_lora["throughput"]["tokens_per_second"]
    print(f"  tokens/sec:   baseline={t_b:.2f}   "
          f"zero-shot={t_z:.2f} ({100*(t_z-t_b)/t_b:+.2f}%)   "
          f"+LoRA={t_l:.2f} ({100*(t_l-t_b)/t_b:+.2f}%)")

    out = {
        "args": vars(args),
        "conversion": conv_summary,
        "baseline": baseline,
        "cyliformer_zero_shot": zero_shot,
        "cyliformer_lora": with_lora,
        "training": {
            "elapsed_seconds": t_train,
            "final_loss": log[-1]["loss"] if log else None,
            "n_steps": len(log),
        },
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nResults: {args.out}")


if __name__ == "__main__":
    main()
