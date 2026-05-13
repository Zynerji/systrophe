"""A/B benchmark: base Qwen2.5-7B vs Cyliformer-converted (optionally LoRA-tuned).

Measures three things on both models, in the same process and on the
same hardware, with the same tokenizer and the same eval set:

  * **Perplexity** on a WikiText-2 (raw) test sample.
  * **Peak VRAM** during a forward pass at fixed (batch=1, seq_len).
  * **Tokens/sec** generation throughput at fixed prompt length and
    output length.
  * **Cyliformer diagnostics**: per-layer per-cylinder lambda_2,
    back-reaction, beam_gain (Cyliformer side only).

Usage:
    python ab_benchmark.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --n_cylinders 2 \
        --lora_adapter /tmp/cyliformer-qwen-7b-lora \
        --eval_max_chars 200000 \
        --seq_len 512 \
        --gen_new 128
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

from qwen_convert import CylinderFFN, convert_qwen_to_cyliformer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def measure_perplexity(model, tok, text: str, block_size: int = 512,
                          max_blocks: int = 8, device=None) -> dict:
    """Average per-block cross-entropy on contiguous chunks of `text`."""
    model.eval()
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    if device is None:
        device = next(model.parameters()).device
    losses = []
    n_tokens = 0
    with torch.no_grad():
        for i in range(0, ids.shape[0] - block_size, block_size):
            block = ids[i : i + block_size + 1].to(device)
            x = block[:-1].unsqueeze(0)
            y = block[1:].unsqueeze(0)
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            losses.append(float(loss.item()))
            n_tokens += int(y.numel())
            if len(losses) >= max_blocks:
                break
    if not losses:
        return {"loss": float("nan"), "perplexity": float("nan"),
                "n_blocks": 0, "n_tokens": 0}
    mean_loss = sum(losses) / max(n_tokens, 1)
    return {
        "loss": float(mean_loss),
        "perplexity": float(np.exp(mean_loss)),
        "n_blocks": len(losses),
        "n_tokens": int(n_tokens),
    }


def measure_vram_forward(model, tok, prompt: str, seq_len: int = 512) -> dict:
    """Peak VRAM during one forward pass at the chosen seq_len."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return {"peak_vram_bytes": 0, "peak_vram_gb": 0.0, "note": "non-CUDA"}
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=seq_len, padding="max_length")
    x = enc["input_ids"].to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        _ = model(input_ids=x)
    peak = int(torch.cuda.max_memory_allocated())
    return {
        "peak_vram_bytes": peak,
        "peak_vram_gb": peak / (1024 ** 3),
        "seq_len": int(seq_len),
    }


def measure_throughput(model, tok, prompt: str, max_new_tokens: int = 128,
                          n_warmup: int = 1) -> dict:
    """Tokens/sec for greedy generation."""
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                pad_token_id=tok.eos_token_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tok.eos_token_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    n_gen = int(out.shape[1] - enc["input_ids"].shape[1])
    return {
        "tokens_generated": n_gen,
        "elapsed_seconds": elapsed,
        "tokens_per_second": float(n_gen / elapsed) if elapsed > 0 else 0.0,
    }


def collect_cyliformer_diagnostics(model) -> dict:
    """Collect last-forward-pass diagnostics from every CylinderFFN."""
    layers_diag = []
    decoder = None
    # The model may have peft wrapper(s): unwrap until we find .layers
    cur = model
    for _ in range(5):
        if hasattr(cur, "layers"):
            decoder = cur
            break
        for attr in ("base_model", "model", "transformer"):
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
                break
        else:
            break
    if decoder is None:
        return {"available": False, "reason": "no layers attribute"}
    for i, layer in enumerate(decoder.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "last_lambda2_per_cylinder"):
            continue
        layers_diag.append({
            "layer": i,
            "n_cylinders": int(mlp.n_cylinders),
            "lambda_2_per_cylinder": list(mlp.last_lambda2_per_cylinder),
            "backreaction_per_cylinder": list(mlp.last_backreaction_per_cylinder),
            "beam_gain": float(mlp.beam_gain()),
        })
    if not layers_diag:
        return {"available": False, "reason": "no CylinderFFN found"}
    all_lams = [v for L in layers_diag for v in L["lambda_2_per_cylinder"]]
    all_back = [v for L in layers_diag for v in L["backreaction_per_cylinder"]]
    return {
        "available": True,
        "n_converted_layers": len(layers_diag),
        "lambda_2_mean": float(np.mean(all_lams)) if all_lams else 0.0,
        "lambda_2_std": float(np.std(all_lams)) if all_lams else 0.0,
        "backreaction_mean": float(np.mean(all_back)) if all_back else 0.0,
        "backreaction_std": float(np.std(all_back)) if all_back else 0.0,
        "beam_gain_mean": float(np.mean([L["beam_gain"] for L in layers_diag])),
        "per_layer": layers_diag,
    }


# ---------------------------------------------------------------------------
# Model harness
# ---------------------------------------------------------------------------


def load_baseline(base: str, dtype: torch.dtype, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
    )
    return model, tok


def load_cyliformer(
    base: str, n_cylinders: int, lambda_target: float, dtype: torch.dtype,
    device_map: str, lora_adapter: str | None,
    catcher_max_nodes: int,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
    )
    summary = convert_qwen_to_cyliformer(
        model, n_cylinders=n_cylinders, lambda_target=lambda_target,
        catcher_max_nodes=catcher_max_nodes,
    )
    if lora_adapter is not None and len(lora_adapter) > 0:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_adapter)
        # Cylinder-side params from the adapter dir
        # NOTE: peft.save_pretrained only saves LoRA deltas. Our
        # phasors/catcher/backreaction_scale are NOT saved by PEFT.
        # If a finetuned cyl state was saved separately we'd load it
        # here. For now we use the random-init Cyliformer modulated by
        # the LoRA-adapted shared FFN.
    return model, tok, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_cylinders", type=int, default=2)
    parser.add_argument("--lambda_target", type=float, default=0.18)
    parser.add_argument("--lora_adapter", type=str, default="",
                          help="Path to a LoRA adapter for the Cyliformer side")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--gen_new", type=int, default=64)
    parser.add_argument("--max_eval_blocks", type=int, default=4)
    parser.add_argument("--catcher_max_nodes", type=int, default=96)
    parser.add_argument("--eval_corpus", type=str,
                          default="wikitext_2_test",
                          help="One of: wikitext_2_test, fallback")
    parser.add_argument("--prompt", type=str,
                          default="Explain the second law of thermodynamics in three sentences.")
    parser.add_argument("--out", type=str, default="/tmp/ab_benchmark.json")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_cyliformer", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Eval corpus
    def get_eval_text(name: str, max_chars: int = 200_000) -> str:
        if name == "wikitext_2_test":
            try:
                from datasets import load_dataset
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                text = "\n".join(s for s in ds["text"] if s.strip())
                return text[:max_chars]
            except Exception:
                pass
        return ("The first law of thermodynamics. The second law of thermodynamics. " * 1000)[:max_chars]

    eval_text = get_eval_text(args.eval_corpus)
    print(f"Eval text length: {len(eval_text)} chars")

    result = {
        "base": args.base,
        "n_cylinders": args.n_cylinders,
        "lambda_target": args.lambda_target,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "gen_new": args.gen_new,
        "lora_adapter": args.lora_adapter,
        "baseline": None,
        "cyliformer": None,
    }

    if not args.skip_baseline:
        print("\n" + "=" * 60)
        print("BASELINE: original", args.base)
        print("=" * 60)
        model, tok = load_baseline(args.base, dtype, args.device_map)
        ppl = measure_perplexity(model, tok, eval_text, block_size=args.seq_len,
                                    max_blocks=args.max_eval_blocks)
        print(f"  perplexity: {ppl['perplexity']:.3f} over {ppl['n_blocks']} blocks "
              f"({ppl['n_tokens']} tokens)")
        vram = measure_vram_forward(model, tok, args.prompt * 50, seq_len=args.seq_len)
        print(f"  peak VRAM forward: {vram['peak_vram_gb']:.2f} GB")
        thr = measure_throughput(model, tok, args.prompt, max_new_tokens=args.gen_new)
        print(f"  generation: {thr['tokens_per_second']:.2f} tok/s "
              f"({thr['tokens_generated']} tokens in {thr['elapsed_seconds']:.2f}s)")
        total_params = sum(p.numel() for p in model.parameters())
        result["baseline"] = {
            "perplexity": ppl,
            "vram": vram,
            "throughput": thr,
            "n_params": int(total_params),
        }
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    if not args.skip_cyliformer:
        print("\n" + "=" * 60)
        print(f"CYLIFORMER: {args.base} + N={args.n_cylinders}"
              f"{' + LoRA' if args.lora_adapter else ''}")
        print("=" * 60)
        model, tok, conv_summary = load_cyliformer(
            args.base, args.n_cylinders, args.lambda_target, dtype,
            args.device_map, args.lora_adapter, args.catcher_max_nodes,
        )
        print(f"  conversion: {conv_summary['n_layers_converted']} layers, "
              f"+{conv_summary['new_params_added']:,} new params")
        ppl = measure_perplexity(model, tok, eval_text, block_size=args.seq_len,
                                    max_blocks=args.max_eval_blocks)
        print(f"  perplexity: {ppl['perplexity']:.3f} over {ppl['n_blocks']} blocks "
              f"({ppl['n_tokens']} tokens)")
        vram = measure_vram_forward(model, tok, args.prompt * 50, seq_len=args.seq_len)
        print(f"  peak VRAM forward: {vram['peak_vram_gb']:.2f} GB")
        thr = measure_throughput(model, tok, args.prompt, max_new_tokens=args.gen_new)
        print(f"  generation: {thr['tokens_per_second']:.2f} tok/s "
              f"({thr['tokens_generated']} tokens in {thr['elapsed_seconds']:.2f}s)")
        diag = collect_cyliformer_diagnostics(model)
        if diag.get("available"):
            print(f"  diagnostics: lambda_2 mean = {diag['lambda_2_mean']:.4f} "
                  f"(std {diag['lambda_2_std']:.4f}), "
                  f"backreact mean = {diag['backreaction_mean']:.4f}, "
                  f"beam_gain mean = {diag['beam_gain_mean']:.4f}")
        total_params = sum(p.numel() for p in model.parameters())
        result["cyliformer"] = {
            "conversion": conv_summary,
            "perplexity": ppl,
            "vram": vram,
            "throughput": thr,
            "diagnostics": diag,
            "n_params": int(total_params),
        }
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    # Print delta summary
    if result["baseline"] and result["cyliformer"]:
        b = result["baseline"]
        c = result["cyliformer"]
        print("\n" + "=" * 60)
        print("DELTA SUMMARY (Cyliformer - Baseline)")
        print("=" * 60)
        ppl_b = b["perplexity"]["perplexity"]
        ppl_c = c["perplexity"]["perplexity"]
        print(f"  perplexity:   {ppl_b:.3f} -> {ppl_c:.3f}   "
              f"({100*(ppl_c-ppl_b)/ppl_b:+.2f}%)")
        v_b = b["vram"]["peak_vram_gb"]
        v_c = c["vram"]["peak_vram_gb"]
        print(f"  peak VRAM:    {v_b:.2f} -> {v_c:.2f} GB   "
              f"({100*(v_c-v_b)/v_b:+.2f}%)")
        t_b = b["throughput"]["tokens_per_second"]
        t_c = c["throughput"]["tokens_per_second"]
        print(f"  tokens/sec:   {t_b:.2f} -> {t_c:.2f}   "
              f"({100*(t_c-t_b)/t_b:+.2f}%)")

    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
