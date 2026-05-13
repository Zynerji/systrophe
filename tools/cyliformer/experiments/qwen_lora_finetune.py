"""LoRA fine-tune of a Cyliformer-converted Qwen2.5-7B.

Strategy ("LoRA on new params"):
  * **Fully trainable**: every newly-added Cyliformer parameter
    (catcher projection weights + biases, phasors, backreaction_scale).
    These are tiny and per-layer.
  * **LoRA**: applied to the shared FFN linears (gate_proj, up_proj,
    down_proj) so the original Qwen weights are not disturbed but can
    adapt to the cylinder-loop calling convention.
  * **Frozen**: everything else (embeddings, attention, RMSNorm, LM head).

Dataset: a small chunk of a clean text corpus by default (WikiText-2
train split). The goal is *not* to teach the model new facts, but to
let it adjust to the new FFN structure (N rotated views averaged into
one output) and the catcher signal.

Run:
    python qwen_lora_finetune.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --n_cylinders 2 \
        --max_steps 200 \
        --out /tmp/cyliformer-qwen-7b-lora
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

from cyliformer.catcher import LearnedAddressCatcher
from qwen_convert import CylinderFFN, convert_qwen_to_cyliformer


def setup_lora(model, lora_r: int = 8, lora_alpha: int = 16,
                  lora_dropout: float = 0.0):
    """Apply LoRA to gate_proj/up_proj/down_proj. Requires peft."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    return model


def mark_cylinder_params_trainable(model):
    """Ensure phasors / catcher / backreaction_scale stay trainable even
    after PEFT freezes everything else."""
    n_set = 0
    for name, p in model.named_parameters():
        # Catch params added by CylinderFFN -- they appear under
        # ".mlp.phasors", ".mlp.catcher.proj.*", ".mlp.backreaction_scale".
        if any(token in name for token in (
            "mlp.phasors",
            "mlp.catcher.",
            "mlp.backreaction_scale",
            # PEFT may rename to base_model.model.model.layers.<i>.mlp...
        )):
            p.requires_grad = True
            n_set += 1
    return n_set


def get_text_corpus(name: str = "wikitext_2_train", max_chars: int = 200_000) -> str:
    """Tiny corpus loader. By default tries to fetch WikiText-2 via
    datasets, falls back to a built-in lorem ipsum if datasets is
    unavailable."""
    if name == "wikitext_2_train":
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n".join(s for s in ds["text"] if s.strip())
            return text[:max_chars]
        except Exception as exc:  # pragma: no cover
            print(f"datasets unavailable ({exc}); using fallback text")
    # Fallback: repeated boilerplate to at least exercise the trainer
    return ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 500)[:max_chars]


def chunks(text: str, tok, block_size: int):
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    for i in range(0, ids.shape[0] - block_size, block_size):
        yield ids[i : i + block_size + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_cylinders", type=int, default=2)
    parser.add_argument("--lambda_target", type=float, default=0.18)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_weight", type=float, default=0.1)
    parser.add_argument("--corpus_chars", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="/tmp/cyliformer-qwen-7b-lora")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--catcher_max_nodes", type=int, default=96)
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"Loading base model {args.base} (dtype={args.dtype})")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=args.device_map,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print(f"Converting -> Cyliformer with n_cylinders={args.n_cylinders}")
    conv_summary = convert_qwen_to_cyliformer(
        model,
        n_cylinders=args.n_cylinders,
        lambda_target=args.lambda_target,
        catcher_max_nodes=args.catcher_max_nodes,
    )
    print(f"  {conv_summary['n_layers_converted']} layers, "
          f"+{conv_summary['new_params_added']:,} new params")

    print(f"Setting up LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    model = setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    n_new_set = mark_cylinder_params_trainable(model)
    print(f"  marked {n_new_set} Cyliformer-side parameter tensors as trainable")
    model.print_trainable_parameters()

    print(f"Loading corpus (target {args.corpus_chars} chars)")
    text = get_text_corpus("wikitext_2_train", max_chars=args.corpus_chars)
    print(f"  corpus length: {len(text)} chars")

    # Materialise into a list of token blocks for simple iteration
    blocks = list(chunks(text, tok, block_size=args.block_size))
    print(f"  {len(blocks)} blocks of {args.block_size + 1} tokens")
    if len(blocks) == 0:
        raise SystemExit("corpus too small for chosen block_size")

    device = next(model.parameters()).device

    # Optimizer over trainable params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"  trainable parameters: {n_train:,}")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # Train
    model.train()
    log = []
    step = 0
    t_start = time.time()
    while step < args.max_steps:
        for batch_idx in range(len(blocks)):
            block = blocks[batch_idx].to(device)
            input_ids = block[:-1].unsqueeze(0)
            labels = block[1:].unsqueeze(0)
            out = model(input_ids=input_ids, labels=None)
            logits = out.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            # Optional lambda_2 floor penalty across all converted layers
            if args.lambda_weight > 0:
                lam_list = []
                for layer in model.base_model.model.model.layers:
                    cyl = getattr(layer, "mlp", None)
                    if cyl is None or not hasattr(cyl, "last_lambda2_per_cylinder"):
                        continue
                    lam_list.extend(cyl.last_lambda2_per_cylinder)
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
                print(f"step {step:4d}  loss={float(loss.item()):.4f}")
            step += 1
            if step >= args.max_steps:
                break

    elapsed = time.time() - t_start
    print(f"Training done in {elapsed:.1f} s")

    print(f"Saving LoRA adapter to {args.out}")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "lora_finetune.json").write_text(json.dumps({
        "base": args.base,
        "n_cylinders": args.n_cylinders,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "block_size": args.block_size,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "lambda_weight": args.lambda_weight,
        "lambda_target": args.lambda_target,
        "final_loss": log[-1]["loss"] if log else None,
        "elapsed_seconds": elapsed,
        "conversion_summary": conv_summary,
    }, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
