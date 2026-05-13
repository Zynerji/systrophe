#!/usr/bin/env bash
# End-to-end A/B run on a single GPU.
#
# Steps:
#   1. Baseline measurement (perplexity + VRAM + tok/s on raw Qwen2.5-7B).
#   2. Cyliformer zero-shot measurement (convert + benchmark, no LoRA).
#   3. LoRA on new params + light LoRA over shared FFN linears.
#   4. Cyliformer + LoRA measurement.
#
# Each step writes its own results JSON; the final summary is printed.
set -euo pipefail

BASE="${BASE:-Qwen/Qwen2.5-7B-Instruct}"
N_CYL="${N_CYL:-2}"
LAMBDA_TARGET="${LAMBDA_TARGET:-0.18}"
SEQ_LEN="${SEQ_LEN:-512}"
GEN_NEW="${GEN_NEW:-64}"
MAX_BLOCKS="${MAX_BLOCKS:-4}"
LORA_STEPS="${LORA_STEPS:-150}"
LORA_DIR="${LORA_DIR:-/tmp/cyliformer-qwen-7b-lora}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/cyliformer_ab_results}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "${RESULTS_DIR}"
cd "$(dirname "$0")"

echo "== Step 1/3: A/B (no LoRA yet) =="
python ab_benchmark.py \
    --base "${BASE}" \
    --n_cylinders "${N_CYL}" \
    --lambda_target "${LAMBDA_TARGET}" \
    --seq_len "${SEQ_LEN}" \
    --gen_new "${GEN_NEW}" \
    --max_eval_blocks "${MAX_BLOCKS}" \
    --dtype "${DTYPE}" \
    --out "${RESULTS_DIR}/ab_no_lora.json"

echo
echo "== Step 2/3: LoRA fine-tune (Cyliformer side) =="
python qwen_lora_finetune.py \
    --base "${BASE}" \
    --n_cylinders "${N_CYL}" \
    --lambda_target "${LAMBDA_TARGET}" \
    --max_steps "${LORA_STEPS}" \
    --block_size "${SEQ_LEN}" \
    --out "${LORA_DIR}" \
    --dtype "${DTYPE}"

echo
echo "== Step 3/3: A/B with LoRA-tuned Cyliformer =="
python ab_benchmark.py \
    --base "${BASE}" \
    --n_cylinders "${N_CYL}" \
    --lambda_target "${LAMBDA_TARGET}" \
    --seq_len "${SEQ_LEN}" \
    --gen_new "${GEN_NEW}" \
    --max_eval_blocks "${MAX_BLOCKS}" \
    --lora_adapter "${LORA_DIR}" \
    --dtype "${DTYPE}" \
    --skip_baseline \
    --out "${RESULTS_DIR}/ab_with_lora.json"

echo
echo "All results in ${RESULTS_DIR}"
ls -la "${RESULTS_DIR}"
