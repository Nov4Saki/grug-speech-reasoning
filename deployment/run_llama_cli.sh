#!/usr/bin/env bash
# Script to run Grug Speech models locally using llama.cpp
set -euo pipefail

MODEL_PATH="${1:-./Qwen3.5-4B-GrugSpeech-Q4_K_M.gguf}"
LLAMA_CLI="${LLAMA_CLI:-./llama-cli}"
PROMPT="${2:-Explain why DATE(created_at) = '2026-10-01' is slow in SQL and provide the index-friendly fix.}"

echo "Loading model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "--------------------------------------------------------"

"$LLAMA_CLI" \
  -m "$MODEL_PATH" \
  -p "<|im_start|>system\nYou are Qwen 3.5. You reason internally in Grug Speech inside <think> tags.<|im_end|>\n<|im_start|>user\n${PROMPT}<|im_end|>\n<|im_start|>assistant\n" \
  -n 512 \
  --temp 0.6 \
  -ngl 99
