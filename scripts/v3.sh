#!/bin/sh
set -e
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TQDM_DISABLE=1
PY=${PY:-.venv/bin/python}
for m in mlx-community/Qwen2.5-1.5B-Instruct-4bit mlx-community/Qwen2.5-3B-Instruct-4bit mlx-community/Llama-3.2-3B-Instruct-4bit mlx-community/Qwen2.5-7B-Instruct-4bit; do
  $PY -m harness.cli run --model $m --prompt v3
done
