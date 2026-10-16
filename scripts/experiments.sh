#!/bin/sh
# Full experiment: 4 models x 2 prompts greedy, plus a reliability run (5 samples, T=0.7) for Qwen2.5-3B on both prompts.
set -e
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TQDM_DISABLE=1
PY=${PY:-.venv/bin/python}
for m in mlx-community/Qwen2.5-1.5B-Instruct-4bit mlx-community/Qwen2.5-3B-Instruct-4bit mlx-community/Qwen2.5-7B-Instruct-4bit mlx-community/Llama-3.2-3B-Instruct-4bit; do
  for p in v1 v2; do $PY -m harness.cli run --model $m --prompt $p; done
done
for p in v1 v2; do $PY -m harness.cli run --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt $p --temperature 0.7 --samples 5; done
