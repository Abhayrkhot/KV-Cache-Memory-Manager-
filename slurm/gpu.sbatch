#!/usr/bin/env bash
#SBATCH --job-name=pagedkv-gpu
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/pagedkv/logs/%x-%j.out
# GPU correctness suite, then the kernel and serving benchmarks; writes results/<gpu>/.
#   sbatch --constraint=a100@80g slurm/gpu.sbatch
#   sbatch --gres=gpu:h200:1 slurm/gpu.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' )
OUT=results/gpu/$GPU && mkdir -p "$OUT"
# Benchmarks only count if the kernels pass their correctness tests on this GPU.
python -m pytest tests -q -p no:cacheprovider --junitxml="$OUT/tests.xml"
python benchmarks/bench_kernel.py --out "$OUT/kernel.json"
python benchmarks/bench_serving.py --model Qwen/Qwen2.5-1.5B-Instruct --kv-gib 2 --out "$OUT/serving_qwen2.5-1.5b.json"
python benchmarks/bench_serving.py --model Qwen/Qwen2.5-1.5B-Instruct --kv-gib 2 --kv-dtype fp8 --out "$OUT/serving_qwen2.5-1.5b_fp8.json"
python scripts/render_readme.py
