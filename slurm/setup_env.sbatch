#!/usr/bin/env bash
#SBATCH --job-name=pagedkv-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/pagedkv/logs/%x-%j.out
# One-time: create the Python env on scratch. Usage: sbatch slurm/setup_env.sbatch
set -euo pipefail
ROOT=/scratch/$USER/pagedkv
PKV_ENV=${PKV_ENV:-$ROOT/venv}
export PYTHONNOUSERSITE=1 PIP_CACHE_DIR=$ROOT/pip-cache TMPDIR=$ROOT/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$ROOT/logs"
module load python/3.13.5 cuda/12.8.0
[ -d "$PKV_ENV" ] || python3 -m venv "$PKV_ENV"
source "$PKV_ENV/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # brings Triton
pip install -e "$SLURM_SUBMIT_DIR[hf,bench,dev]"
pip install flashinfer-python || echo "flashinfer unavailable: kernel bench will skip it"
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
