# Sourced by the job scripts: modules and the scratch Python env.
module load python/3.13.5 cuda/12.8.0
export PKV_ENV=${PKV_ENV:-/scratch/$USER/pagedkv/venv}
export PYTHONNOUSERSITE=1                      # keep ~/.local packages from shadowing the venv
source "$PKV_ENV/bin/activate"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # model weights off the small home quota
export TRITON_CACHE_DIR=/scratch/$USER/pagedkv/triton-cache
export TMPDIR=/scratch/$USER/pagedkv/tmp && mkdir -p "$TMPDIR"
