#!/usr/bin/env bash
# Run the Triton kernel tests on a machine without a GPU: Triton's interpreter
# executes the kernels with NumPy inside a Linux container (Triton has no macOS wheels).
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pagedkv-triton-interp
DOCKER="docker --context ${DOCKER_CONTEXT:-desktop-linux}"  # needs a local daemon: the repo is bind-mounted
if ! $DOCKER image inspect "$IMAGE" >/dev/null 2>&1; then
  $DOCKER build -t "$IMAGE" - <<'EOF'
FROM python:3.11-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir triton numpy pytest hypothesis
EOF
fi
$DOCKER run --rm -v "$PWD":/src:ro -w /src -e TRITON_INTERPRET=1 -e PYTHONPATH=/src/src \
  -e PYTHONDONTWRITEBYTECODE=1 "$IMAGE" \
  python -m pytest -p no:cacheprovider -q tests/test_triton.py "$@"
