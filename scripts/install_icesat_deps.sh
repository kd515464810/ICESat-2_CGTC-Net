#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements-icesat.txt

# Note: torch wheels are environment-specific (CUDA/CPU).
# If torch installation above fails or picks wrong build, install manually from:
# https://pytorch.org/get-started/locally/

echo "Base ICESat dual-task dependencies installed."
python - <<'PY'
try:
    import torch
    print(f"[OK] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
except Exception as e:
    print(f"[WARN] torch import failed: {e}")
PY
