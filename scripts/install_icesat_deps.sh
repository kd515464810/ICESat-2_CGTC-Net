#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy scikit-learn tqdm

# Optional for soft-DTW alternatives or advanced smoothers
python -m pip install tslearn || true

echo "Base ICESat dual-task dependencies installed."
