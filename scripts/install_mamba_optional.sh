#!/usr/bin/env bash
set -euo pipefail

# Requires: Linux + NVIDIA GPU + torch>=1.12 + cuda>=11.6
python -m pip install --upgrade pip
python -m pip install causal-conv1d>=1.4.0 mamba-ssm

echo "Optional Mamba dependencies installed."
