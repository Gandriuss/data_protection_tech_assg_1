#!/usr/bin/env bash

set -e

echo "===== STARTING ENVIRONMENT REPLICATION ====="

# Setup Virtual Environment with uv
pip install uv
uv venv .venv --system-site-packages
source .venv/bin/activate

# Install dependencies
uv pip install ipykernel pandas scikit-learn torchvision matplotlib tensorboardX

echo "===== SETUP COMPLETE ====="