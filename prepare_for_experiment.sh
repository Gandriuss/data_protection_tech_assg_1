#!/usr/bin/env bash

set -e

echo "===== STARTING ENVIRONMENT REPLICATION ====="

# Setup Virtual Environment with uv
pip install uv
uv venv .venv --system-site-packages
source .venv/bin/activate

# Install dependencies
uv pip install ipykernel pandas scikit-learn torchvision matplotlib tensorboardX


# Download CelebA dataset from Kaggle
mkdir -p data
cd data
curl -L -o ./celeba-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset


# Unzip and cleanup
apt update
apt install zip unzip -y
if [ -f "celeba-dataset.zip" ]; then
    unzip celeba-dataset.zip
    rm celeba-dataset.zip
else
    echo "Warning: celeba-dataset.zip not found in data folder."
fi

echo "===== SETUP COMPLETE ====="
echo "Now download GAN, victim & evaluator model .tar files from"
echo "https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70"