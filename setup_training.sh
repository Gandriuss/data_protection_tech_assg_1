#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "===== STARTING ENVIRONMENT REPLICATION ====="

# 1. Clone the repository
git clone https://github.com/Gandriuss/data_protection_tech_assg_1.git
cd data_protection_tech_assg_1

# 2. Setup Virtual Environment with uv
pip install uv
uv venv .venv --system-site-packages
source .venv/bin/activate

# 3. Install dependencies and fix PATH
uv pip install ipykernel pandas scikit-learn torchvision matplotlib tensorboardX
export PATH="/workspace/data_protection_tech_assg_1/.venv/bin:$PATH"

# 4. Configure Git
git config user.name "Andrius"
git config user.email "andriusresetnikovas1@gmail.com"

# 5. Handle Data
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

# 6. Run Split Script
cd ..
python prepare_train_test_split.py

echo "===== SETUP COMPLETE ====="