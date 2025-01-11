#!/bin/bash

module load StdEnv/2023  cudacore/.12.2.2
module load arrow/15.0.1
module load python/3.11
source /home/haolun/scratch/plugin-decoding/statml/bin/activate
module load python/3.11

# Ensure the script exits immediately if a command fails
set -e

# Assuming the environment is already activated, start directly with package installation
echo "Installing Python packages..."

pip install torch==2.4.1+computecanada
pip install pyyaml transformers pandas datasets scikit-learn tensorboardX
pip install 'accelerate>=0.26.0'

pip install evaluate
pip install nlg-metricverse

pip install --upgrade datasets

pip install numpy pycocoevalcap

pip install nltk rouge_score

echo "Installation complete!"


