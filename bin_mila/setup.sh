#!/bin/bash

source /home/mila/h/haolun.wu/projects/plugin-decoding/statml/bin/activate
module load python/3.10

# Ensure the script exits immediately if a command fails
set -e

# Assuming the environment is already activated, start directly with package installation
echo "Installing Python packages..."

pip install torch==2.4
pip install pyyaml transformers pandas datasets scikit-learn tensorboardX
pip install 'accelerate>=0.26.0'

pip install evaluate
pip install nlg-metricverse

pip install --upgrade datasets

pip install numpy==1.26.0 pycocoevalcap

pip install nltk rouge_score

echo "Installation complete!"


