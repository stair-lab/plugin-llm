#!/bin/bash

cd
sudo apt install -y python3.11
sudo apt install -y python3.11-venv
python3.11 -m venv statml
source statml/bin/activate

pip install torch==2.4
pip install pyyaml transformers pandas datasets scikit-learn tensorboardX
pip install 'accelerate>=0.26.0'

pip install evaluate 
pip install  nlg-metricverse

pip install --upgrade datasets

pip install numpy==1.26.0 pycocoevalcap

pip install nltk rouge_score

cd
deactivate

