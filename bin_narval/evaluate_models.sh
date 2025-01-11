#!/bin/bash

#SBATCH --account=ctb-lcharlin
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4 # 4 cores per GPU
#SBATCH --mem-per-gpu=40G   # Memory per GPU

# Rest of your environment setup
module load StdEnv/2023  cudacore/.12.2.2
module load arrow/15.0.1
module load python/3.11
source /home/haolun/scratch/plugin-decoding/statml/bin/activate
module load python/3.11

# Add virtual environment site-packages to PYTHONPATH
VENV_PATH="/home/haolun/scratch/plugin-decoding/statml"
export PYTHONPATH="/lustre07/scratch/haolun/plugin-decoding:$PYTHONPATH"
export HUGGING_FACE_HUB_TOKEN='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'
# Print Python info after activation to verify
which python
python --version
python -c "import sys; print(sys.path)"

export PYTHONPATH="/lustre07/scratch/haolun/plugin-decoding:$PYTHONPATH"
nvidia-smi

evaluate_models=(
  e2e_nlg_weighted_gpt2-medium_e2e_nlg_cleaned_80_0.0005_8_1.0_0.75_43_gpt2-medium
  # "e2e_nlg_plugin_gpt2-xl_1layer_e2e_nlg_cleaned_80_0.0005_8_10.0_42_gpt2-xl"
  # "e2e_nlg_weighted_gpt2-xl_weight_1_e2e_nlg_cleaned_80_0.0005_8_1.0_1.0_42_gpt2-xl"
  # "e2e_nlg_weighted_gpt2-xl_weight_all_e2e_nlg_cleaned_80_0.0005_8_1.0_0.75_42_gpt2-xl"
  # "cg_weighted_llama_weight_all_common_gen_80_0.0005_8_1.0_0.25_42_gpt2-xl"
  # "corrected_base_5_e2e_nlg_cleaned_gpt2-medium_5_0.0005_8_0.01_42"
  # "corrected_plugin_gpt2_1layer_e2e_nlg_cleaned_80_0.0005_8_10.0_42_gpt2-medium"
  # "corrected_plugin_gpt2_1layer_e2e_nlg_cleaned_80_5e-05_8_10.0_42_corrected_base_1_e2e_nlg_cleaned_gpt2-medium_1_0.0005_8_0.01_42"
  # "corrected_plugin_gpt2_1layer_e2e_nlg_cleaned_80_5e-05_8_10.0_42_corrected_base_2_e2e_nlg_cleaned_gpt2-medium_2_0.0005_8_0.01_42"
  # "corrected_plugin_gpt2_1layer_e2e_nlg_cleaned_80_5e-05_8_10.0_42_corrected_base_5_e2e_nlg_cleaned_gpt2-medium_5_0.0005_8_0.01_42"
)
base_models=(
  "gpt2-medium"
  # "gpt2-xl"
  # ""
  # "gpt2-xl"
  # "gpt2-xl"
  # "meta-llama/Llama-3.1-8B"
  # ""
  # "gpt2-medium"
  # "corrected_base_1_e2e_nlg_cleaned_gpt2-medium_1_0.0005_8_0.01_42"
  # "corrected_base_2_e2e_nlg_cleaned_gpt2-medium_2_0.0005_8_0.01_42"
  # "corrected_base_5_e2e_nlg_cleaned_gpt2-medium_5_0.0005_8_0.01_42"
)
gpus=(
  "0"
  # "0"
  # "1"
  # "2"
  # "3"
  # "4"
  # "5"
  # "6"
  # "7"
)

new_model_weights=(
  "0.75"
  # ""
  # "1"
  # "0.75"
  # "1"
  # "0.25"
  # ""
  # ""
  # ""
  # ""
  # ""
  # ""
)

len_contexts=(
  "0"
  # "0"
  # "0"
  # "0"
  # "0"
  # ""
  # ""
  # ""
  # ""
)

batch_size=8
model_type="gpt2"

for i in "${!evaluate_models[@]}"; do
  evaluate_model_name=${evaluate_models[$i]}
  base_model_name=${base_models[$i]}
  gpu=${gpus[$i]}
  new_model_weight=${new_model_weights[$i]}
  len_context=${len_contexts[$i]}
  
  echo "Evaluating $evaluate_model_name on GPU $gpu"
  
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="/lustre07/scratch/haolun/plugin-decoding:$PYTHONPATH" python ./src/evaluate_models.py \
    --model_type $model_type \
    --evaluate_model_name "$evaluate_model_name" \
    --base_model_name "$base_model_name" \
    --batch_size $batch_size \
    --new_model_weight "$new_model_weight" \
    --len_context "$len_context" \
    --gpu 0 &

done

wait
echo "All model evaluations completed."
