#!/bin/bash

#SBATCH --account=ctb-lcharlin
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8 # 4 cores per GPU
#SBATCH --mem-per-gpu=40G   # Memory per GPU

# Rest of your environment setup
module load StdEnv/2023  cudacore/.12.2.2
module load arrow/15.0.1
module load python/3.11
source /home/haolun/scratch/plugin-decoding/statml/bin/activate
module load python/3.11

# Add virtual environment site-packages to PYTHONPATH
VENV_PATH="/home/haolun/scratch/plugin-decoding/statml"
# export PYTHONPATH="${VENV_PATH}/lib/python3.11/site-packages:/home/haolun/scratch/plugin-decoding/:${PYTHONPATH}"
export PYTHONPATH="/lustre07/scratch/haolun/plugin-decoding:$PYTHONPATH"
export HUGGING_FACE_HUB_TOKEN='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'
nvidia-smi

# Hyperparameter search space
# learning_rates=(5e-4 5e-3)
# weight_decays=(10 1 0.1 0.01)
# new_model_weights=(1)
learning_rates=(5e-4)
weight_decays=(1)
new_model_weights=(0.75)

# Seed for reproducibility
seed=43

batch_size=8
model_type="gpt2"

# File to store results
# results_file="./results/e2e_nlg_weighted_gpt2-medium_cv_weight_1.txt"
results_file="./results/e2e_nlg_weighted_gpt2-medium.txt"
best_loss=9999999
best_params=""

# Empty results file
echo "" > $results_file

# Loop over all combinations of hyperparameters
for lr in "${learning_rates[@]}"; do
    for wd in "${weight_decays[@]}"; do
        for nmw in "${new_model_weights[@]}"; do
            # Create unique output file for this setting
            # log_file="./logs/e2e_nlg_weighted_gpt2-medium_lr${lr}_wd${wd}_nmw${nmw}.log"
      
            # Print PYTHONPATH for debugging
            echo "Current PYTHONPATH: $PYTHONPATH"
            
            # Run the training script with current hyperparameters
            echo "Running with learning_rate=$lr, weight_decay=$wd, new_model_weight=$nmw"
            output=$(python -c "import sys; print('Python path:', sys.path)" && \
                    python ./src/weighted_decoding.py \
                --model_type $model_type \
                --learning_rate $lr \
                --batch_size $batch_size \
                --weight_decay $wd \
                --new_model_weight $nmw \
                --random_seed $seed 2>&1)
            
            validation_loss=$(echo "$output" | grep "eval_loss" | tail -n 1 | awk -F "'eval_loss': " '{print $2}' | awk -F "," '{print $1}')
        
            # Log the results
            echo "learning_rate=$lr, weight_decay=$wd, new_model_weight=$nmw, validation_loss=$validation_loss" >> $results_file
            
            # Check if this is the best validation loss
            if (( $(echo "$validation_loss < $best_loss" | bc -l) )); then
            best_loss=$validation_loss
            best_params="learning_rate=$lr, weight_decay=$wd, new_model_weight=$nmw"
            fi

        done
    done
done

# Output the best parameters
echo "Best hyperparameters: $best_params with validation_loss=$best_loss" >> $results_file