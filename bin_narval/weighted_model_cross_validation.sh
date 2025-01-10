#!/bin/bash

#SBATCH --account=ctb-lcharlin
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4 # 4 cores per GPU
#SBATCH --mem-per-gpu=40G   # Memory per GPU

# Define base directory
RESULTS_DIR="$/home/haolun/projects/ctb-lcharlin/haolun/plugin-decoding/results"
LOGS_DIR="$/home/haolun/projects/ctb-lcharlin/haolun/plugin-decoding/logs"

# Create necessary directories
mkdir -p $RESULTS_DIR
mkdir -p $LOGS_DIR

# Update results file path
results_file="$/home/haolun/projects/ctb-lcharlin/haolun/plugin-decoding/e2e_nlg_weighted_gpt2-medium.txt"

# Rest of your environment setup
source /home/haolun/projects/ctb-lcharlin/haolun/plugin-decoding/statml/bin/activate
module load python/3.10
nvidia-smi

# Hyperparameter search space
# learning_rates=(5e-4 5e-3)
# weight_decays=(10 1 0.1 0.01)
# new_model_weights=(1)
learning_rates=(5e-4)
weight_decays=(1)
new_model_weights=(0.75 0.50 0.25)

# Seed for reproducibility
seed=42

batch_size=8
model_type="gpt2"

# File to store results
# results_file="./results/e2e_nlg_weighted_gpt2-medium_cv_weight_1.txt"
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
      
            # Run the training script with current hyperparameters
            echo "Running with learning_rate=$lr, weight_decay=$wd, new_model_weight=$nmw"
            output=$(PYTHONPATH=/home/haolun/projects/ctb-lcharlin/haolun/plugin-decoding/ python /home/haolun/projects/ctb-lcharlin/haolun/plugin-decoding/src/weighted_decoding.py \
                --model_type $model_type \
                --learning_rate $lr \
                --batch_size $batch_size \
                --weight_decay $wd \
                --new_model_weight $nmw \
                --random_seed $seed 2>&1)

            echo "Raw output:"
            echo "$output"
            
            validation_loss=$(echo "$output" | grep "eval_loss" | tail -n 1 | awk -F "'eval_loss': " '{print $2}' | awk -F "[, }]" '{print $1}')
            
            if [[ -z "$validation_loss" ]]; then
                echo "Validation loss not found. Skipping this run." >> $results_file
                continue
            fi
            
            echo "learning_rate=$lr, weight_decay=$wd, new_model_weight=$nmw, validation_loss=$validation_loss" >> $results_file
            
            if (( $(echo "$validation_loss < $best_loss" | bc -l) )); then
                best_loss=$validation_loss
                best_params="learning_rate=$lr, weight_decay=$wd, new_model_weight=$nmw"
            else
                echo "Comparison failed: validation_loss=$validation_loss, best_loss=$best_loss" >> $results_file
            fi

        done
    done
done

# Output the best parameters
echo "Best hyperparameters: $best_params with validation_loss=$best_loss" >> $results_file