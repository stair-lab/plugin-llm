#!/bin/bash

# Hyperparameter search space
learning_rates=(5e-6 5e-5 5e-4)
weight_decays=(0.01 0.001)

# Seed for reproducibility
seed=42

batch_size=8

# File to store results
results_file="../results/base_model_cross_validation_results.txt"
best_loss=9999999
best_params=""

# Empty results file
echo "" > $results_file

# Loop over all combinations of hyperparameters
for lr in "${learning_rates[@]}"; do
    for wd in "${weight_decays[@]}"; do
      
        # Run the training script with current hyperparameters
        echo "Running with learning_rate=$lr, weight_decay=$wd"
        output=$(PYTHONPATH=/home/ubuntu/decoding/ python ../src/base_model_training.py \
        --learning_rate $lr \
        --batch_size $batch_size \
        --weight_decay $wd \
        --random_seed $seed)

        validation_loss=$(echo "$output" | grep "eval_loss" | tail -n 1 | awk -F "'eval_loss': " '{print $2}' | awk -F "," '{print $1}')
      
        # Log the results
        echo "learning_rate=$lr, weight_decay=$wd, validation_loss=$validation_loss" >> $results_file
        
        # Check if this is the best validation loss
        if (( $(echo "$validation_loss < $best_loss" | bc -l) )); then
        best_loss=$validation_loss
        best_params="learning_rate=$lr, weight_decay=$wd"
        fi

    done
done

# Output the best parameters
echo "Best hyperparameters: $best_params with validation_loss=$best_loss" >> $results_file
