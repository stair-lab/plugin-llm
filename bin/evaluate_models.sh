#!/bin/bash

evaluate_models=(
  "e2e_nlg_plugin_gpt2-xl_1layer_e2e_nlg_cleaned_80_0.0005_8_10.0_42_gpt2-xl"
  "e2e_nlg_weighted_gpt2-xl_weight_1_e2e_nlg_cleaned_80_0.0005_8_1.0_1.0_42_gpt2-xl"
  "e2e_nlg_weighted_gpt2-xl_weight_all_e2e_nlg_cleaned_80_0.0005_8_1.0_0.75_42_gpt2-xl"
)
base_models=(
  "gpt2-xl"
  ""
  "gpt2-xl"
)
gpus=(
  "0"
  "1"
  "2"
)

new_model_weights=(
  ""
  "1"
  "0.75"
)

len_contexts=(
  "0"
  "0"
  "0"
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
  
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=/home/ubuntu/decoding/ python ../src/evaluate_models.py \
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
