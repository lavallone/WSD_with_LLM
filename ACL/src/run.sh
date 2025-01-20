#!/bin/bash

# Define arrays for model names, subtasks, and approaches
model_names=("llama_1b" "gemma_2b" "llama_3b" "phi_mini" "phi_small" "mistral" "llama_8b" "gemma_9b")
subtasks=("selection" "generation")
approaches=("few_shot" "one_shot" "few_shot")

# Loop through each combination of model, subtask, and approach
for approach in "${approaches[@]}"; do
  for subtask in "${subtasks[@]}"; do
    for model in "${model_names[@]}"; do
      echo "Running with model: $model, subtask: $subtask, approach: $approach"
      python3 score.py --shortcut_model_name "$model" --subtask "$subtask" --approach "$approach" --pos ALL --sentence_embedder all-mpnet-base-v2
      #sleep 2
    done
  done
done
