#!/bin/bash

for importance_threshold in 2.5 2 1.5 1
do
    for min_value_threshold in 0.02 0.01 0.002 0.001
    do
    echo "Running with importance_threshold=$importance_threshold"
    
    srun python3 /home/eickhoff/esx670/AcceleratedPathPatching/Patching/AutomatedPathPatching.py \
        --model_name=gpt2 \
        --task=IOI \
        --patching_method=path \
        --N=100 \
        --metric=logits_diff \
        --pruning_circuit=none \
        --importance_threshold=$importance_threshold \
        --min_value_threshold=$min_value_threshold \
        --device=cuda \
        --out_path=results \
        --cache_dir=llm_weights/ \
        --seed=12345432 \
        --save_every_x_steps=50 \
        --calc_FLOPS \
        --save_text
    done
done