#!/bin/bash

for lowest_sparsity in 60
do
    for cliff_point in biggest first fixed
    do
    echo "Running with lowest_sparsity=$lowest_sparsity and cliff_point=$cl"

    srun python3  /home/eickhoff/esx670/AcceleratedPathPatching/Pruning/FLAP/lib/FLAP.py \
        --task=ioi \
        --nsamples=200 \
        --batch_size=25 \
        --model_name=gpt2 \
        --device=cuda \
        --cliff_point=$cliff_point \
        --lowest_sparsity=$lowest_sparsity \
        --step_size=1 \
        --seed=239501 \
        --save_img \
        --save_txt \
        --out_path=results/ \
        --cache_dir=llm_weights 
    done
done