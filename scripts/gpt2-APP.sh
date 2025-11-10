# Compute Phase
srun python3 Patching/AutomatedPathPatching.py \
    --task=ioi \
    --patching_method=path \
    --N=100 \
    --metric=logits_diff \
    --pruning_circuit=hybrid \
    --importance_threshold=2 \
    --min_value_threshold=0.02 \
    --model_name=gpt2 \
    --mode=sqrt \
    --device=cuda \
    --out_path=results \
    --cache_dir=llm_weights/ \
    --seed=12345432 \
    --save_every_x_steps=50 \
    --calc_FLOPS \
    --save_text \