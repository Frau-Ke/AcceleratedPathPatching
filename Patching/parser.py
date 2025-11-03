
import argparse
import os 


parser = argparse.ArgumentParser(description="Replicate Tasks")

def add_args(*args, **kwargs):
    parser.add_argument(*args, **kwargs)

# choose the model
add_args(
    "--model_name", 
    type=str,
    default="gpt2-small",
    choices=[   "gpt2",   
                "gpt2-small", 
                "gpt-medium", 
                "gpt2-large",
                "redwood_attn_2l", 
                "Qwen/Qwen2.5-0.5B",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-7B",
                "Qwen/Qwen2.5-7B-Instruct"]          
)


# task to perform
add_args(
    "--task",
    type=str,
    default="IOI",
    choices=["IOI", 
            "GreaterThan",
            "Induction",
            "GenderedPronouns", 
            "Docstring"
            ]       
)

# metric to use
add_args(
    "--metric",
    type = str,
    default = "logits_diff",
    choices = ["logits_diff",
                "prob_diff",
                "kl_divergence"]
)

add_args("--cache_dir", type=str, default=os.getcwd, help="place to cache model weights" )
add_args("--out_path", default="", type=str)

# dataset arguments
add_args("--device", default="cpu", choices = ["cpu", "cuda"], type=str)
add_args("--N", type=int, default=2)
add_args("--seed", default=1234, type=int)

# what and where to patch 
add_args("--per_position",action='store_true')
add_args("--patch_mlp", action="store_true")
add_args(
    "--patching_method",
    type = str,
    default="path",
    choices=["activation", "path"])

# Automated Path Patching0
add_args("--importance_threshold", default=2, type=float, help="scale * STD is importance threshold")
add_args("--use_old_input", action="store_true", help="If true, intermediate results from previos runs are used and run is continued")
add_args("--save_every_x_steps", default=5, type=int, help="store intermediate result every x steps")
add_args("--min_value_threshold", default=0.02, type=float, help="ignore all heads if max activation is below min_activation_threshold")

# Accelerated Path Patching
add_args("--pruning_circuit", default="none", choices=["none", "vanilla, contrastive, hybrid"], help="Pruning Circuits for APP, if none then PP")

# Plotting, Prinitng, Saving
add_args("--show", action="store_true")
add_args("--save_img", action="store_true")
add_args("--save_text", action="store_true")
add_args("--verbose", action="store_true")
add_args("--calc_FLOPS", action="store_true", help="if true, calculate the FLOPS the Path Patching Algorithm needs")