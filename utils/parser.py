import argparse
import os 

parser = argparse.ArgumentParser(description="Replicate Tasks")

def add_args(*args, **kwargs):
    parser.add_argument(*args, **kwargs)

# Universal arguments
add_args(
    "--model_name", 
    type=str,
    default="gpt2",
    choices=[   "gpt2",   
                "gpt2-small", 
                "gpt2-large",
                "Qwen/Qwen2.5-0.5B",
                "Qwen/Qwen2.5-7B"]          
)

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

add_args("--device", default="cuda", choices = ["cpu", "cuda"], type=str)
add_args("--N", type=int, default=100)
add_args("--batch_size", type=int, default=100)
add_args("--seed", default=1234, type=int)
add_args("--eval_seed", default=193485603, type=int)


add_args("--patch_mlp", action="store_true")
add_args("--calc_FLOP", action="store_true", help="if true, calculate FLOPs")


# Automated Path Patching
add_args("--importance_threshold", default=0, type=float, help="scale * STD is importance threshold")
add_args("--min_value_threshold", default=0, type=float, help="ignore all heads if max activation is below min_activation_threshold")
add_args("--use_old_input", action="store_true", help="If true, intermediate results from previos runs are used and run is continued")
add_args("--save_every_x_steps", default=10, type=int, help="store intermediate result every x steps")

# Accelerated Path Patching
add_args("--pruning_circuit", default="none", choices=["none", "vanilla", "contrastive", "hybrid"], help="Pruning Circuits for APP, if none then PP")

# FLAP
add_args("--lowest_sparsity", type=int, default=0, help="min sparsity")
add_args("--highest_sparsity", type=int, default=100, help="max sparsity")
add_args("--step_size", type=int, default=1, help="step size")
add_args("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", 'N/A'])
add_args("--cliff_point_list", nargs='+', default=["first", "biggest", "fixed"])

# Plotting, Prinitng, Saving
add_args("--show", action="store_true")
add_args("--save_img", action="store_true")
add_args("--save_text", action="store_true")
add_args("--verbose", action="store_true")
