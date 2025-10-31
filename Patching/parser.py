
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
    default="ioi",
    choices=["ioi", 
            "GreaterThan",
            "induction",
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
                "probs",
                "KL_divergence"]
)

add_args("--cache_dir", type=str, default=os.getcwd, help="place to cache model weights" )


add_args("--out_path", default="", type=str)
add_args("--device", default="cpu", choices = ["cpu", "cuda"], type=str)


# dataset arguments
add_args("--N", type=int, default=2)
add_args("--prepend_bos", action="store_true", help="prepend a eos token at the begnning of the sample")
add_args("--seed", default=1234, type=int)


# what and where to patch 
add_args("--per_position",action='store_true')
add_args("--patch_mlp", action="store_true")
add_args(
    "--patching_method",
    type = str,
    default="path",
    choices=["activation", "path", "acdc"])

# Automated Path Patching
add_args("--alpha", default=0.5, type=float)
add_args("--mode", default="sqrt", type=str)
add_args("--scale", default=2, type=float)
add_args("--k", default=1, type=int, help="scaling by number of heads")
add_args("--use_old_input", action="store_true", help="If true, intermediate results from previos runs are used and run is continued")
add_args("--save_every_x_steps", default=5, type=int, help="store intermediate result every x steps")
add_args("--min_activation_threshold", default=0.02, type=float, help="ignore all heads if max activation is below min_activation_threshold")


# Plotting, Prinitng, Saving
add_args("--show", action="store_true")
add_args("--save_img", action="store_true")
add_args("--save_text", action="store_true")
add_args("--verbose", action="store_true")
add_args("--calc_FLOPS", action="store_true", help="if true, calculate the FLOPS the Path Patching Algorithm needs")




## acdc arguments!
add_args("--threshold", default=0.35, type=float)
add_args("--zero_ablation", action="store_true")
add_args("--remove_redundant", action="store_true")
add_args("--reset_network", action="store_true")
add_args('--online_cache_cpu', action="store_true", required=False)
add_args('--corrupted_cache_cpu', action="store_true", required=False)
add_args('--abs_value_threshold', action="store_true")
