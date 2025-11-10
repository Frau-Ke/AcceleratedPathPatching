
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='LLaMA model')    # Huggingface model name
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=2048, help='Number of calibration samples.')
parser.add_argument('--batch_size', type=int, default=50, help="batch size")
parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
parser.add_argument('--remove_heads', type=int, default=8, help='Remove num_heads')
parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
parser.add_argument("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", 'N/A'])
parser.add_argument("--prune_method", type=str, default="flap", choices=["flap", "wanda_sp", "mag_sp"])
parser.add_argument("--cache_dir", default="llm_weights", type=str)
parser.add_argument('--unstr', action="store_true")

"""
METRICS:
1) WIFN: Weighted Input Feature Norm: 
    - assess the effect of weights columns on output feature map (like WANDA and OWL)
2) IFV: Input Feature Variance
    - Variability among features
    - sample variance of each input feature
3) WIFV: Weights Input Feature Variance 
    - used by FLAP
    - influence of an column of the weight matrix on the recovery of the output feature map
    - "Fluctuation metric"
    - sample variance of each input feature weightes with the squared norm of the corresponding weight matrix column

STRUCUTRE
adaptively search the global compression model structures
magnited of metrics vary within different layers

    - UL-UM: Uniform across Layers and Modules 
        - all layers/modules share the same sparsity ratio like in WANDA
    - UL-MM: Uniform across Layer, Manual ratio for Modules
    - AL-MM: Adaptive across Layers, Manuel ratio for Modules
        - amount of attention heads to removed manually set via --remove_heads
        - MLP pruning is done via pruning ratio GLOBALLY over ALL modules (attention-layers, mlp)
    - AL-AM: Adaptive Across both Layers and Modules
        - MLP and attention layer pruning is done via pruning ratio GLOBALLY over ALL modules (attention-layers, mlp)


Layers = pruned via attention heads
Modules = FeedForward block, mlp
Uniform: all share the same sparsity value
manual: using the --remove_heads parameter to remove a specific amount of heads
adaptive:
    - standardize meatric distribution across layers to equal mean and std
    - capture the absolute variation in the output feature map, when input features are replaced with their baseline values

"""

############ New Parser Arguments ##############################
parser.add_argument("--use_mlp", action="store_true", help="prune MLPs")
parser.add_argument("--task", type=str, default="IOI", help="dataset to run pruning on", choices=["IOI", "Induction", "GreaterThan", "GenderedPronouns", "Docstring"])
parser.add_argument("--device", type=str, default="cpu", help="device to run pruning on")
parser.add_argument("--out_path", type=str, default=os.getcwd(),help="path to results")
parser.add_argument("--prepend_bos", action="store_true", help="append bos token at the beginning")
parser.add_argument("--difference_with", type=str, default="None", choices=["None", "corrupted", "baseline"], help="what kind activations to substract from the clean activations")
parser.add_argument("--save_image", action="store_true", help="save created images, if true")
parser.add_argument("--save_txt", action="store_true", help="store text results, if true")
parser.add_argument("--show", action="store_true", help="show created images, if true")
parser.add_argument("--verbose", action="store_true", help="if true, print debug information")
parser.add_argument("--calc_FLOP", action="store_true", help="if true, calculate FLOPs")

parser.add_argument("--cliff_point", type=str, help="how the cliff will be calculated", choices=["first", "smooth_first", "biggest"])
parser.add_argument("--lowest_sparsity", type=int, default=60, help="min sparsity")
parser.add_argument("--highest_sparsity", type=int, default=99, help="max sparsity")
parser.add_argument("--step_size", type=int, default=1, help="step size")