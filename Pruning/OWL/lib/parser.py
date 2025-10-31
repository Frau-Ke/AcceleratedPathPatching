import argparse


########################## for prune ################################
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='LLaMA model')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
parser.add_argument("--sparsity_type", type=str)
parser.add_argument("--prune_method", type=str)
parser.add_argument("--cache_dir", default="llm_weights", type=str )
parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
parser.add_argument('--save', type=str, default=None, help='Path to save results.')
parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
parser.add_argument('--verbose', action="store_true", help="print out?")

parser.add_argument(
    "--device",
    default="cpu",
    type=str,
    help="device, either cuda or cpu",
)
    
parser.add_argument(
    "--out_path",
    type=str,
    default="",
    help="where to store results"
)

########################################### for train
parser.add_argument(
    "--dataset_name",
    type=str,
    default="wikitext",
    help="The name of the dataset to use (via the datasets library).",
)

parser.add_argument(
    "--low_cpu_mem_usage",
    action="store_true",
    help=(
        "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
        "If passed, LLM loading time and RAM consumption will be benefited."
    ),
)

#### saving parameters ##### 
parser.add_argument(
    "--method",
    type=str,
    default=None,

)   

#### data parameters #####


parser.add_argument(
    "--Lamda",
    default=0.08,
    type=float,
    help="Lamda, range defining the layerwise sparsity",
)
    
parser.add_argument(
    '--Hyper_m', 
    nargs='+',
    type=int,
    help="magnitde: identify elements whose outlier score score is M times higher ")

parser.add_argument(
"--outlier_by_activation", action="store_true", help="outlier_by_activation")  

parser.add_argument(
"--outlier_by_wmetric", action="store_true", help="outlier_by_wmetric")  

########################## OWL prune by heads ####################################


parser.add_argument(
    "--prune_by_head",
    action = "store_true",
    help="prune full heads depednding on threshold"
)
   

########################## Testing: prune only one head ####################################


parser.add_argument(
    "--target_layers", 
    default="attn.c_proj",
    type=str,
    help="for testing. Prune only one matrix"
)

parser.add_argument(
    '--target_head',
    nargs='+',
    type=int,
    help="layer and position of head to prune"
)
