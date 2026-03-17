import os
from utils.parser import parser
from Patching.PathPatching import PathPatching
from Patching.AutomatedPathPatching import automated_PP
from utils.data_io import create_folder, save_circuit, load_circuit, save_dict, set_PATH
from Pruning.FLAP.lib.FLAP import hybrid_FLAP
from circuits.circuits_PP import choose_PP_circuit, get_circuit_name


# parser and arguments
args = parser.parse_args()

# ------ get circuits ------
try:
    GT_CIRCUIT = choose_PP_circuit(args.task, args.model_name)
except:
    GT_CIRCUIT = {}
    

# Hybrid FLAP circuits
circuit_df = hybrid_FLAP(
    args=args,
    half_life_metric=True,
    GT_CIRCUIT=GT_CIRCUIT
    )

# evaluate hybrid FLAP circuits, choose one







CIRCUIT = automated_PP(args=args) 