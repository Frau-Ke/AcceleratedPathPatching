import os
from utils.parser import parser
from Patching.PathPatching import PathPatching
from Patching.AutomatedPathPatching import automated_PP
from Evaluation.findBestCircuit import find_best_hybridFLAP_circuit, pareto_analysis
from utils.data_io import create_folder, save_circuit, load_circuit, save_dict, set_PATH
from Pruning.FLAP.lib.FLAP import hybrid_FLAP
from circuits.circuits_PP import choose_PP_circuit
from utils.utils import get_model_parameters
from utils.circuit_functions import  merge_circuits

# parser and arguments
args = parser.parse_args()
total_model_size, _, epochs = get_model_parameters(args.model_name, args.N, input_batch_size=args.batch_size)


# ------ get GT circuits (here retrieved via PP)------
try:
    GT_CIRCUIT = choose_PP_circuit(args.task, args.model_name)  # TODO: replace with function to retrieve stored GT CIRCUIT
except:
    GT_CIRCUIT = {} 

# contrastive and vanilla FLAP is executed over a interval of sparsity ratios and circuits are retrieved based on cliff points
FLAP_circuits_df = hybrid_FLAP(
    args=args,
    half_life_metric=True,
    GT_CIRCUIT=GT_CIRCUIT
    )

# eval all possible comnintations of contrastive/clean FLAP and cliff points
hybridFLAP_eval_df = find_best_hybridFLAP_circuit(args, circuits_dict=FLAP_circuits_df, GT_CIRCUIT=GT_CIRCUIT)

# choose pareto optimal hybrid FLAP circuit
pareto_optimal_point = pareto_analysis(args, df_results=hybridFLAP_eval_df, max_circuit_size=total_model_size/2)
OPT_VANILLA_CIRCUIT = FLAP_circuits_df["vanilla"][pareto_optimal_point["vanilla"]]
OPT_CONTRASTIVE_FLAP_CIRCUIT = FLAP_circuits_df["contrastive"][pareto_optimal_point["contrastive"]]
OPT_HYBRID_FLAP_CIRCUIT = merge_circuits(OPT_VANILLA_CIRCUIT, OPT_CONTRASTIVE_FLAP_CIRCUIT)

# APP is executed over the limited search space introduced by OPT_HYBRID_FLAP_CIRCUIT
CIRCUIT_APP = automated_PP(args=args, INPUT_PRUING_CIRCUIT=OPT_HYBRID_FLAP_CIRCUIT) 

# choose pareto optimal APP circuit
