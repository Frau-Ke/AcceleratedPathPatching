from circuits.circuits_PP import choose_PP_circuit
from Pruning.FLAP.lib.FLAP import hybrid_FLAP
from Evaluation.findBestCircuit import find_best_hybridFLAP_circuit, find_best_PP_circuit, pareto_analysis
from utils.visualization import pareto_curve
from utils.circuit_functions import  merge_circuits
from utils.utils import get_model_parameters
from utils.data_io import get_base_path
from utils.data_io import create_folder, set_PATH, get_base_path
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union


def get_hybrid_FLAP_CIRCUIT(args, verbose=False, save_folder:Optional[str]=None):
    set_PATH(args.out_path)
    if save_folder is None:
        save_folder = get_base_path(args, "FLAP")
    else:
        save_folder = f"{save_folder}/FLAP"
        create_folder(save_folder)
    
    total_model_size, _, epochs = get_model_parameters(args.model_name, args.N, input_batch_size=args.batch_size)
    # ------ get GT circuits (here retrieved via PP)------
    orig_verbose = args.verbose
    args.verbose=verbose
    
    try:
        GT_CIRCUIT = choose_PP_circuit(args.task, args.model_name)  # TODO: replace with function to retrieve stored GT CIRCUIT
    except:
        GT_CIRCUIT = {} 

    # contrastive and vanilla FLAP is executed over a interval of sparsity ratios and circuits are retrieved based on cliff points
    FLAP_circuits_df = hybrid_FLAP(
        args=args,
        half_life_metric=False,
        GT_CIRCUIT=GT_CIRCUIT
        )

    method_name="FLAP"
    # eval all possible comnintations of contrastive/clean FLAP and cliff points
    hybridFLAP_eval_df = find_best_hybridFLAP_circuit(
        args, 
        circuits_dict=FLAP_circuits_df, 
        GT_CIRCUIT=GT_CIRCUIT, 
        method_name=method_name
        )

    # choose pareto optimal hybrid FLAP circuit
    pareto_optimal_point, pareto_frontier = pareto_analysis(
        args, 
        df=hybridFLAP_eval_df, 
        max_circuit_size=total_model_size/2, 
        focus_on_performance=True,
        method_name=method_name,
        out_path= save_folder
        )

    pareto_curve(
        args, 
        df = hybridFLAP_eval_df, 
        best_point = pareto_optimal_point,
        pareto_frontier = pareto_frontier, 
        out_path= save_folder
        )

    OPT_VANILLA_CIRCUIT = FLAP_circuits_df["vanilla"][pareto_optimal_point["vanilla"]]
    OPT_CONTRASTIVE_FLAP_CIRCUIT = FLAP_circuits_df["contrastive"][pareto_optimal_point["contrastive"]]
    OPT_HYBRID_FLAP_CIRCUIT = merge_circuits(OPT_VANILLA_CIRCUIT, OPT_CONTRASTIVE_FLAP_CIRCUIT)
    
    args.verbose=orig_verbose
    return OPT_HYBRID_FLAP_CIRCUIT