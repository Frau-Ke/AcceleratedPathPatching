from circuits.circuits_PP import choose_PP_circuit

from utils.data_io import create_folder, set_PATH, get_base_path
from utils.parser import parser
from utils.visualization import pareto_curve
from utils.utils import get_model_parameters

from Patching.ACDC import run_acdc
from Pruning.FLAP.lib.main import get_hybrid_FLAP_CIRCUIT
from Evaluation.findBestCircuit import find_best_ACDC_circuit, pareto_analysis
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union


def main(args, threshold_values:Optional[list]=None, ):
    set_PATH(args.out_path)

    if threshold_values is None:
        thresholds =  10 ** np.linspace(-4, 0, args.num_threshold)
    else:
        thresholds = threshold_values
        
        
        
    total_model_size, _, epochs = get_model_parameters(args.model_name, args.N, input_batch_size=args.batch_size)
    ACDC_method = "acceleratedACDC" if args.do_accelerate else "ACDC"
    ACDC_out_path = get_base_path(args, ACDC_method)

    # folder name
    exp_name = f"{args.metric}_N-{args.N}"
    exp_folder = f"{args.out_path}/{args.model_name}/{args.task}/{ACDC_method}"
    create_folder(exp_folder)


    # ------ get GT circuits (here retrieved via PP)------
    try:
        GT_CIRCUIT = choose_PP_circuit(args.task, args.model_name)  # TODO: replace with function to retrieve stored GT CIRCUIT
    except:
        GT_CIRCUIT = {} 
        
    if args.do_accelerate:
        HYBRID_FLAP_CIRCUIT = get_hybrid_FLAP_CIRCUIT(args, verbose=False, save_folder=exp_folder)     # TODO: save FLAP results in folder .../ACDC/experiment_folder/Hybrid-FLAP
    else:
        HYBRID_FLAP_CIRCUIT = None
            
    ACDC_circuits_df = run_acdc(args, thresholds=thresholds, HYBRID_FLAP_CIRCUIT=HYBRID_FLAP_CIRCUIT, save_img=False)

    ACDC_method = "acceleratedACDC" if args.do_accelerate else "ACDC"
    ACDC_eval_df = find_best_ACDC_circuit(
        args, 
        thresholds=thresholds,
        circuits_dict=ACDC_circuits_df, 
        GT_CIRCUIT=GT_CIRCUIT, 
        method_name=ACDC_method
        )
    pareto_optimal_point, pareto_frontier = pareto_analysis(
        args, 
        df=ACDC_eval_df,
        max_circuit_size=total_model_size/4,
        min_performance=75, 
        focus_on_performance=False,
        method_name=ACDC_method,
        out_path=ACDC_out_path
        )

    if args.show or args.save_img:
        pareto_curve(
            args, 
            df = ACDC_eval_df, 
            best_point = pareto_optimal_point,
            pareto_frontier = pareto_frontier, 
            out_path =ACDC_out_path
            )

    OPT_ACDC_CIRCUIT = ACDC_circuits_df[pareto_optimal_point["threshold"]]
        
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)