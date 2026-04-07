from utils.parser import parser
from Patching.AutomatedPathPatching import automated_PP
from Evaluation.findBestCircuit import find_best_hybridFLAP_circuit, find_best_PP_circuit, pareto_analysis
from utils.data_io import get_base_path
from circuits.circuits_PP import choose_PP_circuit
from utils.utils import get_model_parameters
from utils.circuit_functions import  merge_circuits
from utils.visualization import pareto_curve
from Pruning.FLAP.lib.main import get_hybrid_FLAP_CIRCUIT


def main(args):
    # parser and arguments
    total_model_size, _, epochs = get_model_parameters(args.model_name, args.N, input_batch_size=args.batch_size)


    # ------ get GT circuits (here retrieved via PP)------
    try:
        GT_CIRCUIT = choose_PP_circuit(args.task, args.model_name)  # TODO: replace with function to retrieve stored GT CIRCUIT
    except:
        GT_CIRCUIT = {} 


    if args.do_accelerate:
        OPT_HYBRID_FLAP_CIRCUIT = get_hybrid_FLAP_CIRCUIT(args, verbose=False)    
    else:
        OPT_HYBRID_FLAP_CIRCUIT = None

    # APP is executed over the limited search space introduced by OPT_HYBRID_FLAP_CIRCUIT
    PP_circuits_df = automated_PP(args=args, INPUT_PRUING_CIRCUIT=OPT_HYBRID_FLAP_CIRCUIT) 

    # choose pareto optimal APP circuit
    PP_eval_df = find_best_PP_circuit(args, circuits_dict=PP_circuits_df, GT_CIRCUIT=GT_CIRCUIT)
    pareto_optimal_point, pareto_frontier = pareto_analysis(args, df=PP_eval_df, max_circuit_size=total_model_size/4, min_performance=75, focus_on_performance=False)

    if args.show or args.save_img:
        pareto_curve(
            args, 
            df = PP_eval_df, 
            best_point = pareto_optimal_point,
            pareto_frontier = pareto_frontier, 
            out_path = get_base_path(args, "APP")
            )

    OPT_APP_CIRCUIT = PP_circuits_df[pareto_optimal_point["maxValue"]][pareto_optimal_point["importance"]]



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
