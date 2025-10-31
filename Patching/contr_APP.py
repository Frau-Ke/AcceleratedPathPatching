import torch as t
from tqdm import tqdm
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union        
#################################################
from Patching.PathPatching import PathPatching
from utils.Visualization import *
#from Patching.ACDC import ACDC
import gc
import pandas as pd
import json

from utils.PatchingMetric import ave_logit_diff
from utils.eval_circuit import *
from circuits_PP import *
from Patching.parser import parser
import pickle
import copy
from utils.model_loader import load_tokenizer, load_hooked_transformer
from utils.dataset_loader import predict_target_token
from utils.utils import create_folder, save_circuit, store_df, save_parser_information, load_parser_information
from circuits_FLAP import choose_FLAP_circuit, choose_hybrid_FLAP_circuit, choose_FLAP_substracted_circuit, choose_FLAP_clean_circuit
from circuits_contr_FLAP import choose_contr_FLAP_circuit
from dataset.loader import load_dataset
from utils.config import set_PATH

def contr_APP(args,
        ks, 
        scales,
        min_activation_thresholds, 
        resid_scale=1.0):
    print(args)
    
    gc.collect()
    t.cuda.empty_cache()
    t.autograd.set_grad_enabled(False)
 
    # ------ load best Pruning circuit ------
    PRUNING_CIRCUIT = choose_contr_FLAP_circuit(args.task, args.model_name)
        
    #----- Load tokenizer and model ------
    eval_tokenizer = load_tokenizer(args.model_name)
    model_hooked = load_hooked_transformer(args.model_name, device=args.device, cache_dir=args.cache_dir)
    
    pp = PathPatching(
        model=model_hooked,
        tokenizer=eval_tokenizer,
        model_name=args.model_name, 
        task=args.task, 
        patching_method=args.patching_method, 
        metric_name=args.metric, 
        N=args.N, 
        device=args.device, 
        patch_mlp=args.patch_mlp,
        seed=args.seed, 
        calc_FLOPS=args.calc_FLOPS
    )

    for k in ks:
        for s in scales:
            for min_acts in min_activation_thresholds:
                
                # ---- set values for next iteration -----
                pp.reset_efficency_metrics()
                args.min_activation_threshold = min_acts
                args.k = k
                args.scale = s


                # ----- create folder structure -----
                if args.mode=="sqrt":
                        result_folder = f"{args.model_name}/{args.task}/contr_APP/min_threshold-{args.min_activation_threshold}/scale_{args.scale}"
                else:
                        result_folder = f"{args.model_name}/{args.task}/contr_APP/min_threshold-{args.min_activation_threshold}/alpha_{args.alpha}-scale_{args.scale}"

                print("result", result_folder)

                if args.out_path == "":
                    subfolder = result_folder
                else:
                    subfolder = os.path.join(args.out_path, result_folder)

                if not os.path.isdir(subfolder):
                    create_folder(subfolder)


                # ------ load intermediate results, if available and requested -------
                if args.use_old_input:
                    try:
                        with open(os.path.join(subfolder, "intermediate.pkl"), "rb") as f:
                            PP_information = pickle.load(f)
                    except:
                        path = os.path.join(subfolder, "intermediate.pkl")
                        raise(f"No intermediate results at {subfolder}. Either point to correct position of intermediate results or set \"use_old_imput=False\".")

                else:
                    PP_information = None
                        
                if args.use_old_input:
                    try:
                        with open(os.path.join(subfolder, "results.json"), "r") as f:
                            old_results = json.load(f)
                    except:
                        path = os.path.join(subfolder, "results.json")
                        raise(f"No intermediate results at {subfolder}. Either point to correct position of intermediate results or set \"use_old_imput=False\".")
                else:
                    old_results = None

                if pp._model.cfg.n_heads * pp._model.cfg.n_layers >= 150:
                    print_vals = True
                else:
                    print_vals=True

                # ------ run automated path patching ------
                if "Qwen" in args.model_name:
                    CIRCUIT = pp.patch_whole_graph_qwen(
                        PRUNING_CIRCUIT=PRUNING_CIRCUIT,
                        subfolder=subfolder,
                        k=args.k,
                        alpha=args.alpha,
                        mode=args.mode,
                        scale=args.scale, 
                        min_activation_threshold=args.min_activation_threshold,
                        save_every_x_steps=args.save_every_x_steps, 
                            
                        verbose=args.verbose,
                        print_vals=print_vals,
                        PP_information=PP_information, 
                        old_results= old_results,
                        save_img=args.save_img, 
                        show_img = args.show, 
                        resid_scale=resid_scale
                        )
                else:
                    CIRCUIT = pp.patch_whole_graph(
                        PRUNING_CIRCUIT=PRUNING_CIRCUIT,
                        subfolder=subfolder,
                        k=args.k,
                        alpha=args.alpha,
                        mode=args.mode,
                        scale=args.scale, 
                        min_activation_threshold=args.min_activation_threshold,
                        save_every_x_steps=args.save_every_x_steps, 
                            
                        verbose=args.verbose,
                        print_vals=print_vals,
                        PP_information=PP_information, 
                        old_results= old_results,
                        save_img=args.save_img, 
                        show_img = args.show, 
                        resid_scale=resid_scale
                        )

               
                results = {
                    "GFLOP":[pp.FLOP_counter],
                    "n_forward_passes":[pp.n_forward_passes], 
                    "comp_time": [pp.elapsed_time],
                    "size_circuit": [circuit_size(CIRCUIT)],
                }

                results = pd.DataFrame(data=results)
                
                if args.save_text:
                    save_circuit(CIRCUIT, subfolder, "circuit.txt")
                    save_parser_information(args, subfolder, "parser_information.txt")
                    store_df(results, subfolder, "results.json")
                    
    return CIRCUIT

if __name__ == "__main__":
    args = parser.parse_args()
    set_PATH(args.out_path)
    ks = [2]
    
    min_activation_thresholds = [0.01, 0.001, 0.02, 0.002]
    scales=[1, 1.5, 2, 2.5]
    resid_scale=2 

    print("min activation threshold", min_activation_thresholds)
    print("scale", scales)
    print("resid scale", resid_scale)
    
    CIRCUIT = contr_APP(     
        args=args,
        ks = ks, 
        scales = scales,
        min_activation_thresholds=min_activation_thresholds, 
        resid_scale=resid_scale
    )