import torch as t
from tqdm import tqdm
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union        
#################################################
from Patching.PathPatching import PathPatching
from utils.visualization import *
#from Patching.ACDC import ACDC
import gc
import pandas as pd
import json

from dataset.loader import load_dataset
from utils.metrics import ave_logit_diff
from utils.eval_circuit import *
from circuits.circuits_PP import *
from circuits.circuits_FLAP import * 
from Patching.parser import parser
import pickle
from utils.model_loader import load_tokenizer, load_hooked_transformer
from utils.dataset_loader import predict_target_token
from utils.data_io import create_folder, save_circuit, store_df, save_parser_information, set_PATH
from circuits.circuits_FLAP import choose_hybrid_FLAP_circuit, choose_contrastive_FLAP_circuit, choose_vanilla_FLAP_circuit

from circuits.circuits_FLAP import *

gc.collect()
t.cuda.empty_cache()
t.autograd.set_grad_enabled(False)

def automated_PP(
    args,
    resid_importance_threshold=2
):
    
    if args.pruning_circuit == "none":    
        PRUNING_CIRCUIT = None
    elif args.pruning_circuit == "hybrid": 
        PRUNING_CIRCUIT = choose_hybrid_FLAP_circuit()#args.task, args.model_name)
    elif args.pruning_circuit == "contrastive": 
        PRUNING_CIRCUIT = choose_contrastive_FLAP_circuit(args.task, args.model_name)
    else:
        PRUNING_CIRCUIT = choose_vanilla_FLAP_circuit(args.task, args.model_name)
    
    pp = PathPatching(
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

    pp.reset_efficency_metrics()
    
    # ----- create folder structure -----
    if args.pruning_circuit == "none":    
        result_folder = f"{args.model_name}/{args.task}/{args.patching_method}/automatic/min_threshold-{args.min_value_threshold}/scale-{args.importance_threshold}"
    else:
        result_folder = f"{args.model_name}/{args.task}/APP_{args.pruning_circuit}/min_threshold-{args.min_value_threshold}/scale-{args.importance_threshold}"


    if args.out_path == "":
        subfolder = result_folder
    else:
        subfolder = os.path.join(args.out_path, result_folder)
        
    print("result", result_folder)
    if not os.path.isdir(subfolder):
        create_folder(subfolder)

    # ------ load intermediate results, if available and requested -------
    if args.use_old_input:
        try:
            with open(os.path.join(subfolder, "intermediate.pkl"), "rb") as f:
                PP_information = pickle.load(f)
            with open(os.path.join(subfolder, "results.json"), "r") as f:
                old_results = json.load(f)
        except:
            raise(f"No intermediate results at {subfolder}. Either point to correct position of intermediate results or set \"use_old_input=False\".")

    else:
        PP_information = None
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
            importance_threshold=args.importance_threshold, 
            min_value_threshold=args.min_value_threshold,
            save_every_x_steps=args.save_every_x_steps, 
            resid_importance_threshold = resid_importance_threshold,
            
            verbose=args.verbose,
            print_vals=print_vals,
        
            PP_information=PP_information, 
            old_results= old_results,
            
            save_img=args.save_img, 
            show_img = args.show, 
            )

    else:
        CIRCUIT = pp.patch_whole_graph(
            PRUNING_CIRCUIT=PRUNING_CIRCUIT,
            subfolder=subfolder,
            importance_threshold=args.importance_threshold, 
            min_value_threshold=args.min_value_threshold,
            save_every_x_steps=args.save_every_x_steps, 
            resid_importance_threshold= resid_importance_threshold, 
            
            verbose=args.verbose,
            print_vals=print_vals,
            
            PP_information=PP_information, 
            old_results= old_results,
            
            save_img=args.save_img, 
            show_img = args.show, 
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
    print(args)
    
    _ = automated_PP(     
        args=args
    )