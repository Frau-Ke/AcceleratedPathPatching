import numpy as np
import torch
import gc
import pandas as pd
from fvcore.nn import FlopCountAnalysis
import time
import warnings 
from functools import partial


from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from jaxtyping import Float, Int, Bool
from torch import Tensor

from Pruning.FLAP.models.hf_llama.modeling_llama import LlamaForCausalLM
from Pruning.FLAP.models.hf_gpt.modeling_gpt2 import GPT2LMHeadModel2Llama
from transformer_lens import HookedTransformer
from Pruning.FLAP.lib.prune import CIRCUIT_from_scores, head_wise_pruning_scores, prune_flap_modular
from Pruning.FLAP.lib.parser import parser

from dataset.loader import load_dataset
from utils.data_io import save_img, create_folder, save_parser_information, save_circuit, store_df, save_parser_information, set_PATH, get_PATH, save_panda_to_text
from utils.metrics import ave_logit_diff
from utils.eval_circuit import batch_evaluate_circiut, print_statistics
from utils.visualization import heat_map_pruning, choose_metric_sparsity_plot_function
from utils.model_loader import get_gpt2_adapt_to_llama, load_tokenizer, load_hooked_transformer, load_transformer
from utils.circuit_functions import TPR, circuit_size, precision
from utils.utils import get_model_parameters


def detect_cliff(values, slope_window=5, min_consec=10, slope_threshold=-0.4):
    values = np.array(values)
    if slope_window == 0:
        slopes=np.diff(values)
    else:
        slopes = np.convolve(np.diff(values), np.ones(slope_window)/slope_window, mode='valid')
    # Find first sustained negative slope
    for i in range(len(slopes) - min_consec):
        if all(slopes[i:i+min_consec] < slope_threshold):
            sustained_idx = i
            break
    else:
        sustained_idx = None
    
    # Find biggest single drop
    biggest_drop_idx = np.argmax(values[:-1] - values[1:])
    
    # take sustained start if earlier, else biggest drop
    if sustained_idx is not None and sustained_idx < biggest_drop_idx:
        return sustained_idx
    else:
        return biggest_drop_idx


def biggest_cliff(results, window, drop_threshold=5):
    drops = []
    for i in range(0, len(results)-window):
        drops.append(abs(results[i] - min(results[i+1:i+window+1])))

    max_drop = max(drops)
    cliff_idx = drops.index(max_drop)
    window_max_drop = drops[cliff_idx:cliff_idx + window]
    window_max_drop = [max_drop - w for w in window_max_drop]
    window_idx = next((val for val in window_max_drop if val >= drop_threshold), None)

    if window_idx == None:
        window_idx = 0
    else:
        window_idx = window_max_drop.index(window_idx) - 1
    cliff_idx += window_idx


    return cliff_idx
    
    
def first_cliff(results,  window, drop_threshold):
    for i in range(0, len(results)-window):
        min_val = min(results[i+1:i+window+1])
        drop = abs(results[i] - min_val)
        # first value with drop > drop_threshold is potential cliff candidate
        if drop >= drop_threshold:
            cliff_idx = i
            # check for valley:
            if results[i] > max(results[i+1:]):
                max_cliff = max(results[i:i+window])
                cliff_idx +=  results[i:i+window].index(max_cliff)
            
            # difference performance of preceding sparsity ratios
            difference_neighbors = [results[cliff_idx + i] - results[cliff_idx + i + 1] for i in range(window)]
            # first difference bigger than drop_threshold is value of cliff points, else 0 and beginng of window is cliff point
            window_val = next((val for val in difference_neighbors if val >= drop_threshold), None)
           
            if window_val == None:
                window_idx = 0
            else:
                window_idx = difference_neighbors.index(window_val)
                cliff_idx += window_idx

            return cliff_idx

    # if max_drop is too conservative, take the biggest total cliff
    print("no cliff is bigger than set max_drop", {drop_threshold})
    return  -1


def moving_average(data, avg_window):
    averaged = []
    for i in range(0, len(data)):
        window_start = max(0, i - avg_window + 1)
        window_data = data[window_start : i+1]
        averaged_value = np.mean(window_data)
        averaged.append(averaged_value)
    return averaged


def identify_cliff_points(
    sparsity_metrics_df:pd.DataFrame, 
    window:float,
    y_variable:str="performance",
    cliff_point:str="first",
    ) -> int:     
    y_metric_list = sparsity_metrics_df[y_variable].tolist()
    max_diff = max(y_metric_list) - min(y_metric_list)
    drop_threshold = max_diff / 10  # drop 10% performance of max difference over a window of length x
    slope_threshold=(max_diff/250) * 0.5 -((10-window)/100)
    slope_threshold=-slope_threshold * 10

    if cliff_point=="first":
        cliff_idx = first_cliff(y_metric_list, window=window, drop_threshold=drop_threshold)
        
    elif cliff_point=="biggest":
        cliff_idx = biggest_cliff(y_metric_list, window=window, drop_threshold=drop_threshold)
        
    elif cliff_point == "smooth_biggest":
        y_metric_list = moving_average(y_metric_list, avg_window=window)
        cliff_idx = biggest_cliff(y_metric_list, window=window, drop_threshold=drop_threshold)
        
    elif cliff_point == "smooth_first":
        y_metric_list = moving_average(y_metric_list, avg_window=window)
        cliff_idx = first_cliff(y_metric_list, window=window, drop_threshold=drop_threshold)
        
    elif cliff_point == "fixed":
        try:
            fixed_sparsity=0.75 
            cliff_idx = sparsity_metrics_df["sparsity_ratio"].tolist().index(fixed_sparsity)
        except:
            warnings.warn(f"Warning: lowest sparsity is higher than fixed cliff point of 0.75. Set fixed cliff point to lowest sparsity")
            cliff_idx = 0
            
    elif cliff_point == "detect_cliff":
        cliff_idx = detect_cliff(y_metric_list, slope_window=window, min_consec=window, slope_threshold=slope_threshold)
        
    elif cliff_point == "smooth_detect_cliff":
        y_metric_list = moving_average(y_metric_list, avg_window=window)
        cliff_idx = detect_cliff(y_metric_list, slope_window=window, min_consec=window, slope_threshold=slope_threshold)
    else:
        raise ValueError(f"Unknown cliff type: {cliff_point}")
    
    return cliff_idx
    

def create_folder_structure(args, cliff_point):
    result_folder =  f"{args.model_name}/{args.task}/Pruning/{cliff_point}/sparsity-min_{args.lowest_sparsity}/"
    if args.out_path == "":
        subfolder = result_folder
    else:
        subfolder = args.out_path + result_folder
    create_folder(subfolder)
    return subfolder
    

def plot_and_save_heatmap(scores, pruning_metric, GT_CIRCUIT, CIRCUIT, performance, subfolder, save_image, title, save_name):
    fig = heat_map_pruning(
                scores, 
                GT_CIRCUIT=GT_CIRCUIT,
                PRUNING_CIRCUIT=CIRCUIT, 
                title=title, #f"{cliff_point} - {args.task} Vanilla FLAP",
                title_pruning_circuit="FLAP",
                title_gt_circuit="Path Patching",
                performance=performance,
                print_scores=False,
                title_temp_scale=pruning_metric)
                
    if save_image:        
        save_img(fig, name=save_name, out_path=subfolder)
    

def plot_and_save_sparstity_performance_curve(sparsity_metrics_df, cliff_value, save_image, title, save_name, subfolder):
    fig = choose_metric_sparsity_plot_function(
        df1=sparsity_metrics_df, 
        cliff_value1=cliff_value,
        y_metric1="performance", 
        y_metric2="TPR", 
        title=title
        )

    if save_image:
        save_img(fig, name=save_name, out_path=subfolder)


def circuit_metrics(CIRCUIT, GT_CIRCUIT, total_model_size):
    true_pos_ratio = TPR(CIRCUIT, GT_circuit=GT_CIRCUIT)*100
    prec = precision(CIRCUIT, GT_CIRCUIT) * 100  
    size=circuit_size(CIRCUIT)
    sparsity=circuit_size(CIRCUIT) / total_model_size
    
    return true_pos_ratio, prec, size, sparsity


def report_loop_information(FLAP_method, cliff_point, extracted_cliff_point, FLAPvsGT_results, LOOP_GFLOP):
    report=f"{'#' * 20} {FLAP_method} --- Cliff: {cliff_point} {'#' * 20} \n \n " +\
    f"{cliff_point} cliff point at: {extracted_cliff_point} \n" +\
    f"{FLAPvsGT_results}\n" +\
    f"This iteration took {LOOP_GFLOP/1e9} GFLOPs \n"      
    print(report)
    
    
def half_life_value(args, sparsity_metrics_df, title=""):
    half_life = sparsity_metrics_df["TPR"].max()/2 
    half_life_idx = sparsity_metrics_df[sparsity_metrics_df["TPR"]<=half_life].index.values[0]
    half_life_sparsity= sparsity_metrics_df["sparsity_ratio"].iloc[half_life_idx]
    
    fig = choose_metric_sparsity_plot_function(
        df1=sparsity_metrics_df, 
        cliff_value1=half_life_sparsity,
        y_metric1="TPR", 
        title=f"half life - FLAP on {args.task} task"
        )
    
    save_img(fig, name=title, out_path=f"{args.out_path}{args.model_name}/{args.task}/Pruning/half_life")
    return half_life_sparsity


def evaluate_FLAP_over_cliff_point(
    args, 
    pruning_method:str,
    pruning_metric:str,
    activations:str,
    cliff_point:str, 
    sparsity_metrics_df:pd.DataFrame, 
    GT_CIRCUIT:dict, 
    INIT_FLOPS:float,
    INIT_COMP_TIME:float,
    FLOPS_BY_MODULE:dict, 
    subfolder:str,
    model, 
    tokenizer,
    y_variable:str="performance"
    ):
        start_time = time.time()
        total_model_size, _, epochs = get_model_parameters(args.model_name, args.N, input_batch_size=args.batch_size)

        # ---- window size is 10% of the toal amount of values
        window=round((args.highest_sparsity-args.lowest_sparsity) / 10)

        if args.calc_FLOP:
            LOOP_FLOPS = len(range(args.lowest_sparsity, args.highest_sparsity)) *  FLOPS_BY_MODULE[""] * epochs # lowest - highest number of evaluation forward passes - CLEAN
        else:
            LOOP_FLOPS = 0
                            
        # ---- identify the cliff points and its associated metrics----
        cliff_idx = identify_cliff_points(sparsity_metrics_df, window, y_variable, cliff_point)
        cliff = sparsity_metrics_df["sparsity_ratio"].iloc[cliff_idx]    
        cliff_metrics = sparsity_metrics_df[sparsity_metrics_df["sparsity_ratio"] == cliff].iloc[0]     

        # ---- retrieve circuit at identified cliff point ----
        CIRCUIT, scores, GFLOPS, n_traversed_l = prune_flap_modular(args, cliff, model, tokenizer, activations, pruning_metric)
        LOOP_FLOPS += GFLOPS
        
        if args.calc_FLOP:
            LOOP_FLOPS += n_traversed_l * FLOPS_BY_MODULE["blocks.0"] * epochs
            
        LOOP_TIME = time.time() - start_time

        # ---- evaluate the retrieved circuit ----
        true_pos_ratio, prec, size, sparsity = circuit_metrics(CIRCUIT, GT_CIRCUIT, total_model_size)

        # ---- save results ----
        loop_results = pd.DataFrame({
            "pruning_type": pruning_method,
            "cliff_point": cliff_point,
            "sparsity_ratio":cliff, 
            "performance": cliff_metrics["performance"], 
            "size": size,
            "sparsity": sparsity, 
            "TPR":true_pos_ratio, 
            "P": prec,
            "FLOP": (LOOP_FLOPS/1e9 + INIT_FLOPS/2)/1e9, 
            "comp_time":  LOOP_TIME + INIT_COMP_TIME/2
            }, index=[0])
    

        # ---- report loop performance ---- 
        if args.verbose:
            FLAPvsGT_results = print_statistics(
                title="*********** FLAP Circuit vs GT Circuit **************",
                ave_logit=cliff_metrics["ave_logit"], 
                performance_achieved=cliff_metrics["performance"],
                CIRCUIT=CIRCUIT, 
                IOI_CIRCUIT=GT_CIRCUIT,
                )
            report_loop_information(
                FLAP_method=pruning_method,
                cliff_point=cliff_point, 
                extracted_cliff_point=cliff, 
                FLAPvsGT_results=FLAPvsGT_results, 
                LOOP_GFLOP=LOOP_FLOPS
            )
            
        # ---- plot and save graphics ----
        if args.show or args.save_img:
            plot_and_save_sparstity_performance_curve(
                sparsity_metrics_df, 
                cliff_value=cliff,
                save_image=args.save_img,
                title=f"{pruning_method} FLAP - {cliff_point} cliff for {args.task}",
                save_name=f"{pruning_method}ROC", 
                subfolder=subfolder
            )    
            
            plot_and_save_heatmap(
                scores,
                pruning_metric,
                GT_CIRCUIT=GT_CIRCUIT, 
                CIRCUIT=CIRCUIT,
                performance=cliff_metrics["performance"], 
                subfolder=subfolder, 
                save_image=args.save_img, 
                title=f"{pruning_method} FLAP {cliff_point} cliff for {args.task}", 
                save_name=f"{pruning_method}Heatmap"
            )

        # ---- save pruning_statistics_df, parser and circuit ----
        if args.save_text:
            save_parser_information(args, subfolder, f"{pruning_method}_parser_info.json")
            save_circuit(CIRCUIT, subfolder, name=f"{pruning_method}_circuit.txt")
            save_panda_to_text(loop_results, out_path=subfolder, name=f"{pruning_method}_result")
            
        return CIRCUIT, cliff


def loop_over_sparsity_intervall(
    args, 
    GT_CIRCUIT,
    model_hooked:HookedTransformer=None,
    eval_dataset=None, 
    ave_logit_gt:float=0, 
    scores:Float[Tensor, "n_layers n_heads"]=None, 
    mlp_scores:Float[Tensor, "n_layers n_heads"]=None,
    mlp_mask:Bool[Tensor, "n_layers n_heads"]=None,
    ):

    results = pd.DataFrame(columns=["size", "sparsity_ratio", "ave_logit" "performance", "TPR", "P"])

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
            
    for i in range(args.lowest_sparsity, args.highest_sparsity, args.step_size):
        TOTAL_GFLOPS = 0
        pruning_ratio = i / 100
        CIRCUIT, _ , GFLOPS = CIRCUIT_from_scores(
            args,
            pruning_ratio=pruning_ratio,
            attn_metric_list=scores, 
            mlp_metric_list=mlp_scores,
            mlp_mask=mlp_mask, 
            n_layers=model_hooked.cfg.n_layers, 
            n_heads=model_hooked.cfg.n_heads, 
            head_dim=model_hooked.cfg.d_head, 
            )
        
        TOTAL_GFLOPS += GFLOPS
        
        ave_logit, performance,_ = batch_evaluate_circiut(
            model = model_hooked, 
            CIRCUIT=CIRCUIT,
            dataset=eval_dataset,
            ave_logit_gt=ave_logit_gt, 
            task=args.task,
            model_name=args.model_name,  
            epochs = int(args.N /args.batch_size), 
            batch_size = args.batch_size 
            )
        

        recall = TPR(CIRCUIT, GT_CIRCUIT)
        prec = precision(CIRCUIT, GT_CIRCUIT)
        size = circuit_size(CIRCUIT)

        if args.verbose:
            print(f"size: {size}, sparsity_ratio: {pruning_ratio}, ave_logit: {ave_logit},  performance: {performance}, TPR:{recall}, P: {prec}")
        
        new_col = pd.DataFrame({"size": size, "sparsity_ratio":pruning_ratio, "ave_logit": ave_logit, "performance":performance, "TPR":recall, "P": prec}, index=[0])
        results = pd.concat([results, new_col], ignore_index=True)

    return results, TOTAL_GFLOPS


def hybrid_FLAP(
    args,
    half_life_metric:bool=True, 
    GT_CIRCUIT:dict={}, 
    ):    
    """For big models, cuda memory is to limited to have both models on cuda(). This functions alternates between hooked transformer for CIRCUIT evaluation and 
    CasualLM transformer for FLAP.
    """
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.empty_cache()
    gc.collect()
    
    # ------ initialize ------
    _, _, epochs = get_model_parameters(args.model_name, args.N, input_batch_size=args.batch_size)
    circuit_df = {}  # save FLAP circuits {"FLAP_method": {"cliff_point": {CIRCUIT}}}

    num_traversed_layers, n_forward_passes = 0, 0
    INIT_FLOPS, INIT_COMP_TIME = 0, 0
    y_variable = "performance"
    start_time = time.time() 
    
    # ------ load Causual model ------
    if "gpt2" in args.model_name:
        model = get_gpt2_adapt_to_llama(args.model_name, args.device)
    elif "Qwen" in args.model_name:
        model = load_transformer(args.model_name, args.device, cache_dir=args.cache_dir)
    
    tokenizer = load_tokenizer(args.model_name)
    
    # ------ clean and corrupted scores   -----
    pruning_metric = "WIFV"
    activations = "clean"  
            
    scores_clean, mlp_scores_clean, mlp_mask_clean, GFLOPS, n_travered_l = head_wise_pruning_scores(
        args, 
        model, 
        tokenizer, 
        activations=activations,
        pruning_metrics=pruning_metric
        )  # clean FLAP scores
    
    INIT_FLOPS += GFLOPS
    num_traversed_layers += n_travered_l
    
    pruning_metric = "WIFN"
    activations = "contrastive"
    
    scores_corr, mlp_scores_corr, mlp_mask_corr, GFLOPS, n_travered_l = head_wise_pruning_scores(
        args, 
        model, 
        tokenizer,
        activations=activations,
        pruning_metrics=pruning_metric
        )  # corrupted FLAP scores

    INIT_FLOPS += GFLOPS
    num_traversed_layers += n_travered_l
    
    INIT_COMP_TIME += time.time() - start_time


    # ---- load Hooked model ----    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    model_hooked = load_hooked_transformer(model_name=args.model_name, device=args.device, cache_dir=args.cache_dir)

    # ------ get dataset ------
    eval_dataset = load_dataset(
        model_name=args.model_name,
        task=args.task, 
        tokenizer=tokenizer,  
        N=args.N, 
        patching_method="path", 
        device=args.device, 
        seed=args.seed, 
        prepend_bos=False
        )
    
    if args.calc_FLOP:
        FLOPS_BY_MODULE = FlopCountAnalysis(model_hooked, eval_dataset.clean_tokens[:args.batch_size, :]).by_module()
        INIT_FLOPS += num_traversed_layers * FLOPS_BY_MODULE["blocks.0"] * epochs
    else:
        FLOPS_BY_MODULE = {}
    
    
    start_time = time.time()
    
    # ------ ave logit of unpruned model ------
    with torch.no_grad():
        hooked_gt = model_hooked(eval_dataset.clean_tokens)
    n_forward_passes += 1
    
    ave_logit_gt = ave_logit_diff(
        hooked_gt, 
        eval_dataset.correct_answers,  
        eval_dataset.wrong_answers,       
        eval_dataset.target_idx.to(args.device), 
        task=args.task, 
        model_name=args.model_name
        )
    
    # ---- evaluate clean and corrupted scores ----
    eval_sparsity_intervall_dn = partial(
        loop_over_sparsity_intervall, 
        args=args, 
        GT_CIRCUIT=GT_CIRCUIT,
        model_hooked=model_hooked,
        eval_dataset=eval_dataset, 
        ave_logit_gt=ave_logit_gt
    )
    
    activations = "clean"  
    clean_sparsity_metrics_df, GFLOPS = eval_sparsity_intervall_dn(   
                        scores=scores_clean, 
                        mlp_scores=mlp_scores_clean,
                        mlp_mask=mlp_mask_clean
                        )
    
                
    INIT_FLOPS += GFLOPS
    
    activations = "contrastive"
    contr_sparsity_metrics_df, GFLOPS = eval_sparsity_intervall_dn(   
                        scores=scores_corr, 
                        mlp_scores=mlp_scores_corr,
                        mlp_mask=mlp_mask_corr, 
                        )
    
    INIT_FLOPS += GFLOPS # FLOPs form FLAP (calculating the metric, standardization...)
    
    if args.calc_FLOP:
        INIT_FLOPS += n_forward_passes * FLOPS_BY_MODULE[""]  * epochs  # forward pass to get gt and gt_circuit
    INIT_COMP_TIME += time.time() - start_time 
        
    pruning_folder = f"{args.out_path}/{args.model_name}/{args.task}/Pruning/"
    if args.save_text:
        store_df(clean_sparsity_metrics_df, pruning_folder, "vanilla_sparsity_metrics.xlsx")
        store_df(contr_sparsity_metrics_df, pruning_folder, "contrastive_sparsity_metrics.xlsx")
    
    del model_hooked
    torch.cuda.empty_cache()
    gc.collect()
    
    # ------ load Causual model ------
    if "gpt2" in args.model_name:
        model = get_gpt2_adapt_to_llama(args.model_name, args.device)
    elif "Qwen" in args.model_name:
        model = load_transformer(args.model_name, args.device, cache_dir=args.cache_dir)
    
    
    if args.verbose:
        print("INIT_FLOPS", INIT_FLOPS/1e9)
        print("INIT COMP TIME", INIT_COMP_TIME)    
    


    # ---- execute Vanilla and Contrastive FLAPs over all cliff points and save the resulting circuits ----
    for cliff_point in args.cliff_point_list:
        
        subfolder = create_folder_structure(args, cliff_point)
        
        eval_cliffs_fn = partial(
            evaluate_FLAP_over_cliff_point,
            cliff_point=cliff_point,
            GT_CIRCUIT=GT_CIRCUIT,
            INIT_FLOPS=INIT_FLOPS, 
            INIT_COMP_TIME=INIT_COMP_TIME, 
            FLOPS_BY_MODULE=FLOPS_BY_MODULE,
            subfolder=subfolder,
            model=model,
            tokenizer=tokenizer,     
            y_variable=y_variable
        )
        
        # ------ Vanilla FLAP ------
        CLEAN_CIRCUIT, clean_cliff = eval_cliffs_fn(
            args, 
            pruning_method="Vanilla", 
            pruning_metric="WIFV",
            activations="clean",
            sparsity_metrics_df=clean_sparsity_metrics_df
        )
        
        if circuit_df.get("vanilla") is None:
            circuit_df["vanilla"] = {cliff_point: CLEAN_CIRCUIT}
        else:
            circuit_df["vanilla"][cliff_point] = CLEAN_CIRCUIT
            
            
        # ------ Contrastive FLAP ------
        CONTR_CIRCUIT, contr_cliff = eval_cliffs_fn(
            args, 
            pruning_method="Contrastive", 
            pruning_metric="WIFN",
            activations="contrastive",
            sparsity_metrics_df=contr_sparsity_metrics_df
        )
        
        if circuit_df.get("contrastive") is None:
            circuit_df["contrastive"] = {cliff_point: CONTR_CIRCUIT}
        else:
            circuit_df["contrastive"][cliff_point] = CONTR_CIRCUIT
        
        fig = choose_metric_sparsity_plot_function(
                df1=contr_sparsity_metrics_df, 
                cliff_value1=contr_cliff,
                df2=clean_sparsity_metrics_df, 
                cliff_value2=clean_cliff,
                y_metric1="performance", 
                y_metric2="TPR", 
                title=f"Vanilla vs Contrastive FLAP: {cliff_point}"
                )

        if args.save_img:    
            save_img(fig, name=f"both_curves", out_path=subfolder)
        
    # ---- Experiment: Half-life metric -----
    if half_life_metric:      

        half_life_sparsity_clean = half_life_value(args, clean_sparsity_metrics_df, "half_life_vanilla") 
        half_life_sparsity_contr = half_life_value(args, contr_sparsity_metrics_df, "half_life_vanilla")

        fig_two_TP = choose_metric_sparsity_plot_function(
            df1=clean_sparsity_metrics_df, 
            cliff_value1=half_life_sparsity_clean,
            df2=contr_sparsity_metrics_df, 
            cliff_value2=half_life_sparsity_contr,
            y_metric1="TPR", 
            title=""
            )
        
        if args.save_img:
            save_img(fig_two_TP, f"{args.out_path}/{args.model_name}/{args.task}/Pruning/half_life", args.task + "_two_TP.png")    

    return circuit_df


if __name__ == "__main__":
    
    args = parser.parse_args()
    hybrid_FLAP(
        args=args,
        half_life_metric=False
    )