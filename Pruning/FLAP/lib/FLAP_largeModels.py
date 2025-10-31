import os 
import numpy as np
import torch
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from Pruning.FLAP.models.hf_llama.modeling_llama import LlamaForCausalLM
from Pruning.FLAP.models.hf_gpt.modeling_gpt2 import load_pretrained_llama_style_gpt2
from Pruning.FLAP.lib.prune import CIRCUIT_from_scores, head_wise_pruning_scores, prune_flap_modular

from circuits_PP import *
from Pruning.FLAP.lib.parser import parser
from logger_config import logger

from transformers import (
    AutoConfig,
    GPT2Tokenizer,
    GPT2Config,
    GPT2Model)
from transformer_lens import utils, HookedTransformer, ActivationCache

from datetime import datetime
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow

from dataset.loader import load_dataset
from utils.utils import save_img, create_folder, save_parser_information, save_circuit, store_df, save_parser_information, load_df
from utils.PatchingMetric import ave_logit_diff
from utils.eval_circuit import *
from utils.Visualization import outline_IoU, outline_IOI, heat_map_sparsity, ROC_curve, TP_curve, two_TP_curve, two_ROC_curve
import time
from fvcore.nn import FlopCountAnalysis
from utils.model_loader import get_gpt2_adapt_to_llama, load_tokenizer, load_hooked_transformer, load_transformer
from utils.config import set_PATH, get_PATH
import numpy as np

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
        drops.append(abs(results[i] - min(results[i+1:i+window])))

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
        min_val = min(results[i+1:i+window])
        drop = abs(results[i] - min_val)
        # first value with drop > drop_threshold is potential cliff candidate
        if drop >= drop_threshold:
            print("idx", i)
            print("results", results[i] )
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


def evaluate_sparsity_ratios(
    args, 
    model_hooked,
    eval_dataset, 
    ave_logit_gt,
    gt_circuit_ave_logit, 
    gt_circuit_performance,
    scores, 
    unstandardized_scores, 
    mlp_scores,
    mlp_mask,
    min_sparsity, 
    max_sparsity
    ):
    
    results = pd.DataFrame(columns=["sparsity_ratio", "res", "performance", "gain", "diff", "TPR"])
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # ------ get circuits ------
    try:
        GT_CIRCUIT = choose_circuit(args.task, args.model_name)
    except:
        GT_CIRCUIT = {}

    gt_circuit_name =  get_circuit_name(args.task)

    
    if args.verbose:
        res_ground_truth = print_statics(
            title=f"*********** GT Circuit of {gt_circuit_name} Task **************",
            ave_logit=gt_circuit_ave_logit, 
            performance_achieved=gt_circuit_performance,
            CIRCUIT=GT_CIRCUIT, 
            IOI_CIRCUIT=GT_CIRCUIT
            )
        print(res_ground_truth)
        
    ave_logit_old = gt_circuit_ave_logit
    
    for i in range(min_sparsity, max_sparsity):
        TOTAL_GFLOPS=0
        args.pruning_ratio = i / 100
        CIRCUIT, _ , GFLOPS = CIRCUIT_from_scores(
            args, 
            attn_metric_list=scores, 
            W_metric_unstand_list=unstandardized_scores, 
            mlp_metric_list=mlp_scores,
            mlp_mask=mlp_mask, 
            n_layers=model_hooked.cfg.n_layers, 
            n_heads=model_hooked.cfg.n_heads, 
            head_dim=model_hooked.cfg.d_head, 
            )
        
        TOTAL_GFLOPS += GFLOPS
        
        ave_logit, performance = batch_evaluate_circiut(
            model = model_hooked, 
            CIRCUIT=CIRCUIT,
            dataset=eval_dataset,
            ave_logit_gt=ave_logit_gt, 
            task=args.task,
            model_name=args.model_name,  
            epochs = int(args.nsamples /args.batch_size), 
            batch_size = args.batch_size 
            )
        
        gain = performance_gain(performance_new=performance, performance_old=gt_circuit_performance)

        diff = ave_logit_old - ave_logit
        true_pos = get_intersection_num(CIRCUIT, GT_CIRCUIT)
        
        if args.verbose:
            print(f"sparsity_ratio: {args.pruning_ratio}, ave_logit_diff: {ave_logit}, performance: {performance}, gain {gain}, diff: {diff}, TPR:{true_pos}")
        
        new_col = pd.DataFrame({"sparsity_ratio":args.pruning_ratio, "res":ave_logit, "performance":performance, "gain":gain, "diff":diff, "TPR":true_pos}, index=[0])
        results = pd.concat([results, new_col], ignore_index=True)
        ave_logit_old = ave_logit

    return results, TOTAL_GFLOPS



def hybrid_FLAP(
    args,
    highest_sparsities=[0, 60],
    lowest_sparsities=[90, 99],
    cliff_functions=["first", "smooth_first", "biggest"], 
    half_life_metric=True, 
    ):    
    """For big models, cuda memory is to limited to have both models on cuda(). This functions alternates between hooked transformer for CIRCUIT evaluation and 
    CasualLM transformer for FLAP.

    Args:
        args (_type_): _description_
        highest_sparsities (list, optional): _description_. Defaults to [0, 60].
        lowest_sparsities (list, optional): _description_. Defaults to [90, 99].
        cliff_functions (list, optional): _description_. Defaults to ["first", "smooth_first", "biggest"].
    """
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

        
    torch.cuda.empty_cache()
    gc.collect()
    epochs = int(args.nsamples / args.batch_size)
    # ------ load Causual model ------
    if "gpt2" in args.model_name:
        model = get_gpt2_adapt_to_llama(args.model_name, args.device)
        n_layers = model.config.n_layer
        n_heads =  model.config.n_head
    elif "Qwen" in args.model_name:
        model = load_transformer(args.model_name, args.device, cache_dir=args.cache_dir)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    
    tokenizer = load_tokenizer(args.model_name)
    
    
    num_traversed_layers = 0
    n_forward_passes = 0
    INIT_FLOPS = 0
    INIT_COMP_TIME = 0
    y_variable = "performance"
    start_time = time.time() 
    
    # ------ clean and corrupted scores   -----
    args.metrics = "WIFV"
    args.difference_with = "None"  
            
    scores_clean, mlp_scores_clean, mlp_mask_clean, unstandardized_scores_clean, GFLOPS, n_travered_l = head_wise_pruning_scores(args, model, tokenizer)  # clean FLAP scores
    INIT_FLOPS += GFLOPS
    num_traversed_layers += n_travered_l
    
    args.metrics = "WIFN"
    args.difference_with = "corrupted"
    
    scores_corr, mlp_scores_corr, mlp_mask_corr, unstandardized_scores_corr, GFLOPS, n_travered_l = head_wise_pruning_scores(args, model, tokenizer)  # corrupted FLAP scores
    INIT_FLOPS += GFLOPS
    num_traversed_layers += n_travered_l
    
    INIT_COMP_TIME += time.time() - start_time
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # ---- load Hooked model ----
    model_hooked = load_hooked_transformer(model_name=args.model_name, device=args.device, cache_dir=args.cache_dir)

    # ------ get dataset ------
    eval_dataset = load_dataset(
        model_name=args.model_name,
        task=args.task, 
        tokenizer=tokenizer,  
        N=args.nsamples, 
        patching_method="path", 
        device=args.device, 
        seed=args.seed, 
        prepend_bos=args.prepend_bos
        )
    if args.calc_FLOP:
        FLOPS_BY_MODULE = FlopCountAnalysis(model_hooked, eval_dataset.clean_tokens[:args.batch_size, :]).by_module()
            
        INIT_FLOPS += num_traversed_layers * FLOPS_BY_MODULE["blocks.0"] * epochs
    # ------ get circuits ------
    try:
        GT_CIRCUIT = choose_circuit(args.task, args.model_name)
    except:
        GT_CIRCUIT = {}
    
    
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

    # ------ ave logit of gt circuit ------
    gt_circuit_ave_logit, gt_circuit_performance = batch_evaluate_circiut(
        model = model_hooked, 
        CIRCUIT=GT_CIRCUIT,
        dataset=eval_dataset,
        ave_logit_gt=ave_logit_gt, 
        task=args.task,
        model_name=args.model_name, 
        epochs = epochs, 
        batch_size = args.batch_size      
        )
    n_forward_passes += 1
    
    # ---- evaluate clean and corrupted scores ----
    ###### Hooked Transformer
    args.difference_with = "None"  
    results_clean, GFLOPS = evaluate_sparsity_ratios(   
                        args, 
                        model_hooked=model_hooked,
                        eval_dataset=eval_dataset, 
                        ave_logit_gt=ave_logit_gt,
                        gt_circuit_ave_logit=gt_circuit_ave_logit,
                        gt_circuit_performance=gt_circuit_performance, 
                        scores=scores_clean, 
                        unstandardized_scores=unstandardized_scores_clean, 
                        mlp_scores=mlp_scores_clean,
                        mlp_mask=mlp_mask_clean, 
                        min_sparsity=min(lowest_sparsities), 
                        max_sparsity=max(highest_sparsities)
                        )
    INIT_FLOPS += GFLOPS
    
    args.difference_with = "corrupted"
    results_corr, GFLOPS = evaluate_sparsity_ratios(   
                        args, 
                        model_hooked=model_hooked,
                        eval_dataset=eval_dataset, 
                        ave_logit_gt=ave_logit_gt,
                        gt_circuit_ave_logit=gt_circuit_ave_logit,
                        gt_circuit_performance=gt_circuit_performance, 
                        scores=scores_corr, 
                        unstandardized_scores=unstandardized_scores_corr, 
                        mlp_scores=mlp_scores_corr,
                        mlp_mask=mlp_mask_corr, 
                        min_sparsity=min(lowest_sparsities), 
                        max_sparsity=max(highest_sparsities)
                        )
    
    INIT_FLOPS += GFLOPS # FLOPs form FLAP (calculating the metric, standardization...)
    if args.calc_FLOP:
        INIT_FLOPS += n_forward_passes * FLOPS_BY_MODULE[""]  * epochs  # forward pass to get gt and gt_circuit
    INIT_COMP_TIME += time.time() - start_time 
    
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
    ##### CasualLM
    for cliff_f in cliff_functions:
        for ls in lowest_sparsities:
            for hs in highest_sparsities:
                print("cliff function", cliff_f)
                print("INIT_COMP_TIME", INIT_COMP_TIME)    
                # ---- window size is 10% of the toal amount of values
                window=round((hs-ls) / 10)
                average_window = window
                slope_window=window
                min_consec=window
                
                if args.verbose:
                    print(f"cliff {cliff_f} from {ls} to {hs}")
                    print("window", window)
            
                args.lowest_sparsity = ls
                args.highest_sparsity = hs
                args.cliff_type = cliff_f

                result_folder =  f"{args.model_name}/{args.task}/Pruning/{args.cliff_type}/sparsity-min_{args.lowest_sparsity}/"

                if args.out_path == "":
                    subfolder = result_folder
                else:
                    subfolder = args.out_path + result_folder

                create_folder(subfolder)                
                final_results = pd.DataFrame(columns=["pruning_type", "sparsity_ratio", "size", "ave_logit_diff", "performance", "gain", "TPR", "FPR", "half_life", "FLOP", "comp_time"])
                

                #----------------------------------------------------------------------------------------------------
                #                   FLAP on clean input - WIFV metric
                #----------------------------------------------------------------------------------------------------
                if args.calc_FLOP:
                    LOOP_FLOPS_CLEAN = len(range(ls, hs)) *  FLOPS_BY_MODULE[""] * epochs # lowest - highest number of evaluation forward passes - CLEAN
                else:
                    LOOP_FLOPS_CLEAN = 0
                start_time = time.time() 
                
                args.metrics = "WIFV"
                args.difference_with = "None"    
                    
                # ----- Cliff ------
                results_clean_loop = results_clean[(results_clean["sparsity_ratio"] >= ls / 100) & (results_clean["sparsity_ratio"] < hs / 100)]#.iloc[:, 0].tolist()
                performance_metric_clean = results_clean_loop[y_variable].tolist()
              
                max_diff = max(results_clean_loop[y_variable]) - min(results_clean_loop[y_variable])
                drop_threshold = max_diff / 10 # drop 10% performance of max difference over a window of length x
                slope_threshold=(max_diff/250) * 0.5 -((10-average_window)/100)
                slope_threshold=-slope_threshold * 10
                
                if args.verbose:
                    print("clean thresholds")
                    print("drop_threshold", drop_threshold)
                    print("slope_threshold", slope_threshold)

                
                if args.cliff_type=="first":
                    cliff_idx = first_cliff(performance_metric_clean, window=window, drop_threshold=drop_threshold)
                elif args.cliff_type=="biggest":
                    cliff_idx = biggest_cliff(performance_metric_clean, window=window, drop_threshold=drop_threshold)
                elif args.cliff_type == "smooth_biggest":
                    performance_metric_clean = moving_average(performance_metric_clean, avg_window=average_window)
                    cliff_idx = biggest_cliff(performance_metric_clean, window=window, drop_threshold=drop_threshold)
                elif args.cliff_type == "smooth_first":
                    performance_metric_clean = moving_average(performance_metric_clean, avg_window=average_window)
                    cliff_idx = first_cliff(performance_metric_clean, window=window, drop_threshold=drop_threshold)
                elif args.cliff_type == "fixed":
                    cliff_idx=-1
                elif args.cliff_type == "detect_cliff":
                    cliff_idx = detect_cliff(performance_metric_clean, slope_window=slope_window, min_consec=min_consec, slope_threshold=slope_threshold)
                elif args.cliff_type == "smooth_detect_cliff":
                    performance_metric_clean = moving_average(performance_metric_clean, avg_window=average_window)
                    cliff_idx = detect_cliff(performance_metric_clean, slope_window=slope_window, min_consec=min_consec, slope_threshold=slope_threshold)
                else:
                    raise ValueError(f"Unknown cliff type: {args.cliff_type}")
            

                # ----- Plotting and Saving -----

                if args.save_txt:   
                    store_df(results_clean_loop, subfolder, "clean_table.xlsx")
                half_life_sparsity_clean = 0
                if half_life_metric:
                    half_life = results_clean_loop["TPR"].max()/2 
                    half_life_idx = results_clean_loop[results_clean_loop["TPR"]<=half_life].index.values[0]
                    half_life_sparsity_clean = results_clean_loop["sparsity_ratio"].iloc[half_life_idx]
                    fig = TP_curve(results_clean_loop, performance_metric_clean, cliff_value=half_life_idx, title="")
                    save_img(fig, name=f"half_life_clean", out_path=f"{args.model_name}/{args.task}/Pruning/half_life")

                if cliff_idx==-1:
                    clean_cliff= 0.75                     
                else:
                    clean_cliff = results_clean_loop["sparsity_ratio"].iloc[cliff_idx]
                
                
                if args.save_img or args.show:
                    fig = ROC_curve(results_clean_loop, performance_metric_clean, cliff_value=clean_cliff,  title=f"{args.cliff_type} - {args.task} task on {args.metrics} metric")
                    if args.save_img:
                        save_img(fig, name=f"clean_ROC", out_path=subfolder)
                                
                if args.verbose:
                    print("cliff at", clean_cliff)

                
                #----------------------------------------------------------------------------------------------------
                #                   Get Circuit by Prunning FLAP at cliff sparisity
                #----------------------------------------------------------------------------------------------------                args.pruning_ratio = clean_cliff
                args.pruning_ratio = clean_cliff
                CIRCUIT_CLEAN, scores, GFLOPS, n_traversed_l = prune_flap_modular(args, model, tokenizer)
                LOOP_FLOPS_CLEAN += GFLOPS
                if args.calc_FLOP:
                    LOOP_FLOPS_CLEAN += n_traversed_l * FLOPS_BY_MODULE["blocks.0"] * epochs
                
                res = results_clean_loop[results_clean_loop["sparsity_ratio"] == clean_cliff].iloc[0]     
                           
                if args.verbose:
                    res_pruned_model = print_statics(
                        title="*********** FLAP Circuit vs GT Circuit **************",
                        ave_logit=res["res"], 
                        performance_achieved=res["performance"],
                        CIRCUIT=CIRCUIT_CLEAN, 
                        IOI_CIRCUIT=GT_CIRCUIT,
                        performance_gain=res["gain"]
                        )
                    print(res_pruned_model)

                true_pos_ratio = TPR(CIRCUIT_CLEAN, GT_circuit=GT_CIRCUIT)*100
                false_pos_ratio = FPR(CIRCUIT_CLEAN, GT_circuit=GT_CIRCUIT)*100
                
                LOOP_TIME_CLEAN = time.time() - start_time
                new_res_col = pd.DataFrame({
                    "pruning_type": "clean",
                    "sparsity_ratio":clean_cliff, 
                    "size": circuit_size(CIRCUIT_CLEAN),
                    "ave_logit_diff":res["res"], 
                    "performance":res["performance"], 
                    "gain":res["gain"], 
                    "TPR":true_pos_ratio, 
                    "FPR": false_pos_ratio,
                    "half_life": half_life_sparsity_clean,
                    "FLOP": (LOOP_FLOPS_CLEAN/1e9 + INIT_FLOPS/2)/1e9, 
                    "comp_time":  LOOP_TIME_CLEAN + INIT_COMP_TIME/2
                    }, index=[0])
                final_results = pd.concat([final_results, new_res_col], ignore_index=True)


                if args.show or args.save_img:
                    fig = heat_map_sparsity(
                        scores, 
                        GT_CIRCUIT,
                        CIRCUIT_CLEAN, 
                        title=f"{args.cliff_type} - {args.task} - Clean FLAP",
                        title_eval_circuit="FLAP",
                        title_compare_circuit="Path Patching",
                        performance=res["performance"],
                        print_vals=False,
                        title_temp_scale=args.metrics)
                        
                    if args.save_img:        
                        save_img(fig, subfolder, "clean_heatmap.png")

                if args.save_txt:
                    save_parser_information(args, subfolder, "clean_parser_info.json")

                if args.save_txt:
                    save_circuit(CIRCUIT_CLEAN, subfolder, name="clean_circuit.txt")

                if args.verbose:
                    print("FLOPS after clean", LOOP_FLOPS_CLEAN/1e9)
                    print("elapsed time for clean loop", LOOP_TIME_CLEAN)

                #----------------------------------------------------------------------------------------------------
                #                   FLAP on corrupted input - WIFN metric
                #----------------------------------------------------------------------------------------------------
                start_time = time.time() 
                if args.calc_FLOP:
                    LOOP_FLOPS_CORR = len(range(ls, hs)) *  FLOPS_BY_MODULE[""] * epochs # lowest - highest number of evaluation forward passes - ABLATED
                else:
                    LOOP_FLOPS_CORR = 0
                    
                args.metrics = "WIFN"
                args.difference_with = "corrupted"
                
                # ----- Cliff ------
                results_corr_loop = results_corr[(results_corr["sparsity_ratio"] >= ls / 100) & (results_corr["sparsity_ratio"] < hs / 100)]
                performance_metric = results_corr_loop[y_variable].tolist()
                                
                max_diff = max(results_corr_loop[y_variable]) - min(results_corr_loop[y_variable])
                drop_threshold = max_diff / 10  # drop 10% performance of max difference over a window of length x
                slope_threshold=(max_diff/250) * 0.5 -((10-average_window)/100)
                slope_threshold=-slope_threshold * 10

                if args.verbose:
                    print("corrupted thresholds")
                    print("drop_threshold", drop_threshold)
                    print("slope_threshold", slope_threshold)


                if args.cliff_type=="first":
                    cliff_idx = first_cliff(performance_metric, window=window, drop_threshold=drop_threshold)
                elif args.cliff_type=="biggest":
                    cliff_idx = biggest_cliff(performance_metric, window=window, drop_threshold=drop_threshold)
                elif args.cliff_type == "smooth_biggest":
                    performance_metric = moving_average(performance_metric, avg_window=average_window)
                    cliff_idx = biggest_cliff(performance_metric, window=window, drop_threshold=drop_threshold)
                elif args.cliff_type == "smooth_first":
                    performance_metric = moving_average(performance_metric, avg_window=average_window)
                    cliff_idx = first_cliff(performance_metric, window=window, drop_threshold=drop_threshold)
                elif args.cliff_type == "fixed":
                    cliff_idx=-1
                elif args.cliff_type == "detect_cliff":
                    cliff_idx = detect_cliff(performance_metric, slope_window=slope_window, min_consec=min_consec, slope_threshold=slope_threshold)
                elif args.cliff_type == "smooth_detect_cliff":
                    performance_metric = moving_average(performance_metric, avg_window=average_window)
                    cliff_idx = detect_cliff(performance_metric, slope_window=slope_window, min_consec=min_consec, slope_threshold=slope_threshold)
                else:
                    raise ValueError(f"Unknown cliff type: {args.cliff_type}")

                # ----- Plotting and Saving -----
                if args.save_txt:   
                    store_df(results_corr_loop, subfolder, "ablated_table.xlsx")
                half_life_sparsity_corr = 0
                if half_life_metric:
                    half_life = results_corr_loop["TPR"].max()/2
                    half_life_idx=results_corr_loop[results_corr_loop["TPR"]<=half_life].index.values[0]
                    half_life_sparsity_corr = results_corr_loop["sparsity_ratio"].iloc[half_life_idx]

                    fig = TP_curve(results_corr_loop, performance_metric, cliff_value=half_life_idx, title="")
                    save_img(fig, name=f"half_life_corr", out_path=f"{args.out_path}/{args.model_name}/{args.task}/Pruning/half_life")
                    
                if cliff_idx==-1:
                    corr_cliff = 0.75
                else:
                    corr_cliff = results_corr_loop["sparsity_ratio"].iloc[cliff_idx]
                
                if args.save_img or args.show:                    
                    fig1 = ROC_curve(results_corr_loop, performance_metric, cliff_value=corr_cliff, title=f"{args.cliff_type} - {args.task} task on {args.metrics} metric")
                    fig2 = two_ROC_curve(results_corr_loop, performance_metric, corr_cliff, results_clean_loop, performance_metric_clean, clean_cliff, title="Cliff Points", p1=None, p2=None)
                    
                if args.save_img:
                        save_img(fig1, name=f"ablated_ROC", out_path=subfolder)
                        save_img(fig2, name=f"both_curves", out_path=subfolder)

                
                if args.verbose:
                    print("cliff at", corr_cliff)



                #----------------------------------------------------------------------------------------------------
                #                   Get Circuit by running FLAP at cliff sparisity
                #----------------------------------------------------------------------------------------------------
                
                args.pruning_ratio = corr_cliff

                CIRCUIT_ABLATED, scores, GFLOPS, n_traversed_l = prune_flap_modular(args, model, tokenizer)
                LOOP_FLOPS_CORR += GFLOPS
                if args.calc_FLOP:
                    LOOP_FLOPS_CORR += n_traversed_l * FLOPS_BY_MODULE["blocks.0"] * epochs
                
                res = results_corr_loop[results_corr_loop["sparsity_ratio"] == corr_cliff].iloc[0]                
                
                if args.verbose:
                    res_pruned_model = print_statics(
                        title="*********** FLAP CIrcuit vs GT Circuit **************",
                        ave_logit=res["res"], 
                        performance_achieved=res["performance"],
                        CIRCUIT=CIRCUIT_ABLATED, 
                        IOI_CIRCUIT=GT_CIRCUIT,
                        performance_gain=res["gain"]
                        )
                    print(res_pruned_model)



                true_pos_ratio = TPR(CIRCUIT_ABLATED, GT_circuit=GT_CIRCUIT)*100
                false_pos_ratio = FPR(CIRCUIT_ABLATED, GT_circuit=GT_CIRCUIT)*100
                LOOP_TIME_CORR = time.time() - start_time
                new_res_col = pd.DataFrame({
                    "pruning_type": "ablate",
                    "sparsity_ratio":corr_cliff, 
                    "size": circuit_size(CIRCUIT_ABLATED),
                    "ave_logit_diff":res["res"], 
                    "performance":res["performance"], 
                    "gain":res["gain"], 
                    "TPR":true_pos_ratio, 
                    "FPR": false_pos_ratio,
                    "half_life":half_life_sparsity_corr,
                    "FLOP": (LOOP_FLOPS_CORR + INIT_FLOPS/2)/1e9, 
                    "comp_time": LOOP_TIME_CORR + INIT_COMP_TIME/2
                    }, index=[0])

                final_results = pd.concat([final_results, new_res_col], ignore_index=True)


                if args.show or args.save_img:
                    fig = heat_map_sparsity(
                        scores, 
                        GT_CIRCUIT,
                        CIRCUIT_ABLATED, 
                        title=f"{args.cliff_type} - {args.task} - contrastive FLAP",
                        title_eval_circuit="FLAP",
                        title_compare_circuit="Path Patching",
                        performance=res["performance"],
                        print_vals=False,
                        title_temp_scale=args.metrics)
                    fig.show()
                    if args.save_img:
                        save_img(fig, subfolder, "contrastive_heatmap.png")

                if args.save_txt:
                    save_parser_information(args, subfolder, "contrastive_parser_info.json")
                
                if args.save_txt:
                    save_circuit(CIRCUIT_ABLATED, subfolder, name="contrastive_circuit.txt")

                if args.verbose:
                    print("FLOPS after corrupted", LOOP_FLOPS_CORR/1e9)
                    print("elapsed time for loop", LOOP_TIME_CORR)

                #----------------------------------------------------------------------------------------------------
                #                  Merge Clean and Corrupted Circuit - Save temporary
                #----------------------------------------------------------------------------------------------------
                if half_life_metric:
                    fig_two_TP = two_TP_curve(results_clean_loop, results_corr_loop, half_life_sparsity_clean, half_life_sparsity_corr)
                    save_img(fig_two_TP, f"{args.out_path}/{args.model_name}/{args.task}/Pruning/half_life", args.task + "_two_TP.png")
                
                
                HYBRID_CIRCUIT = merge_circuits(CIRCUIT_CLEAN, CIRCUIT_ABLATED)
                new_res_col = pd.DataFrame({
                    "pruning_type": "hybrid",
                    "sparsity_ratio":-1, 
                    "size": -1,
                    "ave_logit_diff":-1, 
                    "performance":-1, 
                    "gain":-1, 
                    "TPR":-1, 
                    "FPR": -1,
                    "half_life": -1, 
                    "FLOP":  (INIT_FLOPS + LOOP_FLOPS_CLEAN + LOOP_FLOPS_CORR)/1e9, 
                    "comp_time": INIT_COMP_TIME + LOOP_TIME_CLEAN + LOOP_TIME_CORR
                    }, index=[0])
                
                final_results = pd.concat([final_results, new_res_col], ignore_index=True)
                if args.save_txt:
                    save_circuit(HYBRID_CIRCUIT, subfolder, name="hybrid_circuit.txt")
                store_df(final_results, subfolder, "results.json")
                
    if half_life_metric:
        return 
    
    # hooked model
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # ---- load Hooked model ----
    model_hooked = load_hooked_transformer(model_name=args.model_name, device=args.device, cache_dir=args.cache_dir)  


    for cliff_f in cliff_functions:
        for ls in lowest_sparsities:
            for hs in highest_sparsities:
                
                #if hs == 99 and not ls == 0:
                #    continue
                
                args.lowest_sparsity = ls
                args.highest_sparsity = hs
                args.cliff_type = cliff_f

                
                result_folder =  f"{args.model_name}/{args.task}/Pruning/{args.cliff_type}/sparsity-min_{args.lowest_sparsity}/"

                if args.out_path == "":
                    subfolder = result_folder
                else:
                    subfolder = args.out_path + result_folder

                HYBRID_CIRCUIT = load_circuit(out_path=subfolder, name="hybrid_circuit.txt")
                final_results = load_df(subfolder, name="results.json")
                
                ave_logit, performance = batch_evaluate_circiut(
                    model = model_hooked, 
                    CIRCUIT=HYBRID_CIRCUIT,
                    dataset=eval_dataset,
                    ave_logit_gt=ave_logit_gt, 
                    task=args.task,
                    model_name=args.model_name, 
                    epochs = epochs, 
                    batch_size = args.batch_size 
                    )

                gain = performance_gain(performance_new=performance, performance_old=gt_circuit_performance)

                if args.verbose:
                    res_pruned_model = print_statics(
                        title="*********** Hybrid CIrcuit vs GT Circuit **************",
                        ave_logit=ave_logit, 
                        performance_achieved=performance,
                        CIRCUIT=HYBRID_CIRCUIT, 
                        IOI_CIRCUIT=GT_CIRCUIT,
                        performance_gain=gain
                        )
                    print(res_pruned_model)




                true_pos_ratio = TPR(HYBRID_CIRCUIT, GT_circuit=GT_CIRCUIT)*100
                false_pos_ratio = FPR(HYBRID_CIRCUIT, GT_circuit=GT_CIRCUIT)*100


                # ---- save ----
                if args.save_img or args.show:
                    fig = heat_map_sparsity(
                        torch.zeros((n_layers, n_heads)), 
                        GT_CIRCUIT,
                        HYBRID_CIRCUIT, 
                        title=f"{args.cliff_type} - {args.task} - Hybrid Circuit",
                        title_eval_circuit="FLAP",
                        title_compare_circuit="Path Patching",
                        performance=performance,
                        print_vals=False,
                        title_temp_scale="",
                        scale_on=False)
                    if args.save_img:
                        save_img(fig, subfolder, f"hybrid_heatmap.png")
                    
                # ----- performance metrics -----
                hybrid_res = final_results[final_results["pruning_type"] == "hybrid"].iloc[0]
                new_res_row = pd.DataFrame({
                    "pruning_type": "hybrid",
                    "sparsity_ratio":-1, 
                    "size": circuit_size(HYBRID_CIRCUIT),
                    "ave_logit_diff":ave_logit, 
                    "performance":performance, 
                    "gain":gain, 
                    "TPR":true_pos_ratio, 
                    "FPR": false_pos_ratio,
                    "half_life": -1, 
                    "FLOP": hybrid_res["FLOP"], 
                    "comp_time": hybrid_res["comp_time"]
                    }, index=[0])

                final_results.iloc[2] = new_res_row.iloc[0]
                if args.save_txt:
                    save_circuit(HYBRID_CIRCUIT, subfolder, name="hybrid_circuit.txt")    
                    store_df(final_results, subfolder, "results.json")


if __name__ == "__main__":
    
    args = parser.parse_args()
    lowest_sparsities = [60]
    highest_sparsities = [99]
    cliff_functions =["smooth_biggest", "biggest", "smooth_first", "first", "fixed", "detect_cliff", "smooth_detect_cliff"]
    set_PATH(args.out_path)
    all_tasks=["ioi", "GreaterThan", "GenderedPronouns", "induction", "Docstring"]
    #all_tasks=["induction", "Docstring"]
    for t in all_tasks:
        args.task=t
        hybrid_FLAP(
            args=args,
            lowest_sparsities=lowest_sparsities, 
            highest_sparsities=highest_sparsities,
            cliff_functions=cliff_functions, 
            half_life_metric=False
        )