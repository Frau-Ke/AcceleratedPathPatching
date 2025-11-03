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
from utils.utils import save_img, create_folder, save_parser_information, save_circuit, store_df, save_parser_information
from utils.metrics import ave_logit_diff
from utils.eval_circuit import *
from utils.visualization import outline_IoU, outline_IOI, heat_map_sparsity, ROC_curve
import time
from fvcore.nn import FlopCountAnalysis
from utils.model_loader import get_gpt2_adapt_to_llama, load_tokenizer, load_hooked_transformer, load_transformer

gc.collect()
t.cuda.empty_cache()
t.autograd.set_grad_enabled(False)


average_window=5
window = 5
drop_threshold=0.2


def first_cliff(results):
    max_drop=0
    for i in range(0, len(results)-window):
        drop = abs(results[i] - min(results[i+1:i+window]))
        drop = drop / 100
        if drop > max_drop and drop >= drop_threshold:
            max_drop = drop
            cliff_idx = i
            # check for valley:
            if results[i] > max(results[i+1:]):
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



def hybrid_FLAP(
    args,
    highest_sparsities=[0, 60],
    lowest_sparsities=[90, 99],
    cliff_functions=["first", "smooth_first", "biggest"]
    ):
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    y_variable = "performance"

    # ------ load model ------
    if "gpt2" in args.model_name:
        model = get_gpt2_adapt_to_llama(args.model_name, args.device)
        n_layers = model.config.n_layer
        n_heads =  model.config.n_head
    elif "Qwen" in args.model_name:
        model = load_transformer(args.model_name, args.device, cache_dir=args.cache_dir)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        
    tokenizer = load_tokenizer(args.model_name)

    model_hooked = load_hooked_transformer(model_name=args.model_name, device=args.device)  

    # ------ get circuits ------
    try:
        GT_CIRCUIT = choose_circuit(args.task, args.model_name)
    except:
        GT_CIRCUIT = {}

    gt_circuit_name =  get_circuit_name(args.task)

    # ------ get dataset ------
    eval_dataset = load_dataset(
        model_name=args.model_name,
        task=args.task, 
        tokenizer=tokenizer,  
        N=200, 
        patching_method="path", 
        device=args.device, 
        seed=args.seed + 20, 
        prepend_bos=args.prepend_bos
        )

    # ------ ave logit of unpruned model ------
    # forward pass 1
    with torch.no_grad():
        hooked_gt = model_hooked(eval_dataset.clean_tokens)
        
    ave_logit_gt = ave_logit_diff(
        hooked_gt, 
        eval_dataset.answer_tokens, 
        eval_dataset.target_idx.to(args.device), 
        task=args.task, 
        model_name=args.model_name
        )

    # ------ ave logit of gt circuit ------
    # fowardpass 2
    gt_circuit_ave_logit, gt_circuit_performance = evaluate_circiut(
        model = model_hooked, 
        CIRCUIT=GT_CIRCUIT,
        dataset=eval_dataset,
        ave_logit_gt=ave_logit_gt, 
        task=args.task,
        model_name=args.model_name
        )
    
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


        
    for cliff_f in cliff_functions:
        for ls in lowest_sparsities:
            for hs in highest_sparsities:
                print(f"cliff {cliff_f} from {ls} to {hs}")
                
                TOTAL_GFLOPS = 0
                n_forward_passes = 2 # standard is doing two forward passes
                
                if hs == 99 and not ls == 0:
                    continue
                
                args.lowest_sparsity = ls
                args.highest_sparsity = hs
                args.cliff_type = cliff_f

                
                result_folder =  f"{args.model_name}/{args.task}/Hybrid-FLAP/{args.cliff_type}/sparsity-min_{args.lowest_sparsity}/sparsity-max_{args.highest_sparsity}/"

                if args.out_path == "":
                    subfolder = result_folder
                else:
                    subfolder = args.out_path + result_folder

                create_folder(subfolder)
                save_parser_information(args, subfolder, name="parser_infomation.txt")


                final_results = pd.DataFrame(columns=["pruning_type", "sparsity_ratio", "size", "ave_logit_diff", "performance", "gain", "diff", "TPR", "FLOP", "comp_time"])
                
                start_time = time.time() 


                #----------------------------------------------------------------------------------------------------
                #                   FLAP on clean input - WIFV metric
                #----------------------------------------------------------------------------------------------------

                args.metrics = "WIFV"
                args.difference_with = "None"    
                
                # ------ Headwise pruning Scores -----
                scores, mlp_scores, mlp_mask, unstandardized_scores, GFLOPS, n_traversed_l = head_wise_pruning_scores(args, model, tokenizer)
                TOTAL_GFLOPS += GFLOPS

                # ------ Evaluation of differen sparsities -----
                ave_logit_old = gt_circuit_ave_logit

                results = pd.DataFrame(columns=["sparsity_ratio", "res", "performance", "gain", "diff", "TPR"])

                for i in range(args.lowest_sparsity, args.highest_sparsity, args.step_size):
                    args.pruning_ratio = i / 100
                    CIRCUIT, _ , GFLOPS = CIRCUIT_from_scores(
                        args,
                        attn_metric_list=scores,
                        W_metric_unstand_list=unstandardized_scores,
                        mlp_metric_list=mlp_scores,
                        mlp_mask=mlp_mask,
                        FLOPS=GFLOPS,
                        n_layers=n_layers, 
                        n_heads=n_heads, 
                        head_dim=model_hooked.cfg.d_head, 
                        )
                                            
                    TOTAL_GFLOPS += GFLOPS
                    
                    ave_logit, performance = evaluate_circiut(
                        model = model_hooked, 
                        CIRCUIT=CIRCUIT,
                        dataset=eval_dataset,
                        ave_logit_gt=ave_logit_gt, 
                        task=args.task,
                        model_name=args.model_name
                        )
                    n_forward_passes += 1    

                    
                    gain = performance_gain(performance_new=performance, performance_old=gt_circuit_performance)

                    diff = ave_logit_old - ave_logit
                    true_pos = get_intersection_num(CIRCUIT, GT_CIRCUIT)
                    
                    if args.verbose:
                        print(f"sparsity_ratio: {args.pruning_ratio}, ave_logit_diff: {ave_logit}, performance: {performance}, gain {gain}, diff: {diff}, TPR:{true_pos}")
                    
                    new_col = pd.DataFrame({"sparsity_ratio":args.pruning_ratio, "res":ave_logit, "performance":performance, "gain":gain, "diff":diff, "TPR":true_pos}, index=[0])
                    results = pd.concat([results, new_col], ignore_index=True)
                    ave_logit_old = ave_logit
                    
                    
                # ----- Cliff ------
                performance_metric = results[y_variable]

                if args.cliff_type=="first":
                    cliff_idx = first_cliff(performance_metric)
                elif args.cliff_type=="biggest":
                    cliff_idx = results["diff"][1:].idxmax() - 1
                elif args.cliff_type == "smooth_first":
                    performance_metric = moving_average(performance_metric, avg_window=average_window)
                    cliff_idx = first_cliff(performance_metric)

                if cliff_idx==-1:
                    cliff_idx = results.index[results["sparsity_ratio"] == 0.75].tolist()[0]

                if args.save_txt:   
                    store_df(results, subfolder, "clean_table.xlsx")
                    
                fig = ROC_curve(results, performance_metric, cliff_idx=cliff_idx, title=f"{args.task} task on {args.metrics} metric")
                if args.save_img:
                    save_img(fig, name=f"clean_ROC", out_path=subfolder)
                    
                # ----- Plotting and Saving -----
                cliff = results["sparsity_ratio"][cliff_idx]
                
                
                
                #----------------------------------------------------------------------------------------------------
                #                   Get Circuit by running FLAP at cliff sparisity
                #----------------------------------------------------------------------------------------------------
                
                args.pruning_ratio = cliff

                CIRCUIT_CLEAN, scores, GFLOPS, n_traversed_l = prune_flap_modular(args, model, tokenizer)
                TOTAL_GFLOPS += GFLOPS

                ave_logit, performance = evaluate_circiut(
                    model = model_hooked, 
                    CIRCUIT=CIRCUIT_CLEAN,
                    dataset=eval_dataset,
                    ave_logit_gt=ave_logit_gt,
                    task=args.task,
                    model_name=args.model_name
                    )

                gain = performance_gain(performance_new=performance, performance_old=gt_circuit_performance)

                if args.verbose:
                    res_pruned_model = print_statics(
                        title="*********** FLAP Circuit vs GT Circuit **************",
                        ave_logit=ave_logit, 
                        performance_achieved=performance,
                        CIRCUIT=CIRCUIT_CLEAN, 
                        IOI_CIRCUIT=GT_CIRCUIT,
                        performance_gain=gain
                        )
                    print(res_pruned_model)

                fig = heat_map_sparsity(
                    scores, 
                    GT_CIRCUIT,
                    CIRCUIT_CLEAN, 
                    title=f"{args.task} - Clean FLAP",
                    title_eval_circuit="FLAP",
                    title_compare_circuit=gt_circuit_name,
                    performance=performance,
                    print_vals=False,
                    title_temp_scale=args.metrics)

                true_pos_ratio = TPR(CIRCUIT_CLEAN, GT_circuit=GT_CIRCUIT)*100
                false_pos_ratio = FPR(CIRCUIT_CLEAN, GT_circuit=GT_CIRCUIT)*100

                new_res_col = pd.DataFrame({
                    "pruning_type": "clean",
                    "sparsity_ratio":cliff, 
                    "size": circuit_size(CIRCUIT_CLEAN),
                    "ave_logit_diff":ave_logit, 
                    "performance":performance, 
                    "gain":gain, 
                    "TPR":true_pos_ratio, 
                    "FPR": FPR,
                    "FLOP": TOTAL_GFLOPS/1e9, 
                    "comp_time":  time.time() - start_time
                    }, index=[0])

                final_results = pd.concat([final_results, new_res_col], ignore_index=True)


                if args.show:
                    fig.show()
                    
                if args.save_img:        
                    save_img(fig, subfolder, "clean_heatmap.png")

                if args.save_txt:
                    save_parser_information(args, subfolder, "clean_parser_info.json")

                save_circuit(CIRCUIT_CLEAN, subfolder, name="clean_circuit.txt")


                #----------------------------------------------------------------------------------------------------
                #                   FLAP on corrupted input - WIFN metric
                #----------------------------------------------------------------------------------------------------

                args.metrics = "WIFN"
                args.difference_with = "corrupted"

                # ------ Headwise pruning Scores -----
                scores, mlp_scores, mlp_mask, unstandardized_scores, GFLOPS, n_traversed_l = head_wise_pruning_scores(args, model, tokenizer)
                TOTAL_GFLOPS += GFLOPS

                # ------ Evaluation of differen sparsities -----
                ave_logit_old = gt_circuit_ave_logit
                results = pd.DataFrame(columns=["sparsity_ratio", "res", "performance", "gain", "diff", "TPR"])

                for i in range(args.lowest_sparsity, args.highest_sparsity, args.step_size):
                    args.pruning_ratio = i / 100
                    CIRCUIT, _, GFLOPS= CIRCUIT_from_scores(
                        args, 
                        attn_metric_list=scores, 
                        W_metric_unstand_list=unstandardized_scores, 
                        mlp_metric_list=mlp_scores, 
                        mlp_mask=mlp_mask,
                        FLOPS=GFLOPS,
                        n_layers=n_layers, 
                        n_heads=n_heads, 
                        head_dim=model_hooked.cfg.d_head, 
                        )
                    TOTAL_GFLOPS += GFLOPS
                    
                    ave_logit, performance = evaluate_circiut(
                        model = model_hooked, 
                        CIRCUIT=CIRCUIT,
                        dataset=eval_dataset,
                        ave_logit_gt=ave_logit_gt,
                        task=args.task,
                        model_name=args.model_name
                        )
                    n_forward_passes += 1    
                    
                    gain = performance_gain(performance_new=performance, performance_old=gt_circuit_performance)

                    diff = ave_logit_old - ave_logit
                    true_pos = get_intersection_num(CIRCUIT, GT_CIRCUIT)

                    if args.verbose:
                        print(f"sparsity_ratio: {args.pruning_ratio}, ave_logit_diff: {ave_logit}, performance: {performance}, gain {gain}, diff: {diff}, TPR:{true_pos}")
                    
                    new_col = pd.DataFrame({"sparsity_ratio":args.pruning_ratio, "res":ave_logit, "performance":performance, "gain":gain, "diff":diff, "TPR":true_pos}, index=[0])
                    results = pd.concat([results, new_col], ignore_index=True)
                    ave_logit_old = ave_logit
                    
                # ----- Cliff ------

                performance_metric = results[y_variable]
                # ----- Cliff ------
                if args.cliff_type=="first":
                    cliff_idx = first_cliff(performance_metric)
                elif args.cliff_type=="biggest":
                    cliff_idx = results["diff"][1:].idxmax() - 1
                elif args.cliff_type == "smooth_first":
                    performance_metric = moving_average(performance_metric, avg_window=average_window)
                    cliff_idx = first_cliff(performance_metric)

                if cliff_idx==-1:
                    cliff_idx = results.index[results["sparsity_ratio"] == 0.75].tolist()[0]



                # ----- Plotting and Saving -----
                if args.save_txt:   
                    store_df(results, subfolder, "ablated_table.xlsx")

                fig = ROC_curve(results, performance_metric, cliff_idx=cliff_idx, title=f"{args.task} task on {args.metrics} metric")

                if args.save_img:
                    save_img(fig, name=f"ablated_ROC", out_path=subfolder)
                    

                cliff = results["sparsity_ratio"][cliff_idx]
                
                if args.verbose:
                    print("cliff at", cliff)



                #----------------------------------------------------------------------------------------------------
                #                   Get Circuit by running FLAP at cliff sparisity
                #----------------------------------------------------------------------------------------------------
                
                args.pruning_ratio = cliff

                CIRCUIT_ABLATED, scores, GFLOPS, n_traversed_l = prune_flap_modular(args, model, tokenizer)
                TOTAL_GFLOPS += GFLOPS

                ave_logit, performance = evaluate_circiut(
                    model = model_hooked, 
                    CIRCUIT=CIRCUIT_ABLATED,
                    dataset=eval_dataset,
                    ave_logit_gt=ave_logit_gt,
                    task=args.task,
                    model_name=args.model_name
                    )

                gain = performance_gain(performance_new=performance, performance_old=gt_circuit_performance)

                if args.verbose:
                    res_pruned_model = print_statics(
                        title="*********** FLAP CIrcuit vs GT Circuit **************",
                        ave_logit=ave_logit, 
                        performance_achieved=performance,
                        CIRCUIT=CIRCUIT_ABLATED, 
                        IOI_CIRCUIT=GT_CIRCUIT,
                        performance_gain=gain
                        )
                    print(res_pruned_model)

                fig = heat_map_sparsity(
                    scores, 
                    GT_CIRCUIT,
                    CIRCUIT_ABLATED, 
                    title=f"{args.task} - Ablated FLAP",
                    title_eval_circuit="FLAP",
                    title_compare_circuit=gt_circuit_name,
                    performance=performance,
                    print_vals=False,
                    title_temp_scale=args.metrics)


                true_pos_ratio = TPR(CIRCUIT_ABLATED, GT_circuit=GT_CIRCUIT)*100
                false_pos_ratio = FPR(CIRCUIT_ABLATED, GT_circuit=GT_CIRCUIT)*100

                new_res_col = pd.DataFrame({
                    "pruning_type": "ablate",
                    "sparsity_ratio":cliff, 
                    "size": circuit_size(CIRCUIT_ABLATED),
                    "ave_logit_diff":ave_logit, 
                    "performance":performance, 
                    "gain":gain, 
                    "TPR":true_pos_ratio, 
                    "FPR": false_pos_ratio,
                    "FLOP": -1, 
                    "comp_time": -1
                    }, index=[0])

                final_results = pd.concat([final_results, new_res_col], ignore_index=True)


                if args.show:
                    fig.show()
                    
                if args.save_img:
                    save_img(fig, subfolder, "ablated_heatmap.png")

                if args.save_txt:
                    save_parser_information(args, subfolder, "ablated_parser_info.json")
                    
                save_circuit(CIRCUIT_ABLATED, subfolder, name="ablated_circuit.txt")

                    
                #----------------------------------------------------------------------------------------------------
                #                  Merge Clean and Corrupted Circuit and Evaluate
                #----------------------------------------------------------------------------------------------------

                end_time = time.time()  
                elapsed_time = end_time - start_time

                HYBRID_CIRCUIT = merge_circuits(CIRCUIT_CLEAN, CIRCUIT_ABLATED)
                ave_logit, performance = evaluate_circiut(
                    model = model_hooked, 
                    CIRCUIT=HYBRID_CIRCUIT,
                    dataset=eval_dataset,
                    ave_logit_gt=ave_logit_gt, 
                    task=args.task,
                    model_name=args.model_name
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

                fig = heat_map_sparsity(
                    torch.zeros((n_layers, n_heads)), 
                    GT_CIRCUIT,
                    HYBRID_CIRCUIT, 
                    title=f"{args.task} - Hybrid Circuit",
                    title_eval_circuit="FLAP",
                    title_compare_circuit=gt_circuit_name,
                    performance=performance,
                    print_vals=False,
                    title_temp_scale="",
                    scale_on=False)


                true_pos_ratio = TPR(HYBRID_CIRCUIT, GT_circuit=GT_CIRCUIT)*100
                false_pos_ratio = FPR(HYBRID_CIRCUIT, GT_circuit=GT_CIRCUIT)*100


                # ---- save ----
                if args.show:
                    fig.show()

                if args.save_img:
                    save_img(fig, subfolder, f"hybrid_heatmap.png")
                    
                # ----- performance metrics -----
                EVAL_FLOPS = FlopCountAnalysis(model_hooked, eval_dataset.clean_tokens).total()
                TOTAL_GFLOPS = TOTAL_GFLOPS + n_forward_passes * EVAL_FLOPS
                TOTAL_GFLOPS = TOTAL_GFLOPS /  1e9
                    
                new_res_col = pd.DataFrame({
                    "pruning_type": "hybrid",
                    "sparsity_ratio":-1, 
                    "size": circuit_size(HYBRID_CIRCUIT),
                    "ave_logit_diff":ave_logit, 
                    "performance":performance, 
                    "gain":gain, 
                    "TPR":true_pos_ratio, 
                    "FPR": false_pos_ratio,
                    "FLOP": TOTAL_GFLOPS, 
                    "comp_time": elapsed_time
                    }, index=[0])

                final_results = pd.concat([final_results, new_res_col], ignore_index=True)
                    
                    
                save_circuit(HYBRID_CIRCUIT, subfolder, name="hybrid_circuit.txt")    
                store_df(final_results, subfolder, "results.json")
                save_parser_information(args, subfolder, name="parser_infomation.txt")


if __name__ == "__main__":
    
    args = parser.parse_args()
    lowest_sparsities = [60]
    highest_sparsities = [99]
    cliff_functions =["smooth_biggest", "biggest", "smooth_first", "first", "fixed", "detect_cliff", "smooth_detect_cliff"]


    hybrid_FLAP(
        args=args,
        lowest_sparsities=lowest_sparsities, 
        highest_sparsities=highest_sparsities,
        cliff_functions=cliff_functions
    )