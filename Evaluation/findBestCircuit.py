import os
import pandas as pd
import torch
import numpy as np

from dataset.loader import load_dataset

from utils.metrics import ave_logit_diff, accuracy
from utils.model_loader import load_tokenizer, load_hooked_transformer
from utils.circuit_functions import TPR, precision, circuit_size, merge_circuits
from utils.eval_circuit import  batch_evaluate_circiut
from utils.utils import get_model_parameters
from utils.data_io import store_df

def pareto_analysis(
    args,
    df:pd.DataFrame,
    x_metric:str="size",
    y_metric:str="performance",
    min_performance:int=float("inf"), 
    max_circuit_size:int=float("inf"),
    focus_on_performance=False,
    method_name:str="PP"
    ):
    """Pareto frontier between two metrics

    Args:
        df (pd.DataFrame): df
        x_metric (str) columm name of df,
        y_metric (str): column name of df,
        min_performance (int, optional): minimal performance. Defaults to 75.
        total_model_heads (int, optional): total number of heads in model. Defaults to 144.
        save_image (bool, optional): save_img. Defaults to True.
        out_path (str, optional): out_path. Defaults to "".

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    
    df_task = df[df["task"]== args.task] 

    def pareto_frontier(df, x_metric, y_metric, maximize_y=True, minimize_x=True):
        df_sorted = df.sort_values(by=[x_metric], ascending=minimize_x)
        pareto = []
        best_y = -float("inf") if maximize_y else float("inf")
        for _, row in df_sorted.iterrows():
            y = row[y_metric]
            if maximize_y:
                if y > best_y:
                    pareto.append(row)
                    best_y = y
            else:
                if y < best_y:
                    pareto.append(row)
                    best_y = y
        return pd.DataFrame(pareto)

    pareto = pareto_frontier(df_task, x_metric, y_metric)
    
    try:
        best_point = df_task[df_task[y_metric] >= min_performance].sort_values(by=[x_metric, y_metric], ascending = [True, False]).iloc[0]
        if best_point[x_metric] > max_circuit_size:
            raise Exception

    except:            
        # if no point is over 75%, choose that point on the pareto curve furthest away from line between leftmost and rightmost pareto point
        df_pareto = df_task.loc[pareto.index]
        x = df_pareto[x_metric].values
        y = df_pareto[y_metric].values
        
        # line between leftmost and rightmost pareto point
        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])
        v = p2 - p1
        v_norm = np.linalg.norm(v)

        # only choose pareto points above line
        signed_distances = []
        for i in range(len(x)):
            # distance of each point above line to it
            p = np.array([x[i], y[i]])
            cross = np.cross(v, p - p1)
            signed_distances.append(cross / v_norm)

        signed_distances = np.array(signed_distances)
        valid = signed_distances > 0
        
        def sublistfinder(list, sublist, last_knee=False):
            if not last_knee:
                for i in  range(len(list)-len(sublist)):
                    if (list[i:i+len(sublist)] == sublist).all():
                        return i
            else:
                for i in range(len(list)-len(sublist), 0, -1):
                    if (list[i:i+len(sublist)] == sublist).all():
                        return i
            return -1    
        
        if not any(valid):
            #   Case 1: no valid pareto point:
            # - for FLAP: choose point with highest performance, consistent with constraint circuits_siue <= max_circuit_size
            # - for PP and APP: choose point with smallest circuit
            if focus_on_performance and len(df_pareto)-1 > 0:
                for knee_idx in range(len(df_pareto)-1, 0, -1):
                    if df_pareto.iloc[knee_idx].size <= max_circuit_size:
                        break
            else:
                knee_idx=0
        
        elif not sublistfinder(valid[1:-1], [True, False]) == -1 and not sublistfinder(valid[1:-1], [False, True]) == -1:
            #   Case 2: "Zig-Zagging" pareto point: "optimal line" is crossed multiple times 
            # - for FLAP: choose last point crossing line
            # - for PP and APP: choose argmax
            if focus_on_performance and len(df_pareto)-1 > 0:
                knee_idx = sublistfinder(valid, [True, False], last_knee=True)
            else:
                knee_idx = sublistfinder(valid, [True, False])


        else:  
            #    Case 3: else
            # - take valid pareto point furthest from line 
            knee_idx = np.argmax(signed_distances * valid)
            
            if focus_on_performance:
                if signed_distances[knee_idx] < 1 and df_pareto[x_metric].values[-1] < max_circuit_size:
                    knee_idx = -1
                    
                # if behind knee point is still one very steep point with steeper gradient, take it 
                delta_x = [x[i+1] - x[i] for i in range(len(df_pareto[x_metric].values)-1)]
                delta_y = [y[i+1] - y[i] for i in range(len(df_pareto[y_metric].values)-1)]
                gradient = [dy / dx if dx != 0 else float("-inf") for dx, dy in zip(delta_x, delta_y)] 
                if max(gradient[knee_idx:]) > gradient[knee_idx-1]:
                    grad_idx = gradient.index(max(gradient[knee_idx:])) + 1
                    if grad_idx < max_circuit_size:
                        knee_idx = grad_idx

        best_point = df_pareto.iloc[knee_idx]
        if args.save_text:
        
            with open(f'{args.out_path}/{args.model_name}/results/{method_name}/{method_name}_df.txt', 'w') as f:
                print(f"save df at {args.out_path}/{args.model_name}/results/{method_name}/{method_name}_df.txt")
                f.write(df.to_string(header=False, index=False))

    return best_point, pareto


def find_best_PP_circuit(args, circuits_dict:dict, GT_CIRCUIT):
    _, _, epochs = get_model_parameters(args.model_name, args.N, input_batch_size=args.batch_size)
    tokenizer = load_tokenizer(args.model_name)
    model_hooked = load_hooked_transformer(args.model_name, device=args.device, cache_dir=args.cache_dir)

    # directory to Path Patching
    PP_df_results = pd.DataFrame(columns=["task", "maxValue", "importance", "performance", "accuracy", "size", "TPR", "P"])
    
    if GT_CIRCUIT is None:_
    directory = f"{args.out_path}/{args.model_name}/{args.task}/path/automatic"  
 
    eval_dataset = load_dataset(
        prepend_bos=False,
        task=args.task, 
        patching_method="path",
        tokenizer=tokenizer, 
        N=args.N, 
        device=args.device,
        model_name=args.model_name,
        seed=args.seed
    )

    # ----- Average Logit Difference of the unpatched Model -----
    with torch.no_grad():
        logits_gt = model_hooked(eval_dataset.clean_tokens)
        
    ave_logit_gt = ave_logit_diff(
        logits=logits_gt, 
        correct_answers=eval_dataset.correct_answers, 
        wrong_answers=eval_dataset.wrong_answers,
        target_idx=eval_dataset.target_idx.to(args.device), 
        task=args.task,
        model_name=args.model_name
        )
    
    for ma in args.min_value_threshold:
        for s in args.importance_threshold:
            CIRCUIT = circuits_dict[ma][s]
            size = circuit_size(CIRCUIT)            
            _, performance, acc = batch_evaluate_circiut(
                model = model_hooked, 
                CIRCUIT=CIRCUIT,
                dataset=eval_dataset,
                ave_logit_gt=ave_logit_gt,
                task=args.task,
                model_name=args.model_name, 
                epochs = epochs, 
                batch_size = args.batch_size 
            )    
            
            TP_ratio = TPR(CIRCUIT, GT_CIRCUIT)
            prec = precision(CIRCUIT, GT_CIRCUIT)
            
            new_row = pd.DataFrame({
                "task": [args.task],
                "maxValue": [ma],
                "importance": [s],
                "performance": [performance], 
                "accuracy": [acc],
                "size": [size], 
                "TPR": [TP_ratio], 
                "P": [prec]
            }) 
            PP_df_results = pd.concat([PP_df_results, new_row],  ignore_index=True)

    if args.save_text:
        store_df(PP_df_results, out_path=directory, name="results_pp.json")         
          
    return PP_df_results


def find_best_hybridFLAP_circuit(args, circuits_dict:dict, GT_CIRCUIT):
    
    _, _, epochs = get_model_parameters(args.model_name, args.N, input_batch_size=args.batch_size)
    tokenizer = load_tokenizer(args.model_name)
    model_hooked = load_hooked_transformer(args.model_name, device=args.device, cache_dir=args.cache_dir)
    df_results = pd.DataFrame(columns=["task", "vanilla", "contrastive", "performance", "accuracy", "size", "TPR", "P"])


    # directory to Path Patching
    directory = f"{args.out_path}/{args.model_name}/{args.task}/Pruning" 

    eval_dataset = load_dataset(
        prepend_bos=False,
        task=args.task, 
        patching_method="path",
        tokenizer=tokenizer, 
        N=args.N, 
        device=args.device,
        model_name=args.model_name,
        seed=args.eval_seed
    )

    # ----- Average Logit Difference of the unpatched Model -----
    with torch.no_grad():
        logits_gt = model_hooked(eval_dataset.clean_tokens)
        
    ave_logit_gt = ave_logit_diff(
        logits=logits_gt, 
        correct_answers=eval_dataset.correct_answers, 
        wrong_answers=eval_dataset.wrong_answers,
        target_idx=eval_dataset.target_idx.to(args.device), 
        task=args.task,
        model_name=args.model_name
        )

    for p1 in args.cliff_point_list:
        for p2 in args.cliff_point_list:
                    
            CIRCUIT_VANILLA = circuits_dict["vanilla"][p1]
            CIRCUIT_CONTR = circuits_dict["contrastive"][p2]
            CIRCUIT = merge_circuits(CIRCUIT_VANILLA, CIRCUIT_CONTR)
            
            size = circuit_size(CIRCUIT)  
            _, performance, accuracy = batch_evaluate_circiut(
                model = model_hooked, 
                CIRCUIT=CIRCUIT,
                dataset=eval_dataset,
                ave_logit_gt=ave_logit_gt,
                task=args.task,
                model_name=args.model_name, 
                epochs = epochs, 
                batch_size = args.batch_size 
            )    
            
        
            TP_ratio = TPR(CIRCUIT, GT_CIRCUIT)
            prec = precision(CIRCUIT, GT_CIRCUIT)

            new_row = pd.DataFrame({
                "task": [args.task],
                "vanilla": [p1],
                "contrastive": [p2],
                "performance": [performance],
                "accuracy": [accuracy],
                "size": [size], 
                "TPR": [TP_ratio], 
                "P":[prec]
            }) 
            df_results = pd.concat([df_results, new_row],  ignore_index=True)

        if args.save_text:
            store_df(df_results, out_path=directory, name="results_hybridFLAP.json")           

    return df_results