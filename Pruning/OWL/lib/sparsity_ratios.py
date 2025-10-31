import os 
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import einops

from transformer_lens import utils, HookedTransformer, ActivationCache
from utils import get_hooked_llm, store_df, load_dataset

from OWL.lib.prune_all_hooked import (
    OWL_corrupted_activ_pruning,
)

from OWL.lib.parser import parser 
from OWL.lib.prune_all_hooked import sparsity_ratio
from eval_circuit import *
from PatchingMetric import ave_logit_diff
from OWL.lib.eval import IoU_nodes

IOI_CIRCUIT = {
    11: [2, 9, 10],
    10: [0, 1, 2, 6, 7, 10], 
    9:  [0, 9, 6, 7],
    8:  [6, 10],
    7:  [3, 9],
    6:  [9],
    5:  [5, 8, 9],    
    4:  [11],
    3:  [0],
    2:  [2], 
    0:  [1, 10]
}



def global_circuit(hooked_model, scaler=0.5):
        
    W_O = hooked_model.W_O
    head_means = []
    layer_means = []
    for layer in range(hooked_model.cfg.n_layers):
        for head in range(hooked_model.cfg.n_heads):
            head_sparsity_ratio = sparsity_ratio(W_O[layer][head],value=0)
            head_means.append(head_sparsity_ratio)
        layer_means.append(sparsity_ratio(W_O[layer], value=0.0))

    W_O_total_mean = np.mean(head_means)
    W_O_total_std = np.std(head_means)

    assert np.isclose(np.mean(layer_means), np.mean(head_means))
    
    CIRCUIT_PRUNING_GLOBAL = {}
    for layer in range(hooked_model.cfg.n_layers):
        CIRCUIT_PRUNING_GLOBAL[layer] = []
        for head_idx in range(hooked_model.cfg.n_heads):
            
            # if indicidual head has more weights pruned than the total mean, the difference will be negative and the head immediatelly pruned
            # a head is only in the circuit, if it prunes (significantly) less weights than average 
            # (#pruned_weights_in_head) << (average_pruned_weights_in_model)
            diff_from_mean =  W_O_total_mean  - head_means[(layer*hooked_model.cfg.n_layers)+head_idx]
            if diff_from_mean >= scaler * W_O_total_std:
                CIRCUIT_PRUNING_GLOBAL[layer].append(head_idx)
            else:
                #print("not in circuit", layer, head_idx)
                pass
    return CIRCUIT_PRUNING_GLOBAL              
  
        
def local_circuit(hooked_model, scaler):
    W_O = hooked_model.W_O
    head_sparisty = []
    layer_means = []
    for layer in range(hooked_model.cfg.n_layers):
        for head in range(hooked_model.cfg.n_heads):

            head_sparsity_ratio = sparsity_ratio(W_O[layer][head],value=0)
            head_sparisty.append(head_sparsity_ratio)

        layer_means.append(sparsity_ratio(W_O[layer], value=0.0))

    assert np.isclose(np.mean(layer_means), np.mean(head_sparisty))
    W_O_total_mean = np.mean(head_sparisty)
    W_O_total_std = np.std(head_sparisty)
    CIRCUIT_PRUNING_LOCAL = {}
    for layer in range(hooked_model.cfg.n_layers):
        CIRCUIT_PRUNING_LOCAL[layer] = []
        for head_idx in range(hooked_model.cfg.n_heads):
            index = (layer*hooked_model.cfg.n_layers)+head_idx
            diff_from_mean =  layer_means[layer]  - head_sparisty[(layer*hooked_model.cfg.n_layers)+head_idx]
            if  diff_from_mean >= scaler * W_O_total_std:
                CIRCUIT_PRUNING_LOCAL[layer].append(head_idx)
    return CIRCUIT_PRUNING_LOCAL


def main(name="OWL_sparsity"):
    result = pd.DataFrame(columns=["circuit_type", "Hyper_m", "sparsity_ratio", "Lambda", "scaler", "size", "ave logit diff", "performance", "IoU", "circuit"])
    args = parser.parse_args()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
    model_name = args.model.split("/")[-1]
    args.target_layers=["attn.hook_z"]

    device = args.device
    
    # model to ablate
    model_ablate = get_hooked_llm(args.model, device)
    hooked_tokenizer = AutoTokenizer.from_pretrained("gpt2", force_download=False, use_fast=False)
    hooked_tokenizer.pad_token = hooked_tokenizer.eos_token
    
    
    # load dataset
    dataset = load_dataset(task="ioi", patching_method="path",tokenizer=hooked_tokenizer, N=args.nsamples, device=device)

    # ground truth (unpruned) model, logits, logit difference
    model_gt = get_hooked_llm(args.model, device)
    logits_gt = model_gt(dataset.clean_tokens)
    ave_logit_gt = ave_logit_diff(logits_gt, dataset.answer_tokens.to(device), dataset.target_idx.to(device))
    
    # baseline: IOI cicruit from paper:
    model_ablate_IOI = add_mean_ablation_hook(
    model=model_ablate, 
    means_dataset=dataset.corrupted_dataset,
    circuit=IOI_CIRCUIT 
    )
    
    logits_IOI = model_ablate_IOI(dataset.clean_tokens)
    model_ablate_IOI.reset_hooks(including_permanent=True)
    ave_logits_IOI = ave_logit_diff(logits_IOI,  dataset.answer_tokens.to(device), dataset.target_idx.to(device))
    performance_IOI = performance_achieved(ave_logit_gt, ave_logits_IOI)
    size_IOI = circuit_size(IOI_CIRCUIT)

    new_row = pd.DataFrame({"circuit_type":["NONE"], "Hyper_m":[None], "sparsity_ratio":[None], "Lambda":[None], "scaler":[None], "size":[None], "ave logit diff":[ave_logit_gt], "performance":[100.0], "IoU": [None], "circuit":[None]})
    result = pd.concat([result, new_row], ignore_index=True)
    new_row = pd.DataFrame({"circuit_type":["IOI"], "Hyper_m":[None], "sparsity_ratio":[None], "Lambda":[None], "scaler":[None], "size":[size_IOI], "ave logit diff":[ave_logits_IOI], "performance":[performance_IOI], "IoU": [None], "circuit": [IOI_CIRCUIT]})
    result = pd.concat([result, new_row], ignore_index=True)
                            

    for sparsity_ratio in list(map(lambda x: x/10.0, range(6, 9))):
        for Lambda in list(map(lambda x: x/100.0, range(0, 21, 5))):
            if sparsity_ratio + Lambda == 1:
                Lambda = Lambda - 0.1
            elif sparsity_ratio + Lambda > 1:
                continue
            
            for Hyper_m in list(range(2, 5)):
                
                torch.cuda.empty_cache()
                # values to test:
                args.sparsity_ratio = sparsity_ratio
                args.Lamda = Lambda
                args.Hyper_m = [Hyper_m]

                hooked_model = get_hooked_llm(args.model, device)
                hooked_model.eval()
                hooked_model.training=False
                hooked_model.to(device)
                
                # run OWL on hooked_model             
                OWL_corrupted_activ_pruning(
                    args, 
                    None, 
                    hooked_model, 
                    hooked_tokenizer, 
                    device, 
                    prune_n=0, 
                    prune_m=0, 
                    dataset_name=args.dataset_name,
                    corrupt_knockout=False, 
                    testing=False,
                    per_sublayer=True,
                    )
                
                # faithfullness of the OWL pruned model
                hooked_model.reset_hooks(including_permanent=True)
                logits_OWL = hooked_model(dataset.clean_tokens)
                ave_logit_OWL = ave_logit_diff(logits_OWL,  dataset.answer_tokens.to(device), dataset.target_idx.to(device))
                performance_OWL = performance_achieved(ave_logit_gt, ave_logit_OWL)

                # store the result:
                new_row = pd.DataFrame({"circuit_type":["OWL"], "Hyper_m": [Hyper_m], "sparsity_ratio":[sparsity_ratio], "Lambda":[Lambda], "scaler":[None], "size":[None], "ave logit diff":[ave_logit_OWL], "performance":[performance_OWL], "IoU": [None], "circuit": [None]})
                result = pd.concat([result, new_row], ignore_index=True)
                
                for scaler in list(map(lambda x: x/10.0, range(0, 11, 5))):

                    # head pruning: globally
                    CIRCUIT_PRUNING_GLOBAL = global_circuit(hooked_model, scaler)
                    ## knockout new method, without seq_pos_to_keep dic
                    model_ablate_GLOBAL = add_mean_ablation_hook(
                        model=model_ablate, 
                        means_dataset=dataset.corrupted_dataset,
                        circuit=CIRCUIT_PRUNING_GLOBAL
                    )
                    logits_minimal_global = model_ablate_GLOBAL(dataset.clean_tokens)
                    ave_logit_global = ave_logit_diff(logits_minimal_global,  dataset.answer_tokens.to(device), dataset.target_idx.to(device))
                    performance_global = performance_achieved(ave_logit_gt, ave_logit_global)
                    size_global = circuit_size(CIRCUIT_PRUNING_GLOBAL)
                    IoU_global = IoU_nodes(IOI_CIRCUIT, CIRCUIT_PRUNING_GLOBAL)
                    
                    new_row = pd.DataFrame({"circuit_type":["GLOBAL"], "Hyper_m": [Hyper_m], "sparsity_ratio":[sparsity_ratio], "Lambda":[Lambda], "scaler":[scaler], "size":[size_global], "ave logit diff":[ave_logit_global], "performance":[performance_global], "IoU": [IoU_global], "circuit": [CIRCUIT_PRUNING_GLOBAL]})
                    result = pd.concat([result, new_row], ignore_index=True)
                                 
                    # head pruning: locally
                    CIRCUIT_PRUNING_LOCAL = local_circuit(hooked_model, scaler)
                    model_ablate_LOCAL = add_mean_ablation_hook(
                    model=model_ablate, 
                    means_dataset=dataset.corrupted_dataset,
                    circuit=CIRCUIT_PRUNING_LOCAL
                    )
                    logits_minimal_local = model_ablate_LOCAL(dataset.clean_tokens)
                    ave_logit_local = ave_logit_diff(logits_minimal_local,  dataset.answer_tokens.to(device), dataset.target_idx.to(device))
                    performance_local = performance_achieved(ave_logit_gt, ave_logit_local)
                    size_local = circuit_size(CIRCUIT_PRUNING_LOCAL)
                    IoU_local = IoU_nodes(IOI_CIRCUIT, CIRCUIT_PRUNING_LOCAL)

                    
                    new_row = pd.DataFrame({"circuit_type":["LOCAL"], "Hyper_m": [Hyper_m], "sparsity_ratio":[sparsity_ratio], "Lambda":[Lambda], "scaler":[scaler], "size":[size_local], "ave logit diff":[ave_logit_local], "performance":[performance_local], "IoU": [IoU_local], "circuit": [CIRCUIT_PRUNING_LOCAL]})
                    result = pd.concat([result, new_row], ignore_index=True)
                
    if args.out_path == "":
        out_path = os.path.join(os.getcwd() + "/res/OWL")
    else:
        out_path = args.out_path
        
    store_df(df=result, out_path=out_path, name=name + ".csv")
                    
    
if __name__ == "__main__":
    job_id = os.getenv("JOB_ID", "Unknown")
    main(name=job_id)
   