import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from Pruning.data import get_loaders 
import numpy as np
from pdb import set_trace as st 
from collections import defaultdict
import transformers
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
import math

from functools import partial
import einops

def find_layers(module, layers=[nn.Linear], name='', target_hooks=None):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
       
    return res

def check_sparsity(model):
    #use_cache = model.config.use_cache 
    #model.config.use_cache = False 
    if "GPT2"  in model.__class__.__name__:
        layers = model._modules.get("transformer")._modules.get("h")
    elif "HookedTransformer" in model.__class__.__name__:
        layers = model.blocks
    else:
        layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        if "GPT2"  in model.__class__.__name__:
            subset = find_layers(layer,  layers=[transformers.Conv1D])
        elif "HookedTransformer"  in model.__class__.__name__:
            subset = find_layers(layer, layers=[HookPoint])
        else:
            subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            sub_sub_count = 0
            sub_sub_params = 0
            if "HookedTransformer"  in model.__class__.__name__:
                if name.endswith("input"):
                    continue
                elif  "hook_q" in name:
                    W = model.W_Q[i]
                elif "hook_k" in name:            
                    W = model.W_K[i]
                elif "hook_v" in name:
                    W = model.W_V[i]
                    
                elif "attn.hook_z" in name:
                    W = model.W_O[i]
                
                elif "hook_mlp_in" in name:
                    W = model.W_in[i]
                
                elif "hook_mlp_out" in name:
                    W = model.W_out[i]
                  
                else:
                    continue
                    
            else:
                W = subset[name].weight.data
            
            sub_sub_count = (W==0).sum().item()
            sub_sub_params = W.numel()
            
            if not sub_sub_count == 0.0:
                sub_count += sub_sub_count 
                sub_params += sub_sub_params
                
                count += sub_sub_count 
                total_params += sub_sub_params

    
        if sub_params == 0:
            print(f"layer {i} total sparsity 0")
        else:
            print(f"layer {i} total sparsity {float(sub_count)/sub_params:.6f}")

    #model.config.use_cache = use_cache 
    if total_params == 0:
        return 0
    else:
        return float(count)/total_params 

def check_sparsity_mask(mask):

    W = mask
    count = 0 
    total_params = 0
    count += (W!=0).sum().item()
    total_params += W.numel()
    print(f"density {float(count)/total_params:.6f}")

def check_outlier(mask,threshold):
    W = mask
    
    max_shred=torch.max(W)*threshold
    count = (W>max_shred).sum().item()
    total_params = W.numel()
    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def check_outlier_mean(mask, threshold):
    W = mask
    max_shred=torch.mean(W)*threshold
    count = (W>max_shred).sum().item()
    total_params = W.numel()
    outlier_ratio=float(count)/total_params*100
    return outlier_ratio

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity    


################################### OWL helper functions #############################################
"""
####### CLEANED OWL #######
all subfunctions are places, that are 
    a) redundant AND/OR
    b) points that needed adjustment to add model different from LAMA 
        -> the main problem is, that LAMA weights are [C_in x C_out] shaped and e.g gpt2 weigths are [C_out x C_in] shaped
        Thus matrix transformations at the right points necessary
        -> small problem is different way of obtaining layers and gpt2 uses transformer.Conv1d layers instead of nn.Linear
        -> gpt2 does not need position_ids or attention_mask input on forward pass
"""

def preprocess_input_data(
    dataloader,
    dataset_name,
    args,
    model,
    tokenizer,
    device
    ):
    """ get the input data from the dataloader and pass it trough the embedding matrices

    Args:
        dataloader (dataloader): dataloader for dataset
        dataset_name (str): dataset_name
        args (?): arguments of the parser
        model : model
        tokenizer (tokenizer): tokenizer
        device (str): device

    Returns:
        List: list of input, output, attention_mask, Optional[position_id]
    """

    if dataloader is None:
        dataloader, _ = get_loaders(name=dataset_name,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer, device=device)
    with torch.no_grad():
        if  "GPT2"  in model.__class__.__name__:
            return prepare_calibration_input(model.to(device), args.nsamples, dataloader, device)        
        else:
            return prepare_calibration_input(model, args.nsamples, dataloader, device)
    

def get_transformer_blocks(model):
    """get all layers/blocks of a transformer model

    Args:
        model: input model

    Returns:
        TransofomerBlocks: List of all transformer blocks in the model 
    """

    if  "GPT2"  in model.__class__.__name__:
        return  model._modules.get("transformer")._modules.get("h")
    elif "HookedTransformer" in model.__class__.__name__:
        return model.blocks
    else:
        return model.model.layers


def get_sub_layers(target_layers, block, model):
    """return all layers of one transformer block of a specific type (e.g. nn.Linear, transforomer.Conv1d)
    e.g for GPT2: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
    

    Args:
        target_layers  : layers to return. If None: return all
        block : transformer block
        model : input model

    Returns:
        dict: all layers in  a single transformer block 
    """

    if  "GPT2"  in model.__class__.__name__:
        sub_layers = find_layers(block,  layers=[transformers.Conv1D])
        if target_layers:
            sub_layers_copy = sub_layers.copy()
            for name in sub_layers_copy:
                if name not in target_layers:
                    del sub_layers[name]
        return sub_layers

    elif "HookedTransformer"  in model.__class__.__name__:
        sub_layers = find_layers(block, layers=[HookPoint])
        sub_layers_copy = sub_layers.copy()
        for name in sub_layers_copy:
            if not any(target in name for target in target_layers):
                #if name not in target_layers:
                del sub_layers[name]
                
        return sub_layers

    return find_layers(block)


def wrapping_layers(sub_layers:dict, model):
    """wrapping all sublayers of a block. Allowing to add hooks during 
    forward pass to norm activations

    Args:
        sub_layers (dict): all sublayers in a transformer block
        model: input model

    Returns:
        _type_: _description_
    """

    wrapped_layers = {}
    for name in sub_layers:
        if  "GPT2"  in model.__class__.__name__:
            wrapped_layers[name] = WrappedGPT(sub_layers[name], is_gpt=True)
        else:
            wrapped_layers[name] = WrappedGPT(sub_layers[name], device=model.cfg.device)
    return wrapped_layers


def forward_pass_block(
    inps:Float[Tensor, "batch seq_n model_dim"], 
    outs:Float[Tensor, "batch seq_n model_dim"],
    attention_mask:Optional[Float[Tensor, "batch seq_n model_dim"]],
    block,
    nsamples:int, 
    device:str, 
    model
    ) -> None:

    """foward pass through the give transformer block of the input model.
    Hooks are passed during the forward pass

    Args:
        inps (Float[Tensor, &quot;batch seq_n model_dim&quot;]): input tokens
        attenion_mask (Optional[Float[Tensor, &quot;batch seq_n model_dim&quot;]]): which tokens are to be masked
        block (transformers.TransofomerBlocks): one transformer block
        nsamples (int): number of data samples
        device (str): device of model and data
        model (_type_): input model
    """

    for j in range(nsamples):
        with torch.no_grad():
            inp_gpu = inps[j].unsqueeze(0).to(device)

            if "GPT2"  in model.__class__.__name__ or "HookedTransformer" in model.__class__.__name__ :                
                # add hooks to get qkv 
                out = block(inp_gpu)[0]
            else:
                out = block(inp_gpu, attention_mask=attention_mask.to(device))[0]
    
            outs[j] = out.detach().to(device)  
            del inp_gpu, out
            torch.cuda.empty_cache()


def calculate_WANDA_metric(
    activation, 
    weights, 
    name,
    combined_matrices=False,
    n_heads=12
    )-> Tensor: 
    """WANDA metric =  magnitude of weights * accumulation of all connected input data (= activation data)        

    Args:
        name (str): current sublayer
        sub_layers (dict): all sub layers of a block
        wrapped_layers (dict): all hooked sublayers, storing the normalized activation values
        model : input model

    Returns:
        dict: the scoring metric for each sublayer
    """  
    #print("activations", activation[:, :20])
    #print("weights", weights[:, :768].shape)

    #print("weights", weights[:, :768])
    
    if activation.shape[1] != weights.shape[1]:
        repeat = int(weights.shape[1]/activation.shape[1])
        activation = activation.repeat(1,repeat)
    #print("metric", torch.abs(weights) * activation)
    return torch.abs(weights) * activation

def per_output_pruning(
    args,
    model, 
    W_metric, 
    prune_n, 
    prune_m, 
    layer_sparsity_ratio
    ):
    """ Caculate the masking matrix from the scoring matrix. 
    (W_metric.t().shape[1] * layer_sparsity_ratio) is the amount of weights to be set to 0 in 
    one pruning group depending on the layer_sparsity_ratio. 
    Pruning groups in output direction.

    Args:
        model (model): input model
        W_metric (List[C_in, C_out]): Matrix with scores over all weights
        prune_n (int): number weights to prune 
        prune_m (int): group over which prune_n is evaluated (prune n from m weights) 
        layer_sparsity_ratio (List[C_in, C_out]): _description_

    Returns:
        List: Matrix with all elements True/False (Mask)
    """

    W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

    if prune_n != 0:
    # structured n:m sparsity
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
    else:
        # unstructured pruning
        if "GPT2"  in model.__class__.__name__: 
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1] * layer_sparsity_ratio)]
        
        
        elif "HookedTransformer" in model.__class__.__name__:
            if True:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:,:int(W_metric.shape[1] * layer_sparsity_ratio)]
                
            elif False:
                # for hook z
                W_metric_reshaped = einops.rearrange(W_metric, "n_head head_dim model_dim ->model_dim (n_head head_dim) ")
                W_mask_reshaped = (torch.zeros_like(W_metric_reshaped) == 1)  ## initialize a mask to be all False
                sort_res_reshaped = torch.sort(W_metric_reshaped, dim=-1, stable=True)
                indices_reshaped = sort_res_reshaped[1][:,:int(W_metric_reshaped.shape[1] * layer_sparsity_ratio)]
                W_mask_reshaped.scatter_(1, indices_reshaped, True)
                return einops.rearrange(W_mask_reshaped, "model_dim (n_head head_dim) -> n_head head_dim model_dim", n_head=model.cfg.n_heads)

            else:
                W_metric_reshaped = einops.rearrange(W_metric, "n_head model_dim head_dim ->  model_dim (n_head head_dim)")
                W_mask_reshaped = (torch.zeros_like(W_metric_reshaped) == 1)  ## initialize a mask to be all False
            
                sort_res_reshaped = torch.sort(W_metric_reshaped, dim=-1, stable=True)
                indices_reshaped = sort_res_reshaped[1][:,:int(W_metric_reshaped.shape[1] * layer_sparsity_ratio)]
                W_mask_reshaped.scatter_(1, indices_reshaped, True)
                return einops.rearrange(W_mask_reshaped, "model_dim (n_head head_dim)-> n_head model_dim head_dim", n_head=model.cfg.n_heads)

        W_mask.scatter_(1, indices, True)
    return W_mask


def per_layer_pruning(
    args,
    model, 
    W_metric, 
    prune_n, 
    prune_m, 
    layer_sparsity_ratio
    ):
    """ Caculate the masking matrix from the scoring matrix. 
    (W_metric.t().shape[1] * layer_sparsity_ratio) is the amount of weights to be set to 0 in 
    one pruning group depending on the layer_sparsity_ratio. 
    Pruning groups in output direction.

    Args:
        model (model): input model
        W_metric (List[C_in, C_out]): Matrix with scores over all weights
        prune_n (int): number weights to prune 
        prune_m (int): group over which prune_n is evaluated (prune n from m weights) 
        layer_sparsity_ratio (List[C_in, C_out]): _description_

    Returns:
        List: Matrix with all elements True/False (Mask)
    """
    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
    
    if "GPT2"  in model.__class__.__name__: 
        if prune_n != 0:
        # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            # unstructured pruning
                flat_W = torch.abs(W_metric.flatten())
                k = int(W_metric.numel() * layer_sparsity_ratio)
                _, ind_1D = flat_W.topk(k=k, largest=False)
                rows = torch.div(ind_1D, W_metric.shape[0], rounding_mode='floor')
                cols = ind_1D % W_metric.shape[1]  
          
                W_mask[rows, cols] = True
        
    elif "HookedTransformer" in model.__class__.__name__:
        if len(W_metric.shape) == 3:
            if prune_n != 0:
                #TODO: fill
                pass
            else:
                flat_W = torch.abs(W_metric.flatten())
                k = int(W_metric.numel() * layer_sparsity_ratio)
                _, ind_1D = flat_W.topk(k=k, largest=False)
                dim1 = W_metric.shape[1] * W_metric.shape[2]
                dim2 = W_metric.shape[2]

                batch = torch.div(ind_1D, dim1, rounding_mode='floor')                
                rows = torch.div(ind_1D % dim1, dim2, rounding_mode='floor')
                cols = ind_1D % dim2               
                if args.verbose:
                    unique_elements, counts = torch.unique(batch, return_counts=True)
                    max_index=torch.argmin(counts)
                    
                W_mask[batch, rows, cols] = True

                                
        else:
            if prune_n != 0:
                #TODO: fill
                pass
            else:   
                flat_W = torch.abs(W_metric.flatten())
                k = int(W_metric.numel() * layer_sparsity_ratio)
                _, ind_1D = flat_W.topk(k=k, largest=False)
                
                
                rows = torch.div(ind_1D, W_metric.shape[1], rounding_mode='floor')
                cols = ind_1D % W_metric.shape[1]  
                """ else:
                    rows = torch.div(ind_1D, W_metric.shape[0], rounding_mode='floor')
                    cols = ind_1D % W_metric.shape[0]  """
                W_mask[rows, cols] = True
    return W_mask


def sparsity_ratio_per_block(args, all_block_ratio):
    """scale and normalize the target sparsity ratio for all blocks

    Args:
        args (_type_): parser_arguments
        all_block_ratio (List[float]): target sparsity per block

    Returns:
        List[float]: target sparsity normalized and scaled
    """
    if args.verbose:
        print("length of outlier ratio over all layers, ", len(all_block_ratio))
        print("*"*50, "\noutlier ratio over all layers:")
        print ("before adjustment",all_block_ratio)
        print("mean: ", np.mean(all_block_ratio), "\n max: ", np.max(all_block_ratio), "\n min: ", np.min(all_block_ratio)) 
    all_block_ratio=np.array(all_block_ratio)
    all_block_ratio = ((all_block_ratio - all_block_ratio.min()) * (1/(all_block_ratio.max() - all_block_ratio.min()) * args.Lamda * 2))
    all_block_ratio=all_block_ratio-np.mean(all_block_ratio)+(1-args.sparsity_ratio)
    
    if args.verbose:
        print ("after adjustment: ", all_block_ratio)
        print("mean: ", np.mean(all_block_ratio), "\n max: ", np.max(all_block_ratio), "\n min: ", np.min(all_block_ratio)) 
        print("*"*50)

    return all_block_ratio

def split_whole_weight_matrix_to_components(w_matrix, model_dim, dim=1):
    """ gpt2 store qkv matrix not seperatily, but concatenated.
        Split c_attn (model_dim x [3 x model_dim]) in three seperate matrices q,k,v 
        with each (model_dim x model_dim)    

    Args:
        w_matrix (List[model_dim x (3 x model_dim)]): weight matrix: c_attn
        model_dim (int): model dimension
        dim (int, optional): Direction in which c_attn is to be split. Defaults to 1.

    Returns:
        Tuple(q, k, v): Tuple[List[model_dim x model_dim]])
    """
    return w_matrix.split(model_dim, dim=dim)

def split_component_weight_matrix_to_heads(W, model_dim, n_head, dim=1):
    """ Split a weight matrix in its seperate attention heads. Each attention head has a 
    dimension of model_dim / n_heads. 

    Args:
        comp_weight (List[model_dim x model_dim]): Weight matrix
        model_dim (int): model dimension
        n_head (int): number of heads per attention block
        dim (int, optional):  Direction in which weight matrix is to be split. Defaults to 1.

    Returns:
        Union[List[n_head x model_dim x head_dim], List[n_head x head_dim x model_dim]]: weight matrix split in n_heads seperate attention heads
    """
    head_dim = int(model_dim/n_head)
    W = W.contiguous() 
    if dim == 1:
        return einops.rearrange(W, "m (i h) -> i m h", i=n_head) #[n_head, model_dim, head_dim]
    else:
        return einops.rearrange(W, "m (i h) -> i h m", i=n_head) # [n_head, head_dim, model_dim]

def split_whole_weight_matrix_to_heads(w_matrix, model_dim, n_head, dim=1):   
    """Split a weight matrix in its seperate attention heads.

    Args:
        w_matrix (List): weight matrix
        model_dim (int): model dimension
        n_head (int): number of attention heads in blockkz
        dim (int, optional): Direction in which weight matrix is to be split. Defaults to 1.

    Returns:
        Union[List[n_head x model_dim x head_dim], List[n_head x head_dim x model_dim]]: weight matrix split in n_heads seperate attention heads
    """
    if not w_matrix.shape[0] == w_matrix.shape[1]:
        if dim==1:
            # splitting along the second dimension
            assert(w_matrix.shape[1] == model_dim*3)
        elif dim==0:
            # splitting along the first dimension
            assert(w_matrix.shape[0] == model_dim*3)
        
        qkv_split = split_whole_weight_matrix_to_components(w_matrix, model_dim, dim=dim)

        comp_idx = 0
        res = {}
        for letter in "qkv":
            qkv = qkv_split[comp_idx]
            res[letter] = split_component_weight_matrix_to_heads(qkv, model_dim, n_head, dim=1)
            comp_idx += 1
        return res

    else:
        return split_component_weight_matrix_to_heads(w_matrix, model_dim, n_head, dim=0)

#################### Threshold functions for attention head pruning:

def sparsity_ratio(W_matrix, value=True):
    # sparsity_count -> number of weights that are 0/False/True/value
    sparsity_count = (W_matrix==value).sum().item()
    total_params = W_matrix.numel()
    return sparsity_count/total_params


def sublayer_statistics(heads, value=True):
    ## How many weights are on average zeroed out?
    
    head_sparsity_ratio = []

    for head_idx in range(heads.shape[0]):
        head = heads[head_idx]
        head_sparsity_ratio.append(sparsity_ratio(head, value=value))
    head_sparsity_ratio = np.array(head_sparsity_ratio)
    return np.mean(head_sparsity_ratio), np.std(head_sparsity_ratio)


def over_SD_threshold(head, block_mean, block_sd, SD_threshold=2, value=True):
    head_sparsity = sparsity_ratio(head, value=value)
    return abs(head_sparsity - block_mean) > SD_threshold * block_sd
    
#######################################################################
# Hook functions

def OWL_fn(inp, hook, cache, n_samples, weights=None, test_inps=None, name=None):
    if name != None:
        nsamples = n_samples[name]
        scaler_row = cache[name]
    else:
        nsamples = n_samples[hook.name]
        scaler_row = cache[hook.name]
    
    
    if not test_inps == None:
        if "hook_normalized" in name:
            inp = test_inps[hook.layer()][math.floor(nsamples / 3)]
        else:
            inp = test_inps[hook.layer()][nsamples]

    if len(inp.shape) == 2:
        inp = inp.detach().clone().unsqueeze(0)  
    tmp = inp.shape[0]
    if len(inp.shape) == 4:
        if torch.all(inp[:,:,0] == inp[:,:,1]).item():
            inp = inp[:, :, 0]
            
        elif inp.shape[2] == inp.shape[3]:
            n_samples[hook.name]  += tmp
            cache[hook.name] = scaler_row      
            return  
        else:
            inp = einops.rearrange(inp, "batch seq head_n head_dim -> batch seq (head_n head_dim)")

    if len(inp.shape) == 3:
        inp = inp.reshape((-1, inp.shape[-1]))

    inp = inp.t()
    
    scaler_row *= nsamples / (nsamples+tmp)
    nsamples += tmp
    
    inp = inp.type(torch.float32)
    scaler_row += torch.norm(inp, p=2, dim=1)**2 / nsamples
    if name != None:
        n_samples[name]  += tmp
        cache[name] = scaler_row
    else:
        n_samples[hook.name]  += tmp
        cache[hook.name] = scaler_row
    
def store_activations_fn(activation:Tensor, hook:HookPoint, dic:dict):    
    try:
        dic[hook.name] = torch.cat((dic[hook.name], activation), dim=0)
    except:
        dic[hook.name] = activation

def patch_activation_fn(activation, hook, patch_dic):
    return patch_dic[hook.layer()]

def get_mixed_input_to_matrix_fn(clean_activation, hook, corrupted_activation, W_pruned, W_pruned_inv, patch_dic):
    y_clean = einops.einsum(clean_activation, W_pruned,
                            "batch seq n_head head_dim, n_head head_dim model_dim -> batch seq n_head model_dim")
    
    y_corrupt = einops.einsum(corrupted_activation, W_pruned_inv,
                            "batch seq n_head head_dim, n_head head_dim model_dim -> batch seq n_head model_dim")
    
    y_mixed = y_clean + y_corrupt    
    patch_dic[hook.layer()] = y_mixed

      
def wrapped_layers_hooked_model(
    args,
    dataloader, 
    model:HookedTransformer, 
    device:str=torch.device("cuda:0"), 
    target_hooks:List[str] = ["hook_z", "ln1.hook_normalized"],
    test_inps = None
    ):

    scaler_rows_cache = {}
    n_samples = {}
    collected_inps = {}
    
    # Register the hook to get the inputs
    hook_fn = partial(store_activations_fn, dic=collected_inps)
    model.add_hook('blocks.0.hook_resid_pre', hook_fn, level=1)

    # Register hooks for all layers to collect activations
    for name in model.hook_dict:
        for target in args.target_layers:  
            #if any(substring in name for substring in target_hooks):  
            if target in name:
                
                if "hook_z" in target:
                    scaler_rows_cache[name] = torch.zeros((model.cfg.d_model), device=device)
                    n_samples[name] = 0                
                    hook = partial(OWL_fn, cache=scaler_rows_cache, n_samples=n_samples, weights=None, test_inps=test_inps)
                    model.add_hook(name, hook, level=1)
                else:
                    layer = name.split(".")[1]
                    hook_name = f"blocks.{layer}.ln1.hook_normalized"
                    scaler_rows_cache[name] = torch.zeros((model.cfg.d_model), device=device)
                    n_samples[name] = 0                
                    hook = partial(OWL_fn, cache=scaler_rows_cache, n_samples=n_samples, weights=None, test_inps=test_inps, name=name)
                    model.add_hook(hook_name, hook, level=1)
    
    i=0     
    for batch in dataloader:
        i += 1
        with torch.no_grad():
            _ =  model(batch[0])
    return scaler_rows_cache, collected_inps['blocks.0.hook_resid_pre']

#######################################################################
def to_long_matrix(name:str, weights:Union[Float[Tensor, "n_head model_dim head_dim"], Float[Tensor, "n_head head_dim model_dim"]]):
    if any(substring in name for substring in ["hook_q", "hook_k", "hook_v", "ln1.hook_normalized"]): 
        return einops.rearrange(weights, "n m h -> (n h) m")
        
    elif  "attn.hook_z" in name:
        return einops.rearrange(weights, "n h m -> m (n h)")


def to_head_matrix(name:str, weights:Union[Float[Tensor, "n_head model_dim head_dim"], Float[Tensor, "n_head head_dim model_dim"]], n_heads=12, d_model=768):
    if any(substring in name for substring in ["hook_q", "hook_k", "hook_v"]): 
        return einops.rearrange(weights, "(n h) m -> n m h", n=n_heads)
    elif "hook_z" in name: 
        return einops.rearrange(weights, "m (n h) -> n h m", n=n_heads)
    elif "hook_normalized" in name:
            WQ, WK, WV = split_whole_weight_matrix_to_components(weights, d_model, dim=1)
            WQ =  einops.rearrange(WQ, "(n h) m -> n m h", n=n_heads)
            WK =  einops.rearrange(WK, "(n h) m -> n m h", n=n_heads)
            WV =  einops.rearrange(WV, "(n h) m -> n m h", n=n_heads)
            
            return WQ, WK, WV
        
# reshape hooked transformer weights
def get_weight_matrix(layer:int, name:str, model:HookedTransformer, combined_matrices=False) -> Tensor:
    """weight conversion like at https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/pretrained/weight_conversions/gpt2.py

    Args:
        layer (int): _description_
        name (str): _description_
        letter (Optional[str]): _description_
        model (HookedTransformer): _description_

    Returns:
        Tensor: _description_
    """
   
    if "hook_q" in name: 
        WQ = model.blocks[layer].attn.W_Q
        return to_long_matrix(name=name, weights=WQ)
    
    elif "hook_k" in name:
        WK = model.blocks[layer].attn.W_K
        return to_long_matrix(name=name, weights=WK)
    
    elif "hook_v" in name:
        WV = model.blocks[layer].attn.W_V
        return  to_long_matrix(name=name, weights=WV)
        
    elif  "attn.hook_z" in name:
        WO =  model.blocks[layer].attn.W_O
        return to_long_matrix(name=name, weights=WO)
    
    elif "mlp_in" in name:
        return model.blocks[layer].mlp.W_in
    
    elif "mlp_out" in name:
        return model.blocks[layer].mlp.W_out
    
    elif "hook_normalized" in name:
        WQ = model.blocks[layer].attn.W_Q
        WQ = to_long_matrix(name=name, weights=WQ)
        WK = model.blocks[layer].attn.W_K
        WK = to_long_matrix(name=name, weights=WK)
        WV = model.blocks[layer].attn.W_V
        WV = to_long_matrix(name=name, weights=WV)
        return torch.cat([WQ, WK, WV], dim=1)
    else:
        raise Exception("Defined name of activation is not associated with a transformer weight")

def get_component_after_pruned_matrix(name):
    
    if "hook_q" in name: 
        return "attn.hook_q"
    
    elif "hook_k" in name:

        return "attn.hook_k"
    
    elif "hook_v" in name:
        return  "attn.hook_v"
        
    elif  "attn.hook_z" in name:
        return "attn.hook_result"
    
    else:
        raise Exception("Defined name of activation is not associated with a transformer weight")

    
def set_weight_matrix(W, layer, name, model, combined_matrices=False):
    if not combined_matrices:
    
        if "hook_q" in name: 
            WQ = to_head_matrix(name, W, n_heads=model.cfg.n_heads)
            #einops.rearrange(W, "(n h) m -> n m h", n=model.cfg.n_heads)
            model.blocks[layer].attn.W_Q = nn.Parameter(WQ.to(model.cfg.device))
        elif "hook_k" in name: 
            WK = to_head_matrix(name, W, n_heads=model.cfg.n_heads)
            model.blocks[layer].attn.W_K = nn.Parameter(WK.to(model.cfg.device))
        elif "hook_v" in name: 
            WV = to_head_matrix(name, W, n_heads=model.cfg.n_heads)
            model.blocks[layer].attn.W_V = nn.Parameter(WV.to(model.cfg.device))
        elif "hook_z" in name: 
            WO = to_head_matrix(name, W, n_heads=model.cfg.n_heads)
            model.blocks[layer].attn.W_O = nn.Parameter(WO.to(model.cfg.device))
        elif "mlp_in" in name:
            model.blocks[layer].attn.W_in = nn.Parameter(W.to(model.cfg.device))
        elif "mlp_out" in name:
            model.blocks[layer].attn.W_out = nn.Parameter(W.to(model.cfg.device))  
        elif "hook_normalized" in name:
            WQ, WK, WV = to_head_matrix(name, W, d_model=model.cfg.d_model)
            
            model.blocks[layer].attn.W_Q = nn.Parameter(WQ.to(model.cfg.device))
            model.blocks[layer].attn.W_K = nn.Parameter(WK.to(model.cfg.device))
            model.blocks[layer].attn.W_V = nn.Parameter(WV.to(model.cfg.device))        
        
        else:
            raise Exception("Defined name of activation is not associated with a transformer weight")

    else:
        if W.shape[1] == 3 * model.cfg.d_model:
            WQ, WK, WV = W.split(model.cfg.d_model, dim=1)
            WQ = split_component_weight_matrix_to_heads(WQ, model.cfg.d_model, model.cfg.n_heads, dim=1)
            WK = split_component_weight_matrix_to_heads(WK, model.cfg.d_model, model.cfg.n_heads, dim=1)
            WV = split_component_weight_matrix_to_heads(WV, model.cfg.d_model, model.cfg.n_heads, dim=1)
            assert WQ.shape == model.blocks[layer].attn.W_Q.shape
            assert WK.shape == model.blocks[layer].attn.W_K.shape
            assert WV.shape == model.blocks[layer].attn.W_V.shape
            
            model.blocks[layer].attn.W_Q = nn.Parameter(WQ.to(model.cfg.device))
            model.blocks[layer].attn.W_K = nn.Parameter(WK.to(model.cfg.device))
            model.blocks[layer].attn.W_V = nn.Parameter(WV.to(model.cfg.device))
        else: 
            W = split_component_weight_matrix_to_heads(W, model.cfg.d_model, model.cfg.n_heads, dim=0)
            assert W.shape ==  model.blocks[layer].attn.W_O.shape  
            model.blocks[layer].attn.W_O = nn.Parameter(W.to(model.cfg.device))
        
def per_head_pruning(name, W_metric, scaler, CIRCUIT, layer, d_model=768, n_heads=12):
    W_metric = to_head_matrix(name=name, weights=W_metric, d_model=d_model, n_heads=n_heads)
    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

    total_mean = W_metric.mean()
    total_std = W_metric.std()
    means = torch.zeros(n_heads)
    CIRCUIT[layer] = []
    
    for head in range(n_heads):
        head_mean = W_metric[head].mean()
        means[head]  = W_metric[head].mean()
        if head_mean - total_mean > scaler * total_std: 
            CIRCUIT[layer].append(head)
        else:
            W_mask[head] = torch.ones_like(W_metric[head], dtype=torch.bool)

    W_mask = to_long_matrix(name, W_mask)
    return W_mask, CIRCUIT
    
def prune_head_by_score(W_metric, W_mask, scaler, CIRCUIT, layer):
    W_metric[W_mask] = 0
    W_mask_new = torch.full_like(W_mask, fill_value = True)

    heads, head_dim, model_dim = W_metric.shape
    CIRCUIT[layer] = []
    
    head_scores = []
    for head_idx in range(heads):
        s = W_metric[head_idx].sum()
        head_scores.append(s)

    mean = np.mean(head_scores)
    std =  np.std(head_scores)
    
    for head_idx in range(heads):
        if abs(head_scores[head_idx] - mean) > scaler * std:
            CIRCUIT[layer].append(head_idx)
            W_mask_new[head_idx] =  torch.full_like(W_mask[head_idx], fill_value = False)
            
    return W_mask_new

#######################################################################
# OWL Pruning

def OWL_corrupted_activ_pruning(
    args, 
    dataloader, 
    model, 
    tokenizer, 
    device=torch.device("cuda:0"), 
    prune_n=0, 
    prune_m=0, 
    dataset_name="c4",
    corrupt_knockout=False,
    per_sublayer=False,
    testing=False,
    test_inps=None,
    head_pruning=False,
    head_pruning_by_score=False,
    scaler=1, 
    ):

    if corrupt_knockout:
        if args.verbose:
            print("Getting corrupted activations")
        if testing:      
            # only for testing. If also corrupted input is replaced with clean, we expect the same model behaviour as
            # a completly unpruned model:
            # Y_unpruned = X_clean * W_unpruned
            # W_unpruned = W_pruned + W_invers_pruned
            # if we plug in corrupted activations we run Y_mixed = X_clean * W_pruned + X_corrupt * W_invers_pruned
            # Thus if plugging in X_clean for X_corrupt, original model behaviour has to be restored!
            inps_corrupt,_ = get_loaders(args.dataset_name, args.nsamples, args.seed, model.seqlen, tokenizer, device=device, corrupt=False)
        else:
            inps_corrupt,_ = get_loaders(args.dataset_name, args.nsamples, args.seed, model.seqlen, tokenizer, device=device, corrupt=True)

        corrupted_activations = {}
        for layer in range(model.cfg.n_layers):
            hook_name = f"blocks.{layer}.{args.target_layers[0]}"
            store_corrupted_activations = partial(store_activations_fn, dic=corrupted_activations)
            model.add_hook(hook_name, store_corrupted_activations, level=1)

        for batch in inps_corrupt:
            with torch.no_grad():
                model(batch[0].to(device))
        model.reset_hooks()    
        
    
    ##### calucalte outlier ratio per attention head and mlp
    all_outlier_ratio=np.array([])

    model.reset_hooks(including_permanent=True)
    torch.set_grad_enabled(False)

    model_dim=model.cfg.d_model
    n_head=model.cfg.n_heads
    head_dim = model_dim / n_head
    n_layers = model.cfg.n_layers

    
    dataloader, _ = get_loaders(args.dataset_name, args.nsamples, args.seed, model.seqlen, tokenizer)
    scalar_rows_cache, inps = wrapped_layers_hooked_model(args, dataloader, model, device=device, target_hooks=args.target_layers, test_inps=test_inps )
    outs = torch.zeros_like(inps)
    #attention_mask = attention_mask = [l[1] for l in dataloader]
    attention_mask = None
    blocks = get_transformer_blocks(model)     
    model.reset_hooks()
        
    for i in range(len(blocks)):
        block = blocks[i]
        
        # get all sublayers of block
        sub_layers = get_sub_layers(target_layers=args.target_layers, block=block, model=model)
        # calculate the score function
        layer_wmetric=[]
        
        for name in sub_layers:
            hook_name = f"blocks.{i}.{name}"
            if args.verbose:
                print(f"pruning layer {i} name {hook_name}")
                
            weights = get_weight_matrix(layer=i, name=name, model=model)    
            
            activations = torch.sqrt(scalar_rows_cache[hook_name].reshape((1, -1)))                         
            
            W_metric = calculate_WANDA_metric(activations, weights, name)
            if testing:
                W_metric_test =  einops.rearrange(weights.clone().detach(), "i h m -> (i h) m", i=model.cfg.n_heads)
                W_metric_test = calculate_WANDA_metric(activations, W_metric_test, name, combined_matrices=True)
                
                W_metric_test = einops.rearrange(W_metric_test,"(i h) m -> i h m", i = model.cfg.n_heads)
                assert torch.all(W_metric_test == W_metric)
                
            
            layer_wmetric.append(W_metric)

        ################################## outlier ratio ########################################    
        if per_sublayer:
            #  outlier ratio per sublayer 
            for layer_idx in range(len(layer_wmetric)):
                for out_ratio in args.Hyper_m:
                    outlier_ratio = check_outlier_mean(layer_wmetric[layer_idx], out_ratio)
                    all_outlier_ratio = np.append(all_outlier_ratio, outlier_ratio)

        else:   
            # outlier ratio per block 
            block_wmetric = torch.cat([torch.flatten(x).cpu() for x in layer_wmetric])
            for out_ratio in args.Hyper_m:
                outlier_ratio=check_outlier_mean(block_wmetric, out_ratio)
                all_outlier_ratio = np.append(all_outlier_ratio, outlier_ratio)
    all_outlier_ratio = sparsity_ratio_per_block(args, all_outlier_ratio)
    ###################################### pruning  ############################################
    torch.cuda.empty_cache()
    
    scaler_rows_cache_clean = {}
    n_samples2 = {}
    patch_dic={}
    CIRCUIT={}
    scores_before_pruning = torch.zeros((len(args.target_layers), n_layers, n_head))
    scores_after_pruning = torch.zeros((len(args.target_layers), n_layers, n_head))
    
    blocks = get_transformer_blocks(model)
    for i in range(len(blocks)):
        if args.verbose:
            print("############ LAYER", i ,"#################")
        block = blocks[i]
        # get all sublayers of block
        sub_layers = get_sub_layers(target_layers=args.target_layers, block=block, model=model)

        ####### ADDING HOOKS TO GET CLEAN ACTIVATIONS
        # get the scaler activations by inserting hook function and running model on clean input tokens
        for name in sub_layers:
            for target in args.target_layers:  
                hook_name = f"blocks.{i}.{name}"
                if target in name:
                    
                    if "hook_z" in target:
                        scaler_rows_cache_clean[hook_name] = torch.zeros((model.cfg.d_model), device=device)
                        n_samples2[hook_name] = 0                
                        hook = partial(OWL_fn, cache=scaler_rows_cache_clean, n_samples=n_samples2, weights=None, test_inps=test_inps, name=hook_name)
                        model.add_hook(hook_name, hook, level=1)
                    else:
                        scaler_rows_cache_clean[hook_name] = torch.zeros((model.cfg.d_model), device=device)
                        n_samples2[hook_name] = 0                
                        hook = partial(OWL_fn, cache=scaler_rows_cache_clean, n_samples=n_samples2, weights=None, test_inps=test_inps, name=hook_name)
                        model.add_hook(f"blocks.{i}.ln1.hook_normalized", hook, level=1)
                
                
        forward_pass_block(
            inps = inps, 
            outs = outs,
            attention_mask = attention_mask, 
            block = block,
            nsamples = args.nsamples,
            device=device,
            model = model
            )
        model.reset_hooks()
 
        layer_sparsity_ratio= 1-all_outlier_ratio[i]
        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01
        if layer_sparsity_ratio > 1:
            layers_sparsity_ratio=1
        
            
        if args.verbose:
            print("layer sparsity ratio", layer_sparsity_ratio)
        
        for sub_layer_idx, name in enumerate(sub_layers):
            hook_name = f"blocks.{i}.{name}"
            if args.verbose:
                print(f"pruning layer {i} name {hook_name}")
            
            weights = get_weight_matrix(layer=i, name=name, model=model)        
            weight_matrix = weights.clone().detach()
            weight_matrix_inv = weights.clone().detach()
            activations = torch.sqrt(scaler_rows_cache_clean[hook_name].reshape((1, -1))) 
            #print("activations", activations[:, :20])
            W_metric = calculate_WANDA_metric(activations, weight_matrix, name)
            W_metric_head = to_head_matrix(name=name, weights=W_metric, n_heads=n_head, d_model=model_dim)
            #print("W_metric", W_metric)

            scores_before_pruning[0, i] = torch.mean(W_metric_head, dim=(1, 2))
            
            # OWL metric            
            W_mask = per_layer_pruning(
                args=args,
                model=model,
                W_metric=W_metric,
                prune_n=prune_n,
                prune_m=prune_m,
                layer_sparsity_ratio=layer_sparsity_ratio
                )
            W_total = W_mask.numel()
            
            if head_pruning:
                #W_mask = prune_heads(W_mask, scaler=scaler, CIRCUIT=CIRCUIT, layer=i)
                W_mask, CIRCUIT = per_head_pruning(name=name, W_metric=W_metric, scaler=scaler, CIRCUIT=CIRCUIT, layer=i, d_model=model_dim, n_heads=n_head)

            if head_pruning_by_score:
                W_mask = prune_head_by_score( W_metric.clone(), W_mask.clone(), scaler=scaler, CIRCUIT=CIRCUIT, layer=i)
               
            # actual weight pruning happens here
            weight_matrix[W_mask] = 0
            #print("pruned", weight_matrix[:, :768])
            #print("pruned shape", weight_matrix.shape)
            weight_matrix_inv[~W_mask] = 0
            W_metric_pruned_head = to_head_matrix(name=name, weights=weight_matrix, n_heads=n_head, d_model=model_dim)        

            scores_after_pruning[0, i] = torch.mean(W_metric_pruned_head, dim=(1, 2))
                         
            assert torch.all(weight_matrix + weight_matrix_inv == weights)
            set_weight_matrix(weight_matrix, i, name, model)
         
            

        # add permanent hooks to plug in  corrupted activations where weights are zero
        # pruning via knockout: y = x_clean * weight + x_corrupted * weight_inverted 
        matrix_inv = to_head_matrix(name=name, weights=weight_matrix_inv, n_heads=n_head, d_model=model_dim)
        if corrupt_knockout:
            for name in sub_layers:
                hook_name = f"blocks.{i}.{name}"
                hook_modified = partial(
                    get_mixed_input_to_matrix_fn,
                    corrupted_activation=corrupted_activations[hook_name],
                    W_pruned=to_head_matrix(name=name, weights=weight_matrix, n_heads=n_head, d_model=model_dim),
                    W_pruned_inv= to_head_matrix(name=name, weights=weight_matrix_inv, n_heads=n_head, d_model=model_dim),
                    patch_dic=patch_dic
                )
                            
                model.add_hook(hook_name, hook_modified, level=1)
            
            hook_replace_after_matrix = partial(
                patch_activation_fn,
                patch_dic=patch_dic
            )
        
            next_comp = get_component_after_pruned_matrix(name)
            model.add_perma_hook(f"blocks.{i}.{next_comp}", hook_replace_after_matrix)    
        
       

        forward_pass_block(
            inps = inps, 
            outs = outs,
            attention_mask = attention_mask, 
            block = block,
            nsamples = args.nsamples,
            device=device,
            model = model
            )
            
        inps, outs = outs, inps   

        
    if "GPT2"  in model.__class__.__name__: 
        model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    return  CIRCUIT, scores_before_pruning, scores_after_pruning





def OWL_zero_pruning(
    args, 
    dataloader, 
    model, 
    tokenizer, 
    device=torch.device("cuda:0"), 
    prune_n=0, 
    prune_m=0, 
    dataset_name="c4",
    corrupt_knockout=True,
    per_sublayer=True
    ):
                                                
    unpruned_heads = {}
    all_w_metrics = {}
    
    ##### calucalte outlier ratio per attention head and mlp
    all_outlier_ratio=np.array([])

    model.reset_hooks(including_permanent=True)
    torch.set_grad_enabled(False)

    model_dim=model.cfg.d_model
    n_head=model.cfg.n_heads
    head_dim = model_dim / n_head

    
    dataloader,_ = get_loaders("ioi", args.nsamples, args.seed, model.seqlen, tokenizer)
    scalar_rows_cache, inps = wrapped_layers_hooked_model(dataloader, model, device=device, target_hooks=args.target_layers)
    blocks = get_transformer_blocks(model)       

    for i in range(len(blocks)):
        block = blocks[i]
        
        # get all sublayers of block
        sub_layers = get_sub_layers(target_layers=args.target_layers, block=block, model=model)
        
        # calculate the score function
        layer_wmetric=[]
        
        for name in sub_layers:
            hook_name = f"blocks.{i}.{name}"
            if args.verbose:
                print(f"pruning layer {i} name {hook_name}")
                
            weights = get_weight_matrix(layer=i, name=name, model=model)
            activations = torch.sqrt(scalar_rows_cache[hook_name].reshape((1, -1)))                         
            W_metric = calculate_WANDA_metric(activations, weights, name)
            layer_wmetric.append(W_metric)

        ################################## outlier ratio ########################################    
        if per_sublayer:
            #  outlier ratio per sublayer 
            for layer_idx in range(len(layer_wmetric)):
                for out_ratio in args.Hyper_m:
                    out_ratio = check_outlier_mean(layer_wmetric[layer_idx], out_ratio)
                    all_outlier_ratio = np.append(all_outlier_ratio, out_ratio)

        else:   
            # outlier ratio per block 
            block_wmetric = torch.cat([torch.flatten(x).cpu() for x in layer_wmetric])
            for out_ratio in args.Hyper_m:
                out_ratio=check_outlier_mean(block_wmetric, out_ratio)
                all_outlier_ratio = np.append(all_outlier_ratio, out_ratio)
    all_outlier_ratio = sparsity_ratio_per_block(args, all_outlier_ratio)
    ###################################### pruning  ############################################
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)

    scaler_rows_cache2 = {}
    n_samples2 = {}
    
    blocks = get_transformer_blocks(model)

    for i in range(len(blocks)):
        if args.verbose:
            print("############ LAYER", i ,"#################")
        unpruned_heads[i] = []
        block = blocks[i]
        model.reset_hooks()
        # get all sublayers of block
        sub_layers = get_sub_layers(target_layers=args.target_layers, block=block, model=model)
        
        # implement the hooks
        for name in sub_layers:
            hook_name = f"blocks.{i}.{name}"
            scaler_rows_cache2[hook_name] = torch.zeros((model.cfg.d_model), device=device).to(device)
            n_samples2[hook_name] = 0
            hook = partial(OWL_fn, cache=scaler_rows_cache2, n_samples=n_samples2)
            model.add_hook(hook_name, hook)
            
        outs = model.blocks[i](inps)
        layer_sparsity_ratio= 1-all_outlier_ratio[i]
        
        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01
            
        if args.verbose:
            print("layer sparsity ratio", layer_sparsity_ratio)
            

        for name in sub_layers:
            hook_name = f"blocks.{i}.{name}"

            if args.verbose:
                print(f"pruning layer {i} name {hook_name}")
            
            weight_matrix = get_weight_matrix(layer=i, name=name, letter=None, model=model)            
            activations = torch.sqrt(scaler_rows_cache2[hook_name].reshape((1, -1))) 
            weights = get_weight_matrix(layer=i, name=name, letter=None, model=model)        
            W_metric = calculate_WANDA_metric(activations, weights, name)
            
            # OWL metric            
            W_mask = per_output_pruning(
                args=args,
                model=model,
                W_metric=W_metric,
                prune_n=prune_n,
                prune_m=prune_m,
                layer_sparsity_ratio=layer_sparsity_ratio
                )
            W_total = W_mask.numel()

            all_w_metrics[i] = W_mask.clone().detach()
            
            # actual weight pruning happens here
            weight_matrix[W_mask] = 0
            set_weight_matrix(weight_matrix, i, model)
        
        # forward pass thorugh block without hooks
        outs = model.blocks[i](inps)
        inps, outs = outs, inps   
    if "GPT2"  in model.__class__.__name__: 
        model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    return unpruned_heads, all_w_metrics
