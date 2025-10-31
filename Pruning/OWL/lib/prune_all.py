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

import einops


def find_layers(module, layers=[nn.Linear], name=''):
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
    else:
        layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        if "GPT2"  in model.__class__.__name__:
            subset = find_layers(layer,  layers=[transformers.Conv1D])
        else:
             subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            sub_sub_count = 0
            sub_sub_params = 0
            W = subset[name].weight.data
            sub_sub_count = (W==0).sum().item()
            sub_sub_params = W.numel()

                    
            if not sub_sub_count == 0.0:
                sub_count += sub_sub_count 
                sub_params += sub_sub_params
                
                count += sub_sub_count 
                total_params += sub_sub_params  
                
            print(f"layer {i} name {name} sparsity {float(sub_sub_count)/sub_sub_params:.6f}")
        print(f"layer {i} total sparsity {float(sub_count)/sub_params:.6f}")

    #model.config.use_cache = use_cache 
    return float(count)/total_params 

def check_sparsity_mask(mask):

    W = mask
    count = 0 
    total_params = 0
    count += (W!=0).sum().item()
    total_params += W.numel()
    print(f" density {float(count)/total_params:.6f}")

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


def prepare_calibration_input(model, nsamples, dataloader, device):
    if "GPT2"  in model.__class__.__name__: 
        use_cache = model.config.use_cache
        model.config.use_cache = False
    print(dataloader[0][0].shape[0])
    seqlen =  dataloader[0][0].shape[0]

    layers = get_transformer_blocks(model)

    dtype = next(iter(model.parameters())).dtype
    if "GPT2"  in model.__class__.__name__: 
        inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu")

    else:
        inps = torch.zeros((nsamples, model.seqlen, model.cfg.d_model), dtype=dtype, device="cpu")
    
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            
            #cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
            print("This should not be reached!")  # if here is reached, check data and model. Might be on wrong device
        except ValueError:
            pass 
    layers[0] = layers[0].module

    inps = inps.cpu()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    
    if "GPT2"  in model.__class__.__name__: 
        model.config.use_cache = use_cache

    return inps, outs, attention_mask, None 

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
        
        if target_layers:
            sub_layers_copy = sub_layers.copy()
            for name in sub_layers_copy:
                if not name == "hook_z":
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
            wrapped_layers[name] = WrappedGPT(sub_layers[name], is_gpt=True, layer_name=name)
        else:
            wrapped_layers[name] = WrappedGPT(sub_layers[name], layer_name=name)
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
            #print("j", j, "input", inp_gpu)
            if "GPT2"  in model.__class__.__name__:   
                out = block(inp_gpu)[0]
            else:
                print("with attention mask")             
                out = block(inp_gpu, attention_mask=attention_mask.to(device))[0]
            #print("out", out)

            outs[j] = out.detach().to(device)  
            del inp_gpu, out
            torch.cuda.empty_cache()


def calculate_WANDA_metric(
    name:str,
    sub_layers:dict,
    wrapped_layers: dict,
    model
): 
    """WANDA metric =  magnitude of weights * accumulation of all connected input data (= activation data)        

    Args:
        name (str): current sublayer
        sub_layers (dict): all sub layers of a block
        wrapped_layers (dict): all hooked sublayers, storing theo normalized activation values
        model : input model

    Returns:
        dict: the scoring metric for each sublayer
    """
    print("name", name)

    activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


    weights = torch.abs(sub_layers[name].weight.data)

    weights = weights.t()
    print("weights", weights[:768, :])

    if activation.shape[1] != weights.shape[1]:
        repeat = int(weights.shape[1]/activation.shape[1])
        activation = activation.repeat(1, repeat)
        print("REPEAT!")
    print("activation", activation[:, :10])
    w_metric = weights * activation   
    print("w_metric", w_metric.shape) 
    print("w_metric", w_metric[:768, :])     
    return w_metric


def per_output_pruning(
    args,
    model, 
    W_metric, 
    prune_n, 
    prune_m, 
    layer_sparsity_ratio
    ):
    """ Caculate the masking matrix from the scoring matrix. 
    (W_metric.shape[1] * layer_sparsity_ratio) is the amount of weights to be set to 0 in 
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
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:,:int(W_metric.shape[1] * layer_sparsity_ratio)]
        
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
    (W_metric.shape[1] * layer_sparsity_ratio) is the amount of weights to be set to 0 in 
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
           
        flat_W = torch.abs(W_metric.flatten())
        k = int(W_metric.numel() * layer_sparsity_ratio)
        _, ind_1D = flat_W.topk(k=k, largest=False)
        rows = torch.div(ind_1D, W_metric.shape[1], rounding_mode='floor')
        cols = ind_1D % W_metric.shape[1]         
    
        W_mask[rows, cols] = True
        #W_mask.scatter_(1, indices, True)
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
    all_block_ratio = ((all_block_ratio - all_block_ratio.min()) * (1/(all_block_ratio.max() - all_block_ratio.min()) * args.Lamda*2))
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
        
        qkv_split = split_whole_weight_matrix_to_components(w_matrix, model_dim, dim)

        comp_idx = 0
        res = {}
        for letter in "qkv":
            qkv = qkv_split[comp_idx]
            res[letter] = split_component_weight_matrix_to_heads(qkv, model_dim, n_head, dim=1)
            comp_idx += 1
        return res

    else:
        return split_component_weight_matrix_to_heads(w_matrix, model_dim, n_head, dim=0)

def set_head_in_w_mask(
    W_mask, 
    qkv_comp, 
    head_idx, 
    model_dim, 
    head_dim, 
    value, 
    target_layer
    ):    
    
    """Set all entries of one specific attention head in a weight matrix
    (defined by qkv_comp and head_index) to a value.

    Args:
        W_mask (List[bool]): Matrix to mask weight matrix.
        qkv_comp (Optional[str]): Head of which component to prune (if None -> prune head of all components qkv).
        head_idx (int): Index of head to mask.
        model_dim (int): Model dimension size.
        head_dim (int): Head dimension size.
        value (bool): Value for all entries of attention heads.
        target_layer (str): Layer to be targeted (e.g., 'c_proj' or 'c_attn').

    Returns:
        List[bool]: W_mask with one head set to value.
    """

    if "c_proj" in target_layer:
        start = int(head_dim * head_idx)
        end = int(head_dim * (head_idx + 1))
        W_mask[start:end, :] = value

    elif "c_attn" in target_layer:
        if qkv_comp is None:
            # qkv component is not defined, prune heads in all three components
            for comp_idx in range(3):
                start = int(head_dim * head_idx + comp_idx * model_dim)
                end = int(head_dim * (head_idx + 1) + comp_idx * model_dim)
                W_mask[:, start:end] = value
        else:
            map_letter_to_number = {"q": 0, "k": 1, "v": 2}
            comp_idx = map_letter_to_number[qkv_comp]
            start = int(head_dim * head_idx + comp_idx * model_dim)
            end = int(head_dim * (head_idx + 1) + comp_idx * model_dim)
            W_mask[:, start:end] = value
    else:
        raise ValueError(
            f"Invalid target layer '{target_layer}'. Valid layers are 'c_proj' or 'c_attn'."
        )
    return W_mask

def get_head(
    W,
    qkv_comp, 
    head_idx, 
    model_dim, 
    head_dim, 
    target_layer
    ):
    """get one one specific attention head in a weight matrix
    (defined by qkv_comp and/or head_index) to a value.

    Args:
        W: weight matrix
        qkv_comp Optional(int): idx of component in which is head to return 
        head_idx (int): index of head to mask
        model_dim (int): model dimension
        head_dim (int): head dimension
        target_layer (str): name of the layer to prune

    Returns:
        (List[bool]): W_mask with one head set to value
    """
    if "c_proj" in target_layer:
        start = int(head_dim * head_idx)
        end = int(head_dim * (head_idx + 1))
        head = W[start : end]

    elif "c_attn" in target_layer:
        
        start_comp = qkv_comp *model_dim
        print("qkv", qkv_comp)
        print("modeldim", model_dim)
        end_comp = (qkv_comp + 1) * model_dim

        start = int(head_dim * head_idx + start_comp)
        end = int(head_dim * (head_idx + 1) + start_comp)
        head = W[:, start : end]
    else:
        raise ValueError(f"trying to prune a non-exisitng layer: {target_layer}.")

    return head


#################### Threshold functions for attention head pruning:
def sparsity_ratio(W_matrix, value=True):
    # sparsity_count -> number of weights that are 0/False
    # weights that are False/0 will not be zeroed out
    sparsity_count = (W_matrix==value).sum().item()
    total_params = W_matrix.numel()
    return sparsity_count/total_params


def sublayer_statistics(heads):
    ## How many weights are on average zeroed out?
    
    head_sparsity_ratio = []

    for head_idx in range(heads.shape[0]):
        head = heads[head_idx]

        head_sparsity_ratio.append(sparsity_ratio(head))
    head_sparsity_ratio = np.array(head_sparsity_ratio)
    return np.mean(head_sparsity_ratio), np.std(head_sparsity_ratio)


def over_SD_threshold(head, block_mean, block_sd, SD_threshold=2, value=True):
    head_sparsity = sparsity_ratio(head, value=value)
    return abs(head_sparsity - block_mean) > SD_threshold * block_sd

        
def prune_heads_OWL(
    args, 
    dataloader, 
    model, 
    tokenizer, 
    device=torch.device("cuda:0"), 
    prune_n=0, 
    prune_m=0, 
    dataset_name="c4",
    per_sublayer=False
    ):
        
    unpruned_heads = {}
    all_w_metrics = {}

    ##### calucalte outlier ratio per attention head and mlp
    all_outlier_ratio=np.array([])
    sublayer_outlier=[]


    if "GPT2"  in model.__class__.__name__: 
        model_dim=model.config.hidden_size
        n_head=model.config.n_head
        head_dim = model_dim / n_head
        use_cache = model.config.use_cache 
        model.config.use_cache = False 

    else:
        model_dim=model.cfg.d_model
        n_head=model.cfg.n_heads
        head_dim = model_dim / n_head

    # preprocessing input data
    inps, outs, attention_mask, position_ids = preprocess_input_data(
        dataloader=dataloader,
        dataset_name=dataset_name,
        args=args,
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    blocks = get_transformer_blocks(model)

    for i in range(len(blocks)):
        if args.verbose:
            print("################# layer", i)

        block = blocks[i]

        
        # get all sublayers of block
        sub_layers = get_sub_layers(target_layers=args.target_layers, block=block, model=model)
        # wrapp the sublayers    
        wrapped_layers = wrapping_layers(sub_layers, model)
        # add hooks to layers, called during forward pass 
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
    
    
        handles = []
        for name in wrapped_layers:
            handles.append(sub_layers[name].register_forward_hook(add_batch(name)))

        # forward pass through current block with hooks
        forward_pass_block(
            inps = inps, 
            outs = outs,
            attention_mask = attention_mask, 
            block = block,
            nsamples = args.nsamples,
            device=device,
            model = model
            )

        # remove hooks
        for h in handles:
            h.remove()  
            
        # calculate the score function
        # len_wmetric is the score activation * weight_magnitude for each weight per sublayer
        layer_wmetric=[]
        for name in sub_layers:
            if args.verbose:
                print(f"pruning layer {i} name {name}")
            W_metric = calculate_WANDA_metric(name, sub_layers=sub_layers, wrapped_layers=wrapped_layers, model=model)
            W_metric_test = split_component_weight_matrix_to_heads(W_metric, model_dim, n_head, dim=1)
            
            print("W_metric", W_metric)
            #print("W_,metric_test", W_metric_test)
            layer_wmetric.append(W_metric)
        # foward pass through block without the hooks
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
    if "GPT2"  in model.__class__.__name__: 
        model.config.use_cache = use_cache 
        torch.cuda.empty_cache()

        use_cache = model.config.use_cache 
        model.config.use_cache = False 
   
    inps, outs, attention_mask, position_ids = preprocess_input_data(
        dataloader=dataloader,
        dataset_name=dataset_name,
        args=args,
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    blocks = get_transformer_blocks(model)

    for i in range(len(blocks)):
        if args.verbose:
            print("############ LAYER", i ,"#################")

        unpruned_heads[i] = []
        block = blocks[i]


        # get all sublayers of block
        
        sub_layers = get_sub_layers(target_layers=args.target_layers, block=block, model=model)
        # wrapp the sublayers
        wrapped_layers = wrapping_layers(sub_layers, model)

        # add hooks to layers, called during forward pass 
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(sub_layers[name].register_forward_hook(add_batch(name)))
        # forward pass through current block with hooks
        
        forward_pass_block(
            inps = inps, 
            outs = outs,
            attention_mask = attention_mask, 
            block = block,
            nsamples = args.nsamples,
            device=device,
            model = model
            )

        # remove hooks
        for h in handles:
            h.remove()  
    
        layer_sparsity_ratio= 1-all_outlier_ratio[i]
            
        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01
            
        if args.verbose:
            print("layer sparsity ratio", layer_sparsity_ratio)

        for name in sub_layers:
            if args.verbose:
                print(f"pruning layer {i} name {name}")
            
            # New Metric
            W_metric = calculate_WANDA_metric(
                name=name, 
                sub_layers=sub_layers, 
                wrapped_layers=wrapped_layers, 
                model=model)
         
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
            

            # head pruning for c_proj
            all_w_metrics[i] = W_mask.clone().detach()
            sub_layers[name].weight.data[W_mask.t()] = 0  ## set weights to zero 

            print("prubed shape", sub_layers[name].weight.data.shape)
            print("pruned", sub_layers[name].weight.data.t()[:768,:])
        #wq, wk, wv = split_whole_weight_matrix_to_heads(sub_layers["attn.c_attn"].weight.data, model_dim, n_head, dim=1) 
        #print("wq", wq)
        #print("input before final model pass", inps[0])
        # forward pass thorugh block without hooks
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
    return unpruned_heads, all_w_metrics



def prune_wanda_outlier(
    args, 
    dataloader, 
    model, 
    tokenizer, 
    device=torch.device("cuda:0"), 
    prune_n=0, 
    prune_m=0, 
    dataset_name="c4"    
):
    ##### calucalte outlier ratio
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader,_ = get_loaders("ioi", args.nsamples, args.seed, model.seqlen, tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, args.nsamples, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, args.nsamples, dataloader, device)

    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model._modules.get("transformer")._modules.get("h")

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=[transformers.Conv1D])
        wrapped_layers =  wrapping_layers(subset, model)
        
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()
            
        layer_wmetric=[]

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = calculate_WANDA_metric(
                name=name,
                sub_layers=subset,
                wrapped_layers=wrapped_layers,
                model=model
                )
            #print("W_metric", W_metric)
            
            layer_wmetric.append(W_metric)    
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0))[0]
        inps, outs = outs, inps
        
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in args.Hyper_m:
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)
        all_layer_ratio.append(out_ratio_layer)
    if args.verbose:
        print("length of outlier ratio over all layers, ", len(all_layer_ratio))
        print("*"*50, "\noutlier ratio over all layers:")
        print ("before adjustment",all_layer_ratio)
        print("mean: ", np.mean(all_layer_ratio), "\n max: ", np.max(all_layer_ratio), "\n min: ", np.min(all_layer_ratio)) 
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)

    
    if args.verbose:
        print ("after adjustment: ", all_layer_ratio)
        print("mean: ", np.mean(all_layer_ratio), "\n max: ", np.max(all_layer_ratio), "\n min: ", np.min(all_layer_ratio)) 
        print("*"*50)
    
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader,_ = get_loaders("ioi", args.nsamples, args.seed, model.seqlen, tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, args.nsamples, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, args.nsamples, dataloader, device)
    print("inps",inps)
    
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model._modules.get("transformer")._modules.get("h")


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer, layers=[transformers.Conv1D])
        print("sublayers", subset)

        wrapped_layers = wrapping_layers(subset, model)


        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()

        print("outs", outs)
        for name in subset:
            

            print(f"pruning layer {i} name {name}")
            W_metric = calculate_WANDA_metric(
                name=name,
                sub_layers=subset,
                wrapped_layers=wrapped_layers,
                model=model
                )
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            #print("W_metric", W_metric[0, :10, :10])
            
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01
            if args.verbose:
                print("layer sparsity ratio", layer_sparsity_ratio)

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
            print("W_mask", W_mask)
                
            if "GPT" in model.__class__.__name__:
                subset[name].weight.data[W_mask.t()] = 0  ## set weights to zero 
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0))[0]
        inps, outs = outs, inps
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()