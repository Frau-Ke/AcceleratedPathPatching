import torch 
import torch.nn as nn 
from .layerwrapper import WrappedGPT, BiasGPT
import math
from tqdm import tqdm
from dataset.loader import load_dataset
from fvcore.nn import FlopCountAnalysis
import gc


# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance               -> 0 FLOPS
    'WIFV': Weighted Input Feature Variance     -> 2 * weight.X * weights.Y (elementwise multiplication, add collums) + fluc_inp.numel() (mutliply)
    'WIFN': Weighted Input Feature Norm         -> weight.X * weights.Y (abs) + weights.Y (sqrt) +  weight.x * weight.y (mutliplication matrix, elementwise vector) + weight.X * weight.y (mean)
                                                -> 3 * weight.X * weight.Y + +
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0)
}


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


def prepare_calibration_input(nsamples, model, dataloader, device,  difference_with="None"):
    """
    Prepare inputs for model calibration. 
    
    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded. 
        
    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    #cache = {'i': 0, 'attention_mask': torch.tensor([], device=device), "position_ids": None}
    cache = {'i': 0,  'attention_mask': torch.tensor([], device=device, dtype=torch.bool), "position_ids": None}
    # FLOPS: unembeddding
    class Catcher(nn.Module):
        
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):    
            inps[cache['i']] = inp
            cache['i'] += 1
            
            if kwargs['attention_mask'] == None:
                full_mask = torch.tril(torch.ones((model.seqlen, model.seqlen), dtype=torch.bool, device=device)).view(
                                        1, 1, model.seqlen, model.seqlen)
                    
                cache['attention_mask'] = torch.cat((full_mask,  cache['attention_mask']))
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']))

            cache['position_ids'] = kwargs['position_ids']            
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    if "Qwen" in model.__class__.__name__:
        layers[0].attention_type = layers[0].module.attention_type
        
    if difference_with == "corrupted":
        tokens = dataloader.corrupted_tokens
    else:
        tokens = dataloader.clean_tokens


    for idx in range(dataloader.clean_tokens.size(0)):
        try:
            attn_mask = dataloader.attention_mask[idx].unsqueeze(0).to(device)#.unsqueeze(0).to(device)
            inp = tokens[idx].unsqueeze(0)
            model(inp, attention_mask=attn_mask)
            print("ERROR in prepare_calibration_input(): this should not be reached")
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids, None
    

def forward_pass_with_hook(layer, wrapped_layers, subset, inps, outs, attention_mask, position_ids, position_embeddings, device, nsamples):
    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    # FLOP: calculations trough one layer
    for j in range(nsamples):
        inp_gpu = inps[j].unsqueeze(0).to(device)
        attn_mask = attention_mask[j].unsqueeze(0).to(device)
        with torch.no_grad():
            if position_embeddings is not None:
                out = layer(inp_gpu, attention_mask=attn_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
            else:
                out = layer(inp_gpu, attention_mask=attn_mask, position_ids=position_ids)[0]
        
        outs[j] = out.detach().to(device)
    
    del inp_gpu, out
    torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    return outs

    

def head_wise_pruning_scores(args, model, tokenizer, outlier_heads=[]):
    # FLOPs are calculated as follows:
        # one complete forward pass through the model (embedding, unembedding, attn, mlp, ...)
        #   or two in case of corrupted activations
        # for each layer: calculation in BiasGPT: 1 x mean = inp.numel()
        #                                      - if WIFN: norm = 2 x inp.numel()
        #                                      - else: sum = 1 x inp.numel()                           
        # Metric for each z weight matrix of each layer
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    FLOPS = 0
    
    dataset = load_dataset(
        task=args.task, 
        tokenizer=tokenizer, 
        N=args.nsamples, 
        device=args.device, 
        seed=args.seed, 
        prepend_bos=args.prepend_bos, 
        model_name=args.model_name, 
        remove_target_token=False
        )
        
    model.seqlen = dataset.max_len
    num_traversed_layers = 0
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(args.nsamples, model, dataset, args.device)

    if "Qwen" in args.model_name:
        position_embeddings = model.model.rotary_emb(inps[0].unsqueeze(0), position_ids)

    if args.calc_FLOP:
        num_traversed_layers += 1
        #FLOPS += FlopCountAnalysis(model, dataset.corrupted_tokens).total()  # FLOPS for the forward pass on corrupted input
    
 
    if args.difference_with == "corrupted":
        with torch.no_grad():
            inps_diff, outs_diff, _, _, _= prepare_calibration_input(args.nsamples, model, dataset, args.device, difference_with=args.difference_with)        
        
        if args.calc_FLOP:
            num_traversed_layers += 1
            #FLOPS += FlopCountAnalysis(model, dataset.corrupted_tokens).total()  # FLOPS for the forward pass on corrupted input
 
    layers = model.model.layers
    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list = []
    mlp_mask = []
    W_metric_unstand_list  = []
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    
    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        

        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        
        if args.use_mlp:
            subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            print("is in device map")
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        
        wrapped_layers = {}
        for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], args.metrics)#, dataset.start, dataset.end)    

        # FLOPS: Forward pass through one layer
        outs = forward_pass_with_hook(
            layer=layer, 
            wrapped_layers=wrapped_layers,
            subset=subset,
            inps=inps,
            outs=outs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            device=args.device,
            nsamples=args.nsamples
            )
        num_traversed_layers +=1

        # use as new scores score(clean) - score(corrupted) 
        if not args.difference_with == "None":
            corr_wrapped_layers = {}
            for name in subset:
                corr_wrapped_layers[name] = BiasGPT(subset[name], args.metrics)#, dataset.start, dataset.end)    
                  
            outs_diff = forward_pass_with_hook(
                layer=layer, 
                wrapped_layers=corr_wrapped_layers,
                subset=subset,
                inps=inps_diff,
                outs=outs_diff,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                device=args.device,
                nsamples=args.nsamples
                )
            num_traversed_layers +=1

            
        for name in subset:
            if name == 'self_attn.o_proj':
                # FLOPS for one mean calculation during Forward pass
                if args.calc_FLOP:
                    FLOPS += inps.numel()  
                    
                     
                W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                W_metrc_heads = W_metric.reshape(1, -1, head_dim).mean(dim=2)  # shape [12, 12]

                if args.difference_with == "corrupted":
                    W_metric_corr = metrics[args.metrics](corr_wrapped_layers, subset, name) ** 2
                    W_metric = torch.abs(W_metric - W_metric_corr)
                    
                    W_metrc_corr_heads = W_metric_corr.reshape(1, -1, head_dim).mean(dim=2)  # shape [12, 12]
                    W_metrc_heads = W_metric.reshape(1, -1, head_dim).mean(dim=2)  # shape [12, 12]

                if args.calc_FLOP:
                    if args.metrics == "WIFN":
                        # FLOPs for norm calculation
                        FLOPS += 2 * inps.numel()  # norm = 2 x inp.numel()
                        # FLOPs for WIFN metric
                        FLOPS += 3 * subset[name].weight.data.numel() + subset[name].weight.data.shape[1]
                    
                    elif args.metrics == "WIFV":
                        FLOPS += inps.numel()  # sum = inp.numel()
                        # FLOPS for WIFV metric
                        FLOPS += 2 * subset[name].weight.data.numel() +  wrapped_layers[name].fluc_inp.numel() 
                    else:
                        FLOPS += inps.numel()  # sum = inp.numel()
                
                W_metric_unstand_list.append(W_metric.reshape(-1, head_dim).mean(dim=1))
                attn_metric_list.append(W_metric.cpu())
                
                if next(model.parameters()).is_cuda: # when cuda use half precision
                    attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                else:
                    attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.float32))
            
            wrapped_layers[name].free()
            if not args.difference_with == "None":
                corr_wrapped_layers[name].free()
            
        inps, outs = outs, inps # Use the original output as input to the next layer
        if not args.difference_with == "None":
            inps_diff, outs_diff = outs_diff, inps_diff
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    del dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    return attn_metric_list, mlp_metric_list, mlp_mask, W_metric_unstand_list, FLOPS, num_traversed_layers


def CIRCUIT_from_scores(
    args, 
    attn_metric_list,
    W_metric_unstand_list, 
    mlp_metric_list=None,
    mlp_mask=None,
    FLOPS=0,
    n_layers=12, 
    n_heads=12, 
    head_dim=512, 
    ):
    CIRCUIT = {}

    attn_mask = [] 
    # FLOPs are calculated as follows:
        # standardization of the attn_metric_list 


    #FLOPs: ~ 5 * x.nunmel()
    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    if args.structure in ["AL-MM", "AL-AM"]:
        attn_metric = torch.stack(attn_metric_list) # shape [12, 768]
        attn_metric = standarlization(attn_metric)
        if args.calc_FLOP:
            FLOPS += attn_metric.numel() * 5  # 5 FLOPs for standardization: mean, std, substract, divide

        attn_metric = attn_metric.reshape(n_layers, -1, head_dim).mean(dim=2)  # shape [12, 12]
        
        if args.use_mlp:
            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = standarlization(mlp_metric)
        
        if args.structure == "AL-MM":
            sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
            attn_thres = sorted_attn[-int(args.remove_heads)]
            attn_mask = (attn_metric > attn_thres)  # 1 means retain
            
        
        else:
            if args.use_mlp:
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
            else:
                prune_metric = attn_metric.view(-1)

            sorted_prune, indices = torch.sort(prune_metric, descending=True)
            compression_weight = torch.ones_like(indices)
            compression_weight[indices < attn_metric.numel()] = 512.0 / 3
            threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - args.pruning_ratio)))]
            attn_mask = (attn_metric > threshold)
            if args.use_mlp:
                mlp_mask = (mlp_metric > threshold)
                
    elif args.structure in ["UL-UM", "UL-MM"]:
        attn_metric = torch.zeros(n_layers, n_heads)
        for layer_idx in range(n_layers):
            #layer = layers[layer_idx]
            W_metric = attn_metric_list[layer_idx]
            W_metric = standarlization(W_metric.unsqueeze(0))
            W_metric = W_metric.reshape(-1, head_dim).mean(dim=1)

            attn_metric[layer_idx] = W_metric
            if args.structure == "UL-UM":
                thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio*n_heads)].cpu() 
            else:
                thresh = torch.sort(W_metric.cuda())[0][args.remove_heads // n_layers].cpu()
            W_mask = (W_metric>=thresh)
            attn_mask.append(W_mask)
            
        attn_mask = torch.stack(attn_mask) 
        if args.use_mlp:
            mlp_mask = torch.stack(mlp_mask)
    
    # STEP 3: Adaptive Strucutre Search: search model globally to compress the model 
    for idx in range(n_layers):
        CIRCUIT[idx] = torch.where(attn_mask[idx]==True)[0].tolist()
              
    torch.cuda.empty_cache()
    return CIRCUIT, attn_metric, FLOPS
    
    
    
def prune_flap_modular(args, model, tokenizer, outlier_heads=[]):
    layers = model.model.layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    
    attn_metric_list, mlp_metric_list, mlp_mask, W_metric_unstand_list, FLOPS, n_traversed_layers = head_wise_pruning_scores(
        args, 
        model, 
        tokenizer, 
        outlier_heads
        )

    CIRCUIT, attn_metric, FLOPS = CIRCUIT_from_scores(
        args=args, 
        attn_metric_list=attn_metric_list, 
        W_metric_unstand_list=W_metric_unstand_list, 
        mlp_metric_list=mlp_metric_list, 
        mlp_mask=mlp_mask, 
        FLOPS=FLOPS, 
        n_layers=len(layers), 
        n_heads=num_heads,
        head_dim=head_dim
        )
      
    return CIRCUIT, attn_metric, FLOPS, n_traversed_layers