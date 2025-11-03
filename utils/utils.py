from transformer_lens import HookedTransformer
import torch
import numpy as np
import os
from utils.visualization import plot_activations
from utils.data_io import save_img


def activation_patterns(
    model, 
    tokenizer,
    dataset,  
    CIRCUIT=None, 
    layer_list=None,
    head_list=None,
    activation_type:str="clean",
    save:bool=False,
    show:bool=True,     
    out_path:str="", 
    title:str=""
    ):
    """ Analyse circuit heads based on their activation patterns. Only heads in [layer_list, head_list] are regarded.
    Either CRICUIT or layer_list and head_list must be defined. 
    
    Args:
        model (_type_): model
        tokenizer (_type_): tokenizer
        dataset (_type_): dataset
        CIRCUIT (_type_, optional): Analyse all heads in CIRCUIT. Defaults to None.
        layer_list (_type_, optional): Analyse specific heads with layer in layer list. Defaults to None.
        head_list (_type_, optional):  Analyse specific heads with head in head list. Defaults to None.
        activation_type (str, optional): Activations under certain dataset ["clean", "corrupted", "contrastive"]. Defaults to clean.
        save (bool, optional): save images. Defaults to False.
        show (bool, optional): show images. Defaults to True.
        out_path (str, optional): out_path. Defaults to "".
        title (str, optional): title of images. Defaults to "".

    """
    if CIRCUIT is None and (head_list is None and  layer_list is None):
        raise Exception("Either circuit or head and layer list must be defined")
    
    try:
        start = int(dataset.start[0].item())
        end = int(dataset.end[0].item())
    except:
        raise Exception("add a start and end variable to the dataset")

    # ----- activations of HookedLLM models -----
    if isinstance(model, HookedTransformer):
        score_name_filter = lambda name:name.endswith("attn_scores")

        with torch.no_grad():
            _, cache = model.run_with_cache(
                dataset.clean_tokens[:, start:end],
                return_type=None, 
                names_filter = score_name_filter
                )
        clean_outputs = [cache[name] for name in list(cache.keys()) if name.endswith("attn_scores")]
        
        with torch.no_grad():
            _, corr_cache = model.run_with_cache(
                dataset.corrupted_tokens[:, start:end],
                return_type=None, 
                names_filter = score_name_filter
                )
        corr_outputs = [corr_cache[name] for name in list(corr_cache.keys()) if name.endswith("attn_scores")]
        
        
        if activation_type == "corrupted":
            attentions = corr_outputs
            
        elif activation_type == "contrastive":
            attentions = [abs(corr_att - clean_att) for corr_att, clean_att in zip(corr_outputs, clean_outputs)]
        else:
            attentions = clean_outputs
                
    # ----- activations of CasualLLM models -----
    else:
        with torch.no_grad():
            corr_outputs = model(dataset.corrupted_tokens)
            clean_outputs = model(dataset.clean_tokens)

        if activation_type == "corrupted":
            attentions = corr_outputs.attentions  
            
        elif activation_type == "contrastive":
            
            attentions = [abs(corr_att - clean_att) for corr_att, clean_att in zip(corr_outputs.attentions, clean_outputs.attentions)]
        else:
            attentions = clean_outputs.attentions
        

    # ----- tokens for plotting -----
    if activation_type == "corrupted":
        folder = "corr_activations" 
        tokens  = tokenizer.batch_decode(dataset.corrupted_tokens[0][start:end])
    elif activation_type == "contrastive":
        folder = "diff_activations" 
        tokens  = tokenizer.batch_decode(dataset.clean_tokens[0][start:end])
    else:
        folder = "clean_activations" 
        tokens  = tokenizer.batch_decode(dataset.clean_tokens[0][start:end])
        
        
    out_path =  os.path.join(out_path, folder)

    # ----- get the heads for plotting ------
    if CIRCUIT is not None:
        layers = list(CIRCUIT.keys())
        heads = list(CIRCUIT.values())
    else:
        layers = layer_list
        heads = [head_list] * len(layers)

    for layer_idx, layer in enumerate(layers):
        for head in heads[layer_idx]: 
            print(attentions[layer].shape)

            attention_matrix = attentions[layer][0, head]  

            # mask the upper triangle
            mask = np.triu(np.ones_like(attention_matrix.cpu(), dtype=bool), k=1)
            masked_matrix = np.where(mask, np.nan, attention_matrix.cpu())

            # scale activations (only Hooked Transformer!)         
            if isinstance(model, HookedTransformer):
                min = masked_matrix[~mask].min()
                max = masked_matrix[~mask].max()
                masked_matrix = (masked_matrix - min)/ (max - min)

            # plot
            fig = plot_activations(
                activations=masked_matrix, 
                tokens=tokens, 
                head=(layer, head), 
                x_label="Attended Token",
                y_label="Current Token", 
                title=title
                )
            
            if save:
                save_title= f"{layer}_{head}.png"
                save_img(fig, out_path=out_path, name=save_title)
            
            if show:
                fig.show()

def print_pretty(circle_heads:dict):
    for (key, value) in circle_heads.items():
        print(f"{key} : {value}")
        

#----------------------------------------------------------------------------------------------------
# read ACDC graphs 
#----------------------------------------------------------------------------------------------------

def dict_from_edge_list(title, folder, node_list):
    for idx in range(len(node_list)):
        print( node_list[idx][0][2],  node_list[idx][0][3])
        print( node_list[idx][0][0],  node_list[idx][0][1])

def has_duplicate(l):
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if l[i] == l[j]:
                print(l[i])
                print(l[j])
                return True
    return False 
        
def extract(node_name):
    if  len(node_name.split(".")) == 3:
        _, layer, module_name = node_name.split(".")
        return layer, None, module_name

    else:
        _, layer, block, module_name = node_name.split(".") 
        return layer, block, module_name
    
def get_nodes_from_acdc(node_list):
    # turn the nodes of the acdc pkl file in a set of nodes
    
    nodes = []
    
    for index, row in node_list.iterrows():
        
        child = row["child_node"]
        child_head = row["child_head"]
        layer, block, module = extract(child)
        if len(child_head.as_index ) == 3:   
            node = (layer, block, module, child_head.as_index [2])
        else:
            node = (layer, block, module, None)
        nodes.append(node)
        
         
        parent = row["parent_node"]
        parent_head = row["parent_head"]
        layer, block, module = extract(parent)

        if len(parent_head.as_index ) == 3: 
            node = (layer, block, module, parent_head.as_index[2])
        else:
            node = (layer, block, module, None)
        nodes.append(node)

    return set(nodes)

def get_edges_from_acdc(node_list):
    
    edges = []
    
    for index, row in node_list.iterrows():
        
        child = row["child_node"]
        child_head = row["child_head"]
        layer, block, module = extract(child)
        if len(child_head.as_index ) == 3:   
            node = (layer, block, module, child_head.as_index [2])
        else:
            node = (layer, block, module, None)
        e1 = node
         
        parent = row["parent_node"]
        parent_head = row["parent_head"]
        layer, block, module = extract(parent)

        if len(parent_head.as_index ) == 3: 
            node = (layer, block, module, parent_head.as_index[2])
        else:
            node = (layer, block, module, None)
        e2 = node 
        edges.append((e1, e2))

    return set(edges)
