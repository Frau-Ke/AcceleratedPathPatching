from transformer_lens import HookedTransformer

from functools import partial

from utils.PatchingMetric import *
from transformer_lens import utils, HookedTransformer, ActivationCache
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import json
from utils.Visualization import plot_activations, heat_map_sparsity
import argparse

def activation_patterns_heatmap(
    model, 
    tokenizer,
    dataset,  
    CIRCUIT=None, 
    GT_CIRCUIT=None,
    activation_type=False,
    save=False,
    show=True,     
    out_path="/mnt/lustre/work/eickhoff/esx670/", 
    title=""
    ):
    
    try:
        start = int(dataset.start[0].item())
        end = int(dataset.end[0].item())
    except:
        raise Exception("add a start and end variable to the dataset")
    print("from", start, "to", end)

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
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
            
        with torch.no_grad():
            _, corr_cache = model.run_with_cache(
                dataset.corrupted_tokens[:, start:end],
                return_type=None, 
                names_filter = score_name_filter
                )
        corr_outputs = [corr_cache[name] for name in list(corr_cache.keys()) if name.endswith("attn_scores")]
        
        
        if activation_type == "corrupted":
            attentions = corr_outputs
            
        elif activation_type == "difference":
            attentions = [abs(corr_att - clean_att) for corr_att, clean_att in zip(corr_outputs, clean_outputs)]
        else:
            attentions = clean_outputs
                
    # ----- activations of CasualLLM models -----
    else:
        with torch.no_grad():
            corr_outputs = model(dataset.corrupted_tokens)
            clean_outputs = model(dataset.clean_tokens)

        n_layers = model.config.n_layer
        n_heads = model.config.n_head
        
        if activation_type == "corrupted":
            attentions = corr_outputs.attentions  
            
        elif activation_type == "difference":
            attentions = [abs(corr_att - clean_att) for corr_att, clean_att in zip(corr_outputs.attentions, clean_outputs.attentions)]
        else:
            attentions = clean_outputs.attentions
    # ----- tokens for plotting -----
    if activation_type == "corrupted":
        folder = "corr_activations" 
    elif activation_type == "difference":
        folder = "diff_activations" 
    else:
        folder = "clean_activations" 
        
    out_path =  os.path.join(out_path, folder)
    
    scores = np.zeros((n_layers, n_heads))
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            
            attention_matrix = attentions[layer_idx][0, head_idx]  
            # mask the upper triangle
            mask = np.triu(np.ones_like(attention_matrix.cpu(), dtype=bool), k=1)
            masked_matrix = np.where(mask, np.nan, attention_matrix.cpu())
            print(attention_matrix)
            # scale activations (only Hooked Transformer!) 
            scores[layer_idx, head_idx] = np.nanmean(masked_matrix)#masked_matrix.mean()# attention_matrix.mean()
    # plot
    fig = heat_map_sparsity(
        scores,
        GT_CIRCUIT, 
        CIRCUIT,
        title=f"Average Input Activations",
        title_eval_circuit="FLAP",
        title_compare_circuit="Path Patching",
        performance=None,
        print_vals=False,
        title_temp_scale="average activation"
        )
    
    if save:
        save_title = f"Average Input Activations"
        save_img(fig, out_path=out_path, name=save_title)
    
    if show:
        fig.show()
    


def activation_patterns(
    model, 
    tokenizer,
    dataset,  
    CIRCUIT=None, 
    layer_list=None,
    head_list=None,
    activation_type=False,
    save=False,
    show=True,     
    out_path="/mnt/lustre/work/eickhoff/esx670/", 
    title=""
    ):
    
    try:
        start = int(dataset.start[0].item())
        end = int(dataset.end[0].item())
    except:
        raise Exception("add a start and end variable to the dataset")
    print("from", start, "to", end)

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
            
        elif activation_type == "difference":
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
            
        elif activation_type == "difference":
            
            attentions = [abs(corr_att - clean_att) for corr_att, clean_att in zip(corr_outputs.attentions, clean_outputs.attentions)]
        else:
            attentions = clean_outputs.attentions
        

    # ----- tokens for plotting -----
    if activation_type == "corrupted":
        folder = "corr_activations" 
        tokens  = tokenizer.batch_decode(dataset.corrupted_tokens[0][start:end])
    elif activation_type == "difference":
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


def create_folder(path):
    if not os.path.isdir(path):
        print("create new path", path)
        os.makedirs(path)

def store_df(df, out_path, name):
    create_folder(out_path)
    
    file_path =  os.path.join(out_path, name)
    
    if file_path.endswith(".csv"):
        df.to_csv(file_path, index=False)
    elif file_path.endswith(".xlsx"):
        df.to_excel(file_path, index=False)
    elif file_path.endswith(".json"):
        df.to_json(file_path, orient="records", indent=4)
    else:
        raise ValueError("Unsupported file format. Use .csv, .xlsx, or .json")

    print(f"DataFrame saved as {name} at {out_path}")

def read_df(in_path, name):
    file_path =  os.path.join(in_path, name)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path, orient="records", indent=4)
    else:
        raise ValueError("Unsupported file format. Use .csv, .xlsx, or .json")

    print(f"DataFrame saved as {name} at {in_path}")
    return df

def load_df(in_path, name):
    file_path =  os.path.join(in_path, name)
    print(file_path)
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)

def save_dict(dict, out_path, name):
    file_path =  os.path.join(out_path, name)
    with open(file_path, 'w') as fp:
        json.dump(dict, fp)


def save_circuit(circuit, out_path, name): 
    # Save to file
    if not os.path.exists(out_path):
        create_folder(out_path)
        
    print("saving circuit at", os.path.join(out_path, name))
    circuit = dict(sorted(circuit.items()))

    with open(os.path.join(out_path, name), 'w') as file:
        file.write(str(circuit))

def load_circuit(out_path, name):
    with open(os.path.join(out_path, name), "r") as f:
        circuit = eval(f.read())
    return circuit

def save_img(fig, out_path, name):
    create_folder(out_path)
    file_path =  os.path.join(out_path, name)
    fig.savefig(file_path, bbox_inches='tight')
    print("save img at", file_path)
    
def save_excel(df, out_path, name):
    create_folder(out_path)
    print("saving at", out_path + name + ".xlsx")
    df.to_excel(out_path + name + ".xlsx", index=False)

            
def save_parser_information(args, subfolder, name):
    with open(os.path.join(subfolder, name), 'w') as f:
        json.dump(vars(args), f, indent=2)

def load_parser_information(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return argparse.Namespace(**data)

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
    
def get_nodes_from_OWL(node_list, block, module):
    nodes = []
    for layer, heads in node_list.items():
        for head in heads:
            nodes.append((layer, block, module, head))
    return set(nodes)