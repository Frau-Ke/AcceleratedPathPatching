from IPython.display import Image, display
import numpy as np
import torch
import networkx as nx
import huggingface_hub
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import einops
import yaml
import pandas as pd
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import matplotlib.pyplot as plt
import datetime
import time
from pathlib import Path
from utils.data_io import get_base_path
from utils.circuit_functions import circuit_size

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)

from Patching.acdc.docstring.utils import get_all_docstring_things
from Patching.acdc.acdc_utils import (
    make_nd_dict,
    reset_network,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)
from Patching.acdc.acdc_graphics import (
    build_colorscheme,
    show
)

from fvcore.nn import FlopCountAnalysis
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from Patching.acdc.TLACDCCorrespondence import TLACDCCorrespondence
from Patching.acdc.TLACDCInterpNode import TLACDCInterpNode
from Patching.acdc.TLACDCExperiment import TLACDCExperiment
from Patching.acdc.TLACDCEdge import Edge
from Patching.TaskInterface import TaskInterface
from utils.data_io import save_img, create_folder, save_parser_information, save_circuit, store_df, save_parser_information, set_PATH, get_PATH, save_panda_to_text

class ACDC(TaskInterface):
    def __init__(
        self, 
        args,
        threshold:float,
        remove_redundant:bool,
        reduced_search_space:dict=None,
        model=None,
        tokenizer=None,
        device:str="cuda",
        zero_ablation=False,
        indices_mode:str ="reverse", 
        names_mode:str = "normal",
        single_step:bool = False,
        early_stop:bool = False,
        reset_network = False, 
        corrupted_cache_cpu:bool=False,
        online_cache_cpu:bool =False,
        abs_value_threshold:bool = False,
        add_sender_hooks:bool=True,
        use_pos_embed:bool=False,
        add_receiver_hooks:bool=False,
        hook_verbose:bool=False,
        second_metric:Optional[Callable]=None,
        out_path:str=""
        ):
        super().__init__(
            model_name=args.model_name, 
            task=args.task, 
            patching_method="acdc", 
            metric_name=args.metric, 
            N=args.N, 
            verbose=args.verbose,
            model=model,
            tokenizer=tokenizer,
            device=device, 
            patch_mlp=args.patch_mlp,
            seed=args.seed,
            calc_FLOP=args.calc_FLOP, 
            prepend_bos=False,
            cache_dir=args.cache_dir
            )
        
        
        start_time = time.time() 
        self.single_step = single_step
        self.early_stop = early_stop
        self.threshold = threshold
        
        self._model.reset_hooks()
        self.out_path = out_path
        self.exp_name = f"{self.metric_name}_N-{self.N}"
        
        if reset_network:
            reset_network(task=self.task, device=self._device, model=self._model)
        
        self.ablation = "zero_ablation" if zero_ablation else "corrupted" 
        ACDC_method = "ACDC" if reduced_search_space == None else "acceleratedACDC"
        use_mlp = "withMLPs" if args.patch_mlp else "noMLPs"
        self.task_folder = f"{args.out_path}/{args.model_name}/{args.task}/{ACDC_method}/"
        self.create_folder(self.task_folder)

        self.experiment_folder = f"{self.task_folder}/{self.threshold}/{self.exp_name}/"
        print(self.experiment_folder)
        self.create_folder(self.experiment_folder)
        
        self.exp = TLACDCExperiment(
            model=self._model,
            threshold=self.threshold,
            zero_ablation=zero_ablation,
            ds=self.clean_tokens,
            ref_ds=self.corrupted_tokens,
            metric=self.metric,
            second_metric=second_metric,
            reduced_search_space=reduced_search_space,
            verbose=self.verbose,
            indices_mode=indices_mode,
            corrupted_cache_cpu=corrupted_cache_cpu,
            hook_verbose=hook_verbose,
            online_cache_cpu=online_cache_cpu,
            add_sender_hooks=add_sender_hooks,
            use_pos_embed=use_pos_embed,
            add_receiver_hooks=add_receiver_hooks,
            remove_redundant=remove_redundant,
            names_mode=names_mode,
            abs_value_threshold = abs_value_threshold, 
            experiment_folder=self.experiment_folder, 

        )
        self.CIRCUIT=None
        self.elapsed_time += time.time() - start_time
        
        if args.save_text:
            save_parser_information(args, self.experiment_folder, "parser_info.json")

    
    def run_exp(self, testing:bool=False, max_epochs:int = 10000, save_text:bool=True, show_img:bool=False, save_img:bool=False):
        run_time = time.time()
        
        
        for i in range(max_epochs):
            self.exp.step(testing=testing, early_stop=self.early_stop, show_img=show_img, save_img=save_img, patch_mlp=self.patch_mlp)
            if i == 0:
                self.exp.save_edges(f"{self.experiment_folder}/{self.exp_name}_all_edges.pkl")
            
            if self.exp.current_node is None or self.single_step:
                #show(self.exp.corr, f"{self.experiment_folder}/ACDC_{self.exp_name}.png", show_full_index=False, save_img=True)   
                #display(Image(Path(f"{self.experiment_folder}/ACDC_{self.exp_name}.png")))
                break

            if self.early_stop:
                break
        
        
        self.exp.remove_unsignificant_MLP_connections()  
        show(self.exp.corr, f"{self.experiment_folder}/ACDC_{self.exp_name}.png", show_full_index=False, save_img=True)   
        display(Image(Path(f"{self.experiment_folder}/ACDC_{self.exp_name}.png")))

        self.exp.save_edges(f"{self.experiment_folder}/{self.exp_name}_subgraph_edges.pkl")
        
        self.exp.save_subgraph(
            return_it=True,
            fpath=f"{self.experiment_folder}/{self.exp_name}_subgraph.pth"
        )   
        
        self.elapsed_time += time.time() - run_time
        if self.calc_FLOP:
            self.FLOP_counter += self.exp.n_forward_passes * self.module_FLOPS.total() / 1e9
        
        self.CIRCUIT = self.correspondence_to_dict()
        
        df_efficency_metric = pd.DataFrame(
            {
                "GFLOP": [self.FLOP_counter],
                "n_forward_passes":[ self.exp.n_forward_passes], 
                "comp_time": [self.elapsed_time],                 
            }
            )
        store_df(df_efficency_metric, out_path=self.experiment_folder, name="efficency_metrics.json")
        save_circuit(self.CIRCUIT, self.experiment_folder, "ACDC_circuit.txt")
           
                                    
                
    def correspondence_to_dict(self):
        CIRCUIT = {}
        
        for child_name, rest1 in self.exp.corr.edges.items():
            for child_index, rest2 in rest1.items():
                for parent_name, rest3, in rest2.items():
                    for parent_index, edges in rest3.items():
                        if "attn.hook_result" in child_name:

                                layer_idx = int(child_name.split(".")[1])              
                                head_idx = child_index.hashable_tuple[-1]
                                
                                if CIRCUIT.get(layer_idx) == None:
                                    CIRCUIT[layer_idx] = [head_idx]
                                elif head_idx in CIRCUIT[layer_idx]:
                                    continue
                                else:
                                    CIRCUIT[layer_idx].append(head_idx)
        return CIRCUIT

                
    def reset_network(self):
        reset_network(self.task, self._device, self._model )
        
    def save_subgraph_long(self, path):        
        data = pd.DataFrame(columns=["child_node", "child_head", "parent_node", "parent_head", "effect_size"])
        for child_hook_name in self.exp.corr.edges:
            for child_index in self.exp.corr.edges[child_hook_name]:
                for parent_hook_name in self.exp.corr.edges[child_hook_name][child_index]:
                    for parent_index in self.exp.corr.edges[child_hook_name][child_index][parent_hook_name]:
                        edge = self.exp.corr.edges[child_hook_name][child_index][parent_hook_name][parent_index]                        
                        if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
                            entry = pd.DataFrame({
                            'child_node': [child_hook_name], 
                            'child_head': [child_index],
                            'parent_node': [parent_hook_name],
                            'parent_head': [parent_index],
                            "effect_size": [edge.effect_size]})
                            data = pd.concat([data, entry], ignore_index=False)
        self.create_folder(path)
        data.to_pickle(f"{path}/acdc_circuit_long.pkl")                  
        
def run_acdc(args, thresholds:list, HYBRID_FLAP_CIRCUIT:dict=None, save_img:bool=False):
    torch.autograd.set_grad_enabled(False)
    subfolder = get_base_path(args, "ACDC")

    circuits_dict = {}

    for threshold in thresholds:
        acdc = ACDC(   # TODO: create ACDC object only once. Create possibility to reset it (model, efficency_metrics, metricss) 
                args,
                threshold=threshold,
                second_metric=None,
                reduced_search_space=HYBRID_FLAP_CIRCUIT,
                remove_redundant=args.remove_redundant,
                indices_mode="reverse",
                names_mode="normal",
                zero_ablation=False,
                abs_value_threshold=args.abs_value_threshold,
                out_path=subfolder
                )  
        
        acdc.run_exp(testing=False, max_epochs=args.max_epochs, save_text=False, show_img=args.show, save_img=save_img)
        CIRCUIT = acdc.CIRCUIT
        
        results = {
            "GFLOP": [acdc.FLOP_counter], 
            "n_forward_passes": [acdc.n_forward_passes], 
            "comp_time": [acdc.elapsed_time], 
            "size_circuit": [circuit_size(CIRCUIT)]
        }

        results = pd.DataFrame(data=results)
        
        if args.save_text:
                save_circuit(CIRCUIT, subfolder, "circuit.txt")
                save_parser_information(args, subfolder, "parser_information.txt")
                store_df(results, subfolder, "results.json")
                
        if circuits_dict.get(threshold) is None:
            circuits_dict[threshold] = CIRCUIT
        
    return circuits_dict    