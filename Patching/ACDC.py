import numpy as np
import os
import wandb
import torch
import gc
import acdc
from tqdm import tqdm
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

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
try:
    from acdc.tracr_task.utils import (
        get_all_tracr_things,
        get_tracr_model_input_and_tl_model,
    )
except Exception as e:
    print(f"Could not import `tracr` because {e}; the rest of the file should work but you cannot use the tracr tasks")
from acdc.docstring.utils import get_all_docstring_things
from acdc.acdc_utils import (
    make_nd_dict,
    reset_network,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)
from acdc.acdc_graphics import (
    build_colorscheme,
    show
)

from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment
from TaskInterface import TaskInterface


torch.autograd.set_grad_enabled(False)
class ACDC(TaskInterface):
    def __init__(
        self, 
        model_name: str, 
        task: str, 
        patching_method:str, 
        metric_name:Callable, 
        N:int, 
        threshold:float, 
        remove_redundant:bool,
        out_path:str,
        zero_ablation=False,
        seed:int=1234, 
        indices_mode:str ="reversed", 
        names_mode:str = "normal",
        verbose:bool =False,
        single_step:bool = True,
        early_stop:bool = True,
        reset_network = False, 
        corrupted_cache_cpu:bool=False,
        online_cache_cpu:bool =False,
        abs_value_threshold:bool = False,
        add_sender_hooks:bool=True,
        use_pos_embed:bool=False,
        add_receiver_hooks:bool=False,
        hook_verbose:bool=False,
        second_metric:Optional[Callable]=None,
        device:str="cpu"
        ):
        super().__init__(
            model_name=model_name, 
            task=task, 
            patching_method=patching_method, 
            metric_name=metric_name, 
            N=N, 
            verbose=verbose,
            seed=seed,
            device=device)

        self.single_step = single_step
        self.early_stop = early_stop
        self.threshold = threshold
        
        self._model.reset_hooks()
        self.out_path = out_path
        self.task_folder = self.create_folder_structure()
        exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_name = f"{exp_time}_tau-{self.threshold}_N-{self.N}"
        
        if reset_network:
            acdc.utils.reset_network(task=self.task, device=self._device, model=self._model)
        
        self.exp = TLACDCExperiment(
            model=self._model,
            threshold=self.threshold,
            zero_ablation=zero_ablation,
            ds=self.clean_tokens,
            ref_ds=self.corrupted_tokens,
            metric=self.metric,
            second_metric=second_metric,
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
            abs_value_threshold = abs_value_threshold
        )
    
    def create_folder_structure(self) -> str:
        #result_folder = os.getcwd() + "/res"
        result_folder = self.out_path 
        self.create_folder(result_folder)
        return result_folder
    
    def run_exp(self, testing:bool=False, max_epochs: int = 10000):
        for i in range(max_epochs):
            self.exp.step(testing=testing, early_stop=self.early_stop)
            if i == 0:
                self.exp.save_edges(f"{self.task_folder}/{self.exp_name}_all_edges.pkl")
            
            if self.exp.current_node is None or self.single_step:
                show(self.exp.corr, f"{self.task_folder}/ACDC_{self.exp_name}.png", show_full_index=False, )
                break
            
        show(self.exp.corr, f"{self.task_folder}/{self.exp_name}.png", show_full_index=False)
        self.exp.save_edges(f"{self.task_folder}/{self.exp_name}_subgraph_edges.pkl")
        
        self.exp.save_subgraph(
            return_it=True,
            fpath=f"{self.task_folder}/{self.exp_name}_subgraph.pth"
    )
        
    def reset_network(self):
        acdc.acdc_utils.reset_network(self.task, self._device, self._model )
        
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