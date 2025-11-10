import os
import torch as t
from torch import Tensor
import numpy as np
import einops
import pickle as pkl
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, ActivationCache
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from tqdm import tqdm
import pandas as pd
import itertools
from datetime import datetime
from utils.metrics import *
from Patching.TaskInterface import TaskInterface
from utils.visualization import heat_map_path_patching
from utils.data_io import store_df

# FLOPs and computation time
from fvcore.nn import FlopCountAnalysis
import time


class PathPatching(TaskInterface):
    """
        Find circuits in an autoregressive model.
        Iterate between tracbacking the information flow by their direct logit input 
        and evaluating their influence on the last 
        
    """
    def __init__(
        self, 
        model_name: str, 
        task: str, 
        patching_method:str, 
        metric_name:str,  
        N:int, 
        model=None,
        tokenizer=None,
        device="cpu", 
        patch_mlp=False, 
        seed=1234,
        calc_FLOPS = True, 
        prepend_bos=False,
        cache_dir="llm_weights"
        ) -> None:

        super().__init__(
            model_name=model_name,
            task=task, 
            patching_method=patching_method, 
            metric_name=metric_name,
            N=N,
            model=model, 
            tokenizer=tokenizer,
            device=device, 
            patch_mlp=patch_mlp, 
            seed=seed, 
            calc_FLOPS=calc_FLOPS, 
            prepend_bos=prepend_bos,
            cache_dir=cache_dir
            )

        self.metric_name = metric_name

    
    #----------------------------------------------------------------------------------------------------
    #                   Hook Functions
    #----------------------------------------------------------------------------------------------------

    def freeze_and_patch_mlps(
        self,
        orig_head_vector: Float[Tensor, "batch pos head_index d_head"],                   
        hook: HookPoint,      
        clean_cache: ActivationCache,
        corrupted_cache: ActivationCache,
        head_to_patch: List[int]
        ) -> Float[Tensor, "batch pos head_index d_head"]:
        """ 
        freeze ActivationCache of nodes on the value in the original cache
        patch one head with the value of the new cache
        """
        orig_head_vector[...] = clean_cache[hook.name][...]
        if hook.layer() == head_to_patch[0]:
            orig_head_vector[:, :, head_to_patch[1]] = corrupted_cache[hook.name][:, :, head_to_patch[1]]
        return orig_head_vector
   
    
    def freeze_and_patch_heads(
        self,
        orig_head_vector: Float[Tensor, "batch pos head_index d_head"],                   
        hook: HookPoint,      
        clean_cache: ActivationCache,
        corrupted_cache: ActivationCache,
        head_to_patch: List[int]
        ) -> Float[Tensor, "batch pos head_index d_head"]:
        """ 
        freeze ActivationCache of nodes on the value in the original cache
        patch one head with the value of the new cache
        """
        orig_head_vector[...] = clean_cache[hook.name][...]
        if head_to_patch[0] == None and head_to_patch[1] == None:
            orig_head_vector = corrupted_cache[hook.name]

        elif hook.layer() == head_to_patch[0]:
            if head_to_patch[1] is not None:
                orig_head_vector[:, :, head_to_patch[1]] = corrupted_cache[hook.name][:, :, head_to_patch[1]]
            else:
                orig_head_vector = corrupted_cache[hook.name]
        return orig_head_vector
      
               
    def patch_head_input(
        self,
        orig_activation: Float[Tensor, "batch pos head_idx d_head"],
        hook: HookPoint,
        patched_cache: ActivationCache,
        head_list: List[Tuple[int, int]],
        ) -> Float[Tensor, "batch pos head_idx d_head"]:
        '''
        patch multiple heads in head_list on their respective value in patched_cache
        '''
        is_mlp = False
        heads_to_patch = []
        for layer, head in head_list:
            if head == None:
                is_mlp = True
            heads_to_patch.append(head)
        heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
        if is_mlp:
            orig_activation = patched_cache[hook.name]

        else:
            orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
        return orig_activation
          
     
    #----------------------------------------------------------------------------------------------------
    #                   Patching Functions
    #-----------------------------------------------------------------------------------------------------
    
    def get_path_patch_head_to_final_resid_post(
        self,
        corrupted_cache: Optional[ActivationCache] ,
        clean_cache: Optional[ActivationCache], 
        PRUNING_CIRCUIT: Optional[dict] = None
        ) -> Float[Tensor, "layer head"]:
                
        n_head = self._model.cfg.n_heads + 1 if self.patch_mlp else self._model.cfg.n_heads        
        self._model.reset_hooks()
        results = t.zeros(self._model.cfg.n_layers, n_head, device=self._device, dtype=t.float32)
        n_forward_passes = 0

        # sender is either hook_z or mlp out
        z_pattern = lambda name:name.endswith("z")
        mlp_pattern = lambda name:name.endswith("mlp_out")
                
        # receiver is residual post of last layer
        residual_post = utils.get_act_name("resid_post", self._model.cfg.n_layers - 1)
        residual_post_filter = lambda name: name == residual_post

        # 1) record the attention patterns of all head under the original and the new distribution
        if clean_cache is None:
            _, clean_cache = self._model.run_with_cache(self.clean_tokens, names_filter=z_pattern, return_type=None)
            n_forward_passes += 1
        if corrupted_cache is None:
            _, corrupted_cache = self._model.run_with_cache(self.corrupted_tokens, names_filter=z_pattern, return_type=None)
            n_forward_passes += 1

        
        # 2) use hook functions to freeze all attention pattern, except of the one to replace, cache final result
        for (layer, head) in tqdm(list(itertools.product(
            range(self._model.cfg.n_layers), 
            range(n_head))) , disable = self.verbose
        ):  
            # skip heads where either the layer or the head is not in a given Pruning Circuit
            if PRUNING_CIRCUIT is not None:
                try:
                    if not head in PRUNING_CIRCUIT[layer]:
                        results[layer, head] = float('nan')
                        continue
                except:
                    print("layer", layer, "not in Circuit")
                    results[layer, head] = float('nan')
                    continue
            
            n_forward_passes += 1
            if self.is_mlp((layer, head)):
                hook_fn = partial(self.freeze_and_patch_heads, clean_cache=clean_cache, corrupted_cache=corrupted_cache, head_to_patch=[layer, None])
                self._model.add_hook(mlp_pattern, hook_fn, level=1)
                _, resid_cache = self._model.run_with_cache(self.clean_tokens, names_filter=residual_post_filter, return_type=None)
                
            else:
                hook_fn = partial(self.freeze_and_patch_heads, clean_cache=clean_cache, corrupted_cache=corrupted_cache, head_to_patch=[layer, head])
                self._model.add_hook(z_pattern, hook_fn, level=1)                
                _, resid_cache = self._model.run_with_cache(self.clean_tokens, names_filter=residual_post_filter, return_type=None)
                
            # 3)  apply LinearNorm and Unembedd the result
            resid_linearized = self._model.ln_final(resid_cache[residual_post])
            resid_unembedded = self._model.unembed(resid_linearized)
            # Save the results
            results[layer, head] = self.metric(logits=resid_unembedded)    

        if self.calc_FLOPS:
            FLOP_per_forward_pass = self.FLOPS_till_layer(self._model.cfg.n_layers)
            self.n_forward_passes += n_forward_passes
            self.FLOP_counter += FLOP_per_forward_pass * n_forward_passes
        
        return results


    def get_path_patch_head_to_heads(
        self,
        receiver_heads,
        receiver_input: str,
        corrupted_cache: Optional[ActivationCache] = None,
        clean_cache: Optional[ActivationCache] = None, 
        PRUNING_CIRCUIT: Optional[dict] = None
        ) -> Float[Tensor, "layer head"]:

        """ 
            Path patch from head to head:
            receiver_heads = heads where changes are to be obserced
            reveiver_input = componment too hook
        """
        self._model.reset_hooks()
    
        n_head = self._model.cfg.n_heads + 1 if self.patch_mlp else self._model.cfg.n_heads   
        
        """
        try:
            if receiver_input in ["k", "v"]:
                n_head = self._model.cfg.n_key_value_heads
        except:
            pass
        """
        z_pattern = lambda name:name.endswith("z")
        mlp_pattern = lambda name:name.endswith("mlp_out")        

        receiver_layers = set(next(zip(*receiver_heads)))
        receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
        receiver_hook_names_filter = lambda name: name in receiver_hook_names
        
        results = t.zeros(max(receiver_layers), n_head, device=self._device, dtype=t.float32)
        n_forward_passes = 0
        
        # 1) cache activation patterns under original and new distribution  
        if clean_cache is None:
            _, clean_cache = self._model.run_with_cache(self.clean_tokens, names_filter=z_pattern, return_type=None)
            n_forward_passes += 1

        if corrupted_cache is None:
            _, corrupted_cache = self._model.run_with_cache(self.corrupted_tokens, names_filter=z_pattern, return_type=None)
            n_forward_passes += 1
        # 2) patch sender node, freeze all other nodes, cache receiver node
        for (sender_layer, sender_head) in tqdm(list(itertools.product(
            range(max(receiver_layers)),
            range(n_head)))):
            if not PRUNING_CIRCUIT is None:
                try:
                    if not sender_head in PRUNING_CIRCUIT[sender_layer]:
                        results[sender_layer, sender_head] = float('nan')
                        continue
                except:
                    print("layer", sender_layer, "not in Circuit")
                    results[sender_layer, sender_head] = float('nan')
                    continue

                    
                    

            n_forward_passes += 1
            if self.is_mlp((sender_layer, sender_head)):
                hook_fn = partial(self.freeze_and_patch_heads, 
                                  clean_cache=clean_cache, 
                                  corrupted_cache=corrupted_cache, 
                                  head_to_patch=[sender_layer, None])
                self._model.add_hook(mlp_pattern, hook_fn, level=1)
                _, patched_cache = self._model.run_with_cache(self.clean_tokens, names_filter=receiver_hook_names_filter, return_type=None)
            else:
                hook_fn = partial(self.freeze_and_patch_heads, 
                                clean_cache=clean_cache, 
                                corrupted_cache=corrupted_cache, 
                                head_to_patch=[sender_layer, sender_head])
                self._model.add_hook(z_pattern, hook_fn, level=1)
                _, patched_cache = self._model.run_with_cache(self.clean_tokens, 
                                                        names_filter=receiver_hook_names_filter, 
                                                        return_type=None)
            # 3) forward pass with inserting patched activations
            hook_fn = partial(self.patch_head_input, 
                            patched_cache=patched_cache, 
                            head_list=receiver_heads)	
            patched_logits = self._model.run_with_hooks(
                self.clean_tokens,
                fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
                return_type="logits"
                )
            
            # Save the results
            results[sender_layer, sender_head] = self.metric(logits=patched_logits)    
            
        if self.calc_FLOPS:
            FLOP_per_forward_pass = self.FLOPS_till_layer(max(receiver_layers))
            self.n_forward_passes += n_forward_passes
            self.FLOP_counter += FLOP_per_forward_pass * n_forward_passes

        return results 
      
    
    #----------------------------------------------------------------------------------------------------
    #                   Automatic Path Patching
    #----------------------------------------------------------------------------------------------------
    
    def patch_whole_graph(
        self,        
        PRUNING_CIRCUIT:dict=None, 
        subfolder:str=None, 
        importance_threshold:float=1,
        min_value_threshold:float=0.02,
        resid_importance_threshold:float=2,

        save_every_x_steps:int=2, 
        verbose:bool=False, 
        print_vals:bool=True, 
        
        PP_information:dict=None,
        old_results:dict=None, 

        save_img:bool=False, 
        show_img:bool=True, 
        ):
        
        """
        path path through the whole model.
        Repeat while there are still unseen receiver nodes:
            1.) Patch to one receiver node (at beginneing this is resid post) from sender components (z or mlp_out) from previous layers
            2.) Find important (significant influence on metric) sender heads 
            3.) z component directly influenced by qkv and mlp_out by mlp_pre. Define qkv/mlp_pre component of important heads as temporal receivers
            4.) path path to temporal receivers from z/mlp_out components of previous layers
            5.) Define important sender to temporal receivers as new receivers
         """

        total_n_heads = self._model.cfg.n_heads * self._model.cfg.n_layers
        
        # continue from intermediate results 
        start_time = time.time() 
        idx = 0   
             
        # load old, intermediate information, if given
        if PP_information is not None:
            CIRCUIT, receivers, visited_heads = PP_information.values()
        else:
            CIRCUIT={}     
            receivers, visited_heads = [], []
        
        if old_results is not None:
            if len(old_results[0].values()) == 3:
                FLOP_counter_old, n_foward_passes_old, comp_time_old = old_results[0].values()
            else: 
                print("Loading old results, but apperently this job is done!")
            print("done loading")
            self.FLOP_counter += FLOP_counter_old
            self.n_forward_passes += n_foward_passes_old
            self.elapsed_time += comp_time_old
        
        
        # if no intermediate results: start by patching to the residual post component and collect senders
        if len(receivers) == 0: 
            metric_diff = self.get_path_patch_head_to_final_resid_post(
                    corrupted_cache=self.corrupted_cache, 
                    clean_cache=self.clean_cache, 
                    PRUNING_CIRCUIT=PRUNING_CIRCUIT
                    ).cpu()
            
            metric_diff_no_nan = t.nan_to_num(metric_diff, nan=0.0)
            
            senders = self.get_important_heads_distance_variance_threshold(
                metric_diff_no_nan,  
                importance_threshold=resid_importance_threshold, 
                min_value_threshold=min_value_threshold, 
                verbose=verbose
                )
            if PRUNING_CIRCUIT is not None:
                senders = self.filter_senders(senders, PRUNING_CIRCUIT)
            receivers.extend(senders)

            if show_img or save_img:
                self.heatmap_img(
                    metric_diff, 
                    subfolder + "/heatmap", 
                    "resid_post",
                    (self._model.cfg.n_layers-1, None), 
                    senders=senders, 
                    print_scores=print_vals,
                    save_img=save_img, 
                    show_img=show_img
                    )   
            
        while len(receivers) > 0:
            
            if len(receivers) + len(visited_heads) > total_n_heads*0.75:
                print("too many heads to be relevant circuit. Return current circuit")
                end_time = time.time()  
                self.elapsed_time += end_time - start_time
                start_time=end_time
                
                for layer, head in receivers:
                    if layer not in CIRCUIT:
                        CIRCUIT[layer] = []
                    if head not in CIRCUIT[layer]:
                        CIRCUIT[layer].append(head)                
                
                PP_information = {"CIRCUIT": CIRCUIT, "receivers":receivers, "visited_heads":visited_heads}                
                self.save_PP_information(PP_information=PP_information, subfolder=subfolder, name="break_intermediate.pkl")
                
                results = {
                    "GFLOP":[self.FLOP_counter],
                    "n_forward_passes":[self.n_forward_passes], 
                    "comp_time": [self.elapsed_time]
                }
                
                results = pd.DataFrame(data=results)                
                store_df(results, subfolder, "break_results.json")
                return CIRCUIT    
            
            idx += 1     

            if idx % save_every_x_steps == 0:
                end_time = time.time()  
                #print("self elapsedt time before saving", self.elapsed_time)
                self.elapsed_time += end_time - start_time
                start_time=end_time
                    
                PP_information = {"CIRCUIT": CIRCUIT, "receivers":receivers, "visited_heads":visited_heads}                
                self.save_PP_information(PP_information=PP_information, subfolder=subfolder, name="intermediate.pkl")
                
                results = {
                    "GFLOP":[self.FLOP_counter],
                    "n_forward_passes":[self.n_forward_passes], 
                    "comp_time": [self.elapsed_time]
                }
                
                results = pd.DataFrame(data=results)                
                store_df(results, subfolder, "results.json")
            
            receiver_head = receivers.pop(0)
            
            # add receivers head to the circuit
            try:
                if not receiver_head[1] in CIRCUIT[receiver_head[0]]:
                    CIRCUIT[receiver_head[0]].append(receiver_head[1])
            except:
                # no head yet in this layer
                CIRCUIT[receiver_head[0]] = [receiver_head[1]]
        
            
            if  receiver_head in visited_heads: 
                continue
            
            if receiver_head[0] == 0:
                print("skipt zero layers", receiver_head)
                continue

            visited_heads.append(receiver_head)
            
            if verbose:
                print(f"######## receiver head {receiver_head}")
                print("visited heads", visited_heads)
                print("receiver heads", receivers)
            
            for receiver_name in "qkv":
                if verbose:
                   print("receiver name", receiver_name)
                    
                metric_diff = self.get_path_patch_head_to_heads(
                    receiver_input=receiver_name,
                    receiver_heads=[receiver_head],
                    corrupted_cache=self.corrupted_cache, 
                    clean_cache=self.clean_cache, 
                    PRUNING_CIRCUIT=PRUNING_CIRCUIT
                    ).cpu()
                metric_diff_no_nan = t.nan_to_num(metric_diff, nan=0.0)

                # these attention heads have an significant influence on the output  -> all z-component
                senders = self.get_important_heads_distance_variance_threshold(
                    metric_diff_no_nan,
                    importance_threshold=importance_threshold, 
                    min_value_threshold=min_value_threshold,
                    verbose=verbose
                    )
                
                if PRUNING_CIRCUIT is not None:
                    senders = self.filter_senders(senders, PRUNING_CIRCUIT)
                
                if len(senders) > 0:
                    if show_img or save_img:
                        self.heatmap_img(
                            metric_diff, 
                            subfolder + "/heatmap", 
                            receiver_name, 
                            receiver_head, 
                            senders=senders, 
                            print_scores=print_vals, 
                            save_img=save_img, 
                            show_img=show_img
                            ) 
                
                senders = [head for head in senders if head not in visited_heads and head not in receivers]

    
                if len(senders) > 0:                    
                    receivers.extend(senders)

                
        end_time = time.time()  
        self.elapsed_time += end_time - start_time
        
        return CIRCUIT
    
    
    def patch_whole_graph_qwen(
        self,        
        PRUNING_CIRCUIT:dict=None, 
        subfolder:str=None, 
        importance_threshold:float=1,
        min_value_threshold:float=0.02,
        resid_importance_threshold:float=2,

        save_every_x_steps:int=2, 
        verbose:bool=False, 
        print_vals:bool=True, 
        
        PP_information:dict=None,
        old_results:dict=None, 

        save_img:bool=False, 
        show_img:bool=True, 
        ):
        
        """
        path path through the whole model.
        Repeat while there are still unseen receiver nodes:
            1.) Patch to one receiver node (at beginneing this is resid post) from sender components (z or mlp_out) from all previous layers
            2.) Find important (significant influence on metric) sender heads 
            3.) z component directly influenced by qkv and mlp_out by mlp_pre. Define qkv/mlp_pre component of important heads as new receivers
            4.) path path to new receivers from z/mlp_out components of all previous layers
         """

        total_n_heads = self._model.cfg.n_heads * self._model.cfg.n_layers

        start_time = time.time() 

        group_size = int(self._model.cfg.n_heads / self._model.cfg.n_key_value_heads)
        if verbose:
            print(f"for Qwen model with {self._model.cfg.n_heads} q-kheads and {self._model.cfg.n_key_value_heads} k/v-heads the group size is {group_size}")

        # load old, intermediate information, if given
        if PP_information is not None:
            print("load old information")
            CIRCUIT, receivers, visited_heads, visited_kv_heads = PP_information.values()
            print("done loading")
        else:
            CIRCUIT={}     
            receivers, visited_heads, visited_kv_heads = [], [], []
        
        if old_results is not None:
            print("loading old results")
            if len(old_results[0].values()) == 3:
                FLOP_counter_old, n_foward_passes_old, comp_time_old = old_results[0].values()
            else: 
                FLOP_counter_old, n_foward_passes_old, comp_time_old, _, _ = old_results[0].values()
            print("done loading old results")

    
            self.FLOP_counter += FLOP_counter_old
            self.n_forward_passes += n_foward_passes_old
            self.elapsed_time += comp_time_old
        
        # if no intermediate results: start by patching to the residual post component from the z-component of all layers, collect senders
        if len(receivers) == 0: 
            
            # patch to receiver
            metric_diff = self.get_path_patch_head_to_final_resid_post(
                    corrupted_cache=self.corrupted_cache, 
                    clean_cache=self.clean_cache, 
                    PRUNING_CIRCUIT=PRUNING_CIRCUIT
                    ).cpu()
            metric_diff_no_nan = t.nan_to_num(metric_diff, nan=0.0)
            
            # get senders
            senders = self.get_important_heads_distance_variance_threshold(
                metric_diff_no_nan, 
                importance_threshold=resid_importance_threshold, 
                min_value_threshold=min_value_threshold, 
                verbose=verbose
                )
            if PRUNING_CIRCUIT is not None:
                senders = self.filter_senders(senders, PRUNING_CIRCUIT)
            
            # senders become receivers
            receivers.extend(senders)

            if save_img or show_img:
                self.heatmap_img(
                    metric_diff, 
                    subfolder + "/heatmap", 
                    "resid_post",
                    (self._model.cfg.n_layers-1, None), 
                    senders=senders, 
                    print_scores=print_vals,
                    save_img=save_img, 
                    show_img=show_img
                    )   
            
        save_idx = 0
        while len(receivers) > 0:
            if len(receivers) + len(visited_heads) > total_n_heads:
                end_time = time.time()  
                self.elapsed_time += end_time - start_time
                start_time=end_time
                
                for layer, head in receivers:
                    if layer not in CIRCUIT:
                        CIRCUIT[layer] = []
                    if head not in CIRCUIT[layer]:
                        CIRCUIT[layer].append(head)                
                
                PP_information = {"CIRCUIT": CIRCUIT, "receivers":receivers, "visited_heads":visited_heads}                
                self.save_PP_information(PP_information=PP_information, subfolder=subfolder, name="break_intermediate.pkl")
                
                results = {
                    "GFLOP":[self.FLOP_counter],
                    "n_forward_passes":[self.n_forward_passes], 
                    "comp_time": [self.elapsed_time]
                }
                
                results = pd.DataFrame(data=results)                
                store_df(results, subfolder, "break_results.json")
                return CIRCUIT    
            
            # save intermediate result every x steps
            save_idx += 1     
            if save_idx % save_every_x_steps == 0:
                end_time = time.time()  
                self.elapsed_time += end_time - start_time
                start_time=end_time
                
                PP_information = {"CIRCUIT": CIRCUIT, "receivers":receivers, "visited_heads":visited_heads, "visited_kv_heads": visited_kv_heads}                
                self.save_PP_information(PP_information=PP_information, subfolder=subfolder, name="intermediate.pkl")
                
                results = {
                    "GFLOP":[self.FLOP_counter],
                    "n_forward_passes":[self.n_forward_passes], 
                    "comp_time": [self.elapsed_time]
                }
                
                results = pd.DataFrame(data=results)                
                store_df(results, subfolder, "results.json")
            
            
            # get new receiver head
            receiver_head = receivers.pop(0)
            
            # add receivers head to the circuit
            try:
                if not receiver_head[1] in CIRCUIT[receiver_head[0]]:
                    CIRCUIT[receiver_head[0]].append(receiver_head[1])
            except:
                # no head yet in this layer
                CIRCUIT[receiver_head[0]] = [receiver_head[1]]
        
            # skip, if alreadyy patched to or zero layer -> no senders
            if  receiver_head in visited_heads: 
                continue
            
            if receiver_head[0] == 0:
                print("skipt zero layers", receiver_head)
                continue

            visited_heads.append(receiver_head)
            
            if verbose:
                print(f"######## receiver head {receiver_head}")
                print("visited heads", visited_heads)
                print("receiver heads", receivers)
            
            # all qkv components of the receiver head are receiver components
            for receiver_name in "qkv":
                if verbose:
                   print("receiver name", receiver_name)


                if receiver_name in["k", "v"]:
                    # kv component
                    # only for GQA models: k/v heads are grouped -> map head to the fitting k/v head
                    grouped_receiver_head = (receiver_head[0], int(receiver_head[1]/group_size))
                    if grouped_receiver_head in visited_kv_heads:
                        if verbose:
                            print(f"receiver head {grouped_receiver_head} already patched to for component {receiver_name}")
                        continue
                    
                    visited_kv_heads.append(grouped_receiver_head)
                    
                    if verbose:
                        print("original receiver head", receiver_head, "and its grouped equivalent", grouped_receiver_head)
                    if grouped_receiver_head[1] < 0 or grouped_receiver_head[1] >=  self._model.cfg.n_key_value_heads:
                        raise Exception("Something went wrong trying to map the q-heads to the k-heads. Remeber for Grouped Query Attention models like Qwen mutiple query heads are mapped to some k- or v heads")
                    
                    # patch to receiver
                    metric_diff = self.get_path_patch_head_to_heads(
                        receiver_input=receiver_name,
                        receiver_heads=[grouped_receiver_head],
                        corrupted_cache=self.corrupted_cache, 
                        clean_cache=self.clean_cache, 
                        PRUNING_CIRCUIT=PRUNING_CIRCUIT
                        ).cpu()
                                        
                    metric_diff_no_nan = t.nan_to_num(metric_diff, nan=0.0)
                    # get senders     
                    senders = self.get_important_heads_distance_variance_threshold(
                        metric_diff_no_nan,
                        importance_threshold=importance_threshold, 
                        min_value_threshold=min_value_threshold,
                        verbose=verbose
                        )
                                        
                    if PRUNING_CIRCUIT is not None:
                        senders = self.filter_senders(senders, PRUNING_CIRCUIT)
                    
                    # plot and save image
                    if len(senders) > 0:
                        if save_img or show_img:
                            self.heatmap_img(
                                metric_diff, 
                                subfolder + "/heatmap", 
                                receiver_name, 
                                grouped_receiver_head, 
                                senders=senders, 
                                print_scores=print_vals, 
                                save_img=save_img, 
                                show_img=show_img
                                ) 
                        
                    
                else:
                    # q_heads
                    # patch to receiver
                    metric_diff = self.get_path_patch_head_to_heads(
                        receiver_input=receiver_name,
                        receiver_heads=[receiver_head],
                        corrupted_cache=self.corrupted_cache, 
                        clean_cache=self.clean_cache, 
                        PRUNING_CIRCUIT=PRUNING_CIRCUIT
                        ).cpu()
                    
                    metric_diff_no_nan = t.nan_to_num(metric_diff, nan=0.0)
                    # get senders
                    senders = self.get_important_heads_distance_variance_threshold(
                        metric_diff_no_nan,
                        importance_threshold=importance_threshold, 
                        min_value_threshold=min_value_threshold,
                        verbose=verbose
                        )                    
                    
                    if PRUNING_CIRCUIT is not None:
                        senders = self.filter_senders(senders, PRUNING_CIRCUIT)
                    
                    # plot and save image
                    if len(senders) > 0:
                        self.heatmap_img(
                            metric_diff, 
                            subfolder + "/heatmap", 
                            receiver_name, 
                            receiver_head, 
                            senders=senders, 
                            print_scores=print_vals, 
                            save_img=save_img, 
                            show_img=show_img
                            ) 
                    
              # senders become receivers
                senders = [head for head in senders if head not in visited_heads and head not in receivers]
                if len(senders) > 0:                    
                    receivers.extend(senders)

                
        end_time = time.time()  
        self.elapsed_time += end_time - start_time
        
        return CIRCUIT
    
    #----------------------------------------------------------------------------------------------------
    #                   Finding Sender Heads
    #----------------------------------------------------------------------------------------------------
     
    def get_important_heads_distance_variance_threshold(
        self, 
        metric_diff:Float[Tensor, "layer head"], 
        alpha=0.1, 
        mode="linear", 
        importance_threshold=2, 
        min_value_threshold=0.02,
        verbose=False
        ) -> List:
        
        num_layer, num_head = metric_diff.shape
        target_layer = num_layer
    
        mean_activation = t.nanmean(metric_diff)
        sd_activation = metric_diff[~t.isnan(metric_diff)].std()
        new_heads = []
        
        # maximum value threshold
        max_activation = t.max(t.abs(metric_diff[~t.isnan(metric_diff)]))
        if max_activation <= min_value_threshold:
            if verbose:
                print("max activation is really small:", max_activation)
            return []
        
        for layer in range(num_layer):
            distance = abs((layer + 1) - target_layer)
            
            if mode=="linear":
                base_threshold = importance_threshold + alpha * distance
            elif mode=="exp":
                base_threshold = importance_threshold * t.exp(t.tensor(alpha * distance))
            elif mode=="log":
                base_threshold = importance_threshold * (1 + alpha * t.log(t.tensor(distance)))
            elif mode == "sqrt":
                # importance_threshold
                n =  metric_diff[:layer + 1, :][~t.isnan(metric_diff[:layer + 1, :])].numel() # count all not ablated heads fpf previous layers
                base_threshold =   (importance_threshold + 2 / np.sqrt(n))
            else:
                base_threshold = importance_threshold
            
            threshold = base_threshold *  t.abs(sd_activation)
            
            for head in range(num_head):
                diff = t.abs(metric_diff[layer][head]) - t.abs(mean_activation)
                # comment in, if no negative activation heads
                #diff = -metric_diff[layer][head] - t.abs(mean_activation)

                if diff > threshold:
                    new_heads.append( (layer, head))
        return new_heads 
    
    def filter_senders(self, senders: list, PRUNING_CIRCUIT:dict):
        filtered_senders = []
        for layer, head in senders:
            if head in PRUNING_CIRCUIT.get(layer):
                filtered_senders.append((layer, head))
        return filtered_senders
    
    
    #----------------------------------------------------------------------------------------------------
    #                   FLOPs Calculation
    #----------------------------------------------------------------------------------------------------

    def FLOPS_till_layer(self, layer):
        filtered_dict={}
        for i in range(layer):
            name="blocks." + str(i)
            filtered_dict[name] = self.module_FLOPS[name]
        filtered_dict["unembed"] = self.module_FLOPS["unembed"]
        sum_filtered_FLOP = sum(filtered_dict.values())
        return sum_filtered_FLOP  / 1e9
          
    
    #----------------------------------------------------------------------------------------------------
    #                   Plotting and Saving
    #----------------------------------------------------------------------------------------------------

    def heatmap_img(self, metric_diff, subfolder, receiver, receiver_head, senders=[], print_scores=True, show_img=False, save_img=False):
        
        if receiver == "resid_post":
            title = f"receiver: resid_post at layer {receiver_head[0]}"
            name = f"resid_post_{receiver_head[0]}"
        elif len(receiver_head) > 1:
            title = f"receiver: attn_head_component {receiver} at {receiver_head}"
            name = f"att_head_{receiver_head}_{receiver}"
        
        elif self.is_mlp(receiver_head[0]):
            title = f"receiver: mlp_pre at layer {receiver_head[0]}"
            name = f"mlp_pre_{receiver_head[0]}"
        else:
            title = f"receiver: attn_head_component {receiver} at {receiver_head}"
            name = f"att_head_{receiver_head}_{receiver}"
        
        heat_map_path_patching(
            metric_diff, 
            title = title, 
            color_axis_title = self.metric_name, 
            show=show_img, 
            save=save_img,
            subfolder=subfolder, 
            name= name,
            senders=senders,
            print_scores=print_scores)
   
   
    def save_edge(
        self, 
        metric_diff: Float[Tensor, "layer head"],
        receiver_head: Tuple[int, int], 
        receiver: str,
        sender_heads:List[Tuple[int, int]],
        sender_qkv: Optional[str] = None
        ):
        """ Save an edge of ([receiver_name, receiver_head, sender_name, sender_head], effect_size) to self.edges
        receiver components are 'qkv' or 'mlp_pre'
        sender components are 'z' or 'mlp_out'        
        """
        receiver_name = utils.get_act_name(receiver, receiver_head[0])
        if  receiver == "resid_post" or self.is_mlp(receiver_head):
            receiver_index = TorchIndex([None])
        else:
            receiver_index = TorchIndex([None, None, receiver_head[1]])
        
        for sender_head in sender_heads:    
            if not sender_qkv:
                sender = self.get_sender_component(sender_head)   
            else:
                sender = sender_qkv  

            sender_name = utils.get_act_name(sender, sender_head[0])
            if sender == "mlp_out": 
                sender_index = TorchIndex([None])
            else:
                sender_index = TorchIndex([None, None, sender_head[1]])
            
            effect_size = metric_diff[sender_head[0], sender_head[1]].item()
            entry =[[receiver_name, receiver_index, sender_name, sender_index], effect_size]
            #print("new entry", entry)
            self.edges.append(entry)
       
                    
    def save_PP_information(self, PP_information, subfolder, name):
        with open(os.path.join(subfolder, name), "wb") as f:
            pkl.dump(PP_information, f)

    #----------------------------------------------------------------------------------------------------
    #                   Helper Functions
    #----------------------------------------------------------------------------------------------------

    def extend_receiver_list(
        self,
        orig_receivers: List[Tuple[str, Tuple[int, int]]], 
        new_receivers: List[Tuple[int, int]], 
        visited_heads: List[Tuple[str, Tuple[int, int]]], 
        new_receiver_name="z"
        )-> List[Tuple[str, Tuple[int, int]]]:
        """ Add new receivers to the orig_receiver list iff they are not present in the orig_receiver list or in the visited_heads list"""
        for nr in new_receivers:
            if self.is_mlp(nr):
                if ("mlp_pre", nr) not in orig_receivers and ("mlp_pre", nr) not in visited_heads:
                    orig_receivers.extend([("mlp_pre", nr)])
            else:
                if (new_receiver_name, nr) not in orig_receivers and (new_receiver_name, nr) not in visited_heads:
                    orig_receivers.extend([(new_receiver_name, nr)])
        return orig_receivers


    def has_duplicate(self, lst):
        return len(lst) != len(set(lst))


    def get_sender_component(self, head:Union[Tuple[int, int], Tuple[int, None]]):
        """return mlp_out if head is an mlp, else return z"""
        
        if self.is_mlp(head):
            return "mlp_out"
        else:
            return "z"
      
      
    def is_mlp(self, head:Union[Tuple[int, int], Tuple[int, None]]) -> bool:
        """if head position == number of model heads, the head is an mlp
            (e.g. model with 12 heads. 0-11 are attention heads, 12 is mlp)
        """
        return True if head[1] == self._model.cfg.n_heads else False
        