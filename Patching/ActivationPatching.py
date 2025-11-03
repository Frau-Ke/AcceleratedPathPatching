import torch as t
from torch import Tensor
import numpy as np
import einops
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from tqdm import tqdm
import itertools
#from utils.metrics import *
#from utils.visualization import *
from Patching.TaskInterface import TaskInterface


class ActivationPatching(TaskInterface):
    """
        Find circuits in an autoregressive model.
        Iterate between tracbacking the information flow by their direct logit input 
        and evaluating their influence on the last 
    """
    def __init__(self, model_name: str, task: str, patching_method:str, metric_name:Callable, N:int, hook_position:str, device="cpu") -> None:
        super().__init__(model_name=model_name, task=task, patching_method=patching_method, metric_name=metric_name, N=N, device=device)
        self.hook_position = hook_position
     
    # Hook Function
    def patch_residual_component(
        self,
        corrupted_residual_component: Float[Tensor, "batch pos d_model"],
        hook: HookPoint,
        pos: int,
    ) -> Float[Tensor, "batch pos d_model"]:
        '''
        Patches a given sequence position in the residual stream, using the value
        from the clean cache.
        '''
        patched = corrupted_residual_component.clone()
        patched[:, pos, :] =  self.clean_cache[hook.name][:, pos, :]
        return patched
    
    # Hook Function  
    def patch_head_vector(
        self,
        corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
        head_index: int,
    ) -> Float[Tensor, "batch pos head_index d_head"]:
        '''
        Patches the output of a given head (before it's added to the residual stream) at
        every sequence position, using the value from the clean cache.
        '''
        patched = corrupted_head_vector.clone()
        patched[:, :, head_index] = self.clean_cache[self.hook_position, hook.layer()][:, :, head_index]
        return patched
   
    def get_act_patch_resid_pre(self) -> Float[Tensor, "layer pos"]:
        '''
        Returns an array of results of patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        '''
        self._model.reset_hooks()
        _, seq_len = self.corrupted_tokens.shape
        results = t.zeros((self._model.cfg.n_layers, seq_len)).to(self._device)
        for (layer, pos) in tqdm(list(itertools.product(
            range(self._model.cfg.n_layers), 
            range(seq_len)))
        ):
            hook_name = utils.get_act_name("resid_pre", layer)
            hook_fn = partial(self.patch_residual_component, pos=pos)
            pre_residual_patching_result = self._model.run_with_hooks(self.corrupted_tokens, fwd_hooks=[(hook_name, hook_fn)])

            # save results:
            results[layer, pos] = self.metric(logits=pre_residual_patching_result)
        return results
       
    def get_act_patch_attn_head_out_all_pos(self) -> Float[Tensor, "layer head"]:
        '''
        Returns an array of results of patching at all positions for each head in each
        layer, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        '''
        self._model.reset_hooks()
        results = t.zeros((self._model.cfg.n_layers, self._model.cfg.n_heads)).to(self._device)
        for (layer, head) in tqdm(list(itertools.product(
            range(self._model.cfg.n_layers),
            range(self._model.cfg.n_heads)
        ))):
            hook_name = utils.get_act_name(self.hook_position, layer)
            hook_fn = partial(self.patch_head_vector, head_index=head)
            patched_logits = self._model.run_with_hooks(self.corrupted_tokens, 
                                                        fwd_hooks=[(hook_name, hook_fn)], 
                                                        return_type="logits")
            
            # save results:
            results[layer, head] = self.metric(logits=patched_logits)            
        return results
    
    def act_patch_per_position_per_componment(
        self,
    ) -> Float[Tensor, "layer pos"]:
        '''
        Returns an array of results of patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        '''
        self._model.reset_hooks()
        results = t.zeros(self._model.cfg.n_layers, self.corrupted_tokens.size(1), device=self._device, dtype=t.float32)
        
        for layer in tqdm(range(self._model.cfg.n_layers)):
            for position in range(self.corrupted_tokens.shape[1]):
                hook_fn = partial(self.patch_residual_component, pos=position)
                patched_logits = self._model.run_with_hooks(
                    self.corrupted_tokens,
                    fwd_hooks = [(utils.get_act_name(self.hook_position, layer), hook_fn)],
                )
                results[layer, position] = self.metric(logits=patched_logits)

        return results