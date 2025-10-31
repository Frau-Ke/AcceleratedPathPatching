from jaxtyping import Float
from torch import Tensor
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import torch as t
import torch.nn.functional as F
from utils.PatchingMetric import *
import os
from functools import partial
import pickle as pkl
import datetime
from dataset.loader import load_dataset
from utils.dataset_loader import init_metric_and_cache_average
from utils.model_loader import load_hooked_transformer, load_tokenizer
from fvcore.nn import FlopCountAnalysis

class TaskInterface:
    def __init__(
        self, 
        model_name:str , 
        task: str, 
        patching_method: str, 
        metric_name: str, 
        N: int, 
        model=None, 
        tokenizer=None,
        verbose:bool=False, 
        seed:int=1234, 
        device:str="cpu", 
        patch_mlp = False,
        calc_FLOPS = True, 
        prepend_bos = False,
        cache_dir="llm_weights"
        ) -> None:
        t.set_grad_enabled(False)
        
        self.patch_mlp = patch_mlp 
        
        self._device = device
        self.N = N
        self.task = task
        self.patching_method = patching_method
        self.verbose=verbose
        self.model_name = model_name
        self.seed=seed
        
        # metrics: FLOP, Runtime
        self.calc_FLOPS = calc_FLOPS
        self.FLOP_counter = 0
        self.n_forward_passes = 0
        self.elapsed_time = 0


        # load the model and tokenizer
        if model is None:
            self._model: HookedTransformer = load_hooked_transformer(self.model_name, self._device, self.patching_method, cache_dir=cache_dir)
        else:
            self._model = model
        
        if tokenizer is None:
            self.tokenizer=load_tokenizer(self.model_name)
        else:
            self.tokenizer = tokenizer
        
        
        # dataset
        self.dataset: Union[IOI_dataset, PairedFacts] = load_dataset(
            task=task, 
            patching_method=patching_method,
            tokenizer=self.tokenizer, 
            N=N,
            device=device,
            seed=self.seed, 
            prepend_bos=prepend_bos, 
            model_name= model_name
            )

        self.target_idx: Optional[Float[Tensor, "batch 2"]] = self.dataset.target_idx   #["batch seq_position"] at this position are the corrupted tokens
        self.clean_tokens: Int[Tensor, "batch pos-1"] = self.dataset.clean_tokens
        self.corrupted_tokens:  Int[Tensor, "batch pos-1"] = self.dataset.corrupted_tokens
        self.answer_tokens:  Int[Tensor, "batch 2"] = self.dataset.answer_tokens
        
        # cached activations
        self._model.reset_hooks(including_permanent=True)

        self.clean_logits, self.clean_cache = self._model.run_with_cache(self.clean_tokens)
        self.corrupted_logits, self.corrupted_cache = self._model.run_with_cache(self.corrupted_tokens)
        
        self.clean_distribution_average = None
        self.corrupted_distribution_average = None
        
        # metric        
        self.metric = init_metric_and_cache_average(
            clean_logits=self.clean_logits,
            corrupted_logits=self.corrupted_logits,
            task=task, 
            patching_method=self.patching_method,
            metric_name=metric_name,
            dataset=self.dataset, 
            model_name=self.model_name,
            )   
        
        if self.calc_FLOPS:
            FLOP_per_forward_pass = FlopCountAnalysis(self._model, self.clean_tokens)
            print("FLOP per forwad", FLOP_per_forward_pass)
            self.module_FLOPS = FLOP_per_forward_pass.by_module()
            self.n_forward_passes += 2
            self.FLOP_counter += 2 * FLOP_per_forward_pass.total() / 1e9
        
        self.clean_distribution_average = None
        self.corrupted_distribution_average = None
        
    
    def reset_efficency_metrics(self):
        self.FLOP_counter = 0
        self.n_forward_passes = 0
        self.elapsed_time = 0
    
 
    def tok_to_labels(self, toks) -> str:
        """ get the tokens of one prompt as labels. Only used for plotting
        Args:
            toks Int[Tensor, "1 seq"]: _description_

        Returns:
            str: list of tokens as string for plotting
        """
        strings = []
        for t in toks:
            strings.append(self._model.to_single_str_token(t.item()))
        return strings

    def create_folder(self, path:str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
    def save(self, obj, obj_name, path:str) -> None:
        self.create_folder(path)
        with open(obj_name, 'wb') as output_file:
            pkl.dump(obj, output_file)
    