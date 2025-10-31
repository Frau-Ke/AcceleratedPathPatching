import torch as t
import pandas as pd
import numpy as np
import random
import dataset.docstring_prompts as prompts
from torch.utils.data import TensorDataset


class Docstring():
    def __init__(
        self, 
        model_name:str,
        N:int, 
        device:str, 
        seed:int, 
        tokenizer, 
        prepend_bos:bool=False
        ) -> None:
        self.model_name = model_name
        self.device = device
        self.N = N
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        t.manual_seed(self.seed)
        
        docstring_ind_prompt_kwargs = dict(
            n_matching_args=3, n_def_prefix_args=1, n_def_suffix_args=2, n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2
            )
        raw_prompts = [
            prompts.docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=i, tokenizer=tokenizer)
            for i in range(self.N)
            ]   
        
        batched_prompts = prompts.BatchedPrompts(prompts=raw_prompts, tokenizer=tokenizer, prepend_bos=prepend_bos, device=self.device)
        
        # clean tokens
        self.clean_tokens = batched_prompts.clean_tokens
        self.clean_input = tokenizer.batch_decode(self.clean_tokens, padding=True, return_tensors="pt")

        # corrupted tokens
        self.corrupted_tokens = batched_prompts.corrupt_tokens["random_random"]
        self.corrupted_input = tokenizer.batch_decode(self.corrupted_tokens)

        # get the correct and wrong answers
        self.correct_answers = batched_prompts.correct_tokens
        self.wrong_answers = batched_prompts.wrong_tokens
        
        self.answer_tokens = {}        
        self.answer_tokens["correct"] = self.correct_answers
        self.answer_tokens["wrong"] = self.wrong_answers
        
        # max_len and groups
        self.max_len = max(
            [
                len(tokenizer(prompt).input_ids)
                for prompt in self.clean_input 
            ]
        )        
        self.groups = [np.array(range(len(self.clean_tokens)))]
        
        
        # attention masks
        self.attention_mask = tokenizer(self.clean_input, padding=True, return_tensors="pt").attention_mask
        self.target_idx = t.stack((t.arange(self.clean_tokens.size(0)), t.full((self.clean_tokens.size(0),), fill_value=-1)) , dim=1)
        
        self.start = t.zeros(self.N)
        self.end = t.full((self.N,), fill_value=self.max_len)

        self.dataset = TensorDataset(
            self.clean_tokens,       # [N, seq_len], 
            self.corrupted_tokens,   # [N, seq_len]
            self.attention_mask,     # [N, seq_len]
            self.correct_answers,    # [N, 1]
            self.wrong_answers,      # [N, 13]
            self.target_idx,         # [N, 2]
        )
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx): 
        return {
            "clean_tokens": self.clean_tokens[idx],
            "corrupted_tokens": self.corrupted_tokens[idx],
            "mask": self.attention_mask[idx],
            "correct_answers": self.correct_answers[idx],
            "wrong_answers": self.wrong_answers[idx],
            "target_idx": self.target_idx[idx],
        }