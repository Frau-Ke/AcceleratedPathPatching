import huggingface_hub
import torch
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import numpy as np
from utils.dataset_loader import start_of_prompt, end_of_prompt



def shuffle_tensor(tens, seed=42):
    """Shuffle tensor along first dimension"""
    torch.random.manual_seed(seed)
    return tens[torch.randperm(tens.shape[0])]


def word_idx_dict(
    prompts, 
    tokenizer, 
    model_name:str="gpt2", 
    prepend_bos:bool=False
    ) -> dict:
    
    word_idx_dict = {}
    word_idx_dict["START"] = []
    word_idx_dict["END"] = []
    word_idx_dict["TARGET"] = []
    
    for toks in prompts:
        if "Qwen" in model_name:
            start_idx = start_of_prompt(toks, tokenizer=tokenizer, start_text="<|im_start|>user\n")
            end_idx = end_of_prompt(toks, tokenizer=tokenizer, end_text="<|im_end|>\n")
        else:
            start_idx = 0
            end_idx = prompts.size(1)
        
        word_idx_dict["START"].append(start_idx)
        word_idx_dict["END"].append(end_idx)
        word_idx_dict["TARGET"].append(end_idx-1)
        
    
    return [
        int(prepend_bos) + torch.tensor(word_idx_dict[idx_type])
        for idx_type in ["START", "END", "TARGET"]
    ]
    
    
def get_word_idx_dict(
    prompts,
    tokenizer, 
    model_name:str="gpt2", 
    prepend_bos:bool=False
    ):
    start_idx, end_idx, year_idx, target_year_idx = word_idx_dict(prompts, tokenizer, model_name, prepend_bos)

    return {
        "START":start_idx,
        "END":end_idx,
        "TARGET": year_idx,
    }

    
    


class Indcution():
    def __init__(self, N:Optional[int], seq_len:int, device:str, seed:int, model_name) -> None:
        self.N = N
        self.seq_len = seq_len
        self.device = device   
        N_range = slice(0, self.N)
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Induction task of the form "A B ... A __" -> target is B
        self.mask_repeat = self.get_corrupted_dataset()[N_range, :seq_len].contiguous()
        self.clean_tokens = self.get_clean_dataset().contiguous().to(torch.int64)
        
        if model_name != "redwood_attn_2l":
            self.clean_tokens = self.remove_one_dim(self.clean_tokens)
            self.mask_repeat = self.remove_one_dim(self.mask_repeat)
            
        self.corrupted_tokens = shuffle_tensor(self.clean_tokens, seed=self.seed).contiguous()
        self.target_idx = torch.nonzero(self.mask_repeat)
        self.answer_tokens = torch.stack((self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1]],
                                           self.corrupted_tokens[self.target_idx[:, 0], self.target_idx[:, 1]]), dim=1)

        assert self.clean_tokens.shape == self.corrupted_tokens.shape


    def get_clean_dataset(self):
        good_induction_candidates_fname = huggingface_hub.hf_hub_download(
            repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
        )
        good_induction_candidates = torch.load(good_induction_candidates_fname, map_location=self.device)
        if not self.N is None:
            return good_induction_candidates
        else:
            return good_induction_candidates[:self.N, :self.seq_len]
        
        
    def get_corrupted_dataset(self):
        mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
            repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
        )
        mask_repeat_candidates = torch.load(mask_repeat_candidates_fname, map_location=self.device)
        mask_repeat_candidates.requires_grad = False

        if not self.N is None:
            return mask_repeat_candidates
        else:
            return mask_repeat_candidates[:self.N, :self.seq_len]
        
    def remove_one_dim(self, dataset):
        """replace the costume [END] and [BEGIN] token of this dataset by the standard [endoftext] token"""
        for sentence in dataset:
            for word in range(len(sentence)):
                if sentence[word] == 50258 or sentence[word] == 50257: 
                    sentence[word] = 50256
        return dataset

