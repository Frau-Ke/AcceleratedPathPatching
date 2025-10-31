import torch
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

class Dataset():
    def __init__(N:int, tokenizer, device:str="cuda", seed:int=1234, prepend_bos:bool=False):
        """To make any dataset usable with the rest of the code, the listed class variables of Dataset() 
        must be overwritten/implemented by every new subclass

        Args:
            N (int): number of samples
            tokenizer (ModelTokenizer): tokenizer of the model
            device (str, optional): device. Defaults to "cuda".
            seed (int, optional): Defaults to 1234.
            prepend_bos (bool, optional): Append a BOS token at the beginning of the sample. Defaults to False.
        """
        
        self.N = N
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed
        
        self.clean_tokens:Float[Tensor, "batch seq_len"] = None
        self.corrupted_tokens: Float[Tensor, "batch seq_len"] = None
        self.answer_tokens: Float[Tensor, "batch 2"] = None
        self.target_idx: Float[Tensor, "batch 2"] = None
        self.groups: list(array) =  None
        self.max_len: int = 0
        self.attention_mask: Float[Tensor, "batch seq_len"] = None