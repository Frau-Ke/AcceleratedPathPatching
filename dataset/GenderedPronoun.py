import torch
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import random
import numpy as np
from utils.dataset_loader import start_of_prompt, end_of_prompt
from torch.utils.data import TensorDataset

templates = [
    "So [NAME] is a really great friend, isn't",
    "So [NAME] is such a good cook, isn't", 
    "So [NAME] is a very good athlete, isn't",
    "So [NAME] is really nice, isn't",
    "So [NAME] is always full of energy, isn't",
    "So [NAME] is really thoughtful, isn't",
    "So [NAME] is quite the storyteller, isn't",
    "So [NAME] is a bit of a genius, isn't",
    "So [NAME] is always on time, isn't",
    "So [NAME] is great with kids, isn't",
    "So [NAME] is incredibly patient, isn't",
    "So [NAME] is a natural leader, isn't",
    "So [NAME] is really creative, isn't",
    "So [NAME] is a really good student, isn't", 
    "So [NAME] is really clever, isnt't"
    ]

male_names = [
    "John",
    "David",
    "Mark",
    "Paul",
    "Ryan",
    "Gary", #X
    "Jack",
    "Sean",
    "Carl",
    "Joe",    
]
female_names = [
    "Mary",
    "Lisa",
    "Anna",
    "Sarah",
    "Amy",
    "Carol",
    "Karen",
    "Susan",
    "Julie",
    "Judy"
]

# TODO: check prediction probability for he and she in context with neutral names
# should be ~50:50 with other 
"""
neutral_names = [
    "Pat", 
    "Kit",
    "Sky", 
    "Lux",
    "Ash"
]
"""
neutral_names = [
    "Person"
]


#neutral_names = ["person"]

responses = ['he', 'she']

def filter_names(names: List[str], tokenizer) -> List[str]:
    """
    Filter for names which have exactly one tokens when tokenized
    """
    NAMES = []
    for name in names:
        toks = tokenizer.encode(" " + name)
        if len(toks) == 1:
            NAMES.append(name)
    return NAMES



def gen_clean_prompts(templates: list[str], male_names:list[str], female_names:list[str], nouns_dict:dict=None, N:int=40):
    """generate random prompts from the templates, names and nouns_dict.
    In general, templates are of the form [NAME] [VERB] ... [NOUN]. [NAME]
    Names are filtered, s.t only names that are tokenized with two tokens are included.
    
    Args:
        templates (list[str]): List of templates
        names (list[str]): List of names
        nouns_dict (dict): Dictionary of the form {"PLACES": list[PLACES], "TIMES": list[TIMES], "OBJECTS": list[OBJECTS]}
        N (int): number of prompts to generate
    """
    n_gen = 0
    prompts = []
    while n_gen < N:
        temp = random.choice(templates)
        temp_id = templates.index(temp)
        
        if random.random() < 0.5:
            # male example
            name = random.choice(male_names)
            pronoun = responses[0]
        else:
            # female example
            name = random.choice(female_names)
            pronoun = responses[1]
        
        prompt = temp
        induction_prompt = {}
    
        prompt = prompt.replace("[NAME]", name)

        induction_prompt["text"] = prompt + " " + pronoun
        induction_prompt["name"] = name
        induction_prompt["pronoun"] = pronoun
        induction_prompt["template_idx"] = temp_id
        prompts.append(induction_prompt)
        n_gen += 1
    return prompts

def gen_corrupted_prompts(prompts: list[dict], tokenizer, templates: list[str], neutral_names:list[str], prepend_bos: bool=False):

    corrupted_pronmpts = []
    for prompt in prompts:
        template_idx = prompt["template_idx"]
        correct_pronounn = prompt["pronoun"]
        wrong_pronoun = responses[1] if correct_pronounn == responses[0] else responses[0]
        temp = templates[template_idx]
        
        prompt = temp
        neutral_name = random.choice(neutral_names)
        prompt = prompt.replace("[NAME]", neutral_name)
        prompt_split = prompt.split(" ")[1:]
        prompt_split.insert(0, "That")
        prompt=" ".join(prompt_split)
        
        corrupted_prompt = {}
        corrupted_prompt["text"] = f"{prompt} {correct_pronounn}" 
        corrupted_prompt["name"] = neutral_name
        corrupted_prompt["pronoun"] = wrong_pronoun
        corrupted_prompt["template_idx"] = template_idx
        corrupted_pronmpts.append(corrupted_prompt)
    return corrupted_pronmpts
        

def get_name_idx(prompts, tokens, tokenizer, idx_types=["START", "END", "NAME", "PRONOUN"]):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)
    she_toks = tokenizer(" she").input_ids[0]
    he_toks =  tokenizer(" he").input_ids[0]

    for prompt, toks in zip(prompts, tokens):
        
        start_idx = start_of_prompt(toks, tokenizer=tokenizer, start_text="<|im_start|>user\n")
        end_idx = end_of_prompt(toks, tokenizer=tokenizer, end_text="<|im_end|>\n")    

        name = tokenizer(" " + prompt["name"]).input_ids[0]
        name_idx = torch.nonzero(toks == name).item()
        pronoun_idx = torch.nonzero(toks == she_toks) if torch.nonzero(toks == he_toks).size(0) == 0 else torch.nonzero(toks == he_toks)
        
        name_idx_dict["START"].append(start_idx)
        name_idx_dict["END"].append(end_idx)
        name_idx_dict["NAME"].append(name_idx)
        name_idx_dict["PRONOUN"].append(pronoun_idx - 1)
        
    return [
        torch.tensor(name_idx_dict[idx_type])
        for idx_type in idx_types
    ]

def get_idx_dict(prompts, tokens, tokenizer):
    start_idx, end_idx, name_idx, pronoun_idx = get_name_idx(prompts, tokens, tokenizer)
    return {
        "START":start_idx,
        "END": end_idx,
        "NAME": name_idx,
        "PRONOUN": pronoun_idx
    }

class GenderedPronoun():
    def __init__(
        self, 
        model_name:str, 
        N:int, 
        tokenizer, 
        device:str="cuda", 
        seed:int=1234, 
        prepend_bos:bool=False
        ):
        
        print(f"loading GenderedPronouns dataset for {model_name}")
        
        self.N = N
        self.device = device
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        male_names_filtered = filter_names(names=male_names, tokenizer=tokenizer)
        female_names_filtered = filter_names(names=female_names, tokenizer=tokenizer)
        neutral_names_filtered = filter_names(names=neutral_names, tokenizer=tokenizer)
        

        # get the clean prompts, input and tokens
        self.clean_prompts = gen_clean_prompts(templates, male_names_filtered, female_names_filtered, N=self.N)
        
        self.clean_input = [prompt["text"] for prompt in self.clean_prompts]        
        
        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt
            for prompt in self.clean_input
        ]
        
        #if "Qwen" in model_name:
        if False:
            texts_clean = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for prompt in texts
            ]
        else:
            texts_clean = texts

        
        self.clean_tokens = torch.Tensor(self.tokenizer(texts_clean, padding=True).input_ids).long().to(self.device)
        correct_pronount_tokenIDs = [self.tokenizer.encode(" " + prompt["pronoun"])[0] for prompt in self.clean_prompts]
        # get the corrupted prompts, input and tokens
        
        self.corrupted_prompts = gen_corrupted_prompts(self.clean_prompts, self.tokenizer, templates, neutral_names= neutral_names_filtered, prepend_bos=prepend_bos)
        self.corrupted_input = [prompt["text"] for prompt in self.corrupted_prompts]

        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt
            for prompt in self.corrupted_input
        ]

        #if "Qwen" in model_name:
        if False:

            texts_corrupted = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                    ) for prompt in texts
                ]
        else: 
            texts_corrupted = texts
        
        self.corrupted_tokens = torch.Tensor(self.tokenizer(texts_corrupted, padding=True).input_ids).long().to(self.device)
        wrong_pronoun_tokedIDs = [self.tokenizer.encode(" " + prompt["pronoun"])[0] for prompt in self.corrupted_prompts]        
        
        # get the answer tokens and target idx
        
        self.word_idx_dict = get_idx_dict(self.clean_prompts, self.clean_tokens, self.tokenizer)
        self.target_idx = torch.stack((torch.arange(self.clean_tokens.size(0))
                                        , self.word_idx_dict["PRONOUN"]), dim=1)
        
        self.start = self.word_idx_dict["START"]
        self.end = self.word_idx_dict["END"]
        
        self.answer_tokens = torch.zeros([self.N, 2], dtype=torch.int64).to(self.device)
        self.answer_tokens[:, 0] = torch.Tensor(correct_pronount_tokenIDs)  # TODO: decide what wrong answer is         
        self.answer_tokens[:, 1] = torch.Tensor(wrong_pronoun_tokedIDs)  # TODO: decide what wrong answer is         

        
        # get max_len and groups
        self.max_len = self.clean_tokens.size(1)
        
        all_ids = [prompt["template_idx"] for prompt in self.clean_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])
        
        self.attention_mask = self.tokenizer(self.clean_input, padding=True, return_tensors="pt").attention_mask
                
        self.correct_answers = self.answer_tokens[:, 0]
        self.wrong_answers = self.answer_tokens[:, 1]
        
        if not torch.all(self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1] + 1] == self.correct_answers):
            print(tokenizer.batch_decode(self.answer_tokens[:, 0]))
            print(tokenizer.batch_decode(self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1] + 1]))
            raise Exception("Target idx does not align with the position of the IOI in at least one of the senteces.")

        self.dataset = TensorDataset(
            self.clean_tokens,       # [N, seq_len], 
            self.corrupted_tokens,   # [N, seq_len]
            self.attention_mask,     # [N, seq_len]
            self.correct_answers,    # [N, 1]
            self.wrong_answers,      # [N, 1]
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