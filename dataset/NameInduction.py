import torch 
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import random
import numpy as np
from utils.dataset_loader import start_of_prompt, end_of_prompt
from torch.utils.data import TensorDataset


UNFILTERED_NAMES = [
    "Abigail", "Adam", "Adrian", "Aiden", "Alec", "Alexa", "Alexander", "Alexis", "Alice", "Alicia",
    "Allison", "Alyssa", "Amelia", "Amy", "Andrew", "Angela", "Anna", "Anthony", "Aria", "Ariana",
    "Aubrey", "Audrey", "Austin", "Ava", "Avery", "Bella", "Benjamin", "Blake", "Brianna", "Brooklyn",
    "Caleb", "Cameron", "Camila", "Carlos", "Carter", "Charles", "Charlotte", "Chloe", "Christian", "Christopher",
    "Claire", "Clara", "Cole", "Colin", "Connor", "Daniel", "David", "Delilah", "Dominic", "Dylan",
    "Easton", "Eleanor", "Elena", "Eli", "Eliana", "Elijah", "Elizabeth", "Ella", "Ellie", "Emily",
    "Emma", "Eric", "Ethan", "Eva", "Evelyn", "Everett", "Ezra", "Faith", "Finn", "Gabriel",
    "Gavin", "Genesis", "Gianna", "Grace", "Grayson", "Hailey", "Hannah", "Harper", "Hazel", "Henry",
    "Hudson", "Hunter", "Ian", "Isaac", "Isabel", "Isabella", "Isaiah", "Isla", "Jack", "Jackson",
    "Jacob", "Jade", "James", "Jasmine", "Jason", "Jasper", "Jayden", "Jeremiah", "Jessica", "John",
    "Jonathan", "Jordan", "Joseph", "Joshua", "Josiah", "Julia", "Julian", "Kaitlyn", "Kayla", "Kylie",
    "Landon", "Leah", "Leo", "Levi", "Liam", "Lillian", "Lily", "Lincoln", "Logan",
    "Lucas", "Lucy", "Luke", "Madeline", "Madison", "Maria", "Mason", "Mateo", "Matthew", "Maya",
    "Melanie", "Michael", "Mia", "Micah", "Mila", "Miles", "Naomi", "Natalie", "Nathan", "Nathaniel",
    "Nevaeh", "Nicholas", "Nina", "Noah", "Nolan", "Nora", "Nova", "Oliver", "Olivia", "Owen",
    "Paisley", "Parker", "Penelope", "Peyton", "Quinn", "Rachel", "Reagan", "Riley", "Robert", "Ruby",
    "Ryan", "Sadie", "Samantha", "Samuel", "Sarah", "Savannah", "Scarlett", "Sebastian", "Serenity", "Sienna",
    "Silas", "Skylar", "Sofia", "Sophia", "Stella", "Steven", "Taylor", "Theodore", "Thomas", "Tristan",
    "Tyler", "Valentina", "Vera", "Victoria", "Violet", "Vivian", "Walker", "Weston", "William", "Willow",
    "Wyatt", "Xavier", "Zachary", "Zara", "Zoe", "Zoey"
]

PLACES = [
    "kitchen",
    "restaurant",
    "library",
    "office",
    "school",
    "garden",
    "park",
    "store",
    "airport",
    "train station",
    "hospital",
    "cafe",
    "bakery",
    "theater",
    "stadium",
    "beach",
    "zoo",
    "museum",
    "garage",
    "supermarket"
]

TIMES = [
    "night",
    "noon",
    "5 o'clock",
    "midnight",
]


OBJECTS = [
    "ring",
    "cup",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

TEMPLATES = [
    "Then, [NAME] walked to the [PLACE]. This time, [NAME]",
    "Later on, [NAME] bought a [OBJECT] from the store. Here, [NAME]",
    "After dinner, [NAME] played in the [PLACE]. It was certain, that [NAME]",
    "Once again, [NAME] wrote a letter at [TIME]. Then [NAME]",
    "This morning, [NAME] visited the [PLACE]. There, [NAME]",
    "Earlier, [NAME] dropped the [OBJECT] in the [PLACE]. It was certain, that [NAME]",
    "At [TIME], [NAME] left the [PLACE]. Then, [NAME]",
    "Then, [NAME] saw a [OBJECT] near the [PLACE]. After that, [NAME]",
    "While walking, [NAME] passed by the [PLACE]. At that point, [NAME]",
    "Before breakfast, [NAME] checked the [OBJECT]. Just then, [NAME]",
    "In the evening, [NAME] relaxed at the [PLACE]. So, [NAME]",
    "Then, [NAME] watched the stars at [TIME]. From there, [NAME]",
    "Earlier, [NAME] brought snacks to the [PLACE]. At that time, [NAME]",
    "After lunch, [NAME] painted a [OBJECT]. There again, [NAME]",
    "Before that, [NAME] ran through the [PLACE] at [TIME]. Then again, [NAME]",
    "During the trip, [NAME] lost the [OBJECT]. As usual, [NAME]",
    "Eventually, [NAME] sent a message at [TIME]. At last, [NAME]",
    "Later that day, [NAME] cleaned the [PLACE]. Afterward, [NAME]"
]



def filter_names(names: List[str], tokenizer) -> List[str]:
    """
    Filter for names which have exactly two tokens when tokenized
    """
    NAMES = []
    for name in names:
        toks = tokenizer.encode(" " + name)
        if len(toks) == 2:
            NAMES.append(name)
    return NAMES


def gen_clean_prompts(templates: list[str], names:list[str], nouns_dict:dict, N:int, tokenizer, remove_target_token:bool=True):
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
        name = random.choice(names)
        
        prompt = temp
        induction_prompt = {}
        nouns = {}

        for k in nouns_dict:
            prompt = prompt.replace("[" + k + "]", random.choice(nouns_dict[k]))
    
        prompt = prompt.replace("[NAME]", name)
        if remove_target_token:
            text_split = prompt.split(" ")
            induction_name = tokenizer.tokenize(" " + name)[1]
            
            text_split[-1] = text_split[-1].replace(induction_name, "")
            prompt = " ".join(text_split)            
 
        induction_prompt["text"] = prompt
        induction_prompt["name"] = name
        induction_prompt["induction"] = name
        induction_prompt["template_idx"] = temp_id
        prompts.append(induction_prompt)
        n_gen += 1
    return prompts

def get_name_idx(prompts, tokens, tokenizer, idx_types=["START", "END", "NAME1", "NAME2", "INDUCTION"], prepend_bos:bool=False):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)

    
    for prompt, toks in zip(prompts, tokens):
        
        start_idx = start_of_prompt(toks, tokenizer=tokenizer, start_text="<|im_start|>user\n")
        end_idx = end_of_prompt(toks, tokenizer=tokenizer, end_text="<|im_end|>\n")    
            
        name1 = tokenizer(" " + prompt["name"]).input_ids[0]
        name_occurence = torch.nonzero(toks == name1)
        name1_idx = name_occurence[0].item()
        name2_idx = name1_idx + 1
        induction_idx = name_occurence[1].item()
        
        name_idx_dict["START"].append(start_idx)
        name_idx_dict["END"].append(end_idx)
        name_idx_dict["NAME1"].append(name1_idx)
        name_idx_dict["NAME2"].append(name2_idx)
        name_idx_dict["INDUCTION"].append(induction_idx)    
        
    return [
        torch.tensor(name_idx_dict[idx_type])
        for idx_type in idx_types
    ]

def get_idx_dict(prompts, tokens, tokenizer, prepend_bos:bool=False, model_mame:str="gpt2"):
    start_idx, end_idx, name1_idx, name2_idx, induction_idx = get_name_idx(prompts, tokens, tokenizer, prepend_bos=prepend_bos)

    return {
        "START": start_idx,
        "END": end_idx,
        "NAME1": name1_idx,
        "NAME2": name2_idx,
        "INDUCTION": induction_idx
    }

def gen_corrupted_prompts(prompts, tokenizer, names, prepend_bos:bool=False, seed:int=12342349, remove_target_token:bool=True):
    random.seed(seed)
    np.random.seed(seed)
    corrupted_prompts = []
    for prompt in prompts:
        
        clean_name = prompt["name"]        
        corrupted_name1 = random.choice(names)
        corrupted_name2 = random.choice(names)
        
        while clean_name == corrupted_name1 or clean_name == corrupted_name2 or corrupted_name2==corrupted_name1:
            corrupted_name1 = random.choice(names)
            corrupted_name2 = random.choice(names)

        text = prompt["text"]
        text = text.replace(clean_name, corrupted_name1, 1)
        text = text.replace(clean_name, corrupted_name2, 1)
        text_split = text.split(" ")

        if remove_target_token:
            corrupted_name_toks2 = tokenizer(" " + corrupted_name2).input_ids
            text = " ".join(text_split[:-1]) +  tokenizer.decode(corrupted_name_toks2[0])
        else:
            text = " ".join(text_split[:-1]) + " " + corrupted_name2

        corrupted_prompt = {}
        corrupted_prompt["text"] = text
        corrupted_prompt["name"] = corrupted_name1
        corrupted_prompt["induction"] = corrupted_name2
        corrupted_prompt["template_idx"] = prompt["template_idx"]
        corrupted_prompts.append(corrupted_prompt)
        
    return corrupted_prompts

class NameInduction():
    def __init__(
        self, 
        model_name:str, 
        N:Optional[int], 
        tokenizer, 
        device:str, 
        seed:int, 
        prepend_bos:bool=False, 
        remove_target_token:bool=True, 
    ) -> None:    
        print(f"Loading Induction Dataset with {model_name}")
        
        
        self.N = N
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
  
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device 
        self.remove_target_token=remove_target_token
        
        NAMES = filter_names(UNFILTERED_NAMES, tokenizer)
        
        
        # get the clean prompts, input and tokens
        self.clean_prompts = gen_clean_prompts(TEMPLATES, NAMES, {"PLACE": PLACES, "TIME": TIMES, "OBJECT": OBJECTS}, self.N, tokenizer, remove_target_token=self.remove_target_token)
        self.clean_input = [prompt["text"] for prompt in self.clean_prompts]
        if False:
            texts_clean = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for prompt in self.clean_input
            ]
        else:
            texts_clean = [
                (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
                for prompt in self.clean_prompts
            ]
            
        clean_induction_tokedIDs = [self.tokenizer.encode(" " + prompt["induction"])[1] for prompt in self.clean_prompts]
        self.clean_tokens = torch.Tensor(self.tokenizer(texts_clean, padding=True).input_ids).long().to(self.device)


        # get the corrupted prompts, input and tokens    
        self.corrupted_prompts = gen_corrupted_prompts(self.clean_prompts, self.tokenizer, NAMES, prepend_bos=prepend_bos, remove_target_token=self.remove_target_token)
        self.corrupted_input = [prompt["text"] for prompt in self.corrupted_prompts]
        if False:
            texts_corrupted = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                    ) for prompt in self.corrupted_input
                ]
        else:
            texts_corrupted = [
                (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
                for prompt in self.corrupted_prompts
            ]
        corrupted_induction_tokedIDs = [self.tokenizer.encode(" " + prompt["induction"])[1] for prompt in self.corrupted_prompts]
        self.corrupted_tokens = torch.Tensor(self.tokenizer(texts_corrupted, padding=True).input_ids).long().to(self.device)

        # get target_idx and answer tokens
        self.word_idx_dict = get_idx_dict(self.clean_prompts ,self.clean_tokens, self.tokenizer, prepend_bos=prepend_bos)
        
        self.target_idx = torch.stack((torch.arange(self.clean_tokens.size(0))
                                        , self.word_idx_dict["INDUCTION"]), dim=1)
        self.answer_tokens = torch.zeros([len(self.clean_tokens), 2], dtype=torch.int64).to(self.device)
        self.answer_tokens[:, 0] = torch.Tensor(clean_induction_tokedIDs)  # TODO: decide what wrong answer is 
        self.answer_tokens[:, 1] = torch.Tensor(corrupted_induction_tokedIDs)
        
        # max_len and groups
        self.max_len = self.clean_tokens.size(1)
        self.start = self.word_idx_dict["START"]
        self.end = self.word_idx_dict["END"]
        
        all_ids = [prompt["template_idx"] for prompt in self.clean_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])
            
        self.attention_mask = self.tokenizer(texts_clean, padding=True, return_tensors="pt").attention_mask
        
        if self.remove_target_token:
            if not torch.all(self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1]] == self.clean_tokens[self.target_idx[:,0], self.word_idx_dict["NAME1"]]):
                print(tokenizer.batch_decode(self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1]]))
                print(tokenizer.batch_decode(self.clean_tokens[self.target_idx[:,0], self.word_idx_dict["NAME1"]]))
                
        elif not torch.all(self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1] + 1] == self.answer_tokens[:, 0]):
            print(tokenizer.batch_decode(self.answer_tokens[:, 0]))
            print(tokenizer.batch_decode(self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1] + 1]))
            raise Exception("Target idx does not align with the position of the IOI in at least one of the senteces.")
        
        self.correct_answers = self.answer_tokens[:, 0]
        self.wrong_answers = self.answer_tokens[:, 1]
        self.dataset = TensorDataset(
            self.clean_tokens,        # [N, seq_len], 
            self.corrupted_tokens,    # [N, seq_len]
            self.attention_mask,      # [N, seq_len]
            self.correct_answers,     # [N, 1]
            self.wrong_answers,       # [N, 1]            
            self.target_idx,          # [N, 2]
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