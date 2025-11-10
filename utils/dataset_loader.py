from transformers import AutoTokenizer
import torch
from jaxtyping import Float
from torch import Tensor
from utils.metrics import *
from functools import partial


def predict_target_token(model, dataset, tokenizer:AutoTokenizer, device:str="cuda", n=1, use_corrupted:bool=False):
    """Predict the Target token for samples of the clean and corrupted dataset. Only used for testing purposes.

    Args:
        model (_type_): model
        dataset (_type_): dataset
        tokenizer (AutoTokenizer): tokenizer
        device (str, optional): device. Defaults to "cuda".
        n (int, optional): number of smaples. Defaults to 1.
        use_corrupted (bool, optional): If true, corrupted dataset, else clean dataset. Defaults to False.
    """
    
    if use_corrupted:
        tokens = dataset.corrupted_tokens.to(device)
    else:
        tokens = dataset.clean_tokens.to(device)
        
    # Get the logits
    with torch.no_grad():
        outputs = model(tokens[:n])
    try:
        outputs = outputs.logits
    except:
        pass
    
    starts = dataset.start
    logits = outputs[dataset.target_idx[:n, 0], dataset.target_idx[:n, 1]]  # Logits for the last token
    
    for j in range(n):
        target_idx = dataset.target_idx[j, 1]
        # Get top 10 tokens and probabilities
        probs = torch.softmax(logits[j], dim=-1)
        top_probs, top_indices = torch.topk(probs, 10)
        # Decode and print results
        print("Top 10 Predictions")
        print("Prompt", j, ":",  "".join(tokenizer.decode(tokens[j])))#, starts[j]:target_idx+1])))
        for i in range(10):
            token_id = top_indices[i].item()
            token_prob = top_probs[i].item()
            print(f"{tokenizer.decode(token_id):<10} | Probability: {token_prob:.4f}")
        print("\n")
        
    
    
    """ initialize the metric used.
    
    Choose between logit_diff, probability and KL_divergence metric
    For activation patching the goal is to restore performance from a corrupted run (corrupted - patched)
    For path patching the goal is to preserve performance from a clean run (patched - clean)
    
    Calculate clean_distribution_taverage and corrupted_distribution_average depending on the choosen metric
    
    Args:
        metric_name (str): name of the metric
        patching_method (str): activation method to be used

    Returns:
        Callable: chosen metric
    """
    
def init_metric_and_cache_average(
    clean_logits: Float[Tensor, "batch seq token_embed"],
    corrupted_logits: Float[Tensor, "batch seq token_embed" ],
    task:str, 
    patching_method: str,
    metric_name: str,
    dataset, 
    model_name: str = "gpt2",
    ) :
    """Initialize the metric and get the average performance under the clean and corrupted dataset.
    Args:
        clean_logits (Float[Tensor,  "batch seq token_embed"]): clean_logits
        corrupted_logits (Float[Tensor,  "batch seq token_embed"]): corrupted_logits
        task (str): task. ["ioi", "GreaterThan", "GenderedPronouns", "induction", "Docstring"]
        patching_method (str): patching method. ["path_patching", "activation_patching", "acdc"]
        metric_name (str): name of the metric. ["logits_diff", "probs", "KL_divergence"]
        dataset (_type_): dataset
        model_name (str, optional): model_name. Defaults to "gpt2".

    Returns:
        _type_: _description_
    """
    
    ### Average Logits Difference
    if metric_name == "logits_diff":
        assert task in ["IOI", "Induction", "GreaterThan", "GenderedPronouns", "Docstring"], "Logits difference metric can only be used with the tasks ioi, docstring or greaterThan"

        clean_distribution_average = ave_logit_diff(
            clean_logits, 
            correct_answers=dataset.correct_answers,
            wrong_answers=dataset.wrong_answers,      
            target_idx=dataset.target_idx, 
            task=task,
            model_name=model_name
            )
        
        corrupted_distribution_average = ave_logit_diff(
            corrupted_logits, 
            correct_answers=dataset.correct_answers,
            wrong_answers=dataset.wrong_answers,    
            target_idx=dataset.target_idx,
            task=task,
            model_name=model_name
            )
                
        if patching_method=="activation":

            metric = partial(logit_diff_restore_performance,
                            corrupted_logit_diff = corrupted_distribution_average,
                            clean_logit_diff = clean_distribution_average,
                            correct_answers=dataset.correct_answers,
                            wrong_answers=dataset.wrong_answers,  
                            target_idx = dataset.target_idx, 
                            model_name=model_name, 
                            task=task
                            )
        elif patching_method == "path" or  patching_method == "acdc":

            metric = partial(logit_diff_preserve_performance,
                                corrupted_logit_diff = corrupted_distribution_average,
                                clean_logit_diff = clean_distribution_average,
                                correct_answers=dataset.correct_answers,
                                wrong_answers=dataset.wrong_answers,  
                                target_idx = dataset.target_idx, 
                                model_name=model_name, 
                                task=task
                                )
    
    
    ### Probability
    elif metric_name == "probs": 
        if task == "GreaterThan":
            base_val = logprobs_greater_than(
                    logits= dataset.clean_logits,
                    tokens = dataset.clean_tokens,
                    answer_tokens = dataset.answer_tokens,
                    target_idx = dataset.target_idx,
                    valid_years_dict = dataset.VALID_YEARS, 
                    year_tokens_to_year = dataset.YEAR_TOKENS_TO_YEAR,
                    years_to_year_token = dataset.YEARS_TO_YEAR_TOKEN,
                    century_tokens_to_century = dataset.CENTURY_TOKENS_TO_CENTURY)

            metric = partial(logprobs_greater_than_base_val,
                    base_val = base_val,
                    tokens = dataset.clean_tokens,
                    answer_tokens = dataset.answer_tokens,
                    target_idx = dataset.target_idx,
                    valid_years_dict = dataset.VALID_YEARS, 
                    year_tokens_to_year = dataset.YEAR_TOKENS_TO_YEAR,
                    years_to_year_token = dataset.YEARS_TO_YEAR_TOKEN,
                    century_tokens_to_century = dataset.CENTURY_TOKENS_TO_CENTURY
            )  
                    
        elif patching_method == "activation":
            corrupted_distribution_average = logprobs_of_right_answer(
                corrupted_logits, 
                dataset.answer_tokens, 
                dataset.target_idx,
                True
            )
            
            metric = partial(logprobs_compare_to_base_val,
                                base_val=corrupted_distribution_average,
                                answer_tokens=dataset.answer_tokens,
                                target_idx = dataset.target_idx,
                                last_seq_element_only = True
                                )
                            
        elif patching_method == "path" or patching_method == "acdc":
            clean_distribution_average = logprobs_of_right_answer(
                clean_logits, 
                dataset.answer_tokens, 
                dataset.target_idx,
                True
            )
                                                                    
            metric = partial(
                logprobs_compare_to_base_val,
                base_val= clean_distribution_average,
                answer_tokens=dataset.answer_tokens,
                target_idx = dataset.target_idx,
                last_seq_element_only = True
                )
            
            clean_distribution_average = logprobs_of_right_answer(
                clean_logits, 
                dataset.answer_tokens, 
                dataset.target_idx,
                True
            )
            
            metric = partial(
                logprobs_compare_to_base_val,
                base_val= clean_distribution_average,
                answer_tokens=dataset.answer_tokens,
                target_idx = dataset.target_idx,
                last_seq_element_only = True
                )
    
    
    ### KL Divergence
    elif metric_name == "KL_divergence":
        if patching_method == "activation":
            clean_distribution_average = KL_divergence(
                                logits=corrupted_logits, 
                                base_logits=clean_logits,
                                target_idx=dataset.target_idx,
                                answer_tokens = dataset.answer_tokens,
                                last_seq_element_only = True,
                                use_only_target=True
                                )
            
            metric = partial(KL_div_compare_to_base_val,
                                base_logits = clean_logits,
                                base_KL_div = clean_distribution_average,
                                target_idx = dataset.target_idx,                                 
                                answer_tokens=dataset.answer_tokens,
                                last_seq_element_only = True,
                                use_only_target=True
                                )
        
        elif patching_method == "path":
            # base logits are corrupted logits, bc patch patching based on clean run and patches in corrupted values
            corrupted_distribution_average = KL_divergence(
                                logits=clean_logits, 
                                base_logits=corrupted_logits,
                                target_idx=dataset.target_idx,
                                answer_tokens = dataset.answer_tokens,
                                last_seq_element_only = True,
                                use_only_target=True
                                )

            metric = partial(KL_div_compare_to_base_val,
                                base_logits =corrupted_logits,
                                base_KL_div = corrupted_distribution_average,
                                target_idx = dataset.target_idx,                                 
                                answer_tokens=dataset.answer_tokens,
                                last_seq_element_only =True,
                                use_only_target=True
                                )
    
    
        elif patching_method == "acdc":   
            # base logits are the clean logits, bc acdc takes corrupted run and patches in the clean logits           
            metric = partial(
                KL_divergence,
                base_logits = clean_logits,
                target_idx = dataset.target_idx,
                last_seq_element_only = True,
                use_only_target = True         
            )   
    return metric


def print_whole_dataset(dataset):
    print("device", dataset.device)
    print("max length", dataset.max_len)
    print("clean input", dataset.clean_input)
    print("corrupted input", dataset.corrupted_input)
    print("clean tokens", dataset.clean_tokens)
    print("corrupted tokens", dataset.corrupted_tokens)
    print("answer tokens", dataset.answer_tokens)
    print("target idx", dataset.target_idx)
    print("attention mask", dataset.attention_mask)


def start_of_prompt(token:list, tokenizer:AutoTokenizer, start_text:str):
    """Find the beginning of a prompt. 
    For GPT2 very easy: most of the times idx=0.
    For Qwen-Instruct: find "<|im_start|>user\n" token. After that the prompt starts    
    Args:
        prompts (list(str)): one tokenized prompt
        tokenizer (_type_): tkenizer
        start_token (_type_): string that signals the start of the prompt (for Qwen: "<|im_start|>user\n")
    """
    device=token.device
    start_tokens = torch.tensor(tokenizer(start_text).input_ids).to(device)
    window_size = start_tokens.size(0)
    for idx in range(token.size(0) - window_size):
        if torch.all(token[idx:idx+window_size] == start_tokens):
            return idx + window_size 
    return 0
    #raise Exception(f"Start Tokens {start_tokens} not in prompt {token} ")
        
        
def end_of_prompt(token:list, tokenizer:AutoTokenizer, end_text:str, device="cuda"):
    """Find the end of a prompt. 
    For GPT2 very easy: most of the times idx=-1.

    Args:
        prompts (list(str)): one tokenized prompt
        tokenizer (_type_): tkenizer
        start_token (_type_): string that signals the start of the prompt
    """
    token = token.to(device)
    end_tokens = torch.tensor(tokenizer(end_text).input_ids).to(device)
    window_size = end_tokens.size(0)
    for idx in range(token.size(0), window_size+1, -1):
        if torch.all(token[idx-window_size:idx] == end_tokens):
            return idx-window_size
    return token.size(0)
    #raise Exception(f"Start Tokens {end_tokens} not in prompt {token} ")
        
