import torch as t
from torch import Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
        
        
## Logit Difference
def ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answers: Union[Int[Tensor, "batch 1"], Dict],
    wrong_answers: Union[Int[Tensor, "batch 1"], Dict],
    target_idx: Optional[Int[Tensor, "batch 2"]],
    per_prompt=False,
    task="",
    model_name="gpt2",
    pad_token: Optional[int] = 50256,
    corrupted_target_idx: Optional[Int[Tensor, "batch 2"]] = None, 
) -> Union[Float[Tensor, "batch"], float]:
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    if task=="GreaterThan":
        answer_logit_diff= ave_logits_diff_greater_than(
            logits=logits,
            correct_answer_tokens=correct_answers,
            wrong_answer_tokens=wrong_answers,
            target_idx=target_idx, 
            model_name=model_name            
        )
    elif task=="Docstring":
        """
        "We measure whether an intervention changes the difference between the logit
        of the correct answer C and the highest wrong-answer logit 
        (maximum logit of all other definition and docstring argument names including 
        corrupted variants, i.e. A, B, rand1, rand2, ..., maximum recalculated every time)"
        cited from https://www.lesswrong.com/posts/u6KXXmKFbXfWzoAXn#Results__The_Docstring_Circuit
        """
        
        answer_logit_diff = ave_logit_diff_dosctring(
            logits=logits, 
            correct_answer_tokens=correct_answers,
            wrong_answer_tokens=wrong_answers,
            target_idx=target_idx, 
            model_name=model_name,
        )
        
        
    else:
        answer_logit_diff = ave_logits_diff_vanilla(
            logits=logits, 
            correct_answer_tokens=correct_answers,
            wrong_answer_tokens=wrong_answers,
            target_idx=target_idx,
        )
     # Find logit difference
    return answer_logit_diff if per_prompt else answer_logit_diff.mean().item() 


# from https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/docstring/utils.py
def ave_logit_diff_dosctring(
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answer_tokens: Union[Int[Tensor, "batch 2"], Dict],
    wrong_answer_tokens: Union[Int[Tensor, "batch 2"], Dict],
    target_idx: Optional[Int[Tensor, "batch 2"]],
    model_name: Optional[str] = "gpt2") -> Union[Float[Tensor, "batch"], float]:
    
    #correct_labels = answer_tokens["correct"]
    ##wrong_labels = answer_tokens["wrong"]
    correct_logits = logits[t.arange(len(logits)), -1, correct_answer_tokens]
    incorrect_logits = logits[t.arange(len(logits)).unsqueeze(-1), -1, wrong_answer_tokens]

    answer = -(correct_logits - incorrect_logits.max(dim=-1).values)
    return answer


def ave_logits_diff_vanilla(    
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answer_tokens: Int[Tensor, "batch 1"],
    wrong_answer_tokens: Int[Tensor, "batch 1"],
    target_idx: Optional[Int[Tensor, "batch 2"]],
    corrupted_target_idx: Optional[Int[Tensor, "batch 2"]] = None
    ) -> Union[Float[Tensor, "batch"], float]:
    
    if target_idx is not None and corrupted_target_idx is not None:
        # needed for the Gendered Pronouns task, where the target_idx of the clean and the corrupted prompts is different 
        right_predicition_logits: Float[Tensor, "batch"] = logits[target_idx[:, 0], target_idx[:, 1], correct_answer_tokens]
        wrong_prediction_logits: Float[Tensor, "batch"] = logits[corrupted_target_idx[:, 0], corrupted_target_idx[:, 1], wrong_answer_tokens]

    elif target_idx is not None:
        right_predicition_logits: Float[Tensor, "batch"] = logits[target_idx[:, 0], target_idx[:, 1], correct_answer_tokens]
        wrong_prediction_logits: Float[Tensor, "batch"] = logits[target_idx[:, 0], target_idx[:, 1], wrong_answer_tokens]

    else:
        right_predicition_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), -1, correct_answer_tokens]
        wrong_prediction_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), -1, wrong_answer_tokens]

    return right_predicition_logits - wrong_prediction_logits
    
    
def ave_logits_diff_greater_than(
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answer_tokens: Int[Tensor, "batch max_year"],
    wrong_answer_tokens: Int[Tensor, "batch (100 - min_year)"],
    target_idx: Optional[Int[Tensor, "batch 2"]],
    model_name: Optional[str] = "gpt2",
    ) -> Union[Float[Tensor, "batch"], float]:
    
    N = logits.size(0)
    right_predicition_logits: Float[Tensor, "batch"] = t.zeros(N)
    wrong_prediction_logits: Float[Tensor, "batch"] = t.zeros(N)
    
    for i in range(N):
        tar_id = target_idx[i]
        correct_answers = correct_answer_tokens[i]        
        wrong_answers =wrong_answer_tokens[i]

    
        if "gpt2" in model_name:
            # gpt2 model
            correct_answers = correct_answers[correct_answers != -1 ]
            wrong_answers = wrong_answers[wrong_answers != -1 ]

            # If answer_tokens is not a tensor, it is a list of the form [[right_answer], [wrong_answer]]
            # Necessary for the GreaterThan task
            right_predicition_logits[i] = logits[tar_id[0], tar_id[1], correct_answers.unsqueeze(0)].mean()
            wrong_prediction_logits[i] = logits[tar_id[0], tar_id[1], wrong_answers.unsqueeze(0)].mean()
      
        elif "Qwen2.5" in model_name:
            # Qwen model
            correct_answers = correct_answers[correct_answers[:, 0] != -1 ]
            wrong_answers = wrong_answers[wrong_answers[:, 0] != -1]
            right_predicition_logits[i] = logits[i, tar_id[1], correct_answers[:, 0]].mean() + logits[i, tar_id[1] + 1, correct_answers[:, 1]].mean()       

            wrong_prediction_logits[i] = logits[i, tar_id[1], wrong_answers[:, 0]].mean() + logits[i, tar_id[1] + 1, wrong_answers[:, 1]].mean()

        else:
            raise Exception(f"Greater Than metric not implemented for {model_name}. Check how years are tokenized by model and adapt this function accordingly.")
        
    return right_predicition_logits - wrong_prediction_logits
    

def logit_diff_preserve_performance(
    logits: Float[Tensor, "batch seq d_vocab"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    correct_answers: Int[Tensor, "batch 1"],
    wrong_answers: Int[Tensor, "batch 1"],
    target_idx: Int[Tensor, "batch 2"], 
    model_name:str="gpt2", 
    task=""
) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = ave_logit_diff(
        logits=logits, 
        correct_answers=correct_answers, 
        wrong_answers= wrong_answers,
        target_idx=target_idx,
        task=task,
        model_name=model_name
        )
    
    return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def logit_diff_restore_performance(
    logits: Float[Tensor, "batch seq d_vocab"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
    correct_answers: Int[Tensor, "batch 1"],
    wrong_answers: Int[Tensor, "batch 1"],    target_idx: Int[Tensor, "batch 2"], 
    model_name:str="gpt2", 
    task=""
) -> float:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    Which actuvations are sufficient to recstoring models performance 
    '''
    patched_logit_diff = ave_logit_diff(
        logits=logits, 
        correct_answers=correct_answers, 
        wrong_answer_tokens= wrong_answers,
        target_idx=target_idx,
        task=task,
        model_name=model_name
        )
            
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)


######################################## Probability Difference: ############################################################
def logprobs_of_right_answer(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    target_idx: Int[Tensor, "batch 2"],
    last_seq_element_only: bool=False,
    use_only_target:bool=False
) -> float:
    if last_seq_element_only:
        logprobs =  F.softmax(logits[target_idx[:, 0], target_idx[:, 1]], dim=-1)
    
        return logprobs.mean(-1)
    else:
        logprobs =  F.softmax(logits[target_idx[:, 0], target_idx[:, 1], answer_tokens[:, 0]], dim=-1)
        return logprobs.mean(-1)

def logprobs_compare_to_base_val(
    logits: Float[Tensor, "batch seq d_vocab"],
    base_val: float,
    answer_tokens: Float[Tensor, "batch 2"],
    target_idx: Int[Tensor, "batch 2"],
    last_seq_element_only:bool,
    use_only_target:bool=False
):  
    patched_probs:float = logprobs_of_right_answer(logits=logits, 
                                                   answer_tokens=answer_tokens, 
                                                   target_idx=target_idx, 
                                                   last_seq_element_only=last_seq_element_only,
                                                   use_only_target=use_only_target)
    
    return patched_probs - base_val

def logprobs_greater_than(
    logits: Float[Tensor, "batch seq d_vocab"],
    tokens: Int[Tensor, "batch seq"],
    answer_tokens: Float[Tensor, "batch 2"],
    target_idx: Int[Tensor, "batch 2"],
    valid_years_dict:Dict,  # dict: century to List[year_suffix]
    year_tokens_to_year: Dict, # maps year_suffix_token to year_suffix
    century_tokens_to_century: Dict, 
    years_to_year_token: Dict,  # maps century_token to century 
) -> float:
    logprobs = F.softmax(logits[target_idx[:, 0], target_idx[:, 1]], dim=-1)
    # get the target century and year suffix from the tokens
    century_tokens = tokens[:, -1]   # the last token encodes the century prefix
    century = [century_tokens_to_century.get(century_token) for century_token in century_tokens.tolist()]
    year_suffix_tokens = answer_tokens[:, 0]
    year_suffix = [year_tokens_to_year.get(year_suffix_token) for year_suffix_token in year_suffix_tokens.tolist()]
        
    cumsum_bigger = 0
    cumsum_smaller = 0
    batch = 0

    for cent, year in zip(century, year_suffix):
        valid_years = valid_years_dict[cent]     
        for valid_year in valid_years:
            year_token = years_to_year_token[valid_year]
            if valid_year > year:
                cumsum_bigger += logprobs[batch, year_token]
            else:
                cumsum_smaller += logprobs[batch, year_token]
        batch += 1
    return (cumsum_bigger - cumsum_smaller) / len(logits)
    
    
def logprobs_greater_than_base_val(
    logits: Float[Tensor, "batch seq d_vocab"],
    base_val:float,
    tokens: Int[Tensor, "batch seq"],
    answer_tokens: Float[Tensor, "batch 2"],
    target_idx: Int[Tensor, "batch 2"],
    valid_years_dict:Dict,  # dict: century to List[year_suffix]
    year_tokens_to_year: Dict, # maps year_suffix_token to year_suffix
    century_tokens_to_century: Dict, 
    years_to_year_token: Dict,  # maps century_token to century 
    
    
) -> float:
    patched_logprobs = logprobs_greater_than(logits,
                                            tokens,
                                            answer_tokens,
                                            target_idx,
                                            valid_years_dict,  # dict: century to List[year_suffix]
                                            year_tokens_to_year, # maps year_suffix_token to year_suffix
                                            century_tokens_to_century, 
                                            years_to_year_token)  # maps century_to)
    return   patched_logprobs - base_val
    

    
    

################################### KL DIVERGENCE ################################################

def KL_divergence(
    logits: Float[Tensor, "batch seq d_vocab"], # For normal run vs after unembedding
    base_logits: Float[Tensor, "batch seq d_vocab"],  # targt distribution
    target_idx: Optional[Int[Tensor, "batch 2"]] = None,
    answer_tokens :Optional[Int[Tensor, "batch 2"]] = None,
    last_seq_element_only = False,
    use_only_target: bool = False, # only necessary or induction task 
    use_answer_tokens: bool = False
    ) -> float:
    
    if last_seq_element_only:  # for ioi        
        logits = logits[target_idx[:, 0], target_idx[:, 1]]
        base_logits = base_logits[target_idx[:, 0], target_idx[:, 1]]
    
    
    logprobs = F.log_softmax(logits, dim=-1).detach()
    base_log_probs = F.log_softmax(base_logits, dim=-1).detach() #TODO: in partial
    
    assert logprobs.shape == base_log_probs.shape
    kl_div = F.kl_div(input=logprobs, target=base_log_probs, log_target=True, reduction="none").sum(dim=-1)
    
    if use_only_target:
        assert target_idx is not None
        # use for induction!
        kl_div = kl_div[target_idx[:, 0], target_idx[:, 1]]

    
    elif use_answer_tokens:
        assert answer_tokens is not None and target_idx is not None
        kl_div = kl_div[target_idx[:, 0], 
                        target_idx[:, 1], 
                        answer_tokens[:, 0]]
    return kl_div.mean()
        
def KL_div_compare_to_base_val(
    logits: Float[Tensor, "batch seq d_vocab"],
    base_logits: Float[Tensor, "batch seq d_vocab"],
    base_KL_div: float,
    answer_tokens: Float[Tensor, "batch 2"],
    target_idx: Union[Int[Tensor, "batch 1"], Int[Tensor, "batch 2"]],
    last_seq_element_only = False,
    use_only_target: bool = False, # only necessary for induction task 
    use_answer_tokens: bool = False
    ) -> float:
    patched_KL_divergence = KL_divergence(
        logits=logits, 
        base_logits=base_logits, 
        target_idx=target_idx, 
        answer_tokens=answer_tokens, 
        last_seq_element_only = last_seq_element_only,
        use_only_target = use_only_target, # only necessary or induction task 
        use_answer_tokens = use_answer_tokens,
        )
    return base_KL_div - patched_KL_divergence
