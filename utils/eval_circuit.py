## faithfullness = circuit can perform the task as well as the whole model
import torch
from jaxtyping import Float, Int, Bool
from typing import Literal, Callable
from torch import Tensor
import torch as t
import einops
from functools import partial
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from utils.metrics import ave_logit_diff
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import numpy as np
import gc
from dataset.loader import get_dataloader
from utils.metrics import ave_logit_diff
from utils.circuit_functions import circuit_size, TPR, FPR, precision

#----------------------------------------------------------------------------------------------------
#BEGINN COPY FROM;
#https://colab.research.google.com/github/callummcdougall/ARENA_3.0/blob/main/chapter1_transformer_interp/exercises/part41_indirect_object_identification/1.4.1_Indirect_Object_Identification_solutions.ipynb?t=20250910#scrollTo=Xkge79LtZ7bd
#Adapted to ignore sequence positions, if a head is in the circuit, all sequence positions are given the clean activations
#----------------------------------------------------------------------------------------------------

def get_heads_to_keep(
    N,
    max_len,
    model:HookedTransformer,
    circuit: dict[str, list[tuple[int, int]]],
) -> dict[int, Bool[Tensor, "batch seq head"]]:
    '''
    Returns a dictionary mapping layers to a boolean mask giving the indices of the
    z output which *shouldn't* be mean-ablated.
    Ignoring sequence positions.

    The output of this function will be used for the hook function that does ablation.
    '''
    heads_to_keep = {}
    batch, seq, n_heads = N, max_len, model.cfg.n_heads

    for layer in range(model.cfg.n_layers):
        mask = t.zeros(size=(batch, seq, n_heads))

        try:
            for head_idx in circuit[layer]:
                mask[:, :, head_idx] = 1
        except:
            pass
        heads_to_keep[layer] = mask.bool()

    return heads_to_keep
    
    
##### original IOI- metric: corrupted activations in each head AND position no in circuit 
def get_heads_and_posns_to_keep(
    word_idx,
    N:int,
    max_len:int,
    model:HookedTransformer,
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, str],
) -> dict[int, Bool[Tensor, "batch seq head"]]:
    '''
    Returns a dictionary mapping layers to a boolean mask giving the indices of the
    z output which *shouldn't* be mean-ablated.

    The output of this function will be used for the hook function that does ablation.
    '''
    heads_and_posns_to_keep = {}
    batch, seq, n_heads = N, max_len, model.cfg.n_heads

    for layer in range(model.cfg.n_layers):

        mask = t.zeros(size=(batch, seq, n_heads))

        for (head_type, head_list) in circuit.items():
            seq_pos = seq_pos_to_keep[head_type]
            indices = word_idx[seq_pos]
            for (layer_idx, head_idx) in head_list:
                if layer_idx == layer:
                    mask[:, indices, head_idx] = 1

        heads_and_posns_to_keep[layer] = mask.bool()

    return heads_and_posns_to_keep


def hook_fn_mask_z(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    heads_and_posns_to_keep: dict[int, Bool[Tensor, "batch seq head"]],
    means: Float[Tensor, "layer batch seq head d_head"],
) -> Float[Tensor, "batch seq head d_head"]:
    '''
    Hook function which masks the z output of a transformer head.

    heads_and_posns_to_keep
        dict created with the get_heads_and_posns_to_keep function. This tells
        us where to mask.

    means
        Tensor of mean z values of the means_dataset over each group of prompts
        with the same template. This tells us what values to mask with.
    '''
    # Get the mask for this layer, and add d_head=1 dimension so it broadcasts correctly
    mask_for_this_layer = heads_and_posns_to_keep[hook.layer()].unsqueeze(-1).to(z.device)
    # Set z values to the mean
    z = t.where(mask_for_this_layer, z, means[hook.layer()])
    return z


def compute_means_by_template(
    corrupted_tokens: Float[Tensor, "batch seq"],
    N:int,
    max_len:int,
    groups: list,
    model: HookedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:
    '''
    Returns the mean of each head's output over the means dataset. This mean is
    computed separately for each group of prompts with the same template (these
    are given by dataset.groups).
    '''
    # Cache the outputs of every head
    _, means_cache = model.run_with_cache(
        corrupted_tokens.long(),
        return_type=None,
        names_filter=lambda name: name.endswith("z"),
    )
    # Create tensor to store means
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    batch, seq_len = N, max_len
    means = t.zeros(size=(n_layers, batch, seq_len, n_heads, d_head), device=model.cfg.device, dtype=model.cfg.dtype)
    # Get set of different templates for this data
    
    for layer in range(model.cfg.n_layers):
        z_for_this_layer = means_cache[utils.get_act_name("z", layer)] # [batch seq head d_head]
        
        for template_group in groups:
            
            z_for_this_template = z_for_this_layer[template_group]
            z_means_for_this_template = einops.reduce(z_for_this_template, "batch seq head d_head -> seq head d_head", "mean")
            means[layer, template_group] = z_means_for_this_template
            
    return means


def add_mean_ablation_hook(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch seq"],
    groups: list,
    word_idx: dict[str, int],
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, str] = None,
    is_permanent: bool = True,
    original_IOI = False
) -> HookedTransformer:
    '''
    Adds a permanent hook to the model, which ablates according to the circuit and
    seq_pos_to_keep dictionaries.

    In other words, when the model is run on dataset, every head's output will
    be replaced with the mean over means_dataset for sequences with the same template,
    except for a subset of heads and sequence positions as specified by the circuit
    and seq_pos_to_keep dicts.
    '''

    model.reset_hooks(including_permanent=True)
    # Compute the mean of each head's output on the ABC dataset, grouped by template
    N, max_len = corrupted_tokens.size()

    means = compute_means_by_template(corrupted_tokens, N, max_len, groups, model)
    # Convert this into a boolean map
    if original_IOI:
        assert seq_pos_to_keep is not None
        heads_and_posns_to_keep = get_heads_and_posns_to_keep(word_idx, N, max_len, model, circuit, seq_pos_to_keep)         
        # Get a hook function which will patch in the mean z values for each head, at
        # all positions which aren't important for the circuit

        hook_fn = partial(
            hook_fn_mask_z,
            heads_and_posns_to_keep=heads_and_posns_to_keep,
            means=means
        )

    else:
        heads_to_keep = get_heads_to_keep(N, max_len, model, circuit)     
        # Get a hook function which will patch in the mean z values for each head, at
        # all positions which aren't important for the circuit
        hook_fn = partial(
            hook_fn_mask_z,
            heads_and_posns_to_keep=heads_to_keep,
            means=means
        )
        
    # Apply hook
    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)
    
    return model

#----------------------------------------------------------------------------------------------------
# END COPY
#----------------------------------------------------------------------------------------------------


def evaluate_circiut(
    model, 
    CIRCUIT:dict, 
    dataset, 
    ave_logit_gt:float,
    task="", 
    model_name="gpt2"
    ) -> List:
    """Evaluate how a circuit performs.
    All heads of the model, not inlcuded in the circuit, are patched with corrupted activations. Then the performance of
    the patched model is evaluated anbd compared to the ground_truth average logit difference

    Args:
        model (_type_): model
        CIRCUIT (dict): CIRCUIT
        dataset (_type_): dataset
        ave_logit_gt (float): ground-truth average logit differnve
        task (str, optional): task. Defaults to "".
        model_name (str, optional): model_name. Defaults to "gpt2".

    Returns:
        [float, float]: ave_logit and performance of the model, when only circuit heads are patched with clean input
    """
    
    model.reset_hooks(including_permanent=True)
    try:
        word_idx = dataset.word_idx
        print("using the original IOI metric")
    except:
        word_idx = None
    
    # Permanent hooks, that ablate heads not in circuit with the mean over the corrupted activations 
    model = add_mean_ablation_hook(
        model=model, 
        corrupted_tokens=dataset.corrupted_tokens.to(float),
        groups=dataset.groups,
        word_idx=word_idx,
        circuit=CIRCUIT
        )
    
    # forward pass through ablated model
    device = model.cfg.device
    with torch.no_grad():
        logits = model(dataset.clean_tokens)
    
    # average_logit_difference and performance of ablated model 
    ave_logit = ave_logit_diff(
        logits=logits, 
        correct_answers=dataset.correct_answers,
        wrong_answers=dataset.wrong_answers,
        target_idx=dataset.target_idx.to(device),
        task=task,
        model_name=model_name
        )
    performance = performance_achieved(ave_logit_gt, ave_logit)   
    
    # delete the permanent hooks
    model.reset_hooks(including_permanent=True)
    return ave_logit, performance

def batch_evaluate_circiut(
    model, 
    CIRCUIT:dict, 
    dataset, 
    ave_logit_gt:float, 
    task="", 
    model_name="gpt2", 
    epochs=4, 
    batch_size=50
    )-> List:
    
    """Evaluate how a circuit performs, split the input in batches.
    All heads of the model, not inlcuded in the circuit, are patched with corrupted activations. Then the performance of
    the patched model is evaluated anbd compared to the ground_truth average logit difference

    Args:
        model (_type_): model
        CIRCUIT (dict): CIRCUIT
        dataset (_type_): dataset
        ave_logit_gt (float): ground-truth average logit differnve
        task (str, optional): task. Defaults to "".
        model_name (str, optional): model_name. Defaults to "gpt2".
        epochs (int, optional): epochs. Defaults to 4.
        batch_size (int, optional): batch_size. Defaults to 50.

    Returns:
        List[float, float]: ave_logit and performance of the model, when only circuit heads are patched with clean input
    """
    
    # put the dataset into batches
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    model.reset_hooks(including_permanent=True)
    device = model.cfg.device

    try:
        word_idx = dataset.word_idx
    except:
        word_idx = None

    performances = np.zeros(epochs)
    avg_logits = np.zeros(epochs)
    
    for i, batch_dataset in enumerate(dataloader):    
        torch.cuda.empty_cache()
        gc.collect()
        groups = {}
        
        # restructured the groups to the batches, samples with the same template are in a group
        # average over group is used to ablate
        for dataset_idx, _ in batch_dataset["target_idx"]:
            group_idx = next(i for i, arr in enumerate(dataset.groups) if dataset_idx in arr)
            elem = dataset_idx.item() % batch_size
            try:
                groups[group_idx] = np.append(groups[group_idx], elem)
            except:
                groups[group_idx] = np.array([elem])
                
        groups = list(groups.values())

        # Permanent hooks, that ablate heads not in circuit with the mean over the corrupted activations 
        model = add_mean_ablation_hook(
            model=model, 
            corrupted_tokens=batch_dataset["corrupted_tokens"],
            groups=groups,
            word_idx=word_idx,
            circuit=CIRCUIT
            )
        
        # forward pass trough ablated model
        with torch.no_grad():
            logits = model(batch_dataset["clean_tokens"])
            
        target_idx = batch_dataset["target_idx"]
        target_idx[:, 0] = target_idx[:, 0] % batch_size
        
        # average logit and performance of ablated model 
        ave_logit = ave_logit_diff(
            logits=logits.to(device), 
            correct_answers=batch_dataset["correct_answers"].to(device), 
            wrong_answers=batch_dataset["wrong_answers"].to(device), 
            target_idx=target_idx.to(device),
            task=task,
            model_name=model_name
            )
        
        performance = performance_achieved(ave_logit_gt, ave_logit)   
        
        performances[i] = performance
        avg_logits[i] = ave_logit
        
        model.reset_hooks(including_permanent=True)
    return np.mean(avg_logits), np.mean(performances)

    
def performance_achieved(ave_logit_gt:float, ave_logit:float) -> float:
    """How much percentage of the **ground truth** average logit difference is restored by ave_logit. 
    Calculate (|ave_logit_gt - ave_logit| / |ave_logit_gt|)

    Args:
        ave_logit_gt (float): ground-truth ave_logit_gt obtained under the original, unpatched model
        ave_logit (float): ave_logit obtained by circuit

    Returns:
        float: percentage of restored performance
    """
    if ave_logit_gt > 0:
        performance_achieved = 100 - (ave_logit_gt - ave_logit) / ave_logit_gt * 100
    else:
        performance_achieved = abs(ave_logit_gt - ave_logit) / abs(ave_logit_gt) * 100

    return performance_achieved


def performance_gain(performance_new:float, performance_old:float) -> float:
    """Percentual differnence between performance_new and performance_old. How much better/worse is the peroformance
    under one circuit compared to the performance under another one? 

    Args:
        performance_new (float): old performance
        performance_old (float): new peroformance

    Returns:
        float: perofrmance_gain
    """
    if performance_old == 0:
        performance_gain = ((performance_new - performance_old) / 0.000001) *100
    else:
        performance_gain = ((performance_new - performance_old) / performance_old) *100
    return performance_gain
    
    
    
def print_statistics(title, ave_logit, performance_achieved, CIRCUIT, IOI_CIRCUIT, performance_gain=None, circuit_type=""):
    text =  title + "\n" +\
            f"Average logit difference: {ave_logit:.4f} \n" +\
            f"circuit size: {circuit_size(CIRCUIT)} \n" +\
            f"performance achieved: {performance_achieved:.2f}% \n" +\
            f"TPR: {TPR(CIRCUIT, IOI_CIRCUIT)*100:.2f}% \n" +\
            f"FPR: {FPR(CIRCUIT, IOI_CIRCUIT)*100:.2f}% \n" +\
            f"Precision: {precision(CIRCUIT, IOI_CIRCUIT)*100:.2f}% \n" 

    if performance_gain is not None:
        text = text + f"performance gain {performance_gain:.2f}% \n \n"
    return "\n \n" + text