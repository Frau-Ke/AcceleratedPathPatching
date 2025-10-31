# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
from dataset.IOI_dataset import IOI_dataset
from dataset.GreaterThan import GreaterThan
from dataset.NameInduction import NameInduction
from dataset.GenderedPronoun import GenderedPronoun

# Set random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for NumPy and PyTorch for reproducibility.
    
    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process PTB (Penn Treebank) dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    """
    Load and process PTB (Penn Treebank) dataset.
    
    Args:
        nsamples (int): Number of samples to extract.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for each sample.
        tokenizer (Tokenizer): Tokenizer to use for text encoding.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test set.
    """
    # Load train and test datasets
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # traindata = load_dataset('text', data_files='datasets/wikitext/wiki.train.raw', split="train")
    # testdata = load_dataset('text', data_files='datasets/wikitext/wiki.test.raw', split="train")
    
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process C4 (Common Crawl) dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the C4 (Common Crawl) dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    # traindata = load_dataset('json', data_files={'train': 'datasets/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('json', data_files={'validation': 'datasets/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_ioi(nsamples, seed, seqlen, tokenizer, device="cpu", corrupt=False):
    print("use all same length dataset")
    trainloader = []
    traindata = IOI_dataset(
                patching_method="path",
                tokenizer=tokenizer,
                device = device,
                seed=seed,
                prepend_bos=False
                )
    
    has_seq_len = traindata.target_idx[:, 1] == seqlen - 2
    has_seq_len_idx = torch.nonzero(has_seq_len, as_tuple=True)[0]

    for _ in range(nsamples):
        i = random.randint(0, len(has_seq_len_idx) - 1)
        idx = has_seq_len_idx[i]
        if not corrupt:
            trainenc = torch.cat((traindata.clean_tokens[idx, :-1].to(device), traindata.answer_tokens[idx][0].to(device).unsqueeze(0)), dim=0).unsqueeze(0)
            #print(tokenizer.batch_decode(trainenc[0]))
        else:
            trainenc = torch.cat((traindata.corrupted_tokens[idx, :-1].to(device), traindata.answer_tokens[idx][0].to(device).unsqueeze(0)), dim=0).unsqueeze(0)
            #print(tokenizer.batch_decode(trainenc[0]))
        inp = trainenc[0:seqlen+2]
        tar = inp.clone()
        
        tar[0, :-1] = -100
        trainloader.append((inp, tar))
   
    return trainloader, None

def get_ioi_without_seq_limit(nsamples, seed, max_seqlen, tokenizer, device="cpu", corrupt=False):
    print("using different length!")
    trainloader = []
    traindata = IOI_dataset(
                patching_method="path",
                tokenizer=tokenizer,
                device = device,
                seed=seed,
                prepend_bos=True
                )
    
    for _ in range(nsamples):
        idx = random.randint(0, len(traindata.clean_tokens) - 1)
        if not corrupt:
            trainenc = traindata.clean_tokens[idx, :].to(device).unsqueeze(0)
        else:
            trainenc = traindata.corrupted_tokens[idx, :].to(device)            #print(tokenizer.batch_decode(trainenc[0]))

        tar = torch.full_like(trainenc, -100)
        tar_idx = traindata.target_idx[idx, 1] + 1
        tar[:,tar_idx] = traindata.clean_tokens[idx][tar_idx]
        
        trainloader.append((trainenc, tar))
    return trainloader, None

def get_greater_than(nsamples, seed, max_seqlen, tokenizer, device="cpu", corrupt=False):
    trainloader = []
    dataset = GreaterThan(
                N=nsamples,
                tokenizer=tokenizer,
                device = device,
                seed=seed
                )
    max_seqlen = dataset.max_len
    inputs = dataset.clean_tokens.unsqueeze(0)
    if not corrupt:
        trainenc = dataset.clean_tokens.to(device).unsqueeze(0)
    else:
        trainenc = dataset.corrupted_tokens.to(device)  
    # add a random number between "start year" and 99:
    #tar = torch.full_like(inputs, 0)
    #tar[:,:,-1] = 1

    tar = torch.full_like(inputs, -100)
    tar[:,:,-1] = trainenc[:,:,-1]
    for idx in range(nsamples):
        trainloader.append((inputs[:,idx], tar[:,idx]))
    return trainloader, None

def get_indcution(nsamples, seed, max_seqlen, tokenizer, device="cpu", corrupt=False):
    trainloader = []
    dataset = NameInduction(
        N = nsamples,
        tokenizer=tokenizer,
        device = device,
        seed=seed  
    )
    max_seqlen = dataset.max_len
    target_idx = dataset.target_idx
    inputs = dataset.clean_tokens.unsqueeze(0)
    if not corrupt:
        trainenc = dataset.clean_tokens.to(device).unsqueeze(0)
    else:
        trainenc = dataset.corrupted_tokens.to(device)

    tar = torch.full_like(inputs, -100)
    tar[:, target_idx[:, 0], target_idx[:, 1]] = inputs[:, target_idx[:, 0], target_idx[:, 1]]
    for idx in range(nsamples):
        trainloader.append((inputs[:,idx], tar[:,idx]))
        
    return trainloader, None


def get_gendered_pronouns(nsamples, seed, max_seqlen, tokenizer, device="cpu", corrupt=False):
    trainloader = []
    dataset = GenderedPronoun(
        N = nsamples,
        tokenizer=tokenizer,
        device = device,
        seed=seed,
        prepend_bos = True
    )
    max_seqlen = dataset.max_len
    target_idx = dataset.target_idx
    inputs = dataset.clean_tokens.unsqueeze(0)
    if not corrupt:
        trainenc = dataset.clean_tokens.to(device).unsqueeze(0)
    else:
        trainenc = dataset.corrupted_tokens.to(device)

    tar = torch.full_like(inputs, -100)
    tar[:, target_idx[:, 0], target_idx[:, 1]] = inputs[:, target_idx[:, 0], target_idx[:, 1]]
    for idx in range(nsamples):
        trainloader.append((inputs[:,idx], tar[:,idx]))
        
    return trainloader, None
    

# Function to select the appropriate loader based on dataset name
def get_loaders(name='wikitext2', nsamples=128, seed=0, seqlen=2048, tokenizer=None, device=torch.device("cuda:0"), corrupt=False):
    """
    Select the appropriate loader based on dataset name.

    Args:
        name (str): The name of the dataset ('wikitext2', 'c4', or 'ptb').
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation/test set.
    """
    # Determine which dataset to use based on 'name' parameter and return corresponding loader
    if 'wikitext2' in name:
        print("wikitext2")
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if "ioi" in name:
        return get_ioi_without_seq_limit(nsamples, seed, seqlen, tokenizer, device, corrupt=corrupt)
    #elif "ioi" in name:
    #    return get_ioi(nsamples, seed, seqlen, tokenizer, device, corrupt=corrupt)
    elif "GreaterThan" in name:
        print("greaterthan")
        return get_greater_than(nsamples, seed, seqlen, tokenizer, device, corrupt=corrupt)        
    elif "induction" in name:
        print("induction")
        return get_indcution(nsamples, seed, seqlen, tokenizer, device, corrupt=corrupt)
    elif "GenderedPronoun" in name:
        print("GenderedPronoun")
        return get_gendered_pronouns(nsamples, seed, seqlen, tokenizer, device, corrupt=corrupt)
       
if __name__ == "__main__": 
    get_loaders('wikitext2', seed=0, seqlen=2048, tokenizer=None)

# Note:
# This script is designed to load and process various text datasets for training language models.
# It includes functions to load PTB (Penn Treebank), Wikitext-2, and C4 (Common Crawl) datasets.
# Each loading function returns a trainloader (list of input and target pairs) and encoded validation/test set.
