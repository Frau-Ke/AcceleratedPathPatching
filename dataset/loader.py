from .IOI_dataset import IOI_dataset
from .PairedFacts import PairedFacts
from .Induction import Indcution
from .GreaterThan import GreaterThan, GreaterThanQwen
from .NameInduction import NameInduction
from .GenderedPronoun import GenderedPronoun
from .Docstring import Docstring
from torch.utils.data import DataLoader

def load_dataset(
    task: str, 
    tokenizer, 
    N:int,
    model_name="gpt2",
    patching_method: str = "path", 
    device:str="cpu",
    seed:int=123456,
    prepend_bos:bool=True,
    remove_target_token=False
    ) :

    if task == "ioi":
        dataset = IOI_dataset(
            model_name=model_name,
            N=N,
            patching_method=patching_method,
            tokenizer=tokenizer,
            device=device,
            seed=seed,
            prepend_bos=prepend_bos, 
            remove_target_token=remove_target_token
            ) 
    elif task == "FactualAssociation":
        dataset = PairedFacts(
            N=N,
            patching_method=patching_method,
            tokenizer=tokenizer,
            device=device
            #TODO: add seed!
            )
        
    elif task == "induction":
        dataset = NameInduction(
            model_name=model_name,
            N=N,
            tokenizer=tokenizer,
            device=device,
            seed=seed,
            prepend_bos=prepend_bos, 
            remove_target_token=remove_target_token
            )

    elif task == "GreaterThan":
        if "Qwen" in model_name:
            dataset = GreaterThanQwen(
                model_name=model_name,
                N=N,
                device=device,
                tokenizer=tokenizer,
                seed=seed,
                prepend_bos=prepend_bos
            )        
        else:
            dataset = GreaterThan(
                model_name=model_name,
                N=N,
                device=device,
                tokenizer=tokenizer,
                seed=seed,
                prepend_bos=prepend_bos
            )
        
    elif  task == "GenderedPronouns":
        dataset = GenderedPronoun(
            model_name=model_name,
            N=N,
            tokenizer=tokenizer,
            device=device,
            seed=seed,
            prepend_bos=prepend_bos
        )
    
    elif task == "Docstring":
        dataset = Docstring(
            model_name=model_name,
            N=N,
            tokenizer=tokenizer,
            device=device,
            seed=seed,
            prepend_bos=prepend_bos
        )
        
    else:
        raise Exception(f"Requested Task {task} is not implemented. \n Choose one of [ioi, induction, GreaterThan, GenderedPronoun, Docstring]")
    
    return dataset



 
def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    ) 