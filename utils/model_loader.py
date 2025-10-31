from transformer_lens import HookedTransformer
import torch
from Pruning.FLAP.models.hf_gpt.modeling_gpt2 import load_pretrained_llama_style_gpt2, GPT2LMHeadModel2Llama
from Pruning.FLAP.models.hf_llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

#----------------------------------------------------------------------------------------------------
#                   Loading Functions for Models
#----------------------------------------------------------------------------------------------------


def load_hooked_transformer(
    model_name: str, 
    device:str = "cuda", 
    patching_method = "", 
    cache_dir="llm_weights",
    eval:bool=True
    ) -> HookedTransformer:
    """Load a Hooked Transformer from the transformer lens librarry

    Args:
        model_name (str)
        device (str): cpu or cuda
        patching_method (str, optional):Different Patching Methods. ACDC needs different setting. Defaults to "".

    Returns:
        HookedTransformer: model
    """
    print(f"loading {model_name} as HookedTransformer")
    
    if device == "cpu":
        dtype=torch.float32
        print("use float 32")
    else:
        dtype=torch.float16
        print("use float 16")
    
    model= HookedTransformer.from_pretrained(
        model_name=model_name,
        center_unembed=False,                    # set mean of every output vector W_U to zero
        center_writing_weights=False,            # normalize all input writing to residual stream
        fold_ln=False,                           # regularisation method: Hooked Transformer handels centering & normalization & linear operations all together, factor out te linear part
                                                # almost linear map: variance scalling divides by norm of vector -> norm not linear
        refactor_factored_attn_matrices=False,   # use low-rank matrices W_OV, W_QK instead of W_O and W_V (W_Q and W_K)
        torch_dtype=dtype,
        cache_dir=cache_dir, 

    )
    #CAREFUL
    #model.set_use_split_qkv_input(True)

    if patching_method == "acdc":

        model.set_use_attn_result(True)
        model.set_use_split_qkv_input(True)
        if "use_hook_mlp_in" in model.cfg.to_dict():
            model.set_use_hook_mlp_in(True)      
    
    if eval:
        model.eval()  
    model.to(device)
    return model


def load_transformer(
    model_name:str, 
    device:str="cuda", 
    cache_dir="llm_weigths",
    eval:bool=True, 
    output_attentions=False, 

) -> AutoModelForCausalLM:
    print(f"load {model_name} as CasualLLM")

    if output_attentions:
        model =  AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=device, 
            cache_dir=cache_dir, 
            output_attentions=output_attentions,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
            )
    else:
        model =  AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=device, 
            cache_dir=cache_dir, 
            output_attentions=output_attentions,
            low_cpu_mem_usage=True,
            )
    
    if eval:
        model.eval()
    model.to(device)
        
    return model
    
    
def get_gpt2_adapt_to_llama(
    model_name:str, 
    device:str="cuda", 
    ) -> GPT2LMHeadModel2Llama:
    
    """Get the GPT2 model from hugging face and adapt it to Llama style:
        - structure transformer.h -> model.layers
        - attn -> self_attn
        - c_attn split into q_proj, k_proj, v_proj, c_proj -> o_proj

    Args:
        model (str): model name
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        GPT2LMHeadModel2Llama: GPT2 model adapted to Llama style
    """
    
    print(f"load {model_name} as CasualLLM adapted to the Llama architecture")
    if not "gpt2" in model_name:
        raise Exception("Adapting model to Llama form for FLAP has only been done for gpt2 type models. Manaual testing and control necessary!")
    model =  load_pretrained_llama_style_gpt2(model_name)
    
    model.config.intermediate_size = model.config.n_embd
    model.config.num_attention_heads = model.config.n_head
    
    model.eval()
    model.to(device)
    return model

def get_Llama_transformer(
    model_name:str,
    cache_dir:str="llm_weights", 
    device:str="cuda",
    task=""
) -> LlamaForCausalLM:
    print(f"load {model_name} as Llama model")
    
    if device == "cpu":
        dtype=torch.float32
        device_map="cpu"
    else:
        dtype=torch.float16
        device_map="auto"
    
   
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map=device_map,
       )
    
    for i in range(32):
        model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros_like(model.model.layers[i].self_attn.o_proj.bias, device=device))  # 或 'cuda'
        #model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(torch.zeros_like(model.model.layers[i].mlp.down_proj.bias, device='cpu'))  # 或 'cuda'
        torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        #torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
        

    return model

#----------------------------------------------------------------------------------------------------
#                   Loading Function for Tokenizer
#----------------------------------------------------------------------------------------------------

def load_tokenizer(
    model_name:str, 
    force_download=False
    ) -> AutoTokenizer:
    """Load the Tokenizer of a model

    Args:
        model_name (str): model name

    Returns:
        AutoTokenizer: tokenizer
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=force_download)
    
    if "Qwen" in model_name:
        print("load Qwen tokenizer")
        if tokenizer.pad_token is None:
            tokenizer.pad_token =  tokenizer.special_tokens_map["pad_token"]

        if tokenizer.bos_token is None:
            tokenizer.bos_token =  tokenizer.pad_token
            tokenizer.bos_token_id = tokenizer.pad_token_id
    
    elif "gpt2" in model_name:
        print("load gpt2 tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
    
    elif "llama" in model_name:#
        print("load Llama tokenizer")
        tokenizer.eos_token_id = 0
        tokenizer.pad_token_id = 0
        
    return tokenizer

#----------------------------------------------------------------------------------------------------
#                   Tests 
#----------------------------------------------------------------------------------------------------


def test_gpt2_adapt_to_llama():
    gpt2_adapt = get_gpt2_adapt_to_llama("gpt2")
    gpt_orig =  load_transformer("gpt2", cache_dir="/mnt/lustre/work/eickhoff/esx670/llm_weights")
    
    print("Compare the gpt2 model to the gpt2 model adapted to the Llama acrchitecutre for FLAP")
        
    ### Test structural similarity
    # Check that the weights of original gpt2 and adapted gpt2 are the same.
    layer = 10

    print("##### Test the Strucutral Similarity of layer ", layer, " #####")
    print("q_proj equal?")
    print(torch.all(gpt_orig.transformer.h[layer].attn.c_attn.weight.data[:, :768] == gpt2_adapt.model.layers[layer].self_attn.q_proj.weight.data.t()).item())

    print("k_proj equal?")
    print(torch.all(gpt_orig.transformer.h[layer].attn.c_attn.weight.data[:, 768:2*768] == gpt2_adapt.model.layers[layer].self_attn.k_proj.weight.data.t()).item())

    print("v_proj equal?")
    print(torch.all(gpt_orig.transformer.h[layer].attn.c_attn.weight.data[:, 2*768:3*768] == gpt2_adapt.model.layers[layer].self_attn.v_proj.weight.data.t()).item())

    print("o_proj equal?")
    print(torch.all(gpt_orig.transformer.h[layer].attn.c_proj.weight.data == gpt2_adapt.model.layers[layer].self_attn.o_proj.weight.data.t()).item())
    
    
    ### Test for Functional Similarity
    print("Adapted gpt2 model is functionally equal to original gpt2 model?")

    tokenizer = load_tokenizer("gpt2")
    inputs = tokenizer("Hello, world! This is a huge test.", return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    position_ids = torch.arange(0, input_ids.size(-1)).unsqueeze(0)  # Generate position IDs

    outputs_adapt = gpt2_adapt(
            input_ids = input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    outputs_gpt2 = gpt_orig(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        position_ids=position_ids)

    print("Are the output logits equal?", torch.all(torch.isclose(outputs_adapt.logits, outputs_gpt2.logits)).item())