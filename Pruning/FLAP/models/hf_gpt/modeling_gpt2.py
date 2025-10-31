# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""
import logging
import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
    )
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput
)
from transformers import  GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from transformers import GPT2LMHeadModel,load_tf_weights_in_gpt2

from logger_config import logger

_CHECKPOINT_FOR_DOC = "openai-community/gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

def load_pretrained_llama_style_gpt2(model_name):
    """Loads a standard GPT-2 model and transfers weights to the Llama-style structure."""
    #config = GPT2Config.from_pretrained(model_name)
    config = GPT2Config.from_pretrained(model_name)
    original_model = GPT2LMHeadModel.from_pretrained(model_name)
    original_sd = original_model.state_dict()
    
    # Instantiate your custom model
    custom_model = GPT2LMHeadModel2Llama(config)

    # Transfer weights state_dict by state_dict (carefully matching names)
    custom_sd = custom_model.state_dict()
    custom_sd_template = custom_model.state_dict()
    
    new_state_dict = {}
    mapped_keys = set()
    
    for name, param in custom_sd.items():
        if "self_attn.comp_vector" in name:    
            new_state_dict[name] = param
                
    for name, param in original_sd.items():
        mapped_name = name
        # --- Direct mappings ---
        mapped_name = name.replace("transformer.", "model.") 
        mapped_name = mapped_name.replace("h.", "layers.") 
        mapped_name = mapped_name.replace(".attn.", "self.attn")
        
        is_attention_weight = False

        # --- Attention Layer Weights (The tricky part) ---
        if "attn.c_attn.weight" in name:
            is_attention_weight = True

            # Split Q, K, V weights
            layer_idx = name.split('.')[2] # Extract layer index

            q_w, k_w, v_w = torch.split(param, config.hidden_size, dim=1) # dim=1 for Conv1D weight
            # Map to new layers (adjust path based on your final class structure)
            new_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_w.contiguous().T # Transpose for nn.Linear
            new_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_w.contiguous().T # Transpose for nn.Linear
            new_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_w.contiguous().T # Transpose for nn.Linear
            
            logger.info(f"Splitting and mapping: {name} -> QKV weights for layer {layer_idx}")
            mapped_keys.update([
                f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            ])
        elif "attn.c_attn.bias" in name:
            is_attention_weight = True

            # Split Q, K, V biases
            layer_idx = name.split('.')[2]
            q_b, k_b, v_b = torch.split(param, config.hidden_size, dim=0)
            
            new_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = q_b.contiguous()
            new_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = k_b.contiguous()
            new_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = v_b.contiguous()
            
            logger.info(f"Splitting and mapping: {name} -> QKV biases for layer {layer_idx}")
            mapped_keys.update([
                 f"model.layers.{layer_idx}.self_attn.q_proj.bias",
                 f"model.layers.{layer_idx}.self_attn.k_proj.bias",
                 f"model.layers.{layer_idx}.self_attn.v_proj.bias"
            ])   
            
        elif "attn.c_proj.weight" in name:
            is_attention_weight = True

            # Map output projection weight
            layer_idx = name.split('.')[2]
            new_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = param.contiguous().T # Transpose for nn.Linear
            logger.info(f"Mapping: {name} -> o_proj weight for layer {layer_idx}")
            mapped_keys.update([f"model.layers.{layer_idx}.self_attn.o_proj.weight"])
            
        elif "attn.c_proj.bias" in name:
            
            is_attention_weight = True
            # Map output projection bias
            layer_idx = name.split('.')[2]
            new_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.bias"] = param.contiguous()
            logger.info(f"Mapping: {name} -> o_proj bias for layer {layer_idx}")
            mapped_keys.update([f"model.layers.{layer_idx}.self_attn.o_proj.bias"])

            
        if not is_attention_weight:
            if mapped_name in custom_sd_template:
                if custom_sd_template[mapped_name].shape == param.shape:
                    logger.debug(f"Mapping directly: {name} -> {mapped_name}")
                    new_state_dict[mapped_name] = param
                    mapped_keys.add(mapped_name)
                else:
                    logger.warning(f"[Shape Mismatch] Param {name} (mapped to {mapped_name}) "
                                f"shape {param.shape} != custom model shape "
                                f"{custom_sd_template[mapped_name].shape}. Skipping.")
            elif name in custom_sd_template and custom_sd_template[name].shape == param.shape:
                logger.debug(f"Mapping fallback original name: {name} -> {name}")
                new_state_dict[name] = param
                mapped_keys.add(name)
            else:
                # This case should ideally not happen for standard GPT-2 layers
                # if structure is preserved (MLP, LN, Embeddings, LM Head)
                logger.warning(f"[Not Found] Param {name} (mapped to {mapped_name}) "
                                f"not found in custom model state dict template. Skipping.")
        
    # --- Final Checks and Loading ---
    missing_keys = set(custom_sd_template.keys()) - mapped_keys
    if missing_keys:
        logger.warning(f"Missing keys in the new state dict: {missing_keys}")

    unexpected_keys = set(new_state_dict.keys()) - set(custom_sd_template.keys())
    if unexpected_keys:
        logger.warning(f"Unexpected keys created (will be ignored by load_state_dict with strict=True): {unexpected_keys}")

    logger.info("Loading the constructed state dictionary into the custom model...")
    try:

        load_result = custom_model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"Load state dict result: Missing={load_result.missing_keys}, Unexpected={load_result.unexpected_keys}")
        if load_result.missing_keys:
             logger.error(f"CRITICAL: State dict loading reported missing keys: {load_result.missing_keys}")
        logger.info("Strict state dict loading successful.")

    except Exception as e:
        logger.error(f"Error loading state dict: {e}")
        raise
    
    return custom_model



class GPT2PreTrainedModel2Llama(PreTrainedModel, GenerationMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = load_pretrained_llama_style_gpt2
    base_model_prefix = "model"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block2Llama"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "o_proj.weight" and isinstance(module, GPT2Attention2Llama):
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

class GPT2Attention2Llama(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        
        self.embed_dim = config.hidden_size
        
        self.orig_num_heads = config.num_attention_heads
        self.num_heads = self.orig_num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = getattr(config, 'reorder_and_upcast_attn', False) # Handle potential config diffs
        
        # Transform c_attn to q_proj, k_proj, v_proj
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj.bias.requires_grad=False
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj.bias.requires_grad=False

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj.bias.requires_grad=False


        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.o_proj.bias.requires_grad=False

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()
        
        self.comp_vector = nn.Parameter(torch.zeros(self.embed_dim), requires_grad=False)
        self.pruned_heads = []
       
    def set_comp_vector(self, comp_vector):
        self.comp_vector.data.copy_(comp_vector)
        self.comp_vector.requires_grad = False
    
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape) 
    
    def remove_head(self, head_num):
        self.pruned_heads.append(head_num)
        # for each head that has already been removed with a smaller index than head_num, head_num shrinks
        for head in self.pruned_heads:
            if head < head_num:
                head_num -= 1

        if head_num < 0 or head_num >= self.num_heads:
            raise ValueError(f"Head number {head_num} out of range for {self.num_heads} heads.")
        
        start = self.head_dim * head_num
        end = self.head_dim * (head_num+1)
        
        # get the new weights and biases 
        new_q_weight = torch.vstack((self.q_proj.weight.data[:start], self.q_proj.weight.data[end:]))
        new_k_weight = torch.vstack((self.k_proj.weight.data[:start], self.k_proj.weight.data[end:]))
        new_v_weight = torch.vstack((self.v_proj.weight.data[:start], self.v_proj.weight.data[end:]))
        new_o_weight = torch.hstack((self.o_proj.weight.data[:, :start], self.o_proj.weight.data[:, end:]))
        
        new_q_bias = torch.hstack((self.q_proj.bias.data[:start], self.q_proj.bias.data[end:]))  
        new_k_bias = torch.hstack((self.k_proj.bias.data[:start], self.k_proj.bias.data[end:]))  
        new_v_bias = torch.hstack((self.v_proj.bias.data[:start], self.v_proj.bias.data[end:]))  

        # set models weights and biases to the pruned weights
        self.q_proj.weight.data = new_q_weight
        self.k_proj.weight.data = new_k_weight
        self.v_proj.weight.data = new_v_weight
        self.o_proj.weight.data = new_o_weight
        
        self.q_proj.bias.data = new_q_bias
        self.k_proj.bias.data = new_k_bias
        self.v_proj.bias.data = new_v_bias
        
        # reset the size of the output features and the number of heads
        self.q_proj.out_features = self.q_proj.out_features - self.head_dim
        self.k_proj.out_features = self.k_proj.out_features - self.head_dim
        self.v_proj.out_features = self.v_proj.out_features - self.head_dim
        self.o_proj.in_features = self.o_proj.in_features - self.head_dim
        
        self.num_heads -= 1
        self.embed_dim -= self.head_dim

    def split_heads(self, x, k=False):
        if self.num_heads > 0:
            new_x_shape = x.size()[:-1] + (self.num_heads, x.size(-1) // self.num_heads)
            x = x.view(*new_x_shape) 
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else: 
            return x


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Query, Key, Value are already projected!
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        nd, ns = query.size(-2), key.size(-2)
        
        if not self.is_cross_attention:
            mask = self.bias[:, :, ns - nd : ns, :ns] 
            attn_weights = torch.where(mask.bool(), attn_weights.to(attn_weights.dtype), self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if not self.reorder_and_upcast_attn: # Check config flag
            attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    
        
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)    
        
        query = self.split_heads(query_states)
        key = self.split_heads(key_states, k=True)
        value = self.split_heads(value_states)
                
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
            
        current_key_length = key.size(-2)

        if current_key_length > self.bias.shape[-1]:
            # Recreate bias buffer if needed for longer sequences
            new_bias = torch.tril(torch.ones((current_key_length, current_key_length), dtype=torch.bool, device=hidden_states.device)).view(
            1, 1, current_key_length, current_key_length
            )
            self.bias = new_bias
            
        
        attn_output, attn_weight = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self.merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        comp = self.comp_vector.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H]
        attn_output = attn_output + comp
        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)        
    
        return outputs  # a, present, (attentions)


class GPT2Block2Llama(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * self.hidden_size

        self.ln_1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.self_attn = GPT2Attention2Llama(config=config, layer_idx=layer_idx) # use new attention block
        self.ln_2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention2Llama(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_ids=position_ids
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2Model2Llama(GPT2PreTrainedModel2Llama):
    _supports_param_buffer_assignment = False

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList([GPT2Block2Llama(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.layers[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.layers)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
            

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if getattr(self.config, "gradient_checkpointing", False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_ids=position_ids
                )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class GPT2LMHeadModel2Llama(GPT2PreTrainedModel2Llama, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT2Model2Llama(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.device_map = None
        
        #c 
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
