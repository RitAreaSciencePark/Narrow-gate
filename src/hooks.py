import torch
from copy import deepcopy
import pandas as pd
from typing import List, Callable, Union
from einops import rearrange, einsum
from src.utils import repeat_kv
from functools import partial
from line_profiler import profile
import re
import torch.nn as nn

def parse_module_path(module_path):
    pattern = r'([^\.\[\]]+)(?:\[(\d+)\])?'
    components = []
    for attr, idx in re.findall(pattern, module_path):
        components.append(attr)
        if idx:
            components.append(int(idx))
    return components

# 2. Retrieving the module
def get_module_by_path(model, module_path):
    components = parse_module_path(module_path)
    module = model
    for comp in components:
        if isinstance(comp, str):
            if hasattr(module, comp):
                module = getattr(module, comp)
            else:
                raise AttributeError(f"Module '{type(module).__name__}' has no attribute '{comp}'")
        elif isinstance(comp, int):
            if isinstance(module, (list, nn.ModuleList, nn.Sequential)):
                module = module[comp]
            else:
                raise TypeError(f"Module '{type(module).__name__}' is not indexable")
        else:
            raise ValueError(f"Invalid component '{comp}' in module path")
    return module

def create_dynamic_hook(pyvene_hook: Callable, **kwargs):
    partial_hook = partial(pyvene_hook, **kwargs)

    def wrap(*args, **kwargs):
        return partial_hook(*args, **kwargs)

    return wrap


def embed_hook(module, input, output, cache, cache_key):
    """
    Hook function to extract the embeddings of the tokens. It will save the embeddings in the cache (a global variable out the scope of the function)
    """
    if output is None:
        b = input[0]
    cache[cache_key] = b.data.detach().clone()
    return b

# Define a hook that saves the activations of the residual stream
def save_resid_hook(module, input, output,  cache, cache_key, token_index,  ):
    """
    It save the activations of the residual stream in the cache. It will save the activations in the cache (a global variable out the scope of the function)
    """
    if output is None:
        if isinstance(input, tuple):
            b  = input[0]
        else:
            b  = input
    else:
        if isinstance(output, tuple):
            b = output[0]
        else:
            b = output[0]
        
    cache[cache_key] = b.data.detach().clone()[..., token_index, :]
      # slice the tensor to get the activations of the token we want to extract



def avg_hook(module, input, output,  cache, cache_key, last_image_idx, end_image_idx,  ):
    """
    It save the activations of the residual stream in the cache. It will save the activations in the cache (a global variable out the scope of the function)
    """
    if output is None:
        b  = input[0]
    else:
        b = output[0]
        
        
    img_avg = torch.mean(b.data.detach().clone()[:, 1 : last_image_idx + 1, :], dim=1,)
    text_avg = torch.mean(b.data.detach().clone()[:, end_image_idx:, :], dim=1)
    all_avg = torch.mean(b.data.detach().clone()[:, :, :], dim=1)

    cache[f"avg_{cache_key}"] = torch.cat(
        [img_avg.unsqueeze(1), text_avg.unsqueeze(1), all_avg.unsqueeze(1)], dim=1
    )

def zero_ablation(tensor):
    # Set the attention values to zero
    return torch.zeros_like(tensor)

        
# b.copy_(attn_matrix)
def ablate_attn_mat_hook(module, input, output, ablation_queries: pd.DataFrame,  ):
    """
    Hook function to ablate the tokens in the attention
    mask. It will set to 0 the value vector of the
    tokens to ablate
    """
    # Get the shape of the attention matrix
    b = output
    batch_size, num_heads, seq_len_q, seq_len_k = b.shape

    q_positions = ablation_queries['queries'].iloc[0]

    # Used during generation
    if seq_len_q <  len(q_positions):
        q_positions = 0

    k_positions = ablation_queries['keys'].iloc[0]

    # Create boolean masks for queries and keys
    q_mask = torch.zeros(seq_len_q, dtype=torch.bool, device=b.device)
    q_mask[q_positions] = True  # Set positions to True

    k_mask = torch.zeros(seq_len_k, dtype=torch.bool, device=b.device)
    k_mask[k_positions] = True  # Set positions to TrueW

    # Create a 2D mask using outer product
    head_mask = torch.outer(q_mask, k_mask)  # Shape: (seq_len_q, seq_len_k)

    # Expand mask to match the dimensions of the attention matrix
    # Shape after expand: (batch_size, num_heads, seq_len_q, seq_len_k)
    head_mask = head_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

    # Apply the ablation function directly to the attention matrix
    b[head_mask] = zero_ablation(b[head_mask])
    return b

    
def ablate_tokens_hook_flash_attn(module, input, output, ablation_queries: pd.DataFrame, num_layers: int = 32,  ):
    """ same of ablate_tokens_hook but for flash attention. This apply the ablation on the values vectors instead of the attention mask"""
    b = output
    batch_size, seq, d_model = b.shape
    if seq == 1:
        return b
    values = b.clone().data
    device = values.device
        
    ablation_queries.reset_index(drop=True, inplace=True) # Reset index to avoid problems with casting to tensor
    head_indices = torch.tensor(
        ablation_queries["head"], dtype=torch.long, device=device
    )
    pos_indices = torch.tensor(
        ablation_queries["keys"], dtype=torch.long, device=device
    )
    # if num_layers != len(head_indices) or not torch.all(pos_indices == pos_indices[0]) :
    #     raise ValueError("Flash attention ablation should be done on all heads at the same layer and at the same token position")
    # if seq < pos_indices[0]:
    #     # during generation the desired value vector has already been ablated
    #     return b
    pos_indices = pos_indices[0]
    # Use advanced indexing to set the specified slices to zero
    values[..., pos_indices, :] = 0

    

    b.copy_(values)
    
    #!!dirty fix
    
    return b
            

def ablate_heads_hook(module, input, output, ablation_queries: pd.DataFrame,  ):
    """
    Hook function to ablate the heads in the attention
    mask. It will set to 0 the output of the heads to
    ablate
    """
    b = output[0]
    attention_matrix = b.clone().data

    for head in ablation_queries["head"]:
        attention_matrix[0, head, :, :] = 0

    b.copy_(attention_matrix)
    return b



def ablate_pos_keep_self_attn_hook(module, input, output, ablation_queries: pd.DataFrame,  ):
    """
    Hook function to ablate the tokens in the attention
    mask but keeping the self attn weigths.
    It will set to 0 the row of tokens to ablate except for
    the las position
    """
    b = output[0]
    Warning("This function is deprecated. Use ablate_attn_mat_hook instead")
    attn_matrix = b.data
    # initial_shape = attn_matrix.shape

    for head, pos in zip(
        ablation_queries["head"], ablation_queries["pos_token_to_ablate"]
    ):
        attn_matrix[0, head, pos, :-1] = 0

    b.copy_(attn_matrix)

    return b


def ablate_tokens_hook(*args, **kwargs):
    raise NotImplementedError("This function will be discaderd keeping only for backward compatibility")

def ablate_images_hook(*args, **kwargs):
    raise NotImplementedError("This function will be discaderd keeping only for backward compatibility")
def ablate_image_image_hook(*args, **kwargs):
    raise NotImplementedError("This function will be discaderd keeping only for backward compatibility")

def projected_value_vectors_head(
    module,
    input,
    output,
    layer,
    cache,
    num_attention_heads: int,
    num_key_value_heads: int,
    hidden_size: int,
    d_head:int,
    out_proj_weight,
    out_proj_bias,
    head: Union[str, int] = "all",
    act_on_input = False,
    expand_head: bool = True,
):
    """
    Hook function to extract the values vectors of the heads. It will extract the values vectors and then project them with the final W_O projection
    As the other hooks, it will save the activations in the cache (a global variable out the scope of the function)

    Args:
        - b: the input of the hook function. It's the output of the values vectors of the heads
        - s: the state of the hook function. It's the state of the model
        - layer: the layer of the model
        - head: the head of the model. If "all" is passed, it will extract all the heads of the layer
        - expand_head: bool to expand the head dimension when extracting the values vectors and the attention pattern. If true, in the cache we will have a key for each head, like "value_L0H0", "value_L0H1", ...
                        while if False, we will have only one key for each layer, like "value_L0" and the dimension of the head will be taken into account in the tensor.

    """
    # first get the values vectors
    if output is None:
        b = input
    else:
        b = output
    
    values = b.data.detach().clone()  # (batch, num_heads,seq_len, head_dim)

    # reshape the values vectors to have a separate dimension for the different heads
    values = rearrange(
        values,
        "batch seq_len (num_key_value_heads d_heads) -> batch num_key_value_heads seq_len d_heads",
        num_key_value_heads=num_key_value_heads,
        d_heads=d_head,
    )
    
    #        "batch seq_len (num_key_value_heads d_heads) -> batch seq_len num_key_value_heads d_heads",

    
    values = repeat_kv(values, num_attention_heads // num_key_value_heads)
    
    values = rearrange(
        values,
        "batch num_head seq_len d_model -> batch seq_len num_head d_model",
    )

    # reshape in order to get the blocks for each head
    out_proj_weight = out_proj_weight.t().view(
        num_attention_heads,
        d_head,
        hidden_size,
    )

    # apply bias if present (No in Chameleon)
    if out_proj_bias is not None:
        out_proj_bias = out_proj_bias.view(1, 1, 1, hidden_size)

    # apply the projection for each head
    projected_values = einsum(
        values,
        out_proj_weight,
        "batch seq_len num_head d_head, num_head d_head d_model -> batch seq_len num_head d_model",
    )
    if out_proj_bias is not None:
        projected_values = projected_values + out_proj_bias

    # rearrange the tensor to have dimension that we like more
    projected_values = rearrange(
        projected_values,
        "batch seq_len num_head d_model -> batch num_head seq_len d_model",
    )

    # post-process the values vectors
    if head == "all":
        for head_idx in range(num_attention_heads):
                cache[f"projected_value_L{layer}H{head_idx}"] = projected_values[:, head_idx]
    else:
        cache[f"projected_value_L{layer}H{head}"] = projected_values[:, head]


def avg_attention_pattern_head(
    module,
    input,
    output,
    layer,
    attn_pattern_current_avg,
    batch_idx,
    cache,
    extract_avg_value: bool = False,
    act_on_input = False,
):
    """
    Hook function to extract the average attention pattern of the heads. It will extract the attention pattern and then average it.
    As the other hooks, it will save the activations in the cache (a global variable out the scope of the function)

    Args:
        - b: the input of the hook function. It's the output of the attention pattern of the heads
        - s: the state of the hook function. It's the state of the model
        - layer: the layer of the model
        - head: the head of the model
        - attn_pattern_current_avg: the current average attention pattern
    """
    # first get the attention pattern
    if output is None:
        b = input
    else:
        b = output
        
    attn_pattern = b.data.detach().clone()  # (batch, num_heads,seq_len, seq_len)
    # attn_pattern = attn_pattern.to(torch.float32)
    num_heads = attn_pattern.size(1)
    for head in range(num_heads):
        key = f"pattern_L{layer}H{head}"
        if key not in attn_pattern_current_avg:
            attn_pattern_current_avg[key] = attn_pattern[:, head]
        else:
            attn_pattern_current_avg[key] += (
                 attn_pattern[:, head] - attn_pattern_current_avg[key]
            ) / (batch_idx+1)
        attn_pattern_current_avg[key] = attn_pattern_current_avg[key]
        # var_key = f"M2_pattern_L{layer}H{head}"
        # if var_key not in attn_pattern_current_avg:
        #     attn_pattern_current_avg[var_key] = torch.zeros_like(attn_pattern[:, head])
        # attn_pattern_current_avg[var_key] = attn_pattern_current_avg[var_key] + (attn_pattern[:, head] - attn_pattern_current_avg[key]) * (attn_pattern[:, head] - attn_pattern_current_avg[var_key])
        
        if extract_avg_value:
            value_key = f"projected_value_L{layer}H{head}"
            try:
                values = cache[value_key]
            except KeyError:
                print(f"Values not found for {value_key}")
                return
            # get the attention pattern for the values
            value_norm = torch.norm(values, dim=-1)
            
            norm_matrix = (
                value_norm.unsqueeze(1).expand_as(attn_pattern[:, head]).transpose(1, 2)
            )
            
            norm_matrix = norm_matrix * attn_pattern[:, head]
            
            if value_key not in attn_pattern_current_avg:
                attn_pattern_current_avg[value_key] = norm_matrix
            else:
                attn_pattern_current_avg[value_key] += (
                    norm_matrix - attn_pattern_current_avg[value_key]    
                ) / (batch_idx+1)
                
                
        # remove values from cache
            del cache[value_key]
            
            
def attention_pattern_head(
    module,
    input,
    output,
    layer,
    cache,
    head: Union[str, int] = "all",
    act_on_input = False,
):
    """
    Hook function to extract the attention pattern of the heads. It will extract the attention pattern.
    As the other hooks, it will save the activations in the cache (a global variable out the scope of the function)

    Args:
        - b: the input of the hook function. It's the output of the attention pattern of the heads
        - s: the state of the hook function. It's the state of the model
        - layer: the layer of the model
        - head: the head of the model
        - expand_head: bool to expand the head dimension when extracting the values vectors and the attention pattern. If true, in the cache we will have a key for each head, like "pattern_L0H0", "pattern_L0H1", ...
                        while if False, we will have only one key for each layer, like "pattern_L0" and the dimension of the head will be taken into account in the tensor.

    """
    # first get the attention pattern
    if output is None:
        b = input
    else:
        b = output
        
    attn_pattern = b.data.detach().clone()  # (batch, num_heads,seq_len, seq_len)

    if head == "all":
        for head_idx in range(attn_pattern.size(1)):
            key = f"pattern_L{layer}H{head_idx}"
            cache[key] = attn_pattern[:, head_idx]
    else:
        cache[f"pattern_L{layer}H{head}"] = attn_pattern[:, head]
        
        
        
    