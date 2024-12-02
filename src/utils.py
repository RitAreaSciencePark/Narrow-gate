from dotenv import load_dotenv
import os
from typing import Tuple, Union
import pandas as pd
import torch
from rich import print
from collections import defaultdict
import numpy as np
import pandas as pd
import json
from typing import Callable, Literal
from pathlib import Path
from PIL import Image, ImageOps
import torch.nn.functional as F




def data_path(path: str):
    """
    take a relative path and return the absolute path to the data folder (defined in the .env file) that could be in the scratch or in the local machine
    """
    load_dotenv(dotenv_path=".env")
    load_dotenv(dotenv_path="../.env")
    DATA_DIR = os.getenv("DATA_DIR")
    return f"{DATA_DIR}/{path}"


def GB_per_tensor(shape: Tuple[int], dtype="torch.float32"):
    """
    return the number of GB that a tensor with the given shape and dtype would occupy
    """
    if dtype == "torch.float32":
        bytes_per_element = 4
    elif dtype == "torch.float64":
        bytes_per_element = 8
    elif dtype == "torch.float16":
        bytes_per_element = 2

    for dim in shape:
        bytes_per_element *= dim

    return bytes_per_element / 1e9


def print_dataset_stats(dataset):
    pandas_df = pd.DataFrame(dataset.to_dict())
    print(f"""
Number of samples: {len(pandas_df)}
Number of classes: {len(pandas_df['label'].unique())}
Number of samples per class: {pandas_df.groupby('label').size().mean()}
""")


def get_dataset_dir(
    root_synset_name: str = "entity.n.01",
    tree: bool = False,
    split: str = "validation",
    max_samples_per_class: int = 10,
    n_class: int = 300,
    absolute_path: bool = True,
):
    if absolute_path:
        return data_path(
            f"datasets/imagenet_{root_synset_name}_{'tree' if tree else 'graph'}_{split}_{max_samples_per_class}ImPerClass_{n_class}Class"
        )
    return f"imagenet_{root_synset_name}_{'tree' if tree else 'graph'}_{split}_{max_samples_per_class}ImPerClass_{n_class}Class"


def left_pad(tensor, target_length, pad_value):
    padding_size = target_length - tensor.size(0)
    if padding_size <= 0:
        return tensor
    padding = torch.full(
        (padding_size, *tensor.shape[1:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([padding, tensor], dim=0)


def right_cut(tensor, size, dim):
    if isinstance(dim, int):
        dims = [dim]
        sizes = [size]
    elif isinstance(dim, (tuple, list)):
        dims = dim
        sizes = [size] * len(dims) if isinstance(size, int) else size
    else:
        raise ValueError("dim must be an integer, tuple, or list")

    slices = [slice(None)] * tensor.ndim
    for d, s in zip(dims, sizes):
        slices[d] = slice(-s, None)

    return tensor[tuple(slices)]


def left_cut(tensor, size, dim):
    if isinstance(dim, int):
        dims = [dim]
        sizes = [size]
    elif isinstance(dim, (tuple, list)):
        dims = dim
        sizes = [size] * len(dims) if isinstance(size, int) else size
    else:
        raise ValueError("dim must be an integer, tuple, or list")

    slices = [slice(None)] * tensor.ndim
    for d, s in zip(dims, sizes):
        slices[d] = slice(None, s)

    return tensor[tuple(slices)]



def aggregate_cache_efficient(all_cache):
    aggregated = defaultdict(list)

    # First pass: collect all tensors/data
    for cache in all_cache:
        for key, value in cache.items():
            aggregated[key].append(value)

    # Second pass: process each key
    for key in aggregated:
        if key == "mapping_index":
            aggregated[key] = aggregated[key][0]
        elif key.startswith("pattern_") or key == "input_ids":
            # Keep as list for attention patterns
            continue
        elif key in ["text", "offset", "label", "synset", "root_label", "token_extracted_type", "token_extracted_idx"]:
            # Assuming offset is a list of integers, we can use a list comprehension
            aggregated[key] = [item for sublist in aggregated[key] for item in sublist]
        else:
            # For tensors, use torch.cat with a list comprehension for efficiency
            try:
                aggregated[key] = torch.cat([tensor for tensor in aggregated[key]], dim=0)  # type: ignore
            except:
                try:
                    aggregated[key] = torch.stack([tensor for tensor in aggregated[key]], dim=0)  # type: ignore
                except:
                    try:
                        # get the min of second dimension
                        min_dim = min([tensor.size(1) for tensor in aggregated[key]])
                        aggregated[key] = torch.cat([left_cut(tensor, min_dim, 1) for tensor in aggregated[key]], dim=0)  # type: ignore
                        
                    except:
                        raise ValueError(f"Cannot aggregate key {key}")
    

    return dict(aggregated)


def to_string_tokens(tokens: Union[list, torch.Tensor], processor):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    string_tokens = []
    for tok in tokens:
        string_tokens.append(processor.decode(tok))
    return string_tokens


def cosine_similarity_matrix(A, abs: bool = False):
    abs = False
    if A.ndim > 2:
        A = A.squeeze()
    # Normalize the rows of the matrix A
    A_norm = A / A.norm(dim=1, keepdim=True)

    # Compute the cosine similarity matrix
    cosine_sim_matrix = torch.mm(A_norm, A_norm.t())

    # Take the absolute value to ensure all values are non-negative
    if abs:
        cosine_sim_matrix = torch.abs(cosine_sim_matrix)

    # print(f"shape: {cosine_sim_matrix.shape}")

    # take just the lower triangular part but take the diagonal
    # cosine_sim_matrix = torch.tril(cosine_sim_matrix, diagonal=0)

    return cosine_sim_matrix


def prepare_index_for_slicing(start: int, end: int):
    """
    Utility function to prepare the index for slicing. It takes the start and end index and returns a tuple that can be used for slicing that
    contains the start index and the end index + 1. It support also negative indexing
    """
    assert (
        start < end
    ), f"start index {start} should be smaller than the end index {end}"

    if end < 0:
        end = end + 1


def aggregate_metrics(input_list):
    """
     This function takes a list of dictionaries, where each dictionary represents data for all heads.
    For each head, there's a dictionary of metrics, where each metric has a list of tensors.
    It returns a dictionary of heads, where each head has a dictionary of metrics with aggregated numpy arrays.
    >>> input_data = [
            {
                "head_1": [
                    {
                        "full_density_text_text": [0.1111],
                        "full_density_image_image": [0.1111],
                        "full_density_text->image": [0.1111],
                        "full_density_text_text_no_diag": [0.1111],
                    },
                ],
                "head_2": [
                    {
                        "full_density_text_text": [0.3333],
                        "full_density_image_image": [0.3333],
                        "full_density_text->image": [0.3333],
                        "full_density_text_text_no_diag": [0.3333],
                ]
            },
            {
                "head_1": [
                    {
                        "full_density_text_text": [-0.1111],
                        "full_density_image_image": [-0.1111],
                        "full_density_text->image": [-0.1111],
                        "full_density_text_text_no_diag": [-0.1111],
                ],
                "head_2": [
                    {
                        "full_density_text_text": [0.4444],
                        "full_density_image_image": [0.4444],
                        "full_density_text->image": [0.4444],
                        "full_density_text_text_no_diag": [0.4444],
                ]
            }
        ]

    >>> result = aggregate_metrics(input_data)
    >>> print(result)
    >>> {'head_1': {'full_density_text_text': array([ 0.1111,  0.2222, -0.1111, -0.2222]),
        'full_density_image_image': array([ 0.1111,  0.2222, -0.1111, -0.2222]),
        'full_density_text->image': array([ 0.1111,  0.2222, -0.1111, -0.2222]),
        'full_density_text_text_no_diag': array([ 0.1111,  0.2222, -0.1111, -0.2222])},
        'head_2': {'full_density_text_text': array([0.3333, 0.4444, 0.3333, 0.4444]),
        'full_density_image_image': array([0.3333, 0.4444, 0.3333, 0.4444]),
        'full_density_text->image': array([0.3333, 0.4444, 0.3333, 0.4444]),
        'full_density_text_text_no_diag': array([0.3333, 0.4444, 0.3333, 0.4444])}}

    """
    final_dict = {}

    # Initialize the structure
    for item in input_list:
        for head, metrics in item.items():
            if head not in final_dict:
                final_dict[head] = {metric: [] for metric in metrics}

    # Aggregate the data
    for item in input_list:
        for head, metrics in item.items():
            for metric, tensor_list in metrics.items():
                final_dict[head][metric].extend(tensor_list)

    # Convert lists to numpy arrays, handling BFloat16
    for head in final_dict:
        for metric in final_dict[head]:
            tensor_list = final_dict[head][metric]
            if tensor_list and isinstance(tensor_list[0], torch.Tensor):
                # Convert BFloat16 to Float32 before converting to numpy
                final_dict[head][metric] = np.array(
                    [t.float().cpu().numpy() for t in tensor_list]
                )
            else:
                final_dict[head][metric] = np.array(tensor_list)

    return final_dict


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_code():
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def size_of_object(object):
    from pympler import asizeof

    obj_size_gb = asizeof.asizeof(object) / (1024**3)
    return obj_size_gb


def load_json_config(file_path: str) -> dict:
    """Load and parse the JSON configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r") as file:
        try:
            config = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {file_path}: {e}")

    return config


################################################
# ABLATION UTILS
################################################
def preprocess_ablation_queries(data_list, model_config):
    # Initialize an empty list to store the data for DataFrame
    rows = []
    # Iterate over each dictionary in the input list
    for entry in data_list:
        ablation_type = entry["type"]
        head_layer_couple = entry["head-layer-couple"]
        elem_to_ablate = entry.get("elem-to-ablate", "False")

        if (
            ablation_type == "std"
            or ablation_type == "keep_self_attn"
            or ablation_type == "flash-attn"
        ) and elem_to_ablate == "False":
            raise ValueError(
                "The ablation type 'std' and 'keep_self_attn' require an element to ablate"
            )

        if (
            ablation_type == "image->image" or elem_to_ablate == "image->text"
        ) and elem_to_ablate != "False":
            raise Warning(
                f"elem_to_ablate is not supported with ablation_type={ablation_type}"
            )

        if head_layer_couple[0] == "all":
            for layer in range(model_config.num_hidden_layers):
                for head in range(model_config.num_attention_heads):
                    rows.append(
                        {
                            "elem-to-ablate": elem_to_ablate,
                            "head": head,
                            "layer": layer,
                            "type": ablation_type,
                        }
                    )
        else:
            if len(head_layer_couple) == 0:
                raise ValueError("The head-layer-couple list is empty")

            for i in head_layer_couple:
                head, layer = i
                if head < 0 or head >= model_config.num_attention_heads:
                    raise ValueError(f"Invalid head index {head}")
                if layer < 0 or layer >= model_config.num_hidden_layers:
                    raise ValueError(f"Invalid layer index {layer}")
                # Append a dictionary for each row to be added to the DataFrame
                rows.append(
                    {
                        "elem-to-ablate": elem_to_ablate,
                        "head": head,
                        "layer": layer,
                        "type": ablation_type,
                    }
                )

    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(rows)
    return df


def load_json_config(file_path: str) -> dict:
    """Load and parse the JSON configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r") as file:
        try:
            config = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {file_path}: {e}")

    return config


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of
    torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from
    (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def reshape_tensors(
    tensor: torch.Tensor, hf_model: torch.nn.Module
) -> Tuple[torch.Tensor, dict]:
    # get dimension
    num_heads = hf_model.config.num_attention_heads
    head_dim = hf_model.config.hidden_size // num_heads
    initial_shape = tensor.shape
    bsz, q_len, _ = initial_shape
    num_key_value_heads = hf_model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads
    tensor = tensor.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    tensor = repeat_kv(tensor, num_key_value_groups)
    return tensor, {
        "bsz": bsz,
        "q_len": q_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


# get the position of the tokens alias
def map_token_to_pos(
    token: str,
    _get_token_index: Callable,
    string_tokens: list,
    hf_tokenizer,
    inputs: dict,
) -> int:
    """
    This function takes a token and returns its position in the input_ids.
    Token that starts with '@' are considered as aliases and the function will return the position of the alias in the input_ids.
    """
    if token.startswith("@"):
        return _get_token_index([token[1:]], string_tokens)
    else:
        token_id = hf_tokenizer(token, return_tensors="pt")
        assert token_id["input_ids"][0].numel() == 3, "The token should be a single token"
        result = torch.where(inputs["input_ids"][0] == token_id["input_ids"][0][1])
        if result[0].numel() > 0:
            return result[0][0].item()
        else:
            raise ValueError(f"Token {token} not found in the input_ids")


################################################
# PATCHING UTILS
################################################


def preprocess_patching_queries(
    patching_queries: dict, map_token_to_pos: Callable, model_config
) -> dict:
    """
    Creates a dataframe for the patching hook from a cache object.
    Currently doesn't allow to select specific heads.
    """

    # Initialize an empty list to store the data for DataFrame
    rows = []
    # Iterate over each dictionary in the input list
    for entry in patching_queries:
        if entry["base_activation_index"] is not None:
            token_index = entry["base_activation_index"]
        else:
            token_index = map_token_to_pos(entry["patching_elem"])
        if entry["layers_to_patch"] == "all":
            for layer in model_config.num_hidden_layers:
                rows.append(
                    {
                        "pos_token_to_patch": map_token_to_pos(entry["patching_elem"]),
                        "activation_type": entry["activation_type"],
                        "layer": layer,
                        "patching_activations": entry["patching_activations"][
                            entry["activation_type"].format(layer)
                        ][0, token_index],
                    }
                )
        else:
            for layer in entry["layers_to_patch"]:
                try:
                    rows.append(
                        {
                            "pos_token_to_patch": map_token_to_pos(entry["patching_elem"]),
                            "activation_type": entry["activation_type"],
                            "layer": layer,
                            "patching_activations": entry["patching_activations"][
                                entry["activation_type"].format(layer)
                            ][0, token_index],
                        }
                    )
                except:
                    if entry["activation_type"].format(layer) not in entry["patching_activations"].keys():
                        raise ValueError(f"{entry['activation_type'].format(layer)} not found in the patching_activations {entry['patching_activations'].keys()}")
                    else:
                        # re-raise the exception
                        raise
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(rows)
    return df


def logit_diff(
    base_label_tokens: list,
    target_label_tokens: list,
    target_clean_logits: torch.Tensor,
    target_patched_logits: torch.Tensor,
):
    """
    Computes the difference in logits between the target label and the base label for both the clean and patched models.
    Essentially, given a two list of paired tokens, it computes the difference in logits for each pair for the run with and without the patch.
    Then, it computes the difference between the differences to see how the patch affects the difference in logits between the two labels.
    """
    variations = []
    clean_differences = []
    patched_differences = []

    for s_pos, c_pos in zip(base_label_tokens, target_label_tokens):
        clean_diff = (
         target_clean_logits[:, c_pos] -  target_clean_logits[:, s_pos]
        )  # Shape: [batch_size, d_vocab]
        patched_diff = target_patched_logits[:, c_pos] - target_patched_logits[:, s_pos]
        diff = patched_diff - clean_diff

        clean_differences.append(clean_diff.mean(dim=0))
        patched_differences.append(patched_diff.mean(dim=0))
        variations.append(diff.mean(dim=0))
        
        

    
    return {
        "diff_variation": torch.stack(variations),
        "diff_in_clean": torch.stack(clean_differences),
        "diff_in_patched": torch.stack(patched_differences),
    }


def kl_divergence_diff(base_logits, target_clean_logits, target_patched_logits, epsilon=1e-8):
    """
    Computes KL divergence between the target clean and patched logits and the base logits.
    Essentialy it compute:
        - KL divergence between the target input WITHOUT the patch and the base input
        - KL divergence between the target input WITH the patch and the base input
        - KL divergence between the target input WITH the patch and the target input without the patch
        - The difference between the first two KL divergences in order to see how the patch affects the KL divergence between the two inputs.
    """
    #Convert logits to log probabilities using log_softmax for numerical stability
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    target_clean_log_probs = torch.log_softmax(target_clean_logits, dim=-1)
    target_patched_log_probs = torch.log_softmax(target_patched_logits, dim=-1)
    
    # Convert patched probabilities by adding epsilon to avoid log(0)
    
    kl_clean_base = F.kl_div(target_clean_log_probs, base_log_probs, reduction='batchmean', log_target=True)
    
    # Compute KL divergence between patched and standard distribution
    kl_patched_base = F.kl_div(target_patched_log_probs, base_log_probs, reduction='batchmean', log_target=True)
    
    # Compute KL divergence between patched and clean counterfactual distribution
    kl_patched_clean = F.kl_div(target_patched_log_probs, target_clean_log_probs, reduction='batchmean', log_target=True)
    
    # Calculate the KL divergence difference
    kl_diff = kl_patched_base - kl_clean_base
    
    return {
        'kl_clean|base': kl_clean_base,
        "kl_patched|base": kl_patched_base,
        "kl_patched|clean": kl_patched_clean,
        'kl_difference': kl_diff
    }



def get_attribute_from_name(obj, name):
    """Dynamically get nested attributes from a string name."""
    attributes = name.split(".")
    for attr in attributes:
        if "[" in attr and "]" in attr:
            # Handle array-like access
            attr, index = attr.split("[")
            index = int(index.rstrip("]"))
            obj = getattr(obj, attr)[index]
        else:
            obj = getattr(obj, attr)
    return obj


def resize_img_with_padding(
    image, target_size=(512, 512), background_color=(128, 128, 128)
):
    """
    Resizes the image to the target size.
    - Adds a neutral background if the image is smaller than the target size.
    - Center crops the image if it's larger than the target size.

    Args:
    - image: The input PIL image.
    - target_size: Tuple (width, height) specifying the target size.
    - background_color: Tuple (R, G, B) specifying the background color.

    Returns:
    - A new PIL image resized and adjusted as specified.
    """
    # If the image is smaller than the target size, add padding
    if image.size[0] < target_size[0] or image.size[1] < target_size[1]:
        # Create a new image with the background color
        new_image = Image.new("RGB", target_size, background_color)
        # Calculate the position to paste the original image onto the center of the new image
        paste_position = (
            (target_size[0] - image.size[0]) // 2,
            (target_size[1] - image.size[1]) // 2,
        )
        new_image.paste(image, paste_position)
        return new_image
    else:
        # If the image is larger, perform center cropping
        return ImageOps.fit(
            image, target_size, method=Image.LANCZOS, centering=(0.5, 0.5)
        )
        

def resize_img_if_too_large(img, max_size=500):
    w, h = img.size
    max_aspect_ratio = 5
    if w>max_size or h>max_size:
        aspect_ratio = h / w
        if h>w:
            h=max_size
            w=int(h / aspect_ratio)
        else:
            w=max_size
            h = int(w * aspect_ratio)
        img=img.resize((w, h))
    if h/w > max_aspect_ratio:
        h = int(w*max_aspect_ratio)
        img=img.resize((w,h))
    elif w/h > max_aspect_ratio:
        w = int(h*max_aspect_ratio)
        img=img.resize((w,h))
    
    return img

def log_memory_usage(message):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{message} - Allocated: {allocated}, Reserved: {reserved}")




