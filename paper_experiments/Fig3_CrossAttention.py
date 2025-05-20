import os
from argparse import ArgumentParser
import torch
from dataclasses import dataclass, field
from src.activations import ExtractActivationConfig, Extractor
from pathlib import Path
from rich import print
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

load_dotenv("../.env")
DATA_DIR = Path(os.getenv("DATA_DIR"))

@dataclass
class ModelConfig:
    num_layers: int
    num_heads: int
    img_len: int
    special_tokens: int = -1
    initial_tokens: int = 2
    labels: list = field(default_factory=lambda: ['internal image', 'last-image', 'EOI'])
        

MODEL_CONFIG = {
    "facebook/chameleon-7b": ModelConfig(num_layers=32, num_heads=32, img_len=1024, special_tokens=30,
                                         labels=['32nd token', 'internal image', 'last image', 'EOI']),
    "facebook/chameleon-30b": ModelConfig(num_layers=48, num_heads=64, img_len=1024,  special_tokens=0,
                                          labels=['1st token', 'internal image', 'last image', 'EOI']),
    "llava-onevision-7b": ModelConfig(num_layers=28, num_heads=28, img_len=1488,  initial_tokens=3),
    "mistral-community/pixtral-12b": ModelConfig(num_layers=40, num_heads=32, img_len=1053, special_tokens=1023,
                                       labels=['1025th', 'internal image','[EOL]s', 'EOI']),
    "vila-u": ModelConfig(num_layers=32, num_heads=32, img_len=256),
    "deepseek-ai/Janus-1.3B": ModelConfig(num_layers=24, num_heads=16, img_len=575),
    "Emu3-Gen-hf": ModelConfig(num_layers=32, num_heads=32, img_len=1024)
}


def cross_attention_patterns(model, dict_attention_maps):
    """
    Analyze cross-attention patterns between text and image tokens across model layers and heads.
    This function extracts cross-attention patterns from the model's attention maps and
    computes statistics (mean strength and variance) of how text tokens attend to 
    image tokens across different layers.
    Parameters
    ----------
    model : str
        Identifier for the model to analyze. Must be a key in the MODEL_CONFIG dictionary.
    dict_attention_maps : dict
        Dictionary containing attention maps with keys formatted as "pattern_L{layer}H{head}",
        where {layer} is the layer index and {head} is the attention head index.
    Returns
    -------
    dict
        - 'mean': numpy array of shape (num_layers, img_len) containing the mean attention
                  strength for each image token across layers
        - 'std': numpy array of shape (num_layers, img_len) containing the standard deviation
                 of attention for each image token across layers

    """

    # Get configuration for the current model
    config = MODEL_CONFIG.get(model)
    if config is None:
        raise ValueError(f"Configuration not found for model: {model}")

    index_before_text = config.img_len+config.initial_tokens+1
    str_tokens = np.zeros((config.num_layers, index_before_text ))
    var_tokens = np.zeros((config.num_layers, index_before_text ))

    for L in range(config.num_layers):
        for H in range(config.num_heads):
            name_att = f'pattern_L{L}H{H}'
            tmap = dict_attention_maps[name_att].float()
            crosstalk = tmap[0, index_before_text:,:index_before_text]
            token_strength, token_id = torch.sort(crosstalk.mean(0), descending=True)
            str_tokens[L] += crosstalk.mean(0).cpu().numpy()
            var_tokens[L] += crosstalk.std(0).cpu().numpy()

    dist_res = {'mean':str_tokens, 'std':var_tokens}
    return dist_res


# PLOT

def plot_cross_attention(model, dict_data, save_path):

    config = MODEL_CONFIG.get(model)
    
    # Normalize the data
    # import pdb; pdb.set_trace()
    norm = dict_data['mean'][:,config.initial_tokens:] / dict_data['mean'][:,config.initial_tokens:].sum(1,keepdims=True)
    last_img = config.img_len -1
    end_img = config.img_len 
    
        
        
    if config.special_tokens > 0:
        tot = np.zeros((config.num_layers, 4))
        tot[:,0] = norm[:,config.special_tokens]
        tot[:,1] = norm[:,:end_img].sum(1) - tot[:,0] 
        if model == "mistral-community/pixtral-12b":
            tot[:,2] = norm[:,:end_img].sum(1) - tot[:,0]
        else:
            tot[:,2] = norm[:,last_img] # last image
        tot[:,3] = norm[:,end_img] # EOI
    else:
        tot = np.zeros((config.num_layers, 3))
        tot[:,0] = norm[:,:end_img].sum(1)
        tot[:,1] = norm[:,last_img]
        tot[:,2] = norm[:,end_img]
    
    xs = np.arange(config.num_layers+1)
    cum = np.cumsum(tot,axis=1)
    cum /= cum[:,-1:]
    cum = np.vstack((cum, cum[-1,:]))   
    # Plotting
    plt.grid()
    plt.xlim(0,len(xs))
    plt.ylim(0,1)

    colors = [
            "#4477AA",
            "#EE6677",
            "#228833",
            "#1187AA",
            "#66CCEE",
            "#AA3377",
            "#BBBBBB",
        ]

    plt.title(f'Token relevance in crossmodal communication - {model}')
    plt.ylabel('Cumulative crossmodal attention')
    plt.xlabel('Layer')
    plt.fill_between(xs, 0,        cum[:,0], label=config.labels[0], step='post', color=colors[3])
    plt.fill_between(xs, cum[:,0], cum[:,1], label=config.labels[1], step='post', color=colors[0])
    plt.fill_between(xs, cum[:,1], cum[:,2], label=config.labels[2], step='post', color=colors[1])
    if config.special_tokens > 0:
        plt.fill_between(xs, cum[:,2], cum[:,3], label=config.labels[3], step='post', color=colors[2])

    plt.step(xs, cum[:,0], color='black', where='post')
    plt.step(xs, cum[:,1], color='black', where='post')
    if config.special_tokens > 0:
        plt.step(xs, cum[:,2], color='black', where='post')


    plt.legend()
  
    plt.savefig(save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                    type=str,
                    help=" \
                        facebook/chameleon-7b, \
                        facebook/chameleon-30b, \
                        mistral-community/pixtral-12b, \
                        vila-u, \
                        BAAI/Emu2, \
                        deepseek-ai/Janus-1.3B, \
                        llava-onevision-7b, \
                        Emu3-Gen-hf",
                    required=True)
    parser.add_argument("-o", "--out_dir", type=str, default="out")
    parser.add_argument("-s", "--save", action='store_true', default=False)
    parser.add_argument("--small_size", action='store_true', default=False)

    args = parser.parse_args()
    if args.small_size:
        print("SMALL SIZE")
    print(f"Model: {args.model}\nOutdir: {args.out_dir}\nSave: {args.save}")
    print("Extracting text hidden vectors...")

    if "llava" in args.model:
        map_dataset_parallel_mode = "custom"
    elif "Janus" in args.model:
        map_dataset_parallel_mode = "sequential"
    else:
        map_dataset_parallel_mode = "parallel"
    # if args.small_size:
    #     dataset_dir = f"{DATA_DIR}/datasets/imagenet-text_50_classes_x_50_samples"
    # else:
    #     dataset_dir = f"{DATA_DIR}/datasets/imagenet_short_text_100_classes_x_100_samples"
    
    config = ExtractActivationConfig(
        model_name=args.model,
        input="imagenet-text",
        dataset_hf_name="AnonSubmission/imagenet-text",
        split="train",
        token=["all-text"],
        num_proc=2,
        map_dataset_parallel_mode = map_dataset_parallel_mode,
        device_map="balanced",
        resize_image=[256,256],
    )
    
    model = Extractor(config)
    
    cache = model.extract_cache(
        extract_avg_pattern=True,
    )
    if args.save:
        save_act_path = Path(f"{DATA_DIR}/activations/{args.model.replace('/','_')}/2_CrossAttention")
        save_act_path.mkdir(parents=True, exist_ok=True)
        torch.save(cache, save_act_path / "cache.pt")
    else:
        cross_attn = cross_attention_patterns(args.model, cache)
        save_path = Path(f"{args.out_dir}")
        save_path.mkdir(parents=True, exist_ok=True)
        plot_cross_attention(args.model, cross_attn, save_path / f"cross_attn_{args.model.replace('/','_')}.pdf")