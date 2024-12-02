# %%
from src.metrics.overlap import LabelOverlap
from src.activations import Extractor, ExtractActivationConfig
import argparse
import torch
from tqdm.notebook import tqdm_notebook
from probe_utils import extract_activation,\
                        compute_overlap_last_layer,\
                        plotter
import gc
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="facebook/chameleon-7b, facebook/chameleon-30b, mistral-community/pixtral-12b")
    parser.add_argument("-o", "--outdir", type=str,
                        help="outdir")
    
    args = parser.parse_args()
    model = args.model
    model_to_layer = {
        "facebook/chameleon-7b": 32,
        "facebook/chameleon-30b": 48,
        "mistral-community/pixtral-12b": 40
    }

    config = ExtractActivationConfig(
            model_name=model,
            input="imagenet-text",
            dataset_hf_name="AnonSubmission/dataset_2",
            torch_dtype=torch.bfloat16,
            token=["last"],
            id_num=0,
            attn_implementation="eager",
    )   
    

    gc.collect()
    overlap = LabelOverlap()
    data = []

    model = Extractor(
        config
    )
    num_heads = model.model_config.num_attention_heads
    num_layers = model.model_config.num_hidden_layers
    result_dict = extract_activation(model)
    print(f"{'-'*20}Computing end-image{'-'*20}")
    
    for window_of_ablation in tqdm_notebook(range(0,num_layers,4), desc="Extract_activations and compute_overlap"):
        head_layer_couple = [[i,j] for i in range(num_heads) for j in range(window_of_ablation)]
        ablation_queries = [
                            {
                                "type": "std",
                                "elem-to-ablate": "@end-image",
                                "head-layer-couple": head_layer_couple,
                            }
                        ]    
        result_dict = extract_activation(config, ablation_queries)
        data.append(compute_overlap_last_layer(result_dict, num_layers))
    print("*|*|*|*|*|*|*|*|")
    print(f"Data: {data}")
    print("*|*|*|*|*|*|*|*|")

    print(f"\n\n\n{'-'*20}Computing image-text{'-'*20}")
    for window_of_ablation in tqdm_notebook(range(0,num_layers,4), desc="Extract_activations and compute_overlap"):
        head_layer_couple = [[i,j] for i in range(num_heads) for j in range(window_of_ablation)]
        ablation_queries = [
                            {
                                "type": "block-img-txt",
                                "elem-to-ablate": None,
                                "head-layer-couple": head_layer_couple,
                            }
                        ]    
        result_dict = extract_activation(config, ablation_queries)
        data.append(compute_overlap_last_layer(result_dict, num_layers))
    print("*|*|*|*|*|*|*|*|")
    print(f"Data: {data}")
    print("*|*|*|*|*|*|*|*|")
    names = [
    "end-image",
    "image-text",
    ]
    plotter(data=data,
            title=model,
            ylabel="NO",
            yticks=1.,
            names=names)

