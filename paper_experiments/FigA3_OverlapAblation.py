#!/usr/bin/env python
# %% 
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.activations import Extractor, ExtractActivationConfig
import argparse
import torch
from tqdm import tqdm
from paper_experiments.utils import extract_activation, compute_overlap_last_layer
import gc
from rich import print
from utils import plot_figA3

loop_config = {
    "end-image": [
            {
                "type": "std",
                "elem-to-ablate": "@end-image",
            }
        ],
    "image-text": [
            {
                "type": "block-img-txt",
                "elem-to-ablate": None,
            }
        ]
}
LOG_NAME = "test"  # Optional: set a log name identifier if desired

def main():
    parser = argparse.ArgumentParser(
        description="Run activation extraction with resuming and mode selection."
    )
    parser.add_argument(
        "--model_name", "-m",
        type=str,
        required=True,
        help="Model name"
    )
    parser.add_argument(
        "--outdir", "-o",
        type=str,
        required=True,
        help="Output directory"
    )
    args, unknown = parser.parse_known_args()
    model_name = args.model_name
    outdir = args.outdir


    # Create configuration
    config = ExtractActivationConfig(
        model_name=model_name,
        input="imagenet-text",
        dataset_hf_name="AnonSubmission/imagenet-text",
        torch_dtype=torch.bfloat16,
        token=["last"],
        id_num=0,
        attn_implementation="eager",
        map_dataset_parallel_mode="parallel",
        device_map="balanced",
        resize_image=[512, 512]
    )
    
    gc.collect()
    os.makedirs(os.path.join(outdir, "tmp"), exist_ok=True)
    
    model = Extractor(config)
    num_heads = model.model_config.num_attention_heads
    num_layers = model.model_config.num_hidden_layers

    # Set the filename for progress saving (one file per mode)
    data_file = os.path.join(outdir, "tmp", f"data_{LOG_NAME}.pt") if LOG_NAME else os.path.join(outdir, "tmp", "data.pt")

    # If resume flag is set and the file exists, load the saved base and partial results.
    if os.path.exists(data_file):
        result= torch.load(data_file)
        print(f"Cached with saved base activation and {len(result)} completed iterations.")
    else:
        # Compute base activation (no ablation) only if no resume data is found.
        result = {}
        print(f"Starting new loop. Saving progress to {data_file}")
        result_dict_no_ablation = extract_activation(config, model)
        base_result = compute_overlap_last_layer(result_dict_no_ablation, num_layers)
        print(f"{'-'*20} Base activation computed {'-'*20}")
        
        
        start_val = 8
        stop_val = num_layers + 8
        step = 8
        windows = list(range(start_val, stop_val, step))        
        
        for mode in ["end-image", "image-text"]:
            ablation_queries = loop_config[mode]
            result[mode] = []
            result[mode].append({"window": 0, "result":base_result})
            for window in tqdm(windows, total=len(windows), desc="end-image loop"):
                # head_layer_couple = [[i, j] for i in range(num_heads) for j in range(num_layers)]
                head_layer_couple = [[i, num_layers - j - 1] for i in range(num_heads) for j in range(window)]
                ablation_queries[0]["head-layer-couple"] = head_layer_couple
                
                result_dict = extract_activation(config, model, ablation_queries)
                res = compute_overlap_last_layer(result_dict, num_layers)
                result[mode].append({"window": window, "result": res})
                # Save progress after each iteration
                torch.save( result, data_file)
                print(f"Window {window}: result {res}")
            print(f"'{mode}' loop completed. Data saved to {data_file}")
        
        # Optionally, save the final result to a separate file
        final_data_file = os.path.join(outdir, f"data_{LOG_NAME}.pt") if LOG_NAME else os.path.join(outdir, f"{mode}_data.pt")
        torch.save(result, final_data_file)
        print(f"Final data saved to {final_data_file}")
    # Plotting
    plot_figA3(
        results=result,
        path=os.path.join(outdir, f"figA3_{LOG_NAME}.pdf"),
    )

# %%
if __name__ == "__main__":
    main()
# %%
