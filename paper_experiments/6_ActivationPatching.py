import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# path_to_parents(2)

from argparse import ArgumentParser
import torch
from src.activations import ExtractActivationConfig, Extractor
from rich import print
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path
# TODO: implement for pixtral

load_dotenv("../.env")



def lineplot(
    x_indices,
    dist_y_values,
    dist_y_err: Optional[List],
    base: Optional[float],
    title,
    save_path,
    save=True,
):
        
    large_font = 26
    medium_font = 24
    small_font = 22
    # Define the high-contrast and bright color palettes
    hg_contrast = ["#004488", "#DDAA33", "#BB5566"]
    bright = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]

    # Create the line plot
    plt.figure(figsize=(10.0, 6.15))
    plt.grid(True, which="both", linestyle="-", alpha=0.6)
    # plt.axvspan(axvspan_low, axvspan_high, color=bright[3], alpha=0.3, lw=0)
    if base is None and dist_y_err is None:
        plotting_accuracy = True

    if base is not None:
    # Add the base line
        plt.axhline(
            y=base,
            color=hg_contrast[0],
            linestyle="--",
            label="Without Patching",
            linewidth=4,
        )

    # Plot the line
    plt.plot(
        x_indices,
        dist_y_values,
        "-o",
        color=bright[2],
        alpha=0.99,
        linewidth=4,
        markersize=9,
        label="After Patching",
    )
    if dist_y_err is not None:
        # Plot shaded area for error instead of error bars
        lower_bound = np.array(dist_y_values) - np.array(dist_y_err)
        upper_bound = np.array(dist_y_values) + np.array(dist_y_err)
        plt.fill_between(x_indices, lower_bound, upper_bound, color=bright[2], alpha=0.3)

    # Labels and title
    plt.xlabel("Start indices of patched activations", fontsize=medium_font)
    # plt.ylabel("Similarity to base distribution\n", fontsize=medium_font)
    plt.title(
        title,
        fontsize=large_font,
    )


    # Y-axis customization
    y_ticks = np.linspace(0, 1, 5)
    if plotting_accuracy:
        plt.ylabel("Accuracy", fontsize=medium_font)
        y_tick_labels = (
            ["0"]
            + [f"{round(tick, 2)}" for tick in y_ticks[1:-1]]
            + ["1"]
        )
    y_tick_labels = (
        ["Less\nSimilar"]
        + [f"{round(tick, 2)}" for tick in y_ticks[1:-1]]
        + ["Most\nSimilar"]
    )
    plt.yticks(y_ticks, y_tick_labels, fontsize=small_font)
    plt.legend(fontsize=small_font-4.6, loc="upper right" , ncol=1)
    selected_x_indices = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    selected_x_labels = [f"{i}" for i in selected_x_indices]
    plt.xticks(
        selected_x_indices,    selected_x_labels, rotation=0, ha="center", fontsize=small_font
        )
    plt.tight_layout()

    # Save or return the plot
    if save:
        plt.savefig(save_path, format="pdf", dpi=300)
    else:
        return plt



if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-o","--out_dir", type=str, default=None)
    args = parser.parse_args()
    
    config = ExtractActivationConfig(
        model_name=args.model,
        input="imagenet-with-counterfactuals-2class",
        dataset_hf_name= "AnonSubmission/dataset_3",
        token = ["end-image", "last"],
        num_proc=4,
        split="train"
    )
    model = Extractor(config)
    num_hidden_layers = model.model_config.num_hidden_layers
    
    patching_query_residual_stream = [
        [{
            "patching_elem": "@end-image",
            "layers_to_patch": [i],
            "activation_type": "resid_in_{}"
        }]
        for i in range(num_hidden_layers)
    ]
    
   
    with open(Path(f"{args.out_dir}","all_path_residual.txt"), "w") as f:
        f.write(
            "patching_elem, layers_to_patch, activation_type, activation_path \n"
        )
    for i,query in enumerate(patching_query_residual_stream):
        model.update(config)
        output = model.compute_patching(
            config = config,
            patching_query = query,
            return_logit_diff = True,
            extract_resid_out= False
        )
        torch.save(output, f"{args.out_dir}/activation_{i}")
        with open(f"{args.out_dir }/all_path_residual.txt", "a") as f:
            f.write(
                f"{query[0]['patching_elem']}, {query[0]['layers_to_patch']}, {query[0]['activation_type']},  {args.out_dir}/activation_{i} \n"
            )
        
    print("Extraction of hidden activations for residual stream done!")
    
    patching_query_attention = [
        [{
            "patching_elem": "@end-image",
            "layers_to_patch": [range(i, num_hidden_layers)],
            "activation_type": "attn_in_{}"
        }]
        for i in range(num_hidden_layers)
    ]
    
    if os.stat(f"{args.out_dir}/all_path_attn.txt").st_size == 0:
        with open(f"{args.out_dir}/all_path_attn.txt", "w") as f:
            f.write(
                "patching_elem, layers_to_patch, activation_type, activation_path \n"
            )
    for i,query in enumerate(patching_query_attention):
        model.update(config)
        output = model.compute_patching(
            config = config,
            patching_query = query,
            return_logit_diff = True,
            extract_resid_out= False
        )
        torch.save(output, f"{args.out_dir}/activation_attn_{i}")
        with open(f"{args.out_dir }/all_path_attn.txt", "a") as f:
            f.write(
                f"{query[0]['patching_elem']}, {query[0]['layers_to_patch']}, {query[0]['activation_type']},  {args.out_dir}/activation_attn{i} \n"
            )
            
    print("Extraction of hidden activations for attention done!")
    
    from src.plot.patching import load_activations_df, pre_process_data, barplot_dist_similarity, barplot_logit_diff, lineplot_dist_similarity

    activations = load_activations_df(f"{args.out_dir}/all_path_residual.txt",)
    x_value, y_accuracy, _, _ = pre_process_data(activations, "logit_diff")
    
    data = {
        "layers": [f"{i}" for i, _ in enumerate(y_accuracy)],
        "accuracy_diff": y_accuracy   
    }
    
    lineplot(
        data["layers"],
        data["accuracy_diff"],
        None,
        None,
        title="Residual Stream",
        save_path=f"{args.out_dir}/residual_stream.png",
        save=True
    )
    
    activations = load_activations_df(f"{args.out_dir}/all_path_attn.txt",)
    x_value, y_accuracy, _, _ = pre_process_data(activations, "logit_diff")
    
    data = {
        "layers": [f"{i}" for i, _ in enumerate(y_accuracy)],
        "accuracy_diff": y_accuracy   
    }
    
    lineplot(
        data["layers"],
        data["accuracy_diff"],
        None,
        None,
        title="Attention Stream",
        save_path=f"{args.out_dir}/attention_stream_{args.model.split('/')[1]}.pdf",
        save=True
    )
    
    print("Plots saved!")
    
    print("Done!")
    