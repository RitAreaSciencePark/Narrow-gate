import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# join src module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from easyroutine import path_to_parents, Logger

# path_to_parents(2)


import torch
import pandas as pd
from rich import print
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from src.metrics.residual_stream import ResidualStreamMetricComputer

######################################################################################################
#                                 CONFIGURATION                                                      #
######################################################################################################


IMAGE_TEXT_PATH = "./data/activation_path/flickr_test[:4]_pixtral-12b_flickr-image->text_imagenet-text-ablation_693.txt"
TEXT_IMAGE_PATH = "./data/activation_path/flickr_test[:4]_pixtral-12b_flickr-text->image_imagenet-text-ablation_693.txt"

# SAVE_PATH = Path(
#     "./data/plot/residual_stream_flirckr_test_[:4]_chameleon-7b_last-4_350_DIST_PLOT_RESIDUAL/"
# )
SAVE_PATH = Path(
    "./data",
    "plot",
    "residual_stream_flirckr_test_[:4]_pixtral-12b_flickr_DIST_PLOT_RESIDUAL_851",
)

######################################################################################################

print("Current path: ", os.getcwd())
# create the directory if it does not exist
def plot_distribution_head(dist_head_out, save_path):
    """
    Function to plot the distributions of the head outputs for each layer, head, and modality.
    Creates a plot for each layer with subplots for each head.
    """
    num_layers = len(dist_head_out)
    num_heads = len(dist_head_out[0])
    modalities = list(dist_head_out[0][0].keys())
    colors = ["#6929c4", "#1192e8", "#005d5d"]

    # Find global min and max for support
    global_min = float("inf")
    global_max = float("-inf")

    for layer in range(num_layers):
        for head in range(num_heads):
            for modality in modalities:
                support, _ = dist_head_out[layer][head][modality]
                global_min = min(global_min, support.min())
                global_max = max(global_max, support.max())

    for layer in tqdm(range(num_layers), desc="Plotting Head Distributions"):
        # Find min and max density for this layer
        layer_min_density = float("inf")
        layer_max_density = float("-inf")
        for head in range(num_heads):
            for modality in modalities:
                _, density = dist_head_out[layer][head][modality]
                layer_min_density = min(layer_min_density, density.min())
                layer_max_density = max(layer_max_density, density.max())

        fig, axes = plt.subplots(4, 8, figsize=(40, 20))
        fig.suptitle(f"Head Output Distribution Plots for Layer {layer}", fontsize=16)

        for head in range(num_heads):
            ax = axes[head // 8, head % 8]
            for modality, color in zip(modalities, colors):
                support, density = dist_head_out[layer][head][modality]
                ax.plot(support, density, label=f"{modality}", color=color)
                ax.fill_between(support, density, alpha=0.3, color=color)
            ax.set_title(f"Head {head}", fontsize=10)
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(layer_min_density, layer_max_density)
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.axvline(x=0, color="grey", linestyle=":", linewidth=1)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize="large")

        plt.tight_layout()
        if not os.path.exists(f"{save_path}/head_outputs"):
            os.makedirs(f"{save_path}/head_outputs")
        plt.savefig(f"{save_path}/head_outputs/layer{layer}.png", dpi=300)
        plt.close(fig)

def plot_distributions_resid(dist_resid_out, dist_resid_mid, save_path):
    """
    Function to plot the distributions of the residual stream for each layer and modality.
    Creates a plot for each layer with two subplots, one for the residual stream mid and one for the residual stream out.

    """
    num_layers = len(dist_resid_out)
    modalities = list(dist_resid_out[0].keys())
    colors = ["#6929c4", "#1192e8", "#005d5d"]

    # Find global min and max for support and density
    global_min = float("inf")
    global_max = float("-inf")
    global_min_density = float("inf")
    global_max_density = float("-inf")

    for layer in range(num_layers):
        for modality in modalities:
            if dist_resid_mid is not None:
                support_mid, density_mid = dist_resid_mid[layer][modality]
                global_min = min(global_min, support_mid.min())
                global_max = max(global_max, support_mid.max())
                global_min_density = min(global_min_density, density_mid.min())
                global_max_density = max(global_max_density, density_mid.max())
            support_out, density_out = dist_resid_out[layer][modality]
            global_min = min(global_min, support_out.min())
            global_max = max(global_max, support_out.max())
            global_min_density = min(global_min_density, density_out.min())
            global_max_density = max(global_max_density, density_out.max())

    for layer in tqdm(range(num_layers), desc="Plotting Distributions"):
        fig, (ax_mid, ax_out) = plt.subplots(1, 2, figsize=(20, 10))  # type: ignore
        fig.suptitle(f"Distribution Plots for Layer {layer}", fontsize=16)

        # Plot for resid_mid
        if dist_resid_mid is not None:
            for modality, color in zip(modalities, colors):
                support, density = dist_resid_mid[layer][modality]
                ax_mid.plot(support, density, label=f"{modality}", color=color)
                ax_mid.fill_between(support, density, alpha=0.3, color=color)
            ax_mid.set_title(f"Layer {layer} (Mid)", fontsize=14)
            ax_mid.set_xlabel("Correlation", fontsize=14)
            ax_mid.set_ylabel("Density", fontsize=14)
            ax_mid.legend(fontsize="large")
            ax_mid.set_xlim(global_min, global_max)
            ax_mid.set_ylim(
                global_min_density, global_max_density
            )  # Ensure same y-axis height
            ax_mid.tick_params(axis="both", which="major", labelsize=14)
            ax_mid.xaxis.set_tick_params(width=2)
            ax_mid.yaxis.set_tick_params(width=2)
            ax_mid.axvline(
                x=0, color="grey", linestyle=":", linewidth=2
            )  # Add vertical line at x=0

        # Plot for resid_out
        for modality, color in zip(modalities, colors):
            support, density = dist_resid_out[layer][modality]
            ax_out.plot(support, density, label=f"{modality}", color=color)
            ax_out.fill_between(support, density, alpha=0.3, color=color)
        ax_out.set_title(f"Layer {layer} (Out)", fontsize=14)
        ax_out.set_xlabel("Correlation", fontsize=14)
        ax_out.set_ylabel("Density", fontsize=14)
        ax_out.legend(fontsize="large")
        ax_out.set_xlim(global_min, global_max)
        ax_out.set_ylim(
            global_min_density, global_max_density
        )  # Ensure same y-axis height
        ax_out.tick_params(axis="both", which="major", labelsize=14)
        ax_out.xaxis.set_tick_params(width=2)
        ax_out.yaxis.set_tick_params(width=2)
        ax_out.axvline(
            x=0, color="grey", linestyle=":", linewidth=2
        )  # Add vertical line at x=0

        plt.tight_layout()
        plt.savefig(f"{save_path}/layer{layer}.png", dpi=300)
        plt.close(fig)


def compute_similarity(text_image_path, image_text_path):
    # load the file that contains the residual stream paths
    text_image_resid = pd.read_csv(text_image_path, header=None, names=["name", "path"])
    image_text_resid = pd.read_csv(image_text_path, header=None, names=["name", "path"])

    # load the residual stream and create a dictionary with the residual stream for each modality
    residual_stream = {
        "image": torch.load(
            text_image_resid[text_image_resid["name"] == "residual_stream"][
                "path"
            ].iloc[0].replace(" ", "")
        )["activations"],
        "text": torch.load(
            image_text_resid[image_text_resid["name"] == "residual_stream"][
                "path"
            ].iloc[0].replace(" ", "")
        )["activations"],
    }
    print("Residual Stream Loaded Successfully")
    head_out = None
    if len(text_image_resid[text_image_resid["name"] == "head_out"]["path"]) > 0:
        head_out = {
            "image": torch.load(
                text_image_resid[text_image_resid["name"] == "head_out"]["path"].iloc[
                    0
                ].replace(" ", "")
            )["activations"],
            "text": torch.load(
                image_text_resid[image_text_resid["name"] == "head_out"]["path"].iloc[
                    0
                ].replace(" ", "")
            )["activations"],
        }

    # create the metric computer object
    metric_computer = ResidualStreamMetricComputer(
        residual_stream=residual_stream, resid_mid=False, head_out_resid=head_out
    )

    # get a dictionary with the kde estimation for each layer and modality
    dist_resid_out, dist_resid_mid, dist_head_out = (
        metric_computer.correlation_per_modality(analyze_heads=True)
    )  # dist_resid_out[layer][modality] == (support, density)
    #save dist_resid_out, dist_resid_mid
    return dist_resid_out, dist_resid_mid, dist_head_out
    # torch.save({"dist_resid_out": dist_resid_out, "dist_resid_mid": dist_resid_mid}, SAVE_PATH / "dist_resid.pt")
    # plot_distributions_resid(dist_resid_out, dist_resid_mid, SAVE_PATH)
    # plot_distribution_head(dist_head_out, SAVE_PATH)

    # If you want to create other plots, you can use the dist_resid_out and dist_resid_mid dictionaries as you wish here

def plot_orthogonality_per_layer_lineplot(
    dist_resid_out, title, save_path, save=False
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
    
    
    image_image = [tup["image - image"] for tup in dist_resid_out]
    image_text = [tup["text - image"] for tup in dist_resid_out]
    text_text = [tup["text - text"] for tup in dist_resid_out]
    layers = list(range(0,len(dist_resid_out)))
    
    
    plt.figure(figsize=(10.0, 6.15))
    plt.grid(True, which="both", linestyle="-", alpha=0.6)
    #dist_resid_out is a list of dictionaries, each dictonary contains 
    
    # plt.plot(layers, [tup[0] for tup in image_image], "-o", label="image - image", color=hg_contrast[0], linewidth=4, markersize=10)
    # plt.fill_between(layers, [tup[0] - tup[1][0] for tup in image_image], [tup[0] + tup[1][1] for tup in image_image], alpha=0.3, color=hg_contrast[0])
    
    plt.plot(layers, [tup[0] for tup in image_text], "-o", label="image - text", color=hg_contrast[1], linewidth=4, markersize=10)
    plt.fill_between(layers, [tup[1][0] for tup in image_text], [tup[1][1] for tup in image_text], alpha=0.3, color=hg_contrast[1])
    
    # plt.plot(layers, [tup[0] for tup in text_text], "-o", label="text - text", color=hg_contrast[2], linewidth=4, markersize=10)
    # plt.fill_between(layers, [tup[0] - tup[1][0] for tup in text_text], [tup[0] + tup[1][1] for tup in text_text], alpha=0.3, color=hg_contrast[2])
    
    plt.xlabel("Layer", fontsize=medium_font)
    plt.ylabel("Cosine Similarity", fontsize=medium_font)
    
    plt.legend(fontsize=small_font)
    # if len(layers)== 40:
    #     plt.xticks(ticks=layers[::5] + [39], labels=[str(i) for i in layers[::5]] + [39], fontsize=small_font)
    # if len(layers) == 32:
    plt.xticks(fontsize=small_font)
    plt.yticks(fontsize=small_font)
    #set min and max for y ax (0,1)
    plt.ylim(0,1)
    
    plt.title(title, fontsize=large_font)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_orthogonality_per_layer_lineplot_all_models(
    dist_resid_out_chameleon, dist_resid_out_cham30, dist_resid_out_pixtral, title, save_path, save=False
):
    large_font = 26
    medium_font = 24
    small_font = 22

    # Define the high-contrast and bright color palettes
    hg_contrast = ["#004488", "#DDAA33", "#BB5566"]

    # Extract the data for each model
    image_text_chameleon = [tup["text - image"] for tup in dist_resid_out_chameleon]
    image_text_cham30 = [tup["text - image"] for tup in dist_resid_out_cham30]
    image_text_pixtral = [tup["text - image"] for tup in dist_resid_out_pixtral]

    # Define normalized layer ranges between 0 and 1 for each model
    layers_chameleon = np.linspace(0, 1, len(dist_resid_out_chameleon))
    layers_cham30 = np.linspace(0, 1, len(dist_resid_out_cham30))
    layers_pixtral = np.linspace(0, 1, len(dist_resid_out_pixtral))

    plt.figure(figsize=(10.0, 6.15))
    plt.grid(True, which="both", linestyle="-", alpha=0.6)

    # Plot each model with its respective normalized layer range
    plt.plot(
        layers_chameleon,
        [tup[0] for tup in image_text_chameleon],
        "-o",
        label="Chameleon-7b",
        color=hg_contrast[0],
        linewidth=4,
        markersize=10,
    )
    plt.fill_between(
        layers_chameleon,
        [tup[1][0] for tup in image_text_chameleon],
        [tup[1][1] for tup in image_text_chameleon],
        alpha=0.3,
        color=hg_contrast[0],
    )

    plt.plot(
        layers_cham30,
        [tup[0] for tup in image_text_cham30],
        "-o",
        label="Chameleon-30b",
        color=hg_contrast[1],
        linewidth=4,
        markersize=10,
    )
    plt.fill_between(
        layers_cham30,
        [tup[1][0] for tup in image_text_cham30],
        [tup[1][1] for tup in image_text_cham30],
        alpha=0.3,
        color=hg_contrast[1],
    )

    plt.plot(
        layers_pixtral,
        [tup[0] for tup in image_text_pixtral],
        "-o",
        label="Pixtral",
        color=hg_contrast[2],
        linewidth=4,
        markersize=10,
    )
    plt.fill_between(
        layers_pixtral,
        [tup[1][0] for tup in image_text_pixtral],
        [tup[1][1] for tup in image_text_pixtral],
        alpha=0.3,
        color=hg_contrast[2],
    )

    plt.xlabel("Model Depth", fontsize=medium_font)
    plt.ylabel("Cosine Similarity", fontsize=medium_font)

    plt.legend(fontsize=small_font)
    plt.xticks(
        ticks=np.linspace(0, 1, 5),
        labels=[f"{x:.2f}" for x in np.linspace(0, 1, 5)],
        fontsize=small_font,
    )
    plt.yticks(fontsize=small_font)
    plt.title(title, fontsize=large_font)
    plt.tight_layout()

    if save:
        plt.savefig(save_path)
    else:
        plt.show()

