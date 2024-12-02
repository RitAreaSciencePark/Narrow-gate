import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# path_to_parents(2)

from argparse import ArgumentParser
from dataclasses import dataclass
import torch
from src.activations import ExtractActivationConfig, Extractor
from src.metrics.overlap import PredictOverlap
from src.utils import generate_code, data_path
from pathlib import Path
from rich import print
import pandas as pd
import pickle
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import Literal, List
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from src.metrics.residual_stream import ResidualStreamMetricComputer
from dadapy import Data
from sklearn.metrics import homogeneity_completeness_v_measure
load_dotenv("../.env")


def compute_similarity(text_image_resid, image_text_resid):
    # load the file that contains the residual stream paths

    # load the residual stream and create a dictionary with the residual stream for each modality
    residual_stream = {
        "image": text_image_resid,
        "text": image_text_resid,
    }
    print("Residual Stream Loaded Successfully")
    # head_out = None
    # if len(text_image_resid[text_image_resid["name"] == "head_out"]["path"]) > 0:
    #     head_out = {
    #         "image": torch.load(
    #             text_image_resid[text_image_resid["name"] == "head_out"]["path"]
    #             .iloc[0]
    #             .replace(" ", "")
    #         ),
    #         "text": torch.load(
    #             image_text_resid[image_text_resid["name"] == "head_out"]["path"]
    #             .iloc[0]
    #             .replace(" ", "")
    #         ),
    #     }

    # create the metric computer object
    metric_computer = ResidualStreamMetricComputer(
        residual_stream=residual_stream, resid_mid=False, head_out_resid=None
    )

    # get a dictionary with the kde estimation for each layer and modality
    dist_resid_out, dist_resid_mid, dist_head_out = (
        metric_computer.correlation_per_modality(analyze_heads=True)
    )  # dist_resid_out[layer][modality] == (support, density)
    # save dist_resid_out, dist_resid_mid
    return dist_resid_out


def plot_similarity_per_layer_lineplot(dist_resid_out, title, save_path, save=False):
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
    layers = list(range(0, len(dist_resid_out)))

    plt.figure(figsize=(10.0, 6.15))
    plt.grid(True, which="both", linestyle="-", alpha=0.6)
    # dist_resid_out is a list of dictionaries, each dictonary contains

    # plt.plot(layers, [tup[0] for tup in image_image], "-o", label="image - image", color=hg_contrast[0], linewidth=4, markersize=10)
    # plt.fill_between(layers, [tup[0] - tup[1][0] for tup in image_image], [tup[0] + tup[1][1] for tup in image_image], alpha=0.3, color=hg_contrast[0])

    plt.plot(
        layers,
        [tup[0] for tup in image_text],
        "-o",
        label="image - text",
        color=hg_contrast[1],
        linewidth=4,
        markersize=10,
    )
    plt.fill_between(
        layers,
        [tup[1][0] for tup in image_text],
        [tup[1][1] for tup in image_text],
        alpha=0.3,
        color=hg_contrast[1],
    )

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
    # set min and max for y ax (0,1)
    plt.ylim(0, 1)

    plt.title(title, fontsize=large_font)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()


# HOMOGENEITY IN CLUSTERS
def homogeneity_score(
    model,
    text_acts,
    img_acts,
):
    #
    if model == "facebook/chameleon-7b":
        num_layers = 32
    elif model == "facebook/chameleon-30b":
        num_layers = 48
    elif model == "mistral-community/pixtral-12b":
        num_layers = 40

    max_len = len(img_acts.keys())
    dim_emb = text_acts[f"resid_out_0"].shape[2]
    min_pop = 50
    res = []
    log = []
    #
    for l in range(num_layers):
        # read data for the layer
        text_poss = text_acts[f"resid_out_{l}"].float()
        img_poss = img_acts[f"resid_out_{l}"].float()
        print(text_poss.shape)
        all_poss = torch.cat([text_poss, img_poss], dim=0).view(text_poss.shape[2]*2, dim_emb)
        all_poss = all_poss.numpy()
        # normalization on sphere
        all_poss = all_poss / np.linalg.norm(all_poss, keepdims=True, axis=1)
        # setup of cluster algorithm
        data = Data(all_poss, verbose=True, n_jobs=16)
        id_list, id_error_list, id_distance_list = data.return_id_scaling_gride(
            range_max=128
        )
        purity_NO_txt = (data.dist_indices[:10000, 1:] < 10000).sum() / (10000 * 100)
        purity_NO_img = 1 - (data.dist_indices[10000:, 1:] < 10000).sum() / (
            10000 * 100
        )
        data.set_id(id_list[3])
        data.compute_density_kNN(k=15)
        clusters_ADP = data.compute_clustering_ADP(Z=1.2)
        # create text/img labels
        text_label = np.zeros(text_poss.shape[2]*2)
        text_label[:10000] = 1.0
        # evaluate homogeneity
        log_h = homogeneity_completeness_v_measure(text_label, clusters_ADP)
        res.append(log_h[0])
        log.append(log_h)
        print(log_h)

    return {"homogeneity_score": res, "all_scores": log}


def plot_homogeneity(homogeneity_data, label, save_path: str):
    # Preprocessing of raw data ------------
    # pixtral_homo = np.array(dict_homogeneity_data['pixtral12b'])
    # chameleon7b_homo = np.array(dict_homogeneity_data['chameleon7b'])
    # chameleon34b_homo = np.array(dict_homogeneity_data['chameleon34b'])
    # -----------------------------------------

    colors = ["#004488", "#DDAA33", "#BB5566"]

    plt.title("Homogeneity of clustering on residual stream.")
    plt.xlabel("Model depth")
    plt.ylabel("Homogeneity")
    plt.ylim(-0.05, 1.05)

    plt.plot(np.linspace(0, 1, 32), homogeneity_data, "-o", label=label, c=colors[0])
    plt.grid()
    plt.legend()
    plt.savefig(save_path)
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-o", "--out_dir", type=str, default="out")

    args = parser.parse_args()

    print("Extracting text hidden vectors...")
    config = ExtractActivationConfig(
        model_name=args.model,
        input="flickr-image->text",
        dataset_hf_name="nlphuji/flickr30k",
        split="test[:10000]",
        token=["random-text"],
        num_proc=4,
    )

    model = Extractor(config)
    text_hidden_rappresentation = model.extract_cache(
        extract_resid_out=True, save_input_ids=True
    )

    # # torch save text in temp dir
    # torch.save(text_hidden_rappresentation, f"{args.dir}/text_hidden.pt")

    print("Extracting image hidden vectors..")
    config = ExtractActivationConfig(
        model_name=args.model,
        input="flickr-text->image",
        dataset_hf_name="nlphuji/flickr30k",
        split="test[:10000]",
        token=["random-image"],
        num_proc=4,
    )
    # model = model.to("cpu")
    # del model
    # torch.cuda.empty_cache()
    model.update(config)
    image_hidden_rappresentation = model.extract_cache(
        extract_resid_out=True, save_input_ids=True
    )

    # # torch save text in temp dir
    # torch.save(image_hidden_rappresentation, f"{args.dir}/text_hidden.pt")

    similarity = compute_similarity(
        text_hidden_rappresentation, image_hidden_rappresentation
    )

    plot_similarity_per_layer_lineplot(
        similarity,
        title=f"{args.model}",
        save_path=f"{args.out_dir}/similarity.pdf",
        save=True,
    )

    homogeneity_results = homogeneity_score(
        model=args.model,
        text_acts=text_hidden_rappresentation,
        img_acts=image_hidden_rappresentation,
    )

    plot_homogeneity(
        homogeneity_data=homogeneity_results["homogeneity_score"],
        label=f"{args.model}",
        save_path=f"{args.out_dir}/homogeneity.pdf",
    )
