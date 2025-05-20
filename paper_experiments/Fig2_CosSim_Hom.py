import os
import sys
from pathlib import Path

# Add the project root to Python's path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser
import torch
from src.activations import ExtractActivationConfig, Extractor
from pathlib import Path
from rich import print
from dotenv import load_dotenv
import json
import numpy as np
import matplotlib.pyplot as plt
from src.metrics.residual_stream import ResidualStreamMetricComputer
from dadapy import Data
from sklearn.metrics import homogeneity_completeness_v_measure

load_dotenv("../.env")
load_dotenv(".env")
DATA_DIR = os.getenv("DATA_DIR")

Z = 1.65
K = 16
MODEL_LAYERS = {
    "facebook/chameleon-7b": 32,
    "facebook/chameleon-30b": 48,
    "mistral-community/pixtral-12b": 40,
    "vila-u": 32,
    "deepseek-ai/Janus-1.3B": 24,
    "llava-onevision-7b": 28,
    "Emu3-Gen-hf": 32,
}

def compute_similarity(text_image_resid: torch.Tensor, image_text_resid: torch.Tensor) -> dict:
    """
    Compute the similarity between the text and image residual streams.
    """

    residual_stream = {
        "image": text_image_resid,
        "text": image_text_resid,
    }
    print("Residual Stream Loaded Successfully")
    
    # create the metric computer object
    metric_computer = ResidualStreamMetricComputer(
        residual_stream=residual_stream,
        resid_mid=False,
        head_out_resid=None,
        resid_key="resid_out",
    )

    # get a dictionary with the kde estimation for each layer and modality
    dist_resid_out, dist_resid_mid, dist_head_out = (
        metric_computer.correlation_per_modality(analyze_heads=True)
    )  # dist_resid_out[layer][modality] == (support, density)
    
    
    return dist_resid_out


def plot_similarity_per_layer_lineplot(dist_resid_out: dict, title: str, save_path: str, save=False) -> None:
    large_font = 26
    medium_font = 24
    small_font = 22
    # Define the high-contrast and bright color palettes
    hg_contrast = ["#004488", "#DDAA33", "#BB5566"]
    image_text = [tup["text - image"] for tup in dist_resid_out]
    layers = list(range(0, len(dist_resid_out)))

    plt.figure(figsize=(10.0, 6.15))
    plt.grid(True, which="both", linestyle="-", alpha=0.6)
    
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

    plt.xlabel("Layer", fontsize=medium_font)
    plt.ylabel("Cosine Similarity", fontsize=medium_font)

    plt.legend(fontsize=small_font)
    plt.xticks(fontsize=small_font)
    plt.yticks(fontsize=small_font)
    plt.ylim(0, 1)
    plt.title(title, fontsize=large_font)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()


# HOMOGENEITY IN CLUSTERS
def homogeneity_score(model: str, text_acts: dict, img_acts: dict, resid_key="resid_out") -> dict:
    """
    Compute the homogeneity score for the text and image activations.
    """
    # Get number of layers for the model, default to 32 if not found
    num_layers = MODEL_LAYERS.get(model, 32)    
    n_examples = text_acts[f"{resid_key}_0"].shape[0]
    dim_emb = text_acts[f"{resid_key}_0"].shape[2]
    res = []
    log = []
    for layer in range(num_layers):
        # read data for the layer
        text_poss = text_acts[f"{resid_key}_{layer}"].float()
        img_poss = img_acts[f"{resid_key}_{layer}"].float()
        all_poss = torch.cat([text_poss.squeeze(), img_poss.squeeze()], dim=0).view(
            text_poss.shape[0] * 2, dim_emb
        )
        all_poss = all_poss.numpy()
        data = Data(all_poss, verbose=True, n_jobs=16)
        data.compute_distances(metric="cosine", maxk=128)
        data.distances = data.remove_zero_dists(data.distances)
        id_list, id_error_list, id_distance_list = data.return_id_scaling_gride(
            range_max=128
        )
        data.set_id(id_list[3])
        data.compute_density_kNN(k=K)
        clusters_ADP = data.compute_clustering_ADP(Z=Z)
        
        # create text/img labels
        text_label = np.zeros(text_poss.shape[0] * 2)
        text_label[:n_examples] = 1.0
        
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
    plt.figure()
    colors = ["#004488", "#DDAA33", "#BB5566"]

    plt.title("Homogeneity of clustering.")
    plt.xlabel("Model depth")
    plt.ylabel("Homogeneity")
    plt.ylim(-0.05, 1.05)

    plt.plot(
        np.linspace(0, 1, len(homogeneity_data)),
        homogeneity_data,
        "-o",
        label=label,
        c=colors[0],
    )
    plt.grid()
    plt.legend()
    plt.savefig(save_path)
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        type=str,
                        help=" \
                            facebook/chameleon-7b, \
                            facebook/chameleon-30b, \
                            mistral-community/pixtral-12b, \
                            vila-u, \
                            deepseek-ai/Janus-1.3B, \
                            llava-onevision-7b, \
                            Emu3-Gen-hf",
                        required=True)
    parser.add_argument("-o", "--out_dir", type=str, default="out")
    parser.add_argument("--DEBUG", action="store_true")
    args = parser.parse_args()

    device_map = "balanced"
    dataset_size = 10000 if not args.DEBUG else 100
    
    TEXT_HIDDEN_PATH = Path(f"{DATA_DIR}/activations/similarity/{args.model}/text_hidden.pt")
    IMAGE_HIDDEN_PATH = Path(f"{DATA_DIR}/activations/similarity/{args.model}/image_hidden.pt")
       
    # check out_dir exists
    out_dir_path = Path(args.out_dir) / "Fig2_CosSim_Hom"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    

    #####################################
    #        TEXT HIDDEN VECTORS        #
    #####################################
    # check if text_hidden or image_hidden are there
    if TEXT_HIDDEN_PATH.exists():
        text_hidden_rappresentation = torch.load(TEXT_HIDDEN_PATH)
        print("Loaded text hidden vectors")
    else:
        # create the TEXT_HIDDEN_PATH
        TEXT_HIDDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        print("Extracting text hidden vectors...")
        config = ExtractActivationConfig(
            model_name=args.model,
            input="flickr-image->text",
            dataset_hf_name="nlphuji/flickr30k",
            split=f"test[:{dataset_size}]",
            token=["random-text"],
            num_proc=4,
            device_map=device_map,
            map_dataset_parallel_mode="custom",
            resize_image= [256, 256],
            

        )

        model = Extractor(config)
        text_hidden_rappresentation = model.extract_cache(
            extract_resid_out=True, save_input_ids=True, extract_resid_in=False
        )

        # torch save text in temp dir
        torch.save(text_hidden_rappresentation, TEXT_HIDDEN_PATH)

    #####################################
    #        IMAGE HIDDEN VECTORS       #
    #####################################
    # check if text_hidden or image_hidden are there
    if IMAGE_HIDDEN_PATH.exists():
        image_hidden_rappresentation = torch.load(IMAGE_HIDDEN_PATH)
        print("Loaded image hidden vectors")
    else:
        # create the IMAGE_HIDDEN_PATH (just the directory)
        IMAGE_HIDDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        print("Extracting image hidden vectors..")
        
        config = ExtractActivationConfig(
            model_name=args.model,
            input="flickr-text->image",
            dataset_hf_name="nlphuji/flickr30k",
            split=f"test[:{dataset_size}]",
            token=["random-image"],
            num_proc=4,
            device_map=device_map,
            map_dataset_parallel_mode="custom",
            resize_image= [256,256]
        )

        model.update(config)

        image_hidden_rappresentation = model.extract_cache(
            extract_resid_out=True, save_input_ids=True, extract_resid_in=False
        )

        # torch save text in temp dir
        torch.save(image_hidden_rappresentation, IMAGE_HIDDEN_PATH)

    #####################################
    #        Computing Similarity       #
    #####################################
    
    print("Computing similarity...")

    similarity = compute_similarity(
        text_hidden_rappresentation, image_hidden_rappresentation
    )

    # save similarity in out_dir
    similarity_to_save = {}
    similarity_to_save[args.model] = []
    for layer in similarity:
        similarity_to_save[args.model].append(
            [
                layer["text - image"][0],
                [layer["text - image"][1][0], layer["text - image"][1][1]],
            ]
        )
    
    with open(f"{args.out_dir}/similarity.json", "w") as f:
        json.dump(similarity_to_save, f, indent=4)

    try:
        plot_similarity_per_layer_lineplot(
            similarity,
            title=f"{args.model}",
            save_path=f"{args.out_dir}/similarity.pdf",
            save=True,
        )
    except Exception as e:
        print(f"Error in plot_similarity_per_layer_lineplot: {e}")
        print("Plotting similarity failed. Skipping...")
    
    #####################################
    #        Computing Homogeneity       #
    #####################################

    print("Computing homogeneity...")
    
    homogeneity_results = homogeneity_score(
        model=args.model,
        text_acts=text_hidden_rappresentation,
        img_acts=image_hidden_rappresentation,
        resid_key="resid_out",
    )
    
    # save homogeneity results["homogeneity_score"]
    with open(f"{args.out_dir}/homogeneity.json", "w") as f:
        json.dump(homogeneity_results["homogeneity_score"], f, indent=4)

    try:    
        plot_homogeneity(
            homogeneity_data=homogeneity_results["homogeneity_score"],
            label=f"{args.model}",
            save_path=f"{args.out_dir}/homogeneity.pdf",
        )
    except Exception as e:
        print(f"Error in plot_homogeneity: {e}")
        print("Plotting homogeneity failed. Skipping...")