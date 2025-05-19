
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.metrics.overlap import LabelOverlap
from src.activations import Extractor, ExtractActivationConfig
import argparse
import torch
from paper_experiments.utils import preprocess_label, \
                        compute_knn_euclidian, \
                        extract_activation, \
                        plotter
import gc
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from pathlib import Path
load_dotenv("../.env")
DATA_DIR = Path(os.getenv("DATA_DIR"))

@dataclass
class ModelRunConfig:
    num_layers: int
    num_heads: (
        int  # Retained from original, though not directly used in this script's logic
    )
    img_len: int  # Retained from original

    # Defines what tokens are requested from Extractor.
    # Assumed to be a list of strings/keys that Extractor understands.
    # The order here MUST match the order of plot_labels.
    extractor_tokens: List[str]

    # Defines the labels for plotting, corresponding to extractor_tokens.
    plot_labels: List[str]

    # Specific handling for average image representation
    has_avg_image_plot: bool = True
    # Specific handling for Pixtral's average EOLs
    has_avg_eols_plot: bool = False



MODEL_RUN_CONFIGS = {
    "facebook/chameleon-7b": ModelRunConfig(
        num_layers=32,
        num_heads=32,
        img_len=1024,
        extractor_tokens=["32", "last-image", "end-image"],
        plot_labels=["32nd token",  "last image", "EOI", "internal image",],
    ),
    "facebook/chameleon-30b": ModelRunConfig(
        num_layers=48,
        num_heads=64,
        img_len=1024,
        extractor_tokens=["first-image", "last-image", "end-image"],
        plot_labels=["1st token", "last image", "EOI", "internal image"],
    ),
    "llava-onevision-7b": ModelRunConfig(
        num_layers=28,
        num_heads=28,
        img_len=1488,
        extractor_tokens=["last-image", "end-image"],  
        plot_labels=["last image", "EOI", "internal image"],
    ),
    "mistral-community/pixtral-12b": ModelRunConfig(
        num_layers=40,
        num_heads=32,
        img_len=1053,
        extractor_tokens=["[EOL]s", "last-image", "end-image"],
        plot_labels=["[EOL]s",  "last image", "EOI", "internal image",],
        has_avg_eols_plot=True,
    ),
    "vila-u": ModelRunConfig(
        num_layers=32,
        num_heads=32,
        img_len=256,
        extractor_tokens=["last-image", "end-image"],
        plot_labels=[ "last image", "EOI", "internal image"],
    ),
    "deepseek-ai/Janus-1.3B": ModelRunConfig(
        num_layers=24,
        num_heads=16,
        img_len=575,
        extractor_tokens=["last-image", "end-image"],
        plot_labels=["last image", "EOI", "internal image"],
    ),
    "Emu3-Gen-Finetune": ModelRunConfig(
        num_layers=32,
        num_heads=32,
        img_len=1024,
        extractor_tokens=["last-image", "end-image"],
        plot_labels=["internal image", "last image", "EOI"],
    ),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default = "facebook/chameleon-7b")
    parser.add_argument("-o", "--outdir", type=str,
                        help="outdir")
    parser.add_argument("-s", "--small", action="store_true")
    
    args = parser.parse_args()
    print(f"Model: {args.model}\nOutdir: {args.outdir}")
    if args.small:
        print("SMALL SIZE")
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    model = args.model

    if "llava" in args.model:
        map_dataset_parallel_mode = "custom"
    elif "Janus" in args.model:
        map_dataset_parallel_mode = "sequential"
    else:
        map_dataset_parallel_mode = "parallel"
    
    
    model_config = MODEL_RUN_CONFIGS.get(model)

    num_heads = model_config.num_layers
    num_layers = model_config.num_heads
    if not os.path.exists(f"{args.outdir}/tmp/fig4_tmp.pt"):
        config = ExtractActivationConfig(
            model_name=model,
            input="imagenet-text",
            dataset_hf_name="AnonSubmission/imagenet-text",
            torch_dtype=torch.bfloat16,
            token=model_config.extractor_tokens,
            id_num=0,
            attn_implementation="eager",
            resize_image=[256, 256],
            map_dataset_parallel_mode=map_dataset_parallel_mode,
        )   
    
        gc.collect()
        overlap = LabelOverlap()
        data = []
        

        model = Extractor(
            config
        )
        result_dict = extract_activation(config, model, extract_avg=True)
        activations, labels = result_dict["activations"], result_dict["root_label"]
        labels = preprocess_label(labels)
        tensors = [activations[f"resid_out_{i}"] for i in range(num_layers)]
        avg = [activations[f"avg_resid_out_{i}"] for i in range(num_layers)]


        for i, label in enumerate(model_config.plot_labels):
            print(f"------------ Computing overlap for {label} ------------")
            if label == "internal image":
                # avg image
                t_iter = [
                        compute_knn_euclidian(
                            t[:, 0, :].to("cuda"), t[:, 0, :].to("cuda"), 40
                        )
                        for t in avg
                    ]
                
            elif label == "[EOL]s":
                # avg EOLs
                t_iter = [
                        compute_knn_euclidian(
                            t[:, -2, :].to("cuda"), t[:, -2, :].to("cuda"), 40
                        )
                        for t in avg
                    ]
                
            else:
                
                t_iter = [
                        compute_knn_euclidian(
                            t[:, i, :].to("cuda"), t[:, i, :].to("cuda"), 40
                        )
                        for t in tensors
                    ]

                

            data.append(overlap.main(
                    k=30,
                    tensors=t_iter,
                    labels=labels,
                    number_of_layers=num_layers,
                )
            )
        

        
        os.makedirs(
            f"{args.outdir}/tmp", exist_ok = True
        )
        torch.save(data, f"{args.outdir}/tmp/fig4_tmp.pt")
    else:
        print(f"{'-'*10} Loading from tmp!! {'-'*10}") 
        data = torch.load(f"{args.outdir}/tmp/fig4_tmp.pt")

    # save
    torch.save(data, f"{args.outdir}/fig4_overlap_data.pt")

    
    plotter(data=data,
            title=model,
            ylabel="NO",
            yticks=1.,
            names=model_config.plot_labels,
            path=f"{args.outdir}/fig4_overlap.pdf",
            )





    # if (
    #     args.model == "Emu3-Gen-hf" \
    #     or args.model == "Emu3-Chat-hf" \
    #     or args.model == "Emu3-Stage1-hf"\
    #     or args.model == "Emu3-Gen-Finetune"):
    #     new_config = ExtractActivationConfig(
    #         model_name=args.model,
    #         input="imagenet-text",
    #         dataset_hf_name="RitAreaSciencePark/imagenet_short_text_100_classes_x_100_samples",
    #         torch_dtype=torch.bfloat16,
    #         token=["all-image"],
    #         id_num=0,
    #         attn_implementation="eager",
    #         resize_image=[256, 256],
    #         # split="train[:1000]"
    #     )
    #     model.update(
    #         config
    #     )
    #     result_dict = extract_activation(config, model, extract_avg=True)
    
    #     activations, labels = result_dict["activations"], result_dict["root_label"]
    #     labels = preprocess_label(labels)
    #     tensors = [activations[f"avg_resid_out_{i}"] for i in range(num_layers)]
    #     t_iter = [
    #                 compute_knn_euclidian(
    #                     t[:, 0, :].to("cuda"), t[:, 0, :].to("cuda"), 40
    #                 )  
    #                 for t in tensors
    #             ]
    #     print("Computing Overlap")
    #     data.append(overlap.main(
    #             k=30,
    #             tensors=t_iter,
    #             labels=labels,
    #             number_of_layers=num_layers,
    #         )
    #     )

