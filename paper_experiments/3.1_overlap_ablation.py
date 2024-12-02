# %%
from src.metrics.overlap import LabelOverlap
from src.activations import Extractor, ExtractActivationConfig
import argparse
import torch
from probe_utils import preprocess_label, \
                        compute_knn_euclidian, \
                        extract_activation, \
                        plotter
import gc
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="facebook/chameleon-7b, facebook/chameleon-30b, mistral-community/pixtral-12b"
                        )
    parser.add_argument("-o", "--outdir", type=str,
                        help="outdir")
    
    args = parser.parse_args()
    model = args.model
    token_to_extract={
        "facebook/chameleon-7b": ["32", "last-image", "end-image"],
        "facebook/chameleon-30b": ["first-image", "last-image", "end-image"],
        "mistral-community/pixtral-12b": ["1024", "last-image", "end-image"]
    }
    config = ExtractActivationConfig(
            model_name=model,
            input="imagenet-text",
            dataset_hf_name="AnonSubmission/dataset_2",
            torch_dtype=torch.bfloat16,
            token=token_to_extract[model],
            id_num=0,
            attn_implementation="eager",
    )   
    
    gc.collect()
    overlap = LabelOverlap()
    data = []
    
    print(f"{'-'*20}Computing end-image{'-'*20}")
    model = Extractor(
        config
    )
    num_heads = model.model_config.num_attention_heads
    num_layers = model.model_config.num_hidden_layers
    result_dict = extract_activation(model)
    
    activations, labels = result_dict["activations"], result_dict["root_label"]
    labels = preprocess_label(labels)
    tensors = [activations[f"resid_out_{i}"] for i in range(num_layers)]
    avg = [activations[f"avg_resid_avg_{i}"] for i in range(num_layers)]

    # first image/1025/32
    t_iter = [
            compute_knn_euclidian(
                t[:, 0, :].to("cuda"), t[:, 0, :].to("cuda"), 40
            )  
            for t in tensors
        ]
    print("Computing Overlap")
    data.append(overlap.main(
            k=30,
            tensors=t_iter,
            labels=labels,
            number_of_layers=num_layers,
        )
    )

    # last image
    t_iter = [
            compute_knn_euclidian(
                t[:, 1, :].to("cuda"), t[:, 1, :].to("cuda"), 40
            )
            for t in tensors
        ]

    print("Computing Overlap")
    data.append(overlap.main(
            k=30,
            tensors=t_iter,
            labels=labels,
            number_of_layers=num_layers,
     
        )
    )
    # end image
    t_iter = [
            compute_knn_euclidian(
                t[:, 2, :].to("cuda"), t[:, 2, :].to("cuda"), 40
            )
            for t in tensors
    ]

    print("Computing Overlap")
    data.append(overlap.main(
            k=30,
            tensors=t_iter,
            labels=labels,
            number_of_layers=num_layers,
        )
    )
    # avg img
    t_iter = [
            compute_knn_euclidian(
                t[:, 0, :].to("cuda"), t[:, 0, :].to("cuda"), 40
            )
            for t in avg
        ]

    print("Computing Overlap")
    data.append(overlap.main(
            k=30,
            tensors=t_iter,
            labels=labels,
            number_of_layers=num_layers,
        )
    )
    if model=="mistral-community/pixtral-12b":
        # avg EOLs
        t_iter = [
                compute_knn_euclidian(
                    t[:, -2, :].to("cuda"), t[:, -2, :].to("cuda"), 40
                )
                for t in avg
            ]

        print("Computing Overlap")
        data.append(overlap.main(
                k=30,
                tensors=t_iter,
                labels=labels,
                number_of_layers=num_layers,
            )
        )
    
    names = {
        "facebook/chameleon-7b": [
            "32",
            "last image",
            "end image",
            "avg image",
        ],
        "facebook/chameleon-30b": [
            "first image",
            "last image",
            "end image",
            "avg image",
        ],
        "mistral-community/pixtral-12b": [
            "1024",
            "last image",
            "end image",
            "avg image",
            "avg EOLs"
        ]
    }
    
    plotter(data=data,
            title=model,
            ylabel="NO",
            yticks=1.,
            names=names)

