from src.activations import Extractor
from src.metrics.overlap import LabelOverlap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gc
from jaxtyping import Int
from typing import List
from src.annotations import Array


plot_config = {
    'axes.titlesize': 30,      
    'axes.labelsize': 29,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 10,
    'figure.figsize': (10, 8),
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
}

def average_custom_blocks(y, n):
    """
    For the plots in the main section of the paper we average the value of a certain metric for a specific layer
    over a window of n layers. This is done in order to smooth the profile.
    """
    if n==0:
        return y
    # Initialize lists to store averages
    y_avg = []

    # Handle the first block [0:n]
    
    y_avg.append(np.mean(y[0:n]))
    
    # Handle the second block [0:n+1]
    if len(y) > n:
        
        y_avg.append(np.mean(y[0:n+1]))

    # Handle subsequent blocks [i:n+i] starting from i=1
    for i in range(1, len(y)-1):
        
        y_avg.append(np.mean(y[i:n+i+1]))
    assert len(y_avg) == len(y), f"y_avg:{len(y_avg)}, y:{len(y)}"

    return np.array(y_avg)

def preprocess_label(label_array: Int[Array, "num_instances"],
                     ) -> Int[Array, "num_layers num_instances"]:
    label_array = map_label_to_int(label_array)
    
    return label_array


def map_label_to_int(my_list: List[str]
                     ) -> Int[Array, "num_layers num_instances"]:
    unique_categories = sorted(list(set(my_list)))
    category_to_int = {category: index
                       for index, category in enumerate(unique_categories)}
    numerical_list = [category_to_int[category] for category in my_list]
    numerical_array = np.array(numerical_list)
    return numerical_array



def compute_knn_cosine(X, X_new, k):
    """
    Compute the k-nearest neighbors distances and indices using cosine similarity with PyTorch.

    Parameters:
    - X (torch.Tensor): Reference tensor of shape (N, D)
    - X_new (torch.Tensor): Query tensor of shape (M, D)
    - k (int): Number of nearest neighbors to find

    Returns:
    - distances_k (torch.Tensor): Tensor of shape (M, k) containing cosine distances to the k-nearest neighbors
    - indices_k (torch.Tensor): Tensor of shape (M, k) containing indices of the k-nearest neighbors
    """

    # Normalize the vectors to unit length
    X_norm = X / X.norm(dim=1, keepdim=True)
    X_new_norm = X_new / X_new.norm(dim=1, keepdim=True)

    # Compute cosine similarity
    cosine_sim = torch.mm(X_new_norm, X_norm.t())  # Shape: (M, N)

    # Convert cosine similarity to cosine distance
    cosine_dist = 1 - cosine_sim  # Shape: (M, N)

    # Find the k smallest distances and their indices
    distances_k, indices_k = torch.topk(cosine_dist, k=k, dim=1, largest=False)

    return distances_k.cpu().float().numpy(), indices_k.cpu().numpy()

def compute_knn_euclidian(X, X_new, k):
    """
    Compute the k-nearest neighbors distances and indices using euclidian distance with PyTorch.

    Parameters:
    - X (torch.Tensor): Reference tensor of shape (N, D)
    - X_new (torch.Tensor): Query tensor of shape (M, D)
    - k (int): Number of nearest neighbors to find

    Returns:
    - distances_k (torch.Tensor): Tensor of shape (M, k) containing cosine distances to the k-nearest neighbors
    - indices_k (torch.Tensor): Tensor of shape (M, k) containing indices of the k-nearest neighbors
    """
    X = X.float()
    X_new = X_new.float()
    
    # Compute euclidian distance
    euclidian_dist = torch.cdist(X_new, X, p=2)  # Shape: (M, N)

    # Find the k smallest distances and their indices
    distances_k, indices_k = torch.topk(euclidian_dist, k=k, dim=1, largest=False)

    return distances_k.cpu().float().numpy(), indices_k.cpu().numpy()
def plotter(data,
            title,
            ylabel,
            names,
            yticks=None):
    
    # Set the style
    sns.set_style(
        "whitegrid",
        rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
    )
    # Setup figure and axes for 2 plots in one row
    plt.figure(dpi=200)
    layers = np.arange(0, data[0].shape[0])

    # Set ticks
    
    tick_positions = layers
    

    tick_labels = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
       
    colors = sns.color_palette("tab10", 8)
    
    markerstyle = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']

    for n, m in enumerate(zip(data, names, markerstyle)):
        int_dim, label, markerstyle  = m
        int_dim = average_custom_blocks(int_dim, avg)
        sns.scatterplot(x=layers, y=int_dim, marker=markerstyle, color=colors[n]) #  alpha=alpha[n],
        sns.lineplot(x=layers, y=int_dim, label=label, color=colors[n])

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    
    
    if title:
        plt.title(title)

    if yticks:
        plt.xticks(ticks=tick_positions, labels=tick_labels)
        if isinstance(yticks, list):
            tick_positions_y = np.arange(yticks[0], (yticks[1] + (yticks[1]-yticks[0])/10),
                                         (yticks[1]-yticks[0])/10).round(2)
        else:
            tick_positions_y = np.arange(0, (yticks + yticks/10), yticks/10).round(2)
        plt.yticks(tick_positions_y)

    plt.tick_params(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.rcParams.update(plot_config)
    plt.show()

def extract_activation(config, ablation_queries=None):
    metadata = {
        "model_name": config.model_name,
        "input": config.input,
        "batch_size": config.batch_size,
        "device_map": config.device_map,
        "torch_dtype": config.torch_dtype,
        "token": config.token,
        "num_proc": config.num_proc,
        "split": config.split,
        
    }
    extractor = Extractor(config)
    print(
        f"Extracting activations from {config.model_name} on {config.dataset_dir}..."
    )
    output = extractor.extract_cache(
        extract_resid_out=True,
        ablation_queries=ablation_queries,
    )

    out_dict = {
        "activations": {k: v for k, v in output.items()
                        if any(sub in k for sub in ("resid", "values", "pattern"))
                        },
        "metadata": metadata,
    }
    
    out_dict["token-positions"] = config.token
    out_dict["offset"] = output["offset"]
    out_dict["synset"] = output["synset"]
    out_dict["root_label"] = output["root_label"]
    out_dict["text"] = output["text"]
    out_dict["ablation_queries"] = ablation_queries
    return out_dict

def compute_overlap_last_layer(result_dict, num_layers):
    overlap = LabelOverlap()
    data = []
    tensors, labels = result_dict["activations"], result_dict["root_label"]
    labels = preprocess_label(labels)
    tensors = [tensors[f"resid_out_{i}"] for i in range(num_layers)]
    token_position = -1 
    print("Computing dist matrix")
    t_iter = [
        compute_knn_euclidian(
            t[:, token_position, :].to("cuda"), t[:, token_position, :].to("cuda"), 40
        )
        for t in tensors
    ]

    print("Computing Overlap")
    out = overlap.main(
            k=30,
            tensors=t_iter,
            labels=labels,
            number_of_layers=num_layers,
            # parallel=False
        )
    data.append(out[-1])
    del result_dict
    del tensors
    gc.collect()
    torch.cuda.empty_cache()

    return data


