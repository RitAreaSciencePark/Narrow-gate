from src.annotations import Array, _N_JOBS
from src.error import MetricComputationError, DataRetrievalError
import logging
from dadapy.data import Data
import tqdm
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import logging
from jaxtyping import Float, Int, Bool
from typing import Tuple, List, Dict
import torch


class PointOverlap():
    def __init__(self):
        pass
        
    def main(self,
             k: Int,
             input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             input_j: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             number_of_layers: Int,
             parallel: Bool = True
             ) -> Float[Array, "num_layers"]:
        """
        Compute overlap between two sets of representations.

        Returns:
            pd.DataFrame
                Dataframe containing results
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Computing point overlap")


        try:
            overlap = self.parallel_compute(input_i=input_i,
                                            input_j=input_j,
                                            k=k,
                                            number_of_layers=number_of_layers,
                                            parallel=parallel)
        except DataRetrievalError as e:
            module_logger.error(
                f"Error retrieving data: {e}"
            )
            raise
        except MetricComputationError as e:
            module_logger.error(
                f"Error occured during computation of metrics: {e}"
            )
            raise
        except Exception as e:
            module_logger.error(
                f"Error computing clustering: {e}"
            )
            raise e

        return overlap

    

    def parallel_compute(
            self,
            input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_j:  Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            number_of_layers: Int, 
            k: Int,
            parallel: Bool = True
        ) -> Float[Array, "num_layers"]:
        """
        Compute the overlap between two set of representations for each layer.

        Inputs:
            input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_j: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            k: Int
                the number of neighbours considered for the overlap
        Returns:
            Float[Array, "num_layers"]
        """
        assert (
            len(input_i) == len(input_j)
        ), "The two runs must have the same number of layers"
        process_layer = partial(self.process_layer,
                                input_i=input_i,
                                input_j=input_j,
                                k=k)

        if parallel:
            with Parallel(n_jobs=_N_JOBS) as parallel:
                results = parallel(
                    delayed(process_layer)(layer)
                    for layer in tqdm.tqdm(
                        range(number_of_layers), desc="Processing layers"
                    )
                )
        else:
            results = []
            for layer in range(number_of_layers):
                results.append(process_layer(layer))

        overlaps = list(results)

        return np.stack(overlaps)

    def process_layer(
            self,
            layer: Int,predict_neighborhood_overlap_on_cache,
            # input_i: Float[Array, "num_layers num_instances d_model"] |
            #     Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            # input_j:  Float[Array, "num_layers num_instances d_model"] |
            #     Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_i,
            input_j,
            k: Int
        ) -> Float[Array, "num_layers"]:
        """
        Process a single layer
        Inputs:
            layer: Int
            input_i: Float[Array, "num_layers, num_instances, model_dim"]
            input_j: Float[Array, "num_layers, num_instances, model_dim"]
            k: Int
                the number of neighbours considered for the overlap
        Returns:
            Array
        """

        input_i = input_i[layer]
        input_j = input_j[layer]
        if isinstance(input_i, tuple):
            mat_dist_i, mat_coord_i = input_i
            data = Data(distances=(mat_dist_i, mat_coord_i), maxk=k)
            mat_dist_j, mat_coord_j = input_j
            overlap = data.return_data_overlap(distances=(mat_dist_j,
                                                          mat_coord_j), k=k)
            return overlap
        elif isinstance(input_i, np.ndarray):
            data = Data(coordinates=input_i, maxk=k)
            overlap = data.return_data_overlap(input_j, k=k)
            return overlap


class LabelOverlap():
    def __init__(self):
        pass
    def main(self,
             k: Int,
             tensors: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             labels: Int[Array, "num_layers num_instances"],
             number_of_layers: Int,
             parallel: Bool = True
             ) -> Float[Array, "num_layers"]:
        """
        Compute the agreement between the clustering of the hidden states
        and a given set of labels.
        Output
        ----------
        Float[Array, "num_layers"]
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Computing labels cluster")
        # import pdb; pdb.set_trace()
        try:
            
            output_dict = self.parallel_compute(
                number_of_layers, tensors, labels, k, parallel
            )
        except DataRetrievalError as e:
            module_logger.error(
                f"Error retrieving data: {e}"
            )
            raise
        except MetricComputationError as e:
            module_logger.error(
                f"Error occured during computation of metrics: {e}"
            )
            raise
        except Exception as e:
            module_logger.error(
                f"Error computing clustering: {e}"
            )
            raise e

        return output_dict

    def parallel_compute(
        self, 
        number_of_layers: Int,
        tensors: Float[Array, "num_layers num_instances d_model"] |
        Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
        labels: Int[Array, "num_layers num_instances"],
        k: Int,
        parallel: Bool = True
    ) -> Float[Array, "num_layers"]:
        """
        Compute the overlap between a set of representations and a given labels
        using Advanced Peak Clustering.
        M.dErrico, E. Facco, A. Laio, A. Rodriguez, Automatic topography of
        high-dimensional data sets by non-parametric density peak clustering,
        Information Sciences 560 (2021) 476492.
        Inputs:
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
                It can either receive the hidden states or the distance matrices
            labels: Float[Int, "num_instances"]
            k: Int
                the number of neighbours considered for the overlap

        Returns:
            Float[Array, "num_layers"]
        """        
        process_layer = partial(
            self.process_layer, tensors=tensors, k=k,
        )
        results = []
       
        if parallel:
            # Parallelize the computation of the metric
            # If the program crash try reducing the number of jobs
            with Parallel(n_jobs=_N_JOBS) as parallel:
                results = parallel(
                    delayed(process_layer)(layer,
                                           labels=labels)
                    for layer in tqdm.tqdm(range(number_of_layers),
                                           desc="Processing layers")
                )
        else:
            for layer in tqdm.tqdm(range(number_of_layers)):
                results.append(process_layer(layer,
                                             labels=labels))
               
        overlaps = list(results)

        return np.stack(overlaps)

    def process_layer(
            self, 
            layer: Int,
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            labels: Float[Int, "num_instances"],
            k: Int,
    ) -> Float[Array, "num_layers"]:
        """
        Process a single layer.
        Inputs:
            layer: Int
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
                It can either receive the hidden states or the distance matrices
            labels: Float[Int, "num_instances"]
            k: Int
                the number of neighbours considered for the overlap
        Returns:
            Float[Array, "num_layers"]
        """
        tensors = tensors[layer]
        try:
            # do clustering
            if isinstance(tensors, tuple):
                mat_dist, mat_coord = tensors
                data = Data(distances=(mat_dist, mat_coord), maxk=k)
                overlap = data.return_label_overlap(labels, k=k)
                return overlap
            elif isinstance(tensors, np.ndarray):
                # do clustering
                data = Data(coordinates=tensors, maxk=k)
                overlap = data.return_label_overlap(labels, k=k)
                return overlap
        except Exception as e:
            raise MetricComputationError(f"Error raised by Dadapy: {e}")





import torch
from typing import Dict

class PredictOverlap:
    def __init__(self, target: torch.Tensor, clusters: Dict[str, torch.Tensor], k: int = 40):
        """
        Initializes the PredictOverlap class.

        Args:
            target (torch.Tensor): Tensor of shape (num_target, 4096).
            clusters (Dict[str, torch.Tensor]): Dictionary where keys are cluster labels and values are tensors of shape (num_elem, 4096).
        """
        self.target = target  # shape (num_target, 4096)
        self.clusters = clusters

        # Prepare cluster data
        cluster_tensors = []
        cluster_labels = []
        label_to_idx = {}
        idx = 0
        for label, data in clusters.items():
            num_elems = data.size(0)
            cluster_tensors.append(data)
            cluster_labels.append(torch.full((num_elems,), idx, dtype=torch.long))
            label_to_idx[label] = idx
            idx += 1
        self.cluster_data = torch.cat(cluster_tensors, dim=0)  # shape (total_num_elem, 4096)
        self.cluster_labels = torch.cat(cluster_labels, dim=0)  # shape (total_num_elem,)
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        assert k < self.cluster_data.size(0), "k must be less than the total number of elements."
        self.k = k
        # Move to CUDA if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.cluster_data = self.cluster_data.to(self.device)
        self.cluster_labels = self.cluster_labels.to(self.device)
        self.target = self.target.to(self.device)

    def predict(self, x: torch.Tensor):
        """
        Computes the neighbor overlap for a single vector x.

        Args:
            x (torch.Tensor): Tensor of shape (4096,).

        Returns:
            Dict[str, int]: Dictionary with cluster labels as keys and counts as values.
        """
        x = x.to(self.device)  # shape (4096,)

        # Compute squared Euclidean distances
        distances = torch.sum((self.cluster_data - x) ** 2, dim=1)  # shape (total_num_elem,)

        # Get indices of 40 nearest neighbors
        k = 40
        nearest_indices = torch.topk(distances, k, largest=False).indices  # shape (k,)

        # Get the labels of these neighbors
        neighbor_labels = self.cluster_labels[nearest_indices]  # shape (k,)

        # Count the labels
        unique_labels, counts = neighbor_labels.unique(return_counts=True)

        # Map indices back to labels
        labels = [self.idx_to_label[idx.item()] for idx in unique_labels]
        counts = counts.tolist()

        # Return as a dict {label: count}
        overlap = dict(zip(labels, counts))
        return overlap

    def predict_avg(self):
        """
        Computes the average neighbor overlap across all target vectors.

        Returns:
            Dict[str, float]: Dictionary with cluster labels as keys and average fractions as values.
        """

        num_targets = self.target.size(0)
        batch_size = num_targets  # Adjust according to memory
        
        
        

        # Initialize dict to accumulate fractions
        cluster_fractions = {label: 0.0 for label in self.label_to_idx.keys()}

        for start in range(0, num_targets, batch_size):
            end = min(start + batch_size, num_targets)
            x_batch = self.target[start:end]  # shape (batch_size, 4096)

            # Compute distances between x_batch and cluster_data
            distances = torch.cdist(x_batch.to(torch.float32), self.cluster_data.to(torch.float32))  # shape (batch_size, total_num_elem)

            # For each vector in x_batch, get k nearest neighbors
            _, nearest_indices = torch.topk(distances, self.k, dim=1, largest=False)

            # Get neighbor labels
            neighbor_labels = self.cluster_labels[nearest_indices]  # shape (batch_size, k)

            # For each row in neighbor_labels, compute fractions
            for i in range(neighbor_labels.size(0)):
                labels, counts = neighbor_labels[i].unique(return_counts=True)
                total_counts = counts.sum().item()  # Should be equal to k
                for idx, count in zip(labels, counts):
                    label = self.idx_to_label[idx.item()]
                    fraction = count.item() / total_counts
                    cluster_fractions[label] += fraction / num_targets  # Average over all targets

        return cluster_fractions
