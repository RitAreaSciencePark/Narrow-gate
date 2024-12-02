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
from typing import Tuple
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
            layer: Int,
            input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_j:  Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
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