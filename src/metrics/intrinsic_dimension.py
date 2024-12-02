from src.common.globals_vars import _NUM_PROC, Array
from src.common.error import  UnknownError

from dadapy.data import Data

import tqdm
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from jaxtyping import Float


class IntrinsicDimension():
    def __init__(
        self,
        parallel: bool = True,
    ):
        """
        Initialize the class.
        Inputs:
            parallel: bool
                Whether to use parallel computation.
        """
        self.parallel = parallel

    def compute(
            self, 
            activations: Float[Array, "num_instances num_layers model_dim"]
    ) -> Float[Array, "order_of_nearest_neighbour num_layers"]:
        """
        Collect hidden states of all instances and compute ID
        we employ two different approaches: the one of the last token, 
        the sum of all tokens
        
        Inputs
            activations: Float[Float, "num_instances, num_layers, model_dim"]
        Returns
            Float[Array, "order of nearest neighbour, num_layers"]
                Array with the ID of each layer,
                for each order of nearest neighbour
        """

        id_per_layer_gride = []

        num_layers = activations.shape[1]
        process_layer = partial(
            self.process_layer, algorithm="gride"
        )
        try:
            if self.parallel:
                # Parallel version
                with Parallel(n_jobs=_NUM_PROC) as parallel:
                    id_per_layer_gride = parallel(
                        delayed(process_layer)(activations[:, layer, :])
                        for layer in tqdm.tqdm(range(1, num_layers),
                                           desc="Processing layers")
                    )
            else:
                # Sequential version
                for layer in range(1, num_layers):
                    id_per_layer_gride.append(
                        process_layer(activations[:, layer, :])
                        )
        except Exception as e:
            print(f"Error computing ID  Error: {e}")
            raise UnknownError
        
        id_per_layer_gride.insert(0, np.ones(id_per_layer_gride[-1].shape[0]))
        return np.stack(id_per_layer_gride)
               
    def process_layer(
            self, 
            activations: Float[Array, "num_instances num_layers model_dim"],
            algorithm: str = "gride"
    ) -> Float[Array, "order_of_nearest neighbour"]:
        """
        Process a single layer
        Inputs
            layer: Int
                Layer to process
            activations: Float[Float, "num_instances, num_layers, model_dim"]
                Hidden states of the model
            algorithm: str
                Algorithm to compute the ID
        Returns
        """
        data = Data(activations)

        data.remove_identical_points()

        if algorithm == "2nn":
            raise NotImplementedError
        elif algorithm == "gride":
            return data.return_id_scaling_gride(range_max=1000)[0]


