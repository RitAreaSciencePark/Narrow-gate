from src.annotations import Array, _N_JOBS
from src.error import MetricComputationError, DataRetrievalError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import tqdm
import numpy as np
from functools import partial
import logging
from joblib import Parallel, delayed
from functools import partial
import logging
from jaxtyping import Float, Int, Bool



class LinearProbe():
    def __init__(self):
        pass
    def main(self,
             tensors: Float[Array, "num_instances num_layers d_model"],
             labels: Int[Array, "num_instances num_layers"],
             number_of_layers: Int,
             parallel: Bool = True
             ) -> Float[Array, "num_layers num_layers"]:
        """
        Compute linear probe between hidden states and a given set of label
        Returns:
            Float[Array, "num_layers, num_layers"]: Array of accuracies for each layer
        
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Computing linear probe with label")
        
        try:
            
            output_dict = self.parallel_compute(
                number_of_layers, tensors, labels, parallel
            )
        except DataRetrievalError as e:
            module_logger.error(
                f"Error retrieving data: {e}\n"\
                f"LinearProbe require full tensors, maybe in the location provided there are only distances"
            )
            raise
        except MetricComputationError as e:
            module_logger.error(
                f"Error occured during computation of metrics: {e}"
            )
            raise
        except Exception as e:
            module_logger.error(
                f"Error computing linear probe: {e}"
            )
            raise e

        return output_dict
    
    def parallel_compute(
            self,
            number_of_layers: Int,
            tensors: Float[Array, "num_instances num_layers model_dim"],
            label: Int[Array, "num_layer num_instances"],
            parallel: Bool = True
    ) -> Float[Array, "num_instances num_layers"]:

        """
        Compute the linear probe accuracy for each layer in parallel.
        Inputs:
            number_of_layers: Int
            tensors: Float[Array, "num_instances num_layers model_dim"]
            label: Int[Array, "num_instances num_layers"]    
        Returns:    
            accuracies: Float[Array, "num_instances num_layers"]
            Array of accuracies for each layer
        """
        process_layer = partial(
            self.process_layer, tensors=tensors
        )
        results = []
        if parallel:
            # Parallelize the computation of the metric
            # If the program crash try reducing the number of jobs
            with Parallel(n_jobs=_N_JOBS) as parallel:
                results = parallel(
                    delayed(process_layer)(layer,
                                           label=label)
                    for layer in tqdm.tqdm(range(number_of_layers),
                                           desc="Processing layers")
                )
        else:
            for layer in tqdm.tqdm(range(number_of_layers)):
                results.append(process_layer(layer,
                                             label=label))
        return np.array(results)

    def process_layer(
            self, 
            layer: Int,
            tensors: Float[Array, "num_layers num_instances d_model"],
            label: Float[Int, "num_instances"],
    ) -> Float:
        """
        Process a single layer.
        Inputs:
            layer: Int
            tensors: Float[Array, "num_layers num_instances d_model"]
            label: Float[Int, "num_instances"]
        Returns:
            Float
        """
    
        test_size = 0.3
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(
            tensors[layer],
            label,
            test_size=test_size,
            random_state=random_state
            )
        
        # Create and train the logistic regression model
        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # report = classification_report(y_test, y_pred)
        return accuracy