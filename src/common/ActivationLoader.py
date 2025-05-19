from pathlib import Path
import os
import re
import torch
from einops import rearrange
from src.common.exceptions import DataNotFoundError
from jaxtyping import Float
from typing import  Union
from src.common.globals_vars import  Array


class ActivationLoader():
    """
    Load activations from a directory. Activations per layer are expected to be stored
    in files with the pattern "l{layer_number}_target.pt". The last file is expected to
    contain the logits.
    """
    def __init__(self, path
                 ):
        self.storage_path = Path(path)

    def load(self
             ) -> Union[Float[Array, "num_instances num_layers model_dim"],
                        Float[Array, "num_instances vocab_size"]]:
        """
        Load activations from the storage path.
        Returns:
            activations: Float[Array, "num_instances num_layers model_dim"]
                Activations for each instance, layer, and model dimension.
            logits: Float[Array, "num_instances vocab_size"]
                Logits for each instance and vocabulary size.
        """
        if not self.storage_path.exists() or not self.storage_path.is_dir():
            raise DataNotFoundError(f"Storage path does not exist:"
                                    f"{self.storage_path}")

        files = os.listdir(self.storage_path)

        # Filter files with the specific pattern and range check
        pattern = re.compile(r"l(\d+)_target\.pt")
        filtered_files = [file for file in files if pattern.match(file)]

        # Sort files based on the number in the filename
        filtered_files.sort(key=lambda x: int(pattern.match(x).group(1)))

        # Load tensors and add them to a list
        tensors = [
            torch.load(os.path.join(self.storage_path, file))
            for file in filtered_files
        ]

        # Stack all tensors along a new dimension
        stacked_tensor = torch.stack(tensors[:-1])
        stacked_tensor = rearrange(stacked_tensor, "l n d -> n l d")
        logits = tensors[-1]
        return stacked_tensor, logits