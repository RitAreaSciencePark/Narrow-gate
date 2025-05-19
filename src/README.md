# Src
This folder contains the core code for the experiment. It is divided into the following subfolders:
# `data`
 Contains the code to load/preprocess the data. For now it only contains the code to load/process ImageNet and the relative wordnet graph associated.
# `metrics`
Contains the code to compute the core metrics of the experiments.
# `plot`
Contains the code to plot the results of the experiments. 


# Activation Extractor `activations.py`

## Overview

This code implements an activation extractor for neural network models, specifically designed for multimodal models like Chameleon. It allows for the extraction of various activations and computation of metrics across different layers and attention heads.

## Core Components

1. **ExtractActivationConfig**: A dataclass that holds configuration parameters for the extraction process.

2. **ModelFactory**: Responsible for loading the appropriate model and tokenizer based on the configuration.

3. **DatasetFactory**: Handles loading and preprocessing of different datasets.

4. **PreprocessFunctions**: Contains preprocessing functions for different models and input types.

5. **CollateFunctions**: Provides collate functions for batching data.

6. **Extractor**: The main class that orchestrates the extraction process. It initializes the model and dataset, and provides methods for extracting activations and computing metrics.

## Key Functionalities

- **Extract Activations**: The `extract_cache` method allows for extraction of various activations including residual stream outputs, attention patterns, and head outputs.

- **Compute Metrics**: The `compute_metrics` method calculates metrics such as attention pattern density and value norms on the fly, without needing to save all activations to disk.

## Adding New Datasets

To add support for a new dataset:

1. In the `DatasetFactory` class, add a new method to load your dataset:

```python
@staticmethod
def _load_new_dataset(config: ExtractActivationConfig):
   # Load your dataset here
   return your_dataset, optional_graph
```
2. Update the `load_dataset` method in `DatasetFactory` to include your new dataset:
```python
if config.input == "your-new-dataset":
    return DatasetFactory._load_new_dataset(config)
```

3. In the `PreprocessFunctions` class, add a new method to preprocess your dataset:

```python
@staticmethod
def get_preprocess_fn_for_new_dataset(config: ExtractActivationConfig, hf_tokenizer):
    def preprocess_new_dataset(x):
        # Your preprocessing logic here
        return processed_input
    return preprocess_new_dataset
```
4. Update the `get_preprocess_fn_chameleon` method (or create a new one if you use a different model)  in `PreprocessFunctions` to include your new dataset:
```python
elif config.input == "your-new-dataset":
    return PreprocessFunctions.get_preprocess_fn_for_new_dataset(config, hf_tokenizer)
```

## Adding New Models
1. In the `ModelFactory` class, update the `load_model` method to include your new model:
```python
elif "your-new-model" in config.model:
    return (
        YourNewModelHFClass.from_pretrained(config.model, output_attentions=True),
        YourNewModelHFTokenizer.from_pretrained(config.model),
    )
```
2. Create new preprocessing and collate functions for your model in the `PreprocessFunctions` and `CollateFunctions` classes.
3. In the `Extractor` class, update the `get_extractor_routine` method to include your new model:
```python
elif "your-new-model" in config.model:
    def your_new_model_routine(batch, model, tokenizer, config):
        # Your extraction logic here
        return activations
```
4. If your model requires any metrics or processing, update `compute_metrics` in the `Extractor` accordingly.

## Extractor Routine and Hooks

The extractor routine is a core component of the Activation Extractor, responsible for defining how activations are extracted from the model during a forward pass. It leverages PyTorch hooks to capture intermediate activations without modifying the model architecture.

### How the Extractor Routine Works

1. **Initialization**: The extractor routine is defined in the `get_extractor_routine` method of the `Extractor` class. It's specific to each model type (e.g., Chameleon).

2. **Hook Definition**: The routine defines several hook functions that will be attached to specific layers of the model. These hooks capture and store activations in a `cache` dictionary.

3. **PyVene Integration**: The routine uses the PyVene library to create an `IntervenableModel`, which wraps the original model and allows for easy hook attachment.

4. **Forward Pass**: During the forward pass, the hooks are triggered, capturing the specified activations.

5. **Post-processing**: After the forward pass, the routine may perform additional processing on the captured activations before returning the `cache`.

### Using and Adding Custom Hooks

To use existing hooks or add custom ones:

1. **Existing Hooks**: The current implementation includes hooks for:
  - Residual stream outputs (`save_resid_hook`)
  - Attention head outputs (`output_per_head_hook`)
  - Value vectors (`value_vectors_head`)

2. **Adding a Custom Hook**:
  a. Define your hook function:
  ```python
  def custom_hook(b, s, additional_args):
      # b: the tensor output from the layer
      # s: the model's state
      # Process the activation as needed
      cache[f"custom_activation_{additional_args}"] = process(b.data)
   ```
   b. Create a dynamic hook using `create_dynamic_hook` function:
   ```python
   create_dynamic_hook(model, layer_name, custom_hook, additional_args)
   ```
   c. Add the hook to the `hooks` list in the extractor routine.
   ```python
   hooks.append(
    {
        "component": "model.layers[i].your_target_module",
        "intervention": create_dynamic_hook(
            model, f"model.layers[i].your_target_module", custom_hook, additional_args
        ),
    }
   )
   ```
3. Customizing extraction: modify `chameleon_routine` to include your custom extraction logic. Add parameters to control which activations are extracted
4. Accessing Extracted Data. The extracted activations are stored in the `cache` dictionary, which can be accessed after the extraction process is complete. 
   



# `utils.py`: Contains some utility functions used in the experiment.