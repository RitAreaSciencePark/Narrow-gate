# Metric
This folder contains the class used to compute the core metrics of all the experiment.

## residual_stream.py
This file contains the class `ResidualStreamMetricComputer` that compute the cross similarity between the activations of the residual stream of the model. Given activation from different input modalities, it will compute the cosine similarity between all pairs of activations. It support activations from the residual stream and output of attention heads.

## attention_heads.py
This code implements an AttentionHeadsMetricComputer class for analyzing attention patterns and value norms in transformer models, particularly those with multimodal capabilities (handling both text and image inputs).
### Key Features

- Supports analysis of attention patterns for text-to-text, image-to-image, and text-to-image (or image-to-text) interactions.
- Computes various density metrics for attention patterns.
- Calculates norms for attention-weighted value vectors.
- Handles special tokens and positional encodings.
- Optimized for performance using vectorized operations and caching.

### Main Components
#### Initialization

The class is initialized with batch dimension, multimodal processor (optional), and torch data type.
Special tokens and fixed indices for different modalities are set up based on the processor.

#### Block Extraction

The _get_blocks method divides the attention pattern into text-text, image-image, and text-image blocks based on the input order.

#### Density Metrics Computation

compute_attn_density_metrics calculates various density metrics for each block:

- Full density (with and without diagonal)
- Partial densities excluding specific special tokens
- Column densities for special tokens



#### Value Norm Computation

compute_attn_value_metrics computes norms of attention-weighted value vectors:

- Full norms (with and without diagonal)
- Partial norms excluding specific special tokens
- Column norms for special tokens



#### Batch Processing

block_density and value_norms methods process multiple attention patterns and value vectors in batches.

#### Key Algorithms

- Special Token Handling: Uses vectorized operations to create masks for excluding special tokens efficiently.
- Density Calculation: Sums attention weights and normalizes by the number of rows in the block.
- Value Norm Calculation: Computes weighted sums of value vectors using attention weights, then calculates the norm.

### Adding Special Tokens
Special tokens are added during the initialization of the AttentionHeadsMetricComputer class. They are stored in the special_tokens dictionary. To add a new special token:

Modify the _init_from_processor method.
Add a new entry to the special_tokens dictionary in the format:
pythonCopy"token_name": ("token_string", token_id)
For tokens that consist of multiple IDs, use the format:
pythonCopy"token_name": ("token_string", [token_id1, token_id2])


Example:
```python
special_tokens["new_token"] = ("<new_token>", self.processor.tokenizer.encode("<new_token>")[1])
```
#### Adding Special Positional Tokens
Special positional tokens are handled differently from regular special tokens. They are identified by a prefix "pos:" in their name. To add a new special positional token:

Add the token to the special_tokens dictionary in the _init_from_processor method:
```python
"pos:<modality_token_i>": ("pos:<modality_token_i>", torch.tensor(-1000))
```
where modality should be one of "text" or "image" and i is the position of the token. (For example, "pos:<text_1>" or "pos:<image_2>").

Check if the logic for finding the index of special positional tokens in the _find_index_special_positional_tokens method is correct. Should be correct!!:
```python
def _find_index_special_positional_tokens(self, token, order):
    position = int(token[0].split("_")[-1])
    modality = token[0].split(":")[1].split("_")[0][1:]
    # Add logic for the new positional token
    # Existing logic for other tokens
    elif (order == "image->text" and modality == "image") or (order == "text->image" and modality == "text"):
        idx = position + 1
    elif (order == "text->image" and modality == "image") or (order == "image->text" and modality == "text"):
        idx = position
    return idx
```

Ensure that the get_block_without_special_tokens method correctly handles the new positional token when creating masks.

#### Considerations

When adding new tokens, ensure that the tokenizer and model are compatible with these additions.
Special positional tokens require careful handling in the attention pattern analysis, as their positions may vary depending on the input order (image->text or text->image).
After adding new tokens, thoroughly test the compute_attn_density_metrics and compute_attn_value_metrics methods to ensure they correctly handle the new tokens.

## intrinsic_dimension.py