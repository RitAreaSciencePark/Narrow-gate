# file that contain tcode for computing metrix for attention heads
import torch
from typing import List, Union, Optional, Tuple, Dict, Literal, Any
from transformers import ChameleonProcessor, PixtralProcessor
from src.utils import to_string_tokens
from line_profiler import profile
from einops import einsum


#TODO debug for Pixtral

class AttentionHeadsMetricComputer:
    def __init__(
        self,
        batch_dim: int,
        multimodal_processor: Optional[ChameleonProcessor] = None,
        torch_type: torch.dtype = torch.bfloat16,
    ):
        """
        HOW to Add new Model:
            If you want to add a new model, you need to add the special tokens and the fixed index for the modalities in the _init_from_processor method.
        Args:
            - batch_dim: the dimension of the batch
            - multimodal_processor: the multimodal processor used to process the data. It is used to get the special tokens and the fixed index for the modalities
            - torch_type: the type of the tensor to use. It can be torch.float32 or torch.bfloat16
        """
        self.batch_dim = batch_dim
        self.processor = multimodal_processor
        if isinstance(self.processor, ChameleonProcessor):
            self.model_family = "chameleon"
        elif isinstance(self.processor, PixtralProcessor):
            self.model_family = "pixtral"
        else:
            raise ValueError(
                "The processor is not recognized. Please use a processor from the Chameleon or Pixtral models"
            ) 
        self.special_tokens = self._init_from_processor()
        self.torch_type = torch_type

    def _init_from_processor(
        self,
    ) -> Tuple[
        Dict[str, Union[Tuple[str, int], Tuple[str, List[int]]]],
    ]:
        """
        Initialize the special tokens from the processor that is used (so depending on the model)
        Returns:
            - special_tokens: a dictionary containing the special tokens for each mod
        """
        if isinstance(self.processor, ChameleonProcessor):
            special_tokens = {
                "start-sentence": ("<s>", self.processor.tokenizer.bos_token_id),  # type: ignore
                "start-image": (
                    "<racm3:break>",
                    self.processor.tokenizer.encode("<racm3:break>")[1],  # type: ignore
                ),
                "end-image": ("<eoss>", self.processor.tokenizer.encode("<eoss>")[1]),  # type: ignore
                "dot": ("<dot>", self.processor.tokenizer.encode(".")[1]),  # type: ignore
                "space": ("<space>", self.processor.tokenizer.encode(" ")[1]),  # type: ignore
                "last_token_image": ("<last_token_image>", self.processor.tokenizer.encode("<image><eoss>")[1:]),  # type: ignore
                "pos:<image_token_31>": ("pos:<image_token_31>", torch.tensor(-1000)), # 
                "pos:<text_token_-1>": ("pos:<text_token_-1>", torch.tensor(-1000)),
            }  
            

            self.image_token = self.processor.tokenizer.encode("<image>")[1]  # type: ignore

        elif isinstance(self.processor, PixtralProcessor):
            special_tokens = {
                "start-sentence": ("<s>", self.processor.tokenizer.bos_token_id),  # type: ignore,
                "end-image": ("[IMG_END]", self.processor.tokenizer.encode("[IMG_END]")[0]),  # type: ignore
                "dot": ("<dot>", self.processor.tokenizer.encode(".")[0]),  # type: ignore
                "space": ("<space>", self.processor.tokenizer.encode(" ")[0]),  # type: ignore
                "last_token_image": ("<last_token_image>", self.processor.tokenizer.encode("[IMG][IMG_END]")[0:]),  # type: ignore
            }
            
            self.image_token = self.processor.tokenizer.encode("[IMG]")[0]  # type: ignore
        
        else:
            raise ValueError(
                "The processor is not recognized. Please use a processor from the Chameleon or Pixtral models"
            )

        return special_tokens

    def find_index(self, order, input_ids:torch.Tensor, key) -> int:
        if self.model_family == "chameleon":
            fixed_index = {
                "image->text": {
                    "start_image_idx": 0,
                    "first_token_img_idx": 2,
                    "last_token_img_idx": 1024,  # 1024 tokens for the image, 0-1023. This is the last token of the image!! !!! Watch out when slicing
                    "end_image_idx": 1026,
                    "start_text_token_idx": 1027,
                    "end_text_token_idx": -1,
                    # "pos:<image_token_31>": self._find_index_special_positional_tokens(("pos:<image_token_31>", 31), "image->text"),
                    # "pos:<text_token_-1>": self._find_index_special_positional_tokens(("pos:<text_token_-1>", -1), "image->text"),
                },
                "text->image": {
                    "start_text_token_idx": 0,
                    "end_text_token_idx": -1027,
                    "start_image_idx": -1026,
                    "first_token_img_idx": -1025,
                    "last_token_img_idx": -2,
                    "end_image_idx": -1,
                    # "pos:<image_token_31>": self._find_index_special_positional_tokens(("pos:<image_token_31>", 31), "text->image"),
                    # "pos:<text_token_-1>": self._find_index_special_positional_tokens(("pos:<text_token_-1>", -1), "text->image"),
                },
            }
            
            return fixed_index[order][key]
        elif self.model_family == "pixtral":
            if order == "image->text":
                match key:
                    case "start_image_idx":
                        return 0
                    case "first_token_img_idx":
                        return 1
                    case "last_token_img_idx":
                        # find the position of the last token of the image
                        token = self.special_tokens["end-image"][1]
                        # find the index of the last token of the image in the input_ids
                        index  = torch.where(input_ids == token)[0].item()
                        # return the index int of the last token of the image
                        return int(index-1)
                    case "end_image_idx":
                        # find the position of the last token of the image
                        token = self.special_tokens["end-image"][1]
                        # find the index of the last token of the image in the input_ids
                        index  = torch.where(input_ids == token)[0].item()
                        # return the index int of the last token of the image
                        return int(index)
                    case "start_text_token_idx":
                        # find the position of the first token of the text
                        token = self.special_tokens["end-image"][1]
                        # find the index of the first token of the text in the input_ids
                        index  = torch.where(input_ids == token)[0].item()
                        # return the index int of the first token of the text
                        return int(index+1)
                    case "end_text_token_idx":
                        return -1
            elif order == "text->image":
                raise NotImplementedError("The order text->image is not implemented for the Pixtral model")
                    
                        
        
        

    def _assert_inputs(
        self,
        pattern: Dict[str, List[torch.Tensor]],
        input_ids: List[torch.Tensor],
        order: Literal["image->text", "text->image"],
    ):
        """
        Simple function to perform some checks on the inputs to prevent errors.

        Assert the following:
            - all the patterns have the same length as the input_ids
        """

        # Assert that the length of the patterns is the same as the length of the input_ids
        # sample
        for key in pattern.keys():
            if "pattern" in key:
                for i in range(len(pattern[key])):
                    assert (
                        pattern[key][i].squeeze().shape[0]
                        == input_ids[i].squeeze().shape[0]
                    ), f"Pattern {key} has a different length than the input_ids"

                # assert the order of the modalities
                # assert that the order is correct. If image->text the tokens between [2,1026] shoull be "<image>"
                string_tokens = to_string_tokens(input_ids[i], self.processor)
                if order == "image->text":
                    assert (
                        string_tokens[2:1026] == ["<image>"] * 1024
                    ), f"The order of the modalities is not correct for pattern {key} and example {i}. Expected order: image->text"
                elif (
                    order == "text->image"
                ):  # if text->image the tokens between [-1025, -1] should be "<image>"
                    assert (
                        string_tokens[-1025:-1] == ["<image>"] * 1024
                    ), f"The order of the modalities is not correct for pattern {key} and example {i}. Expected order: text->image"
                break
            else:
                continue


    @profile
    def get_block_without_special_tokens(
        self,
        block: torch.Tensor,
        input_ids_col: torch.Tensor,
        special_tokens: List[Union[Tuple[str, int], Tuple[str, List[int]]]],
        cache: dict,
        return_density: bool = False,
        no_diag: bool = False,
        order: Literal["image->text", "text->image"] = "image->text",
    ) -> Tuple[Union[torch.Tensor, float], Optional[Union[torch.Tensor, float]]]:
        """
        Given a block (subset of the attention pattern), remove the columns corresponding to the special tokens and compute the density of the block.
        Density is computed as the sum of the values in the block divided by the number of rows in the block.

        Args:
            - block: the block for which we want to compute the density
            - input_ids_col: the input_ids of the column (the base of the block)
            - special_tokens: the special tokens to remove from the block
            - cache: a dictionary to store the masks for the special tokens
            - return_density: if True, return the density of the block, if False return the block without the special tokens
            - no_diag: if True, return also the density without the diagonal elements, if False return None

        """
        # Create mask using vectorized operations
        mask = torch.ones(block.shape[2], dtype=torch.bool, device=block.device)
        single_tokens = torch.tensor(
            [token[1] for token in special_tokens if not isinstance(token[1], list)],
            device=input_ids_col.device,
        )
        pair_tokens = [
            token[1] for token in special_tokens if isinstance(token[1], list)
        ]

        # Handle single tokens all at once
        if single_tokens.numel() > 0:
            # check if the token is in the cache, if not compute the mask
            # if f"{single_tokens[0]}" not in cache:
            mask &= ~torch.isin(input_ids_col, single_tokens)
                # cache[f"{single_tokens[0]}"] = mask
            

        # Handle pair tokens
        if pair_tokens:
            pairs = torch.stack([input_ids_col[:-1], input_ids_col[1:]], dim=1)

            for token_1, token_2 in pair_tokens:
                pair_mask = (
                    pairs
                    == torch.tensor([token_1, token_2], device=input_ids_col.device)
                ).all(dim=1)
                mask[:-1][pair_mask] = False
                mask[1:][pair_mask] = False


        # Handle positional tokens
        for token in special_tokens:
            if token[0].startswith("pos:"):
                idx = self._find_index_special_positional_tokens(token, order)
                mask[idx] = False

        # Apply mask and calculate densities
        processed_block = block[:, :, mask]

        if return_density:
            total_sum = processed_block.to(torch.float32).sum(dim=(1, 2))
            num_rows = processed_block.shape[1]
            # num_rows = processed_block[0, 0].numel()
            density = total_sum / num_rows
            # implement the density preserving the first dimension
            # density = total_sum / processed_block.shape[0]

            if no_diag:
                if processed_block.shape[1] != processed_block.shape[2]:
                    diag_sum = processed_block[:,-1,-1]
                else:
                    diag_sum = (
                        torch.diagonal(processed_block, dim1=1, dim2=2)
                        .to(torch.float32)
                        .sum(dim=1)
                    )
                # non_diag_elements = num_rows - processed_block.shape[0,0]
                density_nodiag = (total_sum - diag_sum) / num_rows
            else:
                density_nodiag = None

            return density, density_nodiag

        return processed_block, mask


    def _find_index_special_positional_tokens(self, token, order):
        position = int(token[0].split("_")[-1].split(">")[0])
        modality = token[0].split(":")[1].split("_")[0][1:]
        if position < 0:
            return position
        elif (order == "image->text" and modality == "image") or (order == "text->image" and modality == "text"):
            idx = position + 1
        elif (order == "text->image" and modality == "image") or (order == "image->text" and modality == "text"):
            idx = position
            
        return idx
    @profile
    def _get_column_density(
        self,
        block: torch.Tensor,
        input_ids_col: torch.Tensor,
        token: Union[Tuple[str, int], Tuple[str, List[int]]],
        order: Literal["image->text", "text->image"] = "image->text",
    ) -> torch.Tensor:
        """
        Compute the density of a column in a block

        Formula: sum of the values in the column divided by the number of rows in the block

        Args:
            - block: the block for which we want to compute the density
            - input_ids_col: the input_ids of the column
            - token: the token for which we want to compute the density

        Returns:
            - density of the column
        """
        if token[0].startswith("pos:"):
            idx = self._find_index_special_positional_tokens(token, order)
            
            # check if the position is valid, otherwise return -100.0
            if idx >= block.shape[2]:
                return torch.full((block.shape[0],), -100.0, device=block.device)
            else:
                return block[:, :, idx].to(torch.float32).sum(dim=(1)) / block.shape[1]

            

        if isinstance(token[1], list):
            token_1, token_2 = token[1]
            pairs = torch.stack([input_ids_col[:-1], input_ids_col[1:]], dim=1)
            matches = (
                pairs == torch.tensor([token_1, token_2], device=input_ids_col.device)
            ).all(dim=1)
            idx = torch.where(matches)[0]
        else:
            idx = torch.where(input_ids_col == token[1])[0]

        if idx.numel() == 0:
            # return a tensor of shape (block.shape[0],) with -100.0
            return torch.full((block.shape[0],), -100.0, device=block.device)
            # return torch.tensor(-100.0, device=block.device)

        return block[:, :, idx].to(torch.float32).sum(dim=(1, 2)) / block.shape[1]

    def _get_column_value(
        self,
        value_block: torch.Tensor,
        input_ids_col: torch.Tensor,
        token: Union[Tuple[str, int], Tuple[str, List[int]]],
        order: Literal["image->text", "text->image"],
    ) -> torch.Tensor:
        if  token[0].startswith("pos:"):
            idx = self._find_index_special_positional_tokens(token, order)
            return value_block[:, idx,:]
        
        if isinstance(token[1], list):
            token_1, token_2 = token[1]
            pairs = torch.stack([input_ids_col[:-1], input_ids_col[1:]], dim=1)
            matches = (
                pairs == torch.tensor([token_1, token_2], device=input_ids_col.device)
            ).all(dim=1)
            idx = torch.where(matches)[0]
        else:
            idx = torch.where(input_ids_col == token[1])[0]
            
        if idx.numel() == 0:
            return torch.full((value_block.shape[0], value_block.shape[-1]), -100.0, device=value_block.device)
        
        return value_block[:, idx, :]

    def _get_blocks(
        self,
        attention_pattern: torch.Tensor,
        input_ids: torch.Tensor,
        order: Literal["image->text", "text->image"],
        value_pattern: Optional[torch.Tensor] = None,
        filter_low_values: bool = False,
    ) -> Dict[
        str,
        Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ]:
        """
        Get the blocks for the attention and value patterns. The blocks are divided in three parts: text-text, image-image, text-image.

        Args:
            - attention_pattern: the attention pattern of shape (num_heads, seq_len, seq_len)
            - value_pattern: the value pattern of shape (num_heads, seq_len, model_dim)
            - input_ids: the input_ids of the batch
            - order: the order of the modalities in the input_ids shape (seq_len)

        Returns:
            - blocks: a dictionary containing the blocks for the attention and value patterns. The keys are "text-text", "image-image", "text-image"
        """

        # NICE-TO-HAVE: implement assert on the tokens! In our experiment we required the input ids to NOT HAVE this special token. However, if present, all the computation will be wrong, so double check.
        # NICE-TO-HAVE: Normalization of attention map. To reduce the noise in the attention map we can lower to zero the values that are below a certain threshold. If needed, this is a good place to implement it.

        # assert (
        #     string_tokens[-1] != "<reserved08706>"
        # ), "The input_ids have the special token <reserved08706> at the end. This is not expected. Please remove it. This is a special token added by the Chameleon processor by default. You should disable it."

        # assert (
        #     input_ids[start_image_idx] == self.special_tokens["start-image"][1]
        # ), f"Expected token {self.special_tokens['start-image'][0]} at index {start_image_idx} but got {string_tokens[start_image_idx]}"

        # assert (
        #     input_ids[end_image_idx] == self.special_tokens["end-image"][1]
        # ), f"Expected token {self.special_tokens['end-image'][0]} at index {end_image_idx} but got {string_tokens[end_image_idx]}"

        # assert (
        #     input_ids[0] == self.special_tokens["start-sentence"][1]
        # ), f"Expected token {self.special_tokens['start-sentence'][0]} at index {start_text_token_idx} but got {string_tokens[start_text_token_idx]}"
        # squeeze the pattern and the input_ids to ensure that they have the minimum number of dimensions
        pattern = (
            attention_pattern.squeeze()
        )  # shape (num_head, seq_len, seq_len) or (seq_len, seq_len)
        if filter_low_values:
            # filtering low values
            pattern[pattern <= 0.001] = 0.0
        if (
            pattern.dim() == 2
        ):  # if the pattern is not expanded over the heads, expand it
            pattern = pattern.unsqueeze(0)
        input_ids = input_ids.squeeze()  # shape (seq_len)

        assert (
            pattern.shape[1] == input_ids.shape[0]
        ), "Pattern and input_ids have different lengths"

        # text-text block
        text_text_block = pattern[
            :,
            self.find_index(order,input_ids, "start_text_token_idx") : (
                len(input_ids)
                if self.find_index(order,input_ids, "end_text_token_idx") == -1
                else self.find_index(order,input_ids, "end_text_token_idx") + 1
            ),
            self.find_index(order,input_ids, "start_text_token_idx") : (
                len(input_ids)
                if self.find_index(order,input_ids, "end_text_token_idx") == -1
                else self.find_index(order,input_ids, "end_text_token_idx") + 1
            ),
        ]
        text_text_input_ids = input_ids[
            self.find_index(order,input_ids, "start_text_token_idx") : (
                len(input_ids)
                if self.find_index(order,input_ids, "end_text_token_idx") == -1
                else self.find_index(order,input_ids, "end_text_token_idx") + 1
            )
        ]

        if value_pattern is not None:
            text_text_value_block = value_pattern[
                :,
                self.find_index(order,input_ids,"start_text_token_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_text_token_idx") == -1
                    else self.find_index(order,input_ids,"end_text_token_idx") + 1
                ),
                :,
            ]

        # image-image block
        image_image_block = pattern[
            :,
            self.find_index(order,input_ids,"start_image_idx") : (
                len(input_ids)
                if self.find_index(order,input_ids,"end_image_idx")
                == -1  # if -1, take all the tokens, otherwise take until the end_image_idx + 1 (because the end_image_idx is included)
                else self.find_index(order,input_ids,"end_image_idx") + 1
            ),
            self.find_index(order,input_ids,"start_image_idx") : (
                len(input_ids)
                if self.find_index(order,input_ids,"end_image_idx") == -1
                else self.find_index(order,input_ids,"end_image_idx") + 1
            ),
        ]
        image_image_input_ids = input_ids[
            self.find_index(order,input_ids,"start_image_idx") : (
                len(input_ids)
                if self.find_index(order,input_ids,"end_image_idx") == -1
                else self.find_index(order,input_ids,"end_image_idx") + 1
            )
        ]

        if value_pattern is not None:
            image_image_value_block = value_pattern[
                :,
                self.find_index(order,input_ids,"start_image_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_image_idx") == -1
                    else self.find_index(order,input_ids,"end_image_idx") + 1
                ),
                :,
            ]

        # text-image block
        if order == "text->image":
            image_text_block = pattern[
                :,
                self.find_index(order,input_ids,"start_image_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_image_idx") == -1
                    else self.find_index(order,input_ids,"end_image_idx") + 1
                ),
                self.find_index(order,input_ids,"start_text_token_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_text_token_idx") == -1
                    else self.find_index(order,input_ids,"end_text_token_idx") + 1
                ),
            ]

            image_text_input_ids_x = input_ids[
                self.find_index(order,input_ids,"start_text_token_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_text_token_idx") == -1
                    else self.find_index(order,input_ids,"end_text_token_idx") + 1
                )
            ]
            image_text_input_ids_y = input_ids[
                self.find_index(order,input_ids,"start_image_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_image_idx") == -1
                    else self.find_index(order,input_ids,"end_image_idx") + 1
                )
            ]

            if value_pattern is not None:
                image_text_value_block_x = value_pattern[
                    :,
                    self.find_index(order,input_ids,"start_text_token_idx") : (
                        len(input_ids)
                        if self.find_index(order,input_ids,"end_text_token_idx") == -1
                        else self.find_index(order,input_ids,"end_text_token_idx") + 1
                    ),
                    :,
                ]
                # image_text_value_block_y = value_pattern[
                #     :,
                #     self.find_index(order,input_ids,"start_image_idx") : (
                #         len(input_ids)
                #         if self.find_index(order,input_ids,"end_image_idx") == -1
                #         else self.find_index(order,input_ids,"end_image_idx") + 1
                #     ),
                #     :,
                # ]

        elif order == "image->text":
            image_text_block = pattern[
                :,
                self.find_index(order,input_ids,"start_text_token_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_text_token_idx") == -1
                    else self.find_index(order,input_ids,"end_text_token_idx") + 1
                ),
                self.find_index(order,input_ids,"start_image_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_image_idx") == -1
                    else self.find_index(order,input_ids,"end_image_idx") + 1
                ),
            ]
            image_text_input_ids_x = input_ids[
                self.find_index(order,input_ids,"start_image_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_image_idx") == -1
                    else self.find_index(order,input_ids,"end_image_idx") + 1
                )
            ]
            image_text_input_ids_y = input_ids[
                self.find_index(order,input_ids,"start_text_token_idx") : (
                    len(input_ids)
                    if self.find_index(order,input_ids,"end_text_token_idx") == -1
                    else self.find_index(order,input_ids,"end_text_token_idx") + 1
                )
            ]
            if value_pattern is not None:
                image_text_value_block_x = value_pattern[
                    :,
                    self.find_index(order,input_ids,"start_image_idx") : (
                        len(input_ids)
                        if self.find_index(order,input_ids,"end_image_idx") == -1
                        else self.find_index(order,input_ids,"end_image_idx") + 1
                    ),
                    :,
                ]
                # image_text_value_block_y = value_pattern[
                #     :,
                #     self.fixed_index[order]["start_text_token_idx"] : (
                #         len(input_ids)
                #         if self.fixed_index[order]["end_text_token_idx"] == -1
                #         else self.fixed_index[order]["end_text_token_idx"] + 1
                #     ),
                #     :,
                # ]

        if value_pattern is not None:
            return {
                "text-image": (
                    image_text_block,
                    image_text_input_ids_x,
                    # image_text_input_ids_y,
                    image_text_value_block_x,
                    # image_text_value_block_y,
                ),
                "text-text": (
                    text_text_block,
                    text_text_input_ids,
                    text_text_value_block,
                ),
                "image-image": (
                    image_image_block,
                    image_image_input_ids,
                    image_image_value_block,
                ),
            }
        return {
            "text-image": (
                image_text_block,
                image_text_input_ids_x,
                image_text_input_ids_y,
            ),
            "text-text": (text_text_block, text_text_input_ids),
            "image-image": (image_image_block, image_image_input_ids),
        }

    @profile
    def compute_attn_density_metrics(
        self,
        pattern: torch.Tensor,
        input_ids: torch.Tensor,
        order: Literal["image->text", "text->image"],
        separate_special_tokens: Literal[
            "last_token_modality", "all", "none"
        ],  # not in use, maybe usefull in the future
        filter_low_values: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the metrics for a given pattern and input_ids
        Args:
            - pattern: the pattern for a specific attention head
            - input_ids: the input_ids of the batch
            - order: the order of the modalities in the input_ids
            - separate_special_tokens: if True, it will compute the density separately for the special tokens and the rest of the tokens

        Returns:
            - metric: a dictionary containing the metrics for the pattern. Keys:

        """
        # divide in three blocks: text-text, image-image, text-image
        # the start image and final image are consideres as part of the image block
        # text-text block
        full_blocks = self._get_blocks(
            attention_pattern=pattern,
            input_ids=input_ids,
            order=order,
            filter_low_values=filter_low_values)

        metric = {}

        # compute the full density for each block
        for key, block in full_blocks.items():  # iterate over the blocks
            if (
                key not in self.cache
            ):  # init a cache for the position of the special tokens. It will be used to speed up the computation saving the position for all the different layers
                self.cache[key] = {}

            metric[f"full_density_{key}"], metric[f"full_density_{key}_no_diag"] = (
                self.get_block_without_special_tokens(
                    block[0],
                    block[1],
                    [],
                    cache=self.cache[key],
                    return_density=True,
                    no_diag=True,
                    order=order,
                )
            )

            for token in self.special_tokens.values():
                metric[f"col_density_{token[0]}_{key}"] = self._get_column_density(
                    block[0], block[1], token, order
                )
                (
                    metric[f"partial_density_no_{token[0]}_{key}"],
                    metric[f"partial_density_no_{token[0]}_{key}_no_diag"],
                ) = self.get_block_without_special_tokens(
                    block[0],
                    block[1],
                    [token],
                    cache=self.cache[key],
                    return_density=True,
                    no_diag=True,
                    order=order,
                )

            (
                metric[f"partial_density_no_special_tokens_{key}"],
                metric[f"partial_density_no_special_tokens_no_diag_{key}"],
            ) = self.get_block_without_special_tokens(
                block[0],
                block[1],
                [token for key, token in self.special_tokens.items()],
                cache=self.cache[key],
                return_density=True,
                no_diag=True,
                order=order,
            )

        return metric

    def _compute_block_value_norm(
        self,
        attention_block: torch.Tensor,
        value_block: torch.Tensor,
        input_ids_col: torch.Tensor,
        special_tokens: List[Union[Tuple[str, int], Tuple[str, List[int]]]],
        cache: dict,
        order: Literal["image->text", "text->image"],
        no_diag: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_block, mask = self.get_block_without_special_tokens(
            attention_block, input_ids_col, special_tokens, cache, order=order
        )
        if no_diag:  # put zero in the diagonal
            if processed_block.shape[1] != processed_block.shape[2]: 
                # create a zero tensor with the same shape of the block that just the value [:,-1,-1] is not zero
                processed_block_no_diag = processed_block.clone()
                processed_block_no_diag[:, -1, -1] = 0.0
            else:
                processed_block_no_diag = processed_block.clone()
                # set the diagonal to zero
                processed_block_no_diag = processed_block_no_diag - torch.diag_embed(
                    torch.diagonal(processed_block, dim1=1, dim2=2)
                )
            
        # Apply mask to value_block
        masked_value_block = value_block[:, mask, :]

        # Compute weighted sum
        weighted_sum = einsum(
            processed_block,
            masked_value_block,
            "num_head seq_len_row seq_len_col, num_head seq_len_col model_dim -> num_head model_dim",
        )
        
        weighted_sum_no_diag = einsum(
            processed_block_no_diag,
            masked_value_block,
            "num_head seq_len_row seq_len_col, num_head seq_len_col model_dim -> num_head model_dim",
        )

        # Compute norm
        return torch.norm(weighted_sum, dim=-1),  torch.norm(weighted_sum_no_diag, dim=-1)

        

    def compute_attn_value_metrics(
        self,
        pattern: torch.Tensor,
        values: torch.Tensor,
        input_ids: torch.Tensor,
        order: Literal["image->text", "text->image"],
        separate_special_tokens: Literal["last_token_modality", "all", "none"], # not used
        filter_low_values: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            - pattern: the pattern for a specific attention head/layer of shape: (num_heads, seq_len, seq_len) or (seq_len, seq_len)
            - values: the values for a specific attention head of shape: (num_heads, seq_len, model_dim) or (seq_len, model_dim)
            - input_ids: the input_ids of the batch of shape (seq_len)
            - order: the order of the modalities in the input_ids
            - separate_special_tokens: if True, it will compute the density separately for the special tokens and the rest of the tokens
        """
        metric = {}

        # Compute norms for different blocks (text-text, image-image, text-image)
        full_blocks = self._get_blocks(
            attention_pattern=pattern,
            value_pattern=values,
            input_ids=input_ids,
            order=order,
            filter_low_values=filter_low_values,
        )
        for key, block in full_blocks.items():
            if len(block) != 3:
                raise ValueError(
                    f"Expected 3 elements in the block, got {len(block)} for key {key}"
                )
            attn_block = block[0]
            input_ids_block = block[1]
            value_block = block[2]

            metric[f"full_norm_{key}"], metric[f"full_norm_{key}_no_diag"] = (
                self._compute_block_value_norm(
                    attn_block,
                    value_block,
                    input_ids_block,
                    [],
                    self.cache,
                    order,
                    no_diag=True,
                )
            )

            # Partial block norms (excluding special tokens)
            for token in self.special_tokens.values():
                (
                    metric[f"partial_norm_no_{token[0]}_{key}"],
                    metric[f"partial_norm_no_{token[0]}_{key}_no_diag"],
                ) = self._compute_block_value_norm(
                    attn_block,
                    value_block,
                    input_ids_block,
                    [token],
                    self.cache,
                    order,
                    no_diag=True,
                )

            # Norm without all special tokens
            (
                metric[f"partial_norm_no_special_tokens_{key}"],
                metric[f"partial_norm_no_special_tokens_{key}_no_diag"],
            ) = self._compute_block_value_norm(
                attn_block,
                value_block,
                input_ids_block,
                list(self.special_tokens.values()),
                self.cache,
                order,
                no_diag=True,
            )

            # Column norms for special tokens
            for token in self.special_tokens.values():
                col_attn = self._get_column_density(
                    attn_block, input_ids_block, token, order
                )
                col_value = self._get_column_value(
                    value_block, input_ids_block, token, order
                ).squeeze()
                col_attn, col_value = col_attn.to(torch.float32), col_value.to(
                    torch.float32
                )
                if len(col_value.shape) == 3: #multiple values for the same special tokens
                    col_value = col_value[:,0,:]
                metric[f"col_norm_{token[0]}_{key}"] = torch.norm(
                    einsum(col_attn, col_value, "num_head, num_head model_dim -> num_head model_dim"), dim=-1
                )

        return metric

    @profile
    def block_density(
        self,
        pattern: Dict[str, List[torch.Tensor]],
        input_ids: List[torch.Tensor],
        order: Literal["image->text", "text->image"],
        separate_special_tokens: Literal["last_token_modality", "all", "none"],
        filter_low_values: bool = False,
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """
        Take in input a bunch of pattern from attention heads and compute the block density for each pattern.
        Args:
            - pattern: a dictionary containing the pattern for each attention head. Each value could be a tensor of shape (seq_len, seq_len) or
                       a tensor of shape (num_heads, seq_len, seq_len). This is useful to speed up the computation processing one layer at a time.
                       The keys are expected to be in the format "pattern_LiHj" where i is the layer and j is the head or "pattern_Li" where i is the layer.
            - input_ids: the input_ids of the batch, a list of tensor of shape (seq_len)
            - order: the order of the modalities in the input_ids. It can be "image->text" or "text->image"
            - separate_special_tokens: if True, it will compute the density separately for the special tokens
        Returns:
            - metrics: a dictionary containing the metrics for each attention head. The keys are the attention heads and the values are the list of metrics, one for each element of the batch
        # NICE-TO-HAVE: create a dataclass to store the metrics, instead of using dictionary. A good class should have:
                        - a method to add a new metric
                        - a method to save a metric for all the heads and layers given an example
                        - support the sum (concatenation) of two container object: in this way we can return a single object containing all the metrics and sum with a larger container object outside this function
                        - a automatich way to save the metric in a human way (for example, for each metric I would like to do MetricContainer["metric_name"][layer,head] and get a numpy array with the length of the batch)
 
        """
        # self._assert_inputs(pattern, input_ids, order) # NICE-TO-HAVE : fix the assertion and to support both the expanded and not expanded patterns

        # get all the keys of the pattern
        keys = [key for key in pattern.keys() if "pattern" in key]

        metrics = {}
        # iterate over the keys
        for batch_idx in range(self.batch_dim):
            self.cache = {}
            for layer in keys:
                if layer not in metrics:
                    metrics[layer] = []

                # iterate over the patterns
                single_layer_metric = self.compute_attn_density_metrics(
                    pattern=pattern[layer][batch_idx],
                    input_ids=input_ids[batch_idx],
                    order=order,
                    separate_special_tokens=separate_special_tokens,
                    filter_low_values=filter_low_values,
                )
                metrics[layer].append(single_layer_metric)

        # to cpu
        for key in metrics.keys():
            for i in range(len(metrics[key])):
                for metric_key in metrics[key][i].keys():
                    metrics[key][i][metric_key] = (
                        metrics[key][i][metric_key].detach().cpu()
                    )

        return metrics
        # Should

    def value_norms(
        self,
        cache: Dict[str, List[torch.Tensor]],
        input_ids: List[torch.Tensor],
        order: Literal["image->text", "text->image"],
        separate_special_tokens: Literal["last_token_modality", "all", "none"],
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """
        Args:
            - cache: a dictionary containing the values for each attention head. Each value could be a tensor of shape (seq_len, model_dim) or
                        a tensor of shape (num_heads, seq_len, model_dim). This is useful to speed up the computation processing one layer at a time.
            - input_ids: the input_ids of the batch, a list of tensor of shape (seq_len)
            - order: the order of the modalities in the input_ids. It can be "image->text" or "text->image"
            - separate_special_tokens: [Not USed] if True, it will compute the density separately for the special tokens
        
        Returns:
            - metrics: a dictionary containing the metrics for each attention head and batch element. The keys are the attention heads and the values are the list of metrics, one for each element of the batch
        """
        metrics = {}

        # Get all unique keys for patterns and values
        pattern_keys = [key for key in cache.keys() if key.startswith('pattern_')]
        value_keys = [key for key in cache.keys() if key.startswith('value_')]

        # Ensure pattern and value keys match
        assert set(key.replace('pattern_', '') for key in pattern_keys) == set(key.replace('value_', '') for key in value_keys), \
            "Pattern and value keys do not match"

        for batch_idx in range(self.batch_dim):
            self.cache = {}
            for pattern_key in pattern_keys:
                layer_or_head = pattern_key.split('_', 1)[1]  # This could be 'L0' or 'L0H8'
                value_key = f"value_{layer_or_head}"

                if layer_or_head not in metrics:
                    metrics[value_key] = []

                single_metric = self.compute_attn_value_metrics(
                    cache[pattern_key][batch_idx],
                    cache[value_key][batch_idx],
                    input_ids[batch_idx],
                    order,
                    separate_special_tokens,
                )
                metrics[value_key].append(single_metric)

        # Move results to CPU
        for key in metrics.keys():
            for i in range(len(metrics[key])):
                for metric_key in metrics[key][i].keys():
                    metrics[key][i][metric_key] = metrics[key][i][metric_key].detach().cpu()

        return metrics