from dataclasses import dataclass
from typing import Callable, List, Dict, Optional
import pandas as pd
from enum import Enum
import torch
from functools import partial
from src.hooks import (
    ablate_attn_mat_hook,
    ablate_heads_hook,
    ablate_tokens_hook_flash_attn,
    ablate_resid_massive_act_hook
)
from src.utils import preprocess_ablation_queries

# The current version of AbaltionManager doesn't allow to ablate different kind of tokens in the same layer

class AblationType(Enum):
    STD = "std"
    FULL_HEAD = "full-head"
    BLOCK_IMG_TO_TXT = "block-img-txt"
    BLOCK_IMG_TO_IMG = "block-img-img"
    KEEP_SELF_ATTN = "keep-self-attn"
    BLOCK_IMG_TO_TXT_WOUT_SPECIAL_PIXTRAL = "block-img-txt-wout-special-pixtral"
    FLASH_ATTN = "flash-attn"
    FLASH_ATTN_STD = "flash-attn-std"
    FLASH_BLOCK_IMG_TO_TXT = "flash-block-img-txt"
    FLASH_BLOCK_IMG_TO_IMG = "flash-block-img-img"
    MASSIVE_ACT = "massive-act"



@dataclass
class AblationConfig:
    type: AblationType
    hook_pos: str
    ablation_func: Callable
    hook_func: Callable


class AblationManager:
    def __init__(
        self,
        model_config,
        ablation_queries: pd.DataFrame,
        token_to_pos: Callable,
        model_family: str,
        model_attn_type: str,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.model_config = model_config
        self._register_ablation_configs()
        self.ablation_queries = preprocess_ablation_queries(
            ablation_queries, self.model_config
        )
        self.token_to_pos = token_to_pos
        self.model_family = model_family
        self.inputs = inputs
        self.model_attn_type = model_attn_type

    def _register_ablation_configs(self):
        """Register all ablation configurations."""

        self.ablation_configs = {
            AblationType.STD: AblationConfig(
                type=AblationType.STD,
                hook_pos=self.model_config.attn_matrix_hook_name,
                ablation_func=self._ablation_type_std_func,
                hook_func=ablate_attn_mat_hook,
            ),
            AblationType.FULL_HEAD: AblationConfig(
                type=AblationType.FULL_HEAD,
                hook_pos=self.model_config.attn_matrix_hook_name,
                ablation_func=self._ablation_type_full_head_func,
                hook_func=ablate_heads_hook,
            ),
            AblationType.BLOCK_IMG_TO_TXT: AblationConfig(
                type=AblationType.BLOCK_IMG_TO_TXT,
                hook_pos=self.model_config.attn_matrix_hook_name,
                ablation_func=self._ablation_type_block_img_to_txt_func,
                hook_func=ablate_attn_mat_hook,
            ),
            AblationType.BLOCK_IMG_TO_IMG: AblationConfig(
                type=AblationType.BLOCK_IMG_TO_IMG,
                hook_pos=self.model_config.attn_matrix_hook_name,
                ablation_func=self._ablation_type_block_img_to_img_func,
                hook_func=ablate_attn_mat_hook,
            ),
            AblationType.KEEP_SELF_ATTN: AblationConfig(
                type=AblationType.KEEP_SELF_ATTN,
                hook_pos=self.model_config.attn_matrix_hook_name,
                ablation_func=self._ablation_type_keep_self_attn_func,
                hook_func=ablate_attn_mat_hook,
            ),
            AblationType.BLOCK_IMG_TO_TXT_WOUT_SPECIAL_PIXTRAL: AblationConfig(
                type=AblationType.BLOCK_IMG_TO_TXT_WOUT_SPECIAL_PIXTRAL,
                hook_pos=self.model_config.attn_matrix_hook_name,
                ablation_func=self._ablation_type_block_img_to_txt_wout_special_pixtral_func,
                hook_func=ablate_attn_mat_hook,
            ),
            AblationType.FLASH_ATTN: AblationConfig(
                type=AblationType.FLASH_ATTN,
                hook_pos=self.model_config.attn_value_hook_name,
                ablation_func=self._ablation_type_flash_attn_func,
                hook_func=ablate_tokens_hook_flash_attn,
            ),
            AblationType.FLASH_BLOCK_IMG_TO_TXT: AblationConfig(
                type=AblationType.FLASH_BLOCK_IMG_TO_TXT,
                hook_pos=self.model_config.attn_mask_hook_name,
                ablation_func=self._ablation_type_flash_block_img_to_txt_func,
                hook_func=ablate_tokens_hook_flash_attn,
            ),
            AblationType.FLASH_BLOCK_IMG_TO_IMG: AblationConfig(
                type=AblationType.FLASH_BLOCK_IMG_TO_IMG,
                hook_pos=self.model_config.attn_mask_hook_name,
                ablation_func=self._ablation_type_flash_block_img_to_img_func,
                hook_func=ablate_tokens_hook_flash_attn,
            ),
            AblationType.FLASH_ATTN_STD: AblationConfig(
                type=AblationType.FLASH_ATTN_STD,
                hook_pos=self.model_config.attn_mask_hook_name,
                ablation_func=self._ablation_type_flash_attn_std_func,
                hook_func=ablate_tokens_hook_flash_attn,
            ),
            AblationType.MASSIVE_ACT: AblationConfig(
                type=AblationType.MASSIVE_ACT,
                hook_pos=self.model_config.residual_stream_hook_name,
                ablation_func=self._ablation_type_massive_act_func,
                hook_func=ablate_resid_massive_act_hook,
            ),
            # Add more configurations here
        }

    def _get_layer_specific_queries(
        self, queries: pd.DataFrame, layer: int
    ) -> pd.DataFrame:
        """Get queries specific to a layer."""
        return queries[queries["layer"] == layer]

    def _create_hooks_list(
        self, ablation_type: AblationType, ablation_queries: pd.DataFrame, **kwargs
    ) -> List[Dict]:
        """Create the hooks list that will be passed to Pyvene wrapper."""
        # print("Creating hooks for ablation type: ", ablation_type)
        config = self.ablation_configs[ablation_type]
        return [
            {
                "component": config.hook_pos.format(layer),
                "intervention": partial(
                    config.hook_func,
                    ablation_queries=self._get_layer_specific_queries(
                        ablation_queries, layer
                    ),
                    **kwargs,
                ),
            }
            for layer in ablation_queries["layer"].unique().tolist()
        ]

    def _create_hooks(
        self, ablation_type: AblationType, ablation_queries: pd.DataFrame, **kwargs
    ) -> List[Dict]:
        """Create hooks for a specific ablation type."""
        if not set(ablation_queries["type"]).issubset(set(item.value for item in AblationType)):
            raise ValueError("Invalid ablation type found in ablation queries.")
        type_specific_queries = ablation_queries[
            ablation_queries["type"] == ablation_type.value
        ]
        if type_specific_queries.empty:
            return []
        # assert type_specific_queries == AblationType.STD and self.model_attn_type == "flash_attention_2", "Flash attn only supports flash-attn type of ablation"
        hooks = self.ablation_configs[ablation_type].ablation_func(
            type_specific_queries, **kwargs
        )

        return hooks

    def main(self) -> List[Dict]:
        """Create all necessary hooks based on ablation queries."""
        hooks = []

        for ablation_type in AblationType:
            hooks.extend(self._create_hooks(ablation_type, self.ablation_queries))
        
        
        

        return hooks

    def _ablation_type_std_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        ablation_queries = ablation_queries.copy()
        keys = self.token_to_pos(token=ablation_queries["elem-to-ablate"].values[0])
        ablation_queries["keys"] = [keys] * len(ablation_queries)
        ablation_queries["queries"] = [self.token_to_pos(token="@all")] * len(ablation_queries)

        ablation_queries.drop(columns=["elem-to-ablate"], inplace=True)
        hooks = self._create_hooks_list(AblationType.STD, ablation_queries)
        return hooks

    def _ablation_type_full_head_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        hooks = self._create_hooks_list(AblationType.FULL_HEAD, ablation_queries)
        return hooks

    def _ablation_type_block_img_to_txt_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        ablation_queries = ablation_queries.copy()
        ablation_queries["keys"] = [self.token_to_pos(token="@all-image")] * len(ablation_queries)
        ablation_queries["queries"] = [self.token_to_pos(token="@all-text")] * len(ablation_queries)
        
        ablation_queries.drop(columns=["elem-to-ablate"], inplace=True)
        hooks = self._create_hooks_list(AblationType.BLOCK_IMG_TO_TXT, ablation_queries)
        return hooks

    def _ablation_type_block_img_to_img_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        ablation_queries = ablation_queries.copy()
        ablation_queries["keys"] = [self.token_to_pos(token="@all-image")] * len(ablation_queries)
        ablation_queries["queries"] = [self.token_to_pos(token="@all-image")] * len(ablation_queries)
        
        ablation_queries = ablation_queries.drop(columns=["elem-to-ablate"])
        hooks = self._create_hooks_list(AblationType.BLOCK_IMG_TO_IMG, ablation_queries)
        return hooks

    def _ablation_type_keep_self_attn_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        raise NotImplementedError("Currently not supported")
        # hooks = self._create_hooks_list(AblationType.KEEP_SELF_ATTN, ablation_queries)
        # return hooks

    def _ablation_type_block_img_to_txt_wout_special_pixtral_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        keys = list(set(self.token_to_pos(token="@all-image")) - set(self.token_to_pos(
            token="@special-pixtral"
        )))
        ablation_queries["keys"] = [keys] * len(ablation_queries)
        ablation_queries["queries"] = [self.token_to_pos(token="@all-text")] * len(ablation_queries)

        ablation_queries = ablation_queries.drop(columns=["elem-to-ablate"])
        hooks = self._create_hooks_list(AblationType.BLOCK_IMG_TO_TXT_WOUT_SPECIAL_PIXTRAL, ablation_queries)
        return hooks
    
    def _ablation_type_flash_attn_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        ablation_queries = ablation_queries.copy()
        keys = self.token_to_pos(token=ablation_queries["elem-to-ablate"].values[0])
        ablation_queries["keys"] = [keys] * len(ablation_queries)
        ablation_queries["queries"] = [self.token_to_pos(token="@all")] * len(ablation_queries)
        
        ablation_queries.drop(columns=["elem-to-ablate"], inplace=True)
        hooks = self._create_hooks_list(AblationType.FLASH_ATTN, ablation_queries)
        return hooks
    
    def _ablation_type_flash_attn_std_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        ablation_queries = ablation_queries.copy()
        keys = self.token_to_pos(token=ablation_queries["elem-to-ablate"].values[0])
        ablation_queries["keys"] = [keys] * len(ablation_queries)
        ablation_queries["queries"] = [self.token_to_pos(token="@all")] * len(ablation_queries)
        
        ablation_queries.drop(columns=["elem-to-ablate"], inplace=True)
        hooks = self._create_hooks_list(AblationType.FLASH_ATTN_STD, ablation_queries)
        return hooks
    
    def _ablation_type_flash_block_img_to_txt_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        ablation_queries = ablation_queries.copy()
        ablation_queries["keys"] = [self.token_to_pos(token="@all-image")] * len(ablation_queries)
        ablation_queries["queries"] = [self.token_to_pos(token="@all-text")] * len(ablation_queries)
        
        ablation_queries.drop(columns=["elem-to-ablate"], inplace=True)
        hooks = self._create_hooks_list(AblationType.FLASH_BLOCK_IMG_TO_TXT, ablation_queries)
        return hooks
    
    def _ablation_type_flash_block_img_to_img_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        ablation_queries = ablation_queries.copy()
        ablation_queries["keys"] = [self.token_to_pos(token="@all-image")] * len(ablation_queries)
        ablation_queries["queries"] = [self.token_to_pos(token="@all-image")] * len(ablation_queries)
        
        ablation_queries.drop(columns=["elem-to-ablate"], inplace=True)
        hooks = self._create_hooks_list(AblationType.FLASH_BLOCK_IMG_TO_IMG, ablation_queries)
        return hooks
    
    def _ablation_type_massive_act_func(
        self,
        ablation_queries: pd.DataFrame,
    ):
        ablation_queries = ablation_queries.copy()
        #import pdb; pdb.set_trace()
        ablation_queries["elem-to-ablate"] = [self.token_to_pos(token=ablation_queries["elem-to-ablate"].values[0])] * len(ablation_queries)
        hooks = self._create_hooks_list(AblationType.MASSIVE_ACT, ablation_queries)
        return hooks