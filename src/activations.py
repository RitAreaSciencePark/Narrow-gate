import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize as torchvision_resize
from torchvision.transforms.functional import resized_crop as torchvision_resize_crop
from transformers import (
    ChameleonProcessor,
    ChameleonForConditionalGeneration,
    LlavaForConditionalGeneration,
    PixtralProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    GenerationConfig,
    DynamicCache,
    StaticCache,
)
from typing import ( 
    Union,
    Literal,
    Optional,
    List,
    Dict,
    Callable,
    Any,
)
from datasets import (
    load_from_disk,
    Dataset,
    DatasetDict,
    load_dataset,
    IterableDatasetDict,
    IterableDataset,
)
import concurrent.futures
from src.metrics.attention_heads import AttentionHeadsMetricComputer
import numpy as np
from easyroutine.logger import Logger
from tqdm import tqdm
from dataclasses import dataclass, field
import os
from src.ablation import AblationManager
import random
import json

from src.utils import (
    left_pad,
    aggregate_cache_efficient,
    aggregate_metrics,
    to_string_tokens,
    map_token_to_pos,
    preprocess_patching_queries,
    logit_diff,
    get_attribute_from_name,
    resize_img_with_padding,
    kl_divergence_diff,
    
)
from src.hooks import (
    partial,
    embed_hook,
    save_resid_hook,
    # save_resid_in_hook,
    avg_hook,
    projected_value_vectors_head,
    avg_attention_pattern_head,
    attention_pattern_head,
    ablate_tokens_hook_flash_attn,
    get_module_by_path
)
from rich import print
from functools import partial
from line_profiler import profile
from random import randint
import pandas as pd
import GPUtil
from copy import deepcopy
from pathlib import Path
from src.data.imagenet_classes import CHAMELEON_TOKEN_MAP, PIXTRAL_TOKEN_MAP


DEBUG = os.getenv("DEBUG") == "True"


print(
    "WARNING: This implementation use a fork of the HuggingFace transformer library to perform some experiment"
)
# to avoid running out of shared memory
torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class ExtractActivationConfig:
    model_name: str
    input: Optional[Literal[
        "imagenet-text",
        "imagenet-image",
        "labeled-imagenet",
        "imagenet-text_wn_definitions",
        "flickr-text",
        "flickr-image",
        "flickr-text->image",
        "flickr-image->text",
        "imagenet-with-counterfactuals-2class",
        "imagenet-test"
    ]] = None
    dataset_dir: Optional[Union[str, Path]] = None
    dataset_hf_name: Optional[str] = None
    batch_size: int = 1
    device_map: str = "balanced"
    torch_dtype: torch.dtype = torch.bfloat16
    token: List[str] = field(
        default_factory=lambda: ["last-4"]
    )  #    "last", "last-2",  "last-4", "labeled-imagenet-common",  "last-image", "end-image",  "all-text", "all", "image-32", "all-image"
    num_proc: int = 10
    split: Optional[str] = "train"
    id_num: Optional[int] = 0
    attn_implementation: str = "eager"  # NICE-TO-HAVE: add support for sdpe attention
    map_dataset_parallel_mode: Literal["sequential", "parallel", "multigpu"] = (
        "parallel"
    )
    resize_image: Optional[List] = field(
        default_factory= lambda: [512,512] # [width, height] or None 
    )


@dataclass
class ModelConfig:
    residual_stream_input_hook_name: str
    residual_stream_hook_name: str
    intermediate_stream_hook_name: str
    attn_value_hook_name: str
    attn_in_hook_name: str
    attn_out_hook_name: str
    attn_matrix_hook_name: str

    attn_out_proj_weight: str
    attn_out_proj_bias: str
    embed_tokens: str

    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int


@dataclass
class ModelFactoryConfig:
    model_name: str
    attn_implementation: str
    torch_dtype: torch.dtype
    device_map: str
    num_proc: int


class ModelFactory:
    """
    This class is a factory to load the model and the processor. It supports the following models:
    - Chameleon-7b

    TO ADD A NEW MODEL:
    - add the model in the load_model
    """

    @staticmethod
    def load_model(config: Union[ExtractActivationConfig, ModelFactoryConfig]):
        if config.attn_implementation != "eager":
            print(
                "WARNING: using an attention type different from eager could have unexpected beheviour in some experiment!"
            )
        if "chameleon" in config.model_name:
            model = ChameleonForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=config.torch_dtype,
                device_map=config.device_map,
                attn_implementation=config.attn_implementation,
            )
            processor = ChameleonProcessor.from_pretrained(
                config.model_name,
                torch_dtype=config.torch_dtype,
                device_map=config.device_map,
            )
            model_config = ModelConfig(
                residual_stream_input_hook_name="model.layers[{}].input",
                residual_stream_hook_name="model.layers[{}].output",
                intermediate_stream_hook_name="model.layers[{}].post_attention_layernorm.output",
                attn_value_hook_name="model.layers[{}].self_attn.v_proj.output",
                attn_out_hook_name="model.layers[{}].self_attn.o_proj.output",
                attn_in_hook_name="model.layers[{}].self_attn.input",
                attn_matrix_hook_name="model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_out_proj_weight="model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="model.embed_tokens.input",
                num_hidden_layers=model.config.num_hidden_layers,
                num_attention_heads=model.config.num_attention_heads,
                hidden_size=model.config.hidden_size,
                num_key_value_heads=model.config.num_key_value_heads,
                num_key_value_groups=model.config.num_attention_heads
                // model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
            )

        elif "mistral-community/pixtral-12b" in config.model_name:
            print(
                "WARNING: this implementation support just input of the form [IMG] ... [IMG_END] [TEXT]!!"
            )

            model = LlavaForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=config.torch_dtype,
                device_map=config.device_map,
                attn_implementation=config.attn_implementation,
            )
            processor = PixtralProcessor.from_pretrained(
                config.model_name,
                torch_dtype=config.torch_dtype,
                device_map=config.device_map,
            )
            model_config = ModelConfig(
                residual_stream_input_hook_name="language_model.model.layers[{}].input",
                residual_stream_hook_name="language_model.model.layers[{}].output",
                intermediate_stream_hook_name="language_model.model.layers[{}].post_attention_layernorm.output",
                attn_value_hook_name="language_model.model.layers[{}].self_attn.v_proj.output",
                attn_out_hook_name="language_model.model.layers[{}].self_attn.o_proj.output",
                attn_in_hook_name="language_model.model.layers[{}].self_attn.input",
                attn_matrix_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_out_proj_weight="language_model.model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="language_model.model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="language_model.model.embed_tokens.input",
                num_hidden_layers=model.config.text_config.num_hidden_layers,
                num_attention_heads=model.config.text_config.num_attention_heads,
                hidden_size=model.config.text_config.hidden_size,
                num_key_value_heads=model.config.text_config.num_key_value_heads,
                num_key_value_groups=model.config.text_config.num_attention_heads
                // model.config.text_config.num_key_value_heads,
                head_dim=model.config.text_config.head_dim,
            )

        else:
            raise ValueError("Unsupported model_name")

        return model, processor, model_config


@dataclass
class DatasetFactoryConfig:
    input: Literal[
        "imagenet-text",
        "imagenet-image",
        "labeled-imagenet",
        "imagenet-text_wn_definitions",
        "flickr-text",
        "flickr-image",
        "flickr-text->image",
        "flickr-image->text",
        "imagenet-with-counterfactuals-2class",
    ]
    dataset_dir: Union[str, Path]
    dataset_hf_name: Optional[str] = None
    split: Optional[str] = "train"


class DatasetFactory:
    """
    This class is just a factory that return the correct dataset. Now support:
    - imagenet-image: the imagenet dataset with just images as input
    - imagenet-text: the imagenet dataset with just text as input
    - imagenet-text_wn_definitions: the imagenet dataset with the wn definitions as input
    - flickr: the flickr dataset

    TO ADD A NEW DATASET
    Just add here the option to load a new dataset
    """

    @staticmethod
    def load_dataset(
        config: Union[ExtractActivationConfig, DatasetFactoryConfig],
    ) -> Union[Dataset, DatasetDict, IterableDatasetDict, IterableDataset]:
        if config.input == "imagenet-image":
            return DatasetFactory._load_imagenet_image(config)
        elif config.input == "labeled-imagenet":
            return DatasetFactory._load_labeled_imagenet(config)
        elif config.input == "imagenet-text_wn_definitions":
            return DatasetFactory._load_imagenet_text_wn_definitions(config)
        elif config.input.startswith("flickr"):
            return load_dataset("nlphuji/flickr30k", split=config.split)
        elif "imagenet-text" in config.input or "imagenet-test" in config.input:
            return DatasetFactory._load_imagenet_text(config)
        elif "imagenet-with-counterfactuals-2class" in config.input:
            return DatasetFactory._load_imagenet_with_counterfactual(config)
        else:
            raise ValueError("Unsupported input type: get {}".format(config.input))

    @staticmethod
    def _load_imagenet_with_counterfactual(
        config: Union[ExtractActivationConfig, DatasetFactoryConfig],
    ):
        dataset = DatasetFactory._load_imagenet_text(config)

        unique_labels = sorted(set(dataset["root_label"]))

        if len(unique_labels) % 2 != 0:
            raise ValueError("The number of classes should be even")

        # create a list of cuples of labels [(label1, label2), (label3, label4), ...]
        split_labels = [
            (list(unique_labels)[i], list(unique_labels)[i + 1])
            for i in range(0, len(unique_labels), 2)
        ]
        base_labels = [label[0] for label in split_labels]
        counterfactual_labels = [label[1] for label in split_labels]

        print(f"Counterfactual labels pair:\n {split_labels}")

        dataset_1 = dataset.filter(lambda x: x["root_label"] in base_labels)
        dataset_2 = dataset.filter(lambda x: x["root_label"] in counterfactual_labels)

        return (dataset_1, dataset_2)

    @staticmethod
    def _load_imagenet_text(
        config: Union[ExtractActivationConfig, DatasetFactoryConfig],
    ):
        if config.dataset_hf_name is not None:
            return load_dataset(config.dataset_hf_name, split=config.split)
        return load_from_disk(f"{config.dataset_dir}/imagenet-text.arrow")

    @staticmethod
    def _load_imagenet_image(
        config: Union[ExtractActivationConfig, DatasetFactoryConfig],
    ):
        if config.dataset_hf_name is not None:
            return load_dataset(config.dataset_hf_name, split=config.split)
        return load_from_disk(f"{config.dataset_dir}/imagenet-text.arrow")

    @staticmethod
    def _load_labeled_imagenet(
        config: Union[ExtractActivationConfig, DatasetFactoryConfig],
    ):
        if config.dataset_hf_name is not None:
            return load_dataset(config.dataset_hf_name, split=config.split)
        return load_from_disk(f"{config.dataset_dir}/imagenet-text.arrow")

    @staticmethod
    def _load_imagenet_text_wn_definitions(
        config: Union[ExtractActivationConfig, DatasetFactoryConfig],
    ):
        return DatasetFactory._load_imagenet_image(config)

    @staticmethod
    def get_additional_batch_saver(
        config: Union[ExtractActivationConfig, DatasetFactoryConfig],
    ) -> Callable:
        if config.input == "imagenet-image":
            return DatasetFactory._imagenet_image_batch_saver()
        elif config.input == "labeled-imagenet":
            return DatasetFactory._labeled_imagenet_batch_saver()
        elif (
            config.input == "imagenet-text"
            or config.input == "imagenet-with-counterfactuals-2class"
        ):
            return DatasetFactory._imagenet_text_batch_saver()
        else:
            return DatasetFactory._no_batch_saver()

    @staticmethod
    def _imagenet_image_batch_saver():
        def imagenet_image_saver(batch):
            raise NotImplementedError

        return imagenet_image_saver

    @staticmethod
    def _labeled_imagenet_batch_saver():
        def labeled_imagenet_saver(batch):
            return {
                "offset": [item["offset"] for item in batch],
                "label": [item["label"] for item in batch],
                "synset": [item["synset"] for item in batch],
                "root_label": [item["root_label"] for item in batch],
            }

        return labeled_imagenet_saver

    @staticmethod
    def _imagenet_text_batch_saver():
        def imagenet_text_saver(batch):
            return {
                "text": [item["text"] for item in batch],
                "offset": [item["offset"] for item in batch],
                "label": [item["label"] for item in batch],
                "synset": [item["synset"] for item in batch],
                "root_label": [item["root_label"] for item in batch],
            }

        return imagenet_text_saver

    @staticmethod
    def _no_batch_saver():
        def batch_saver(batch):
            return None

        return batch_saver


class PreprocessFunctions:
    """
    This class contains the preprocess functions for the different models. It supports the following models:
    - Chameleon-7b
    - Pixtral-12b (in progress)

    A preprocess function is a function that tokenize the input of the model based on the input type. It returns a dictionary with the tokenized input of the model
    """

    @staticmethod
    def get_args_chameleon(config):
        common_kwargs = {
            "padding": False,
            "return_tensors": "pt",
            "return_for_text_completion": True,
        }
        image_string = "<image>"
        start_sentence = ""  # chameleon add a start token by default

        return common_kwargs, image_string, start_sentence, lambda img: img

    @staticmethod
    def get_args_pixtral(config):
        common_kwargs = {
            "padding": False,
            "return_tensors": "pt",
        }
        image_string = "[IMG]"
        start_sentence = "<s>"

        if config.resize_image:
            return common_kwargs, image_string, start_sentence, lambda img: torchvision_resize(img, size=config.resize_image)
        else:
            return common_kwargs, image_string, start_sentence, lambda img: img

   
    

    @staticmethod
    def get_preprocess_fn(config: ExtractActivationConfig, hf_tokenizer):
        if "chameleon" in config.model_name.lower():
            common_kwargs, image_string, start_sentence, process_img = (
                PreprocessFunctions.get_args_chameleon(config)
            )
        elif "pixtral" in config.model_name.lower():
            common_kwargs, image_string, start_sentence, process_img = (
                PreprocessFunctions.get_args_pixtral(config)
            )
        else:
            raise ValueError(f"Unsupported model {config.model_name}")

        if config.input == "imagenet-image":

            def tokenize_images(x):
                return hf_tokenizer(
                    text=start_sentence + image_string,
                    images=process_img(x["image"]),
                    **common_kwargs,
                )

            return tokenize_images
        elif config.input == "labeled-imagenet":

            def tokenize_images_text(x):
                return hf_tokenizer(  # type: ignore
                    text=f"{start_sentence}{image_string} {x['text']}",
                    images=process_img(x["image"]),
                    **common_kwargs,
                )

            return tokenize_images_text

        elif config.input in ["imagenet-text", "imagenet-with-counterfactuals-2class"]:
            
            if config.model_name == "facebook/chameleon-30b":
                def tokenize_images_text(x):
                    return hf_tokenizer(  # type: ignore
                        text=f"{start_sentence}{image_string} Answer the question using a single word, number, or short phrase. This animal is a",
                        images = process_img(x["image"]),
                        **common_kwargs,
                    )
                return tokenize_images_text
            else:
                def tokenize_images_text(x):
                    return hf_tokenizer(  # type: ignore
                        text=f"{start_sentence}{image_string} {x['text']}",
                        images=process_img(x["image"]),
                        **common_kwargs,
                    )

                return tokenize_images_text

        
        
        elif config.input == "imagenet-test": #! just a test type for try different things
                
                def tokenize_images_text(x):
                    return hf_tokenizer(  # type: ignore
                        text=f"{start_sentence}{image_string}{x['text']}",
                        images=[process_img(x["image"])],
                        **common_kwargs,
                    )
    
                return tokenize_images_text
        
        elif config.input == "imagenet-text_wn_definitions":

            def tokenize_text_wn_definitions(x):
                inputs = hf_tokenizer(  # type: ignore
                    text=start_sentence + x["definition"],
                    images=None,  # type: ignore
                    **common_kwargs,  # type: ignore
                )
                return inputs

            return tokenize_text_wn_definitions
        elif (
            config.input == "flickr-text"
        ):  # this means that we want to use the flickr dataset with the text only

            def tokenize_flickr_text(x):
                if isinstance(x["caption"][0], list):
                    text = [
                        {
                            start_sentence
                            + batch_text[0]
                            + batch_text[1]
                            + batch_text[2]
                            + batch_text[3]
                            + batch_text[4]
                        }
                        for batch_text in x["caption"]
                    ]
                else:
                    text = (
                        x["caption"][0]
                        + x["caption"][1]
                        + x["caption"][2]
                        + x["caption"][3]
                        + x["caption"][4]
                    )
                inputs = hf_tokenizer(  # type: ignore
                    text=text,
                    images=None,  # type: ignore
                    **common_kwargs,  # type: ignore
                )
                return inputs

            return tokenize_flickr_text
        elif (
            config.input == "flickr-image"
        ):  # this means that we want to use the flickr dataset with the images only

            def tokenize_flickr_image(x):
                inputs = hf_tokenizer(  # type: ignore
                    text=start_sentence + image_string,
                    images=x["image"],
                    **common_kwargs,  # type: ignore
                )
                return inputs

            return tokenize_flickr_image
        elif (
            config.input == "flickr-text->image"
        ):  # this means that we want to use the flickr dataset with input that contains first the text and then the image

            def tokenize_flickr_text_image(x):
                if isinstance(x["caption"][0], list):
                    text = [
                        f"{start_sentence+ batch_text[0] + batch_text[1] + batch_text[2] + batch_text[3] + batch_text[4]} {image_string}"
                        for batch_text in x["caption"]
                    ]
                    images = [batch_image for batch_image in x["image"]]
                else:
                    text = f'{start_sentence+x["caption"][0] + x["caption"][1] + x["caption"][2] + x["caption"][3] + x["caption"][4]} {image_string}'
                    images = process_img(x["image"])
                inputs = hf_tokenizer(  # type: ignore
                    text=text,
                    images=images,
                    **common_kwargs,
                )
                return inputs

            return tokenize_flickr_text_image
        elif (
            config.input == "flickr-image->text"
        ):  # this means that we want to use the flickr dataset with input that contains first the image and then the text

            def tokenize_flickr_image_text(x):
                if isinstance(x["caption"][0], list):
                    text = [
                        f"{start_sentence}{image_string} {batch_text[0] + batch_text[1] + batch_text[2] + batch_text[3] + batch_text[4]}"
                        for batch_text in x["caption"]
                    ]
                    images = [process_img(batch_image) for batch_image in x["image"]]
                else:
                    text = f'{start_sentence}{image_string} {x["caption"][0] + x["caption"][1] + x["caption"][2] + x["caption"][3] + x["caption"][4]}'
                    images = process_img(x["image"])
                inputs = hf_tokenizer(  # type: ignore
                    text=text,
                    images=images,
                    **common_kwargs,
                )
                return inputs

            return tokenize_flickr_image_text
        else:
            raise ValueError(
                f"Unsupported input type: got {config.input} while it should be one of ['imagenet-image', 'imagenet-text', 'imagenet-text_wn_definitions', 'flickr-text', 'flickr-image', 'flickr-text->image', 'flickr-image->text']"
            )


class CollateFunctions:
    """
    This class return the collate function. A collate function it's needed to process the dataset in batches.
    """

    @staticmethod
    def get_collate_fn(config: ExtractActivationConfig, hf_tokenizer):
        def unified_collate_fn(batch):
            max_length = max(item["input_ids"].size(1) for item in batch)

            input_ids = torch.stack(
                [
                    left_pad(
                        item["input_ids"].squeeze(0),
                        max_length,
                        hf_tokenizer.tokenizer.pad_token_id,
                    )
                    for item in batch
                ]
            )

            attention_mask = torch.stack(
                [
                    left_pad(item["attention_mask"].squeeze(0), max_length, 0)
                    for item in batch
                ]
            )

            input_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

            if "pixel_values" in batch[0]:
                pixel_values = torch.stack([item["pixel_values"] for item in batch])
                input_dict["pixel_values"] = pixel_values.to(
                    config.torch_dtype
                ).squeeze(1)
                # reshape the pixel_values tensor to have 3D if input_ids is 1D or 4D otherwise
                if (
                    len(input_ids.shape) == 1
                    and len(input_dict["pixel_values"].shape) >= 4
                ) or (
                    len(input_ids.shape) == 2
                    and len(input_dict["pixel_values"].shape) >= 5
                ):
                    input_dict["pixel_values"] = input_dict["pixel_values"].squeeze(0)

                # Collecting all other keys in each item into a list of dictionaries
            other_args = [
                {
                    k: v
                    for k, v in item.items()
                    if k not in {"input_ids", "attention_mask", "pixel_values"}
                }
                for item in batch
            ]

            return input_dict, other_args

        return unified_collate_fn


class TokenIndex:
    def __init__(self, model_name: str):
        if any(
            ["chameleon" in model_name, "pixtral" in model_name]
        ):
            if "chameleon" in model_name:
                model_type = "chameleon"
            elif "pixtral" in model_name:
                model_type = "pixtral"
        else:
            raise ValueError(
                f"Unsupported model {model_name}. Supported model types are ['pixtral', 'chameleon']"
            )
        self.model_type = model_type

    # @staticmethod
    # def get_token_index_chameleon(
    #     tokens: List[str],
    #     string_tokens: List[str],
    #     return_type: Literal["list", "int", "dict"] = "list",
    # ):
    #     # TODO: implement a more flexible way to get the token index

    #     if isinstance(string_tokens[0], list):
    #         raise ValueError(
    #             "This function is not implemented for batched inputs. Expected list of string get list of list"
    #         )

    #     max_len = len(string_tokens)

    #     token_dict = {
    #         "last": [-1],
    #         "last-2": [-2],
    #         "last-4": [-4],
    #         "labeled-imagenet-common": [-3],  # TODO!! change
    #         "image-33": [32],
    #         "last-image": [1025],
    #         "end-image": [1026],
    #         "all-text": [i for i in range(1026, max_len)],
    #         "all": [i for i in range(0, max_len)],
    #         "all-image": [i for i in range(1, 1026)],
            
    #     }

    #     tokens_positions = []

    #     for token in tokens:
    #         tokens_positions.extend(token_dict[token])
    #     if return_type == "int":
    #         if len(tokens_positions) > 1:
    #             raise ValueError(
    #                 "More than one token requested: return_type should be list, got int"
    #             )
    #         return tokens_positions[0]
    #     if return_type == "dict":
    #         return token_dict
    #     return tokens_positions

    def find_occurrences(self, lst: List[str], target:str) -> List[int]:
        r"""
        return a list of the occurrences of the target in the string
        """
        
        return [i for i, x in enumerate(lst) if x == target]
    
    def categorize_tokens(self, string_tokens: List[str]):
        """
        Categorize token in disjoin set of tokens:
            - image_start_tokens: list of the index of the start image token
            - image_end_tokens: list of the index of the end image token
            - image_tokens: list of the index of the image tokens
            - text_tokens: list of the index of the text tokens
            - special_tokens: list of the index of the special tokens
        """
        image_start_tokens = []
        image_end_tokens = []
        image_tokens = []
        last_line_image_tokens = []
        text_tokens = []
        special_tokens = []
        
        if self.model_type == "pixtral":
            start_image_token = "[IMG]"
            special = "[BREAK]"
            end_image_token = "[IMG_END]"
        elif self.model_type == "chameleon":
            start_image_token = "<racm3:break>"
            special = None
            end_image_token = "<eoss>"

        
        in_image_sequence = False
        
        for i, token in enumerate(string_tokens):
            
            #check for the start
            if token == start_image_token and not in_image_sequence:
                in_image_sequence = True
                image_start_tokens.append(i)
                
            # check for the end    
            elif in_image_sequence and token == end_image_token:
                in_image_sequence = False
                image_end_tokens.append(i)
                last_line_image_tokens.append(i-1)
                
            # cehck for special tokens
            elif in_image_sequence and special is not None and token == special:
                special_tokens.append(i)
            
            # check for image tokens
            elif in_image_sequence:
                image_tokens.append(i)
                
            # check for text tokens
            elif not in_image_sequence:
                text_tokens.append(i)
    
        return {
            "image_start": image_start_tokens,
            "image_end": image_end_tokens,
            "image": image_tokens,
            "last_line_image": last_line_image_tokens,
            "text": text_tokens,
            "special": special_tokens
        }
            

    def get_token_index(
        self,
        tokens: List[str],
        string_tokens: List[str],
        return_type: Literal["list", "int", "dict"] = "list",
    ):
        """
        Unified method to extract token indices based on the model type (pixtral ).

        Args:
            - tokens (List[str]): List of tokens to extract the activations.
            - string_tokens (List[str]): List of string tokens of the input.
            - return_type (Literal): Type of return, either "list", "int", or "dict".

        Returns:
            - Token positions based on the specified return_type.
        """
        token_indexes = self.categorize_tokens(string_tokens)

        tokens_positions = []
        
        token_dict = {
            "last": [-1],
            "last-2": [-2],
            "last-4": [-4],
            "last-image": token_indexes["last_line_image"],
            "end-image": token_indexes["image_end"],
            "all-text": token_indexes["text"],
            "all": [i for i in range(0, len(string_tokens))],
            "all-image": token_indexes["image"],
            "special": token_indexes["special"],
            "random-text": [random.choice(token_indexes["text"])],
            "random-image": [random.choice(token_indexes["image"])],
            "random-image-10": random.sample(token_indexes["image"], 10),
            "special-pixtral": [1052, 1051, 1038,  991, 1037, 1047,],# 1032,  925,  988, 1050, 1046,
                                #1034, 1048, 1040, 1027, 1023, 1022, 1049, 1033, 1041, 1026, 1055,
                                #1053, 1054, 1024,   33, 1056,   66, 1025,] #! hard coded and hard finded by Ema,
            "1025": [1025],
            "eols": token_indexes["image"][32::30]
        }
      
        for token in tokens:
            tokens_positions.extend(token_dict[token])
        
        if return_type == "int":
            if len(tokens_positions) > 1:
                raise ValueError(
                    "More than one token requested: return_type should be list, got int"
                )
            return tokens_positions[0]
        if return_type == "dict":
            return token_dict
        return tokens_positions



def _prepare_for_forward_pass(
    dataset, config, processor, num_proc, batch_size
) -> DataLoader:
    processor_function = PreprocessFunctions.get_preprocess_fn(
        config,
        processor,
    )
    collate_fn = CollateFunctions.get_collate_fn(config, processor)

    
    if DEBUG:
        dataset = dataset.select(range(100))
    
    if config.map_dataset_parallel_mode == "sequential":
        dataset = dataset.map(
            processor_function, batched=False, num_proc=1
        ).with_format("torch")
    elif config.map_dataset_parallel_mode == "parallel":
        dataset = dataset.map(
            processor_function, batched=False, num_proc=num_proc
        ).with_format("torch")
    elif config.map_dataset_parallel_mode == "multigpu":
        # get the number of gpus
        num_gpus = torch.cuda.device_count()
        raise NotImplementedError("Multigpu mode not implemented yet")
        
    else:
        raise ValueError(
            f"Unsupported map_parallel_type: got {config.map_dataset_parallel_mode} while it should be one of ['sequential', 'parallel', 'multigpu']"
        )

    return DataLoader(
        dataset,  # type: ignore
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,  # in case of memory error, set to True
        shuffle=False,
    )


class Extractor:
    """
    Extract activation given a dataset and a model
    How to add a new model:
    - add the model in the load_model method
    - add the model in the get_preprocess_and_collate_function method
    - add the model in the get_extractor_routine method

    How to add a new dataset:
    - add the dataset in the load_dataset method
    - add the dataset support in each preprocess and collate method

    NICE-TO-HAVE: This class is too big, and if we will add more datasets and models, it should be too large to navigate
          We may have to implement a class for each model, and then a factory class that returns the correct class
          However, now it's working so let's keep it as it is until we really need to change it


    """

    def __init__(self, config: ExtractActivationConfig):
        torch.set_grad_enabled(False)
        # if the logs folder does not exist, create it
        if not os.path.exists("logs"):
            os.makedirs("logs")
        self.logger = Logger(
            logname="extractor",
            level="info",
            log_file_path=f"logs/log_{config.id_num}.log",
        )
        self.config = config
        self.hf_model, self.hf_tokenizer, self.model_config = ModelFactory.load_model(
            config
        )

        self.first_device = next(self.hf_model.parameters()).device
        device_num = torch.cuda.device_count()
        self.logger.info(
            f"Model loaded in {device_num} devices. First device: {self.first_device}",
            std_out=True,
        )
        if config.dataset_dir is not None or config.dataset_hf_name is not None:
            self.dataset = DatasetFactory.load_dataset(config)
        else:
            self.logger.warning("Dataset not loaded, some method will not work!", std_out=True)
        self.logger.info("Extractor initialized")
        self.act_type_to_hook_name = {
            "resid_in": self.model_config.residual_stream_input_hook_name,
            "resid_out": self.model_config.residual_stream_hook_name,
            "resid_mid": self.model_config.intermediate_stream_hook_name,
            "attn_out": self.model_config.attn_out_hook_name,
            "attn_in": self.model_config.attn_in_hook_name,
            "values": self.model_config.attn_value_hook_name,
            # Add other act_types if needed
        }
        self.assert_config()
        
    def get_processor(self):
        return self.hf_tokenizer
    
    def get_text_tokenizer(self):
        return self.processor.tokenizer
        
    def eval(self):
        self.hf_model.eval()
        
    def device(self):
        return self.first_device
        
    def assert_config(self):
        ## Add some assert
        if self.config.attn_implementation != "eager":
            self.logger.warning(
                f"Got {self.config.attn_implementation} attention implementation. This implementation is weaky supported. Please use eager to guarantee the correct behavior"
            , std_out=True)
                

    def update(self, new_config):
        if new_config.model_name != self.config.model_name:
            self.logger.info("Model changed, reloading the model", std_out=True)
            self.clear()
            self.hf_model, self.hf_tokenizer, self.model_config = (
                ModelFactory.load_model(new_config)
            )
            self.first_device = next(self.hf_model.parameters()).device
            device_num = torch.cuda.device_count()
            self.logger.info(
                f"Model loaded in {device_num} devices. First device: {self.first_device}",
                std_out=True,
            )
        if (
            new_config.input != self.config.input
            or new_config.dataset_dir != self.config.dataset_dir
            or new_config.dataset_hf_name != self.config.dataset_hf_name
            or new_config.split != self.config.split
        ):
            self.logger.info("Dataset changed, reloading the dataset", std_out=True)
            self.dataset = DatasetFactory.load_dataset(new_config)

        if new_config.batch_size != self.config.batch_size:
            self.logger.info("Batch size changed, reloading the dataset", std_out=True)
            self.config.batch_size = new_config.batch_size
        self.config = new_config
        
        self.assert_config()

    def clear(self):
        # remove self.hf_model from memory
        del self.hf_model
        torch.cuda.empty_cache()
        self.logger.info("Model removed from memory", std_out=True)

    def prepare_for_forward_pass(self, dataset=None) -> DataLoader:
        if dataset is None:
            dataset = self.dataset
        return _prepare_for_forward_pass(
            dataset,
            self.config,
            self.hf_tokenizer,
            self.config.num_proc,
            self.config.batch_size,
        )



    
    def create_hooks(
        self,
        inputs,
        cache: Dict[str, torch.Tensor],
        string_tokens: List[str],   
        attn_heads: Optional[Union[list[dict], Literal["all"]]] = None,
        extract_attn_pattern: bool = False,  
        extract_attn_out: bool = False,
        extract_attn_in: bool = False,
        extract_avg_attn_pattern: bool = False,
        extract_avg_values_vectors_projected: bool = False,
        extract_resid_in: bool = False,
        extract_resid_out: bool = True,  # TODO: change to False (fix cascade)
        extract_values: bool = False,
        extract_resid_mid: bool = False,
        save_input_ids: bool = False,
        extract_head_out: bool = False,
        extract_values_vectors_projected: bool = False,
        extract_avg: bool = False,
        ablation_queries: Optional[pd.DataFrame | None] = None,
        patching_queries: Optional[pd.DataFrame | None] = None,
        batch_idx: Optional[int] = None,
        external_cache: Optional[Dict[str, torch.Tensor]] = None,      
    ):
        """TODO: Rewrite the docstring
        Unique routine to extract the activations of multiple models. It uses both a standard huggingface model and pyvene model, which is a wrapper around the huggingface model
        that allows to set sum hooks around the modules of the model. It supports the following args
        Args:
            - inputs: dictionary with the inputs of the model {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
            - attn_heads: list of dictionaries with the layer and head we want to extract something (e.g. attn pattern or values vectors...). If "all" is passed, it will extract all the heads of all the layers
                          it expect a list of dictionaries with the keys "layer" and "head": [{"layer": 0, "head": 0}, {"layer": 1, "head": 1}, ...] # NICE-TO-HAVE: add a assert to check the format
            - extract_attn_pattern: bool to extract the attention pattern (i.e. the attention matrix)
            - extract_avg_attn_pattern: bool to extract the average attention pattern. It will extract the average of the attention pattern of the heads passed in attn_heads. The average is saved in the external_cache
            - extract_avg_values_vectors: bool to extract the average values vectors. It will extract the average of the values vectors of the heads passed in attn_heads. The average is saved in the external_cache. The computation will be
                                         alpha_ij * ||V_j|| where alpha_ij is the attention pattern and V_j is the values vectors for each element of the batch. The average is computed for each element of the batch. It return a matrix of shape [batch, seq_len, seq_len]
            - extract_intermediate_states: bool to extract the intermediate states of the model (i.e. the hiddden rappresentation between the attention and the MLP)
            - save_input_ids: bool to save the input_ids in the cache
            - extract_head_out: bool to extract the output of the heads. It will extract the output of the heads projected by the final W_O projection.
            - extract_values_vectors: bool to extract the values vectors. It will extract the values vectors projected by the final W_O projection. If X_i is the residual stream of the i layer, it will return W_OV * X_i
            - move_to_cpu: bool to move the activations to the cpu before returning the cache. Sometimes it's useful to move the activations to the cpu to avoid to fill the gpu memory, while sometimes it's better to keep the activations on the gpu to avoid to move them back and forth
            - ablation_queries: dataframe with the ablation queries. The user can configure the ablation through a json file passed to extract_activations.py
            - patching_queries: dataframe with the patching queries.
            - freeze_ablation: if true, the attention weights will be frozen during the ablation.
            - external_cache: dictionary with the activations of the model. If passed, the activations will be saved in this dictionary. This is useful if we want to save average activations of multiple batches
            - idx_batch: index of the batch. It's useful to save the activations in the external_cache or perform mean computation

        Returns:
            - cache: dictionary with the activations of the model. The keys are the names of the activations and the values are the activations themselve
                cache = {
                    "resid_out_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
                    "resid_mid_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
                    "pattern_L0": tensor of shape [num_heads, batch, seq_len, seq_len] with the attention pattern of layer 0,
                    "patten_L0H0": tensor of shape [batch, seq_len, seq_len] with the attention pattern of layer 0 and head 0,
                    "input_ids": tensor of shape [batch, seq_len] with the input_ids,
                    "head_out_0": tensor of shape [batch, seq_len, hidden_size] with the output of the heads of layer 0
                    ...
                    }
        """
        # set the model family
        #

        if "pixtral" in self.config.model_name.lower():
            model_family = "pixtral"
        elif "chameleon" in self.config.model_name.lower():
            model_family = "chameleon"
        else:
            raise ValueError(
                "Model family not recognized. Supported families are 'pixtral' and 'chameleon', get {}".format(
                    self.config.model_name
                )
            )

        if extract_attn_pattern or extract_head_out or extract_values_vectors_projected:
            if (
                attn_heads is None
            ):  # attn_head must be passed if we want to extract the attention pattern or the output of the heads. If not, raise an error
                raise ValueError(
                    "attn_heads must be a list of dictionaries with the layer and head to extract the attention pattern or 'all' to extract all the heads of all the layers"
                )

        # add check for batch size > 1
        if (
            inputs["input_ids"].shape[0] > 1
        ):  # NICE-TO-HAVE: add support for batch size > 1
            raise ValueError("Batch size > 1 is not supported for Pixtral model")


        string_tokens = to_string_tokens(
            inputs["input_ids"].squeeze(), self.hf_tokenizer
        )

        token_index = TokenIndex(self.config.model_name).get_token_index(
            tokens=self.config.token, string_tokens=string_tokens
        )
        last_image_idxs = TokenIndex(self.config.model_name).get_token_index(
            tokens=["last-image"], string_tokens=string_tokens
        )
        end_image_idxs = TokenIndex(self.config.model_name).get_token_index(
            tokens=["end-image"], string_tokens=string_tokens
        )

        # define a dynamic factory hook. It takes a function and the corresponding kwargs and returns a function that pyvene can use. This is necessary to use partial() in the hook function
        # but still be consistent with the type of the function that pyvene expects. It's basically a custom partial function that retuns a function of type FuncType

        
        
        hooks = []
    
        if extract_resid_out:
            hooks += [
                {
                    "component": self.model_config.residual_stream_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_out_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]
        if extract_resid_in:
            hooks += [
                {
                    "component": self.model_config.residual_stream_input_hook_name.format(
                        i
                    ),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_in_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if save_input_ids:
            hooks += [
                {
                    "component": self.model_config.embed_tokens,
                    "intervention": partial(
                        embed_hook,
                        cache=cache,
                        cache_key="input_ids",
                    ),
                }
            ]

        if extract_values:
            hooks += [
                {
                    "component": self.model_config.attn_value_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"values_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extract_attn_in:
            hooks += [
                {
                    "component": self.model_config.attn_in_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"attn_in_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extract_attn_out:
            hooks += [
                {
                    "component": self.model_config.attn_out_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"attn_out_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extract_avg:
            # Define a hook that saves the activations of the residual stream 
            raise NotImplementedError("The hook for the average is not working with token_index as a list")

            hooks.extend(
                [
                    {
                        "component": self.model_config.residual_stream_hook_name.format(
                            i
                        ),
                        "intervention": partial(
                            avg_hook,
                            cache=cache,
                            cache_key="resid_avg_{}".format(i),
                            last_image_idx=last_image_idxs,
                            end_image_idx=end_image_idxs,
                        ),
                    }
                    for i in range(0, self.model_config.num_hidden_layers)
                ]
            )
        if extract_resid_mid:
            hooks += [
                {
                    "component": self.model_config.intermediate_stream_hook_name.format(
                        i
                    ),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_mid_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

            # if we want to extract the output of the heads

        # PATCHING
        if patching_queries:
            token_to_pos = partial(
                map_token_to_pos,
                _get_token_index=TokenIndex(self.config.model_name).get_token_index,
                string_tokens=string_tokens,
                hf_tokenizer=self.hf_tokenizer,
                inputs=inputs,
            )
            patching_queries = preprocess_patching_queries(
                patching_queries=patching_queries,
                map_token_to_pos=token_to_pos,
                model_config=self.model_config,
            )

            def make_patch_tokens_hook(patching_queries_subset):
                """
                Creates a hook function to patch the activations in the
                current forward pass.
                """

                def patch_tokens_hook(module, input, output):
                    if output is None:
                        if isinstance(input, tuple):
                            b = input[0]
                        else:
                            b = input
                    else:
                        if isinstance(output, tuple):
                            b = output[0]
                        else:
                            b = output
                    # Modify the tensor without affecting the computation graph
                    act_to_patch = b.detach().clone()
                    for pos, patch in zip(
                        patching_queries_subset["pos_token_to_patch"],
                        patching_queries_subset["patching_activations"],
                    ):
                        act_to_patch[0, pos, :] = patching_queries_subset[
                            "patching_activations"
                        ].values[0]
                        
                    if output is None:
                        if isinstance(input, tuple):
                            return (act_to_patch, *input[1:])
                        elif input is not None:
                            return act_to_patch
                    else:
                        if isinstance(output, tuple):
                            return (act_to_patch, *output[1:])
                        elif output is not None:
                            return act_to_patch
                    raise ValueError("No output or input found")

                return patch_tokens_hook

            # Group the patching queries by 'layer' and 'act_type'
            grouped_queries = patching_queries.groupby(["layer", "activation_type"])

            for (layer, act_type), group in grouped_queries:
                hook_name_template = self.act_type_to_hook_name.get(
                    act_type[:-3]
                )  # -3 because we remove {}
                if not hook_name_template:
                    raise ValueError(f"Unknown activation type: {act_type}")
                    # continue  # Skip unknown activation types

                hook_name = hook_name_template.format(layer)
                hook_function = partial(make_patch_tokens_hook(group))

                hooks.append(
                    {
                        "component": hook_name,
                        "intervention": hook_function,
                    }
                )

        if ablation_queries is not None:
            # TODO: debug and test the ablation. Check with ale
            token_to_pos = partial(
                map_token_to_pos,
                _get_token_index=TokenIndex(self.config.model_name).get_token_index,
                string_tokens=string_tokens,
                hf_tokenizer=self.hf_tokenizer,
                inputs=inputs,
            )
            if self.config.batch_size > 1:
                raise ValueError("Ablation is not supported with batch size > 1")    
            ablation_manager = AblationManager(
                model_config=self.model_config,
                token_to_pos=token_to_pos,
                inputs=inputs,
                model_family=model_family,
                model_attn_type=self.config.attn_implementation,
                ablation_queries=ablation_queries,
            )
            hooks.extend(ablation_manager.main())

        if extract_values_vectors_projected or extract_avg_values_vectors_projected:
            if attn_heads == "all":  # extract the output of all the heads
                hooks += [
                    {
                        "component": self.model_config.attn_value_hook_name.format(i),
                        "intervention": partial(
                            projected_value_vectors_head,
                            cache=cache,
                            layer=i,
                            num_attention_heads=self.model_config.num_attention_heads,
                            num_key_value_heads=self.model_config.num_key_value_heads,
                            hidden_size=self.model_config.hidden_size,
                            d_head=self.model_config.head_dim,
                            out_proj_weight=get_attribute_from_name(
                                self.hf_model,
                                f"{self.model_config.attn_out_proj_weight.format(i)}",
                            ),
                            out_proj_bias=get_attribute_from_name(
                                self.hf_model,
                                f"{self.model_config.attn_out_proj_bias.format(i)}",
                            ),
                            head=attn_heads,
                        ),
                    }
                    for i in range(0, self.model_config.num_hidden_layers)
                ]
            elif isinstance(attn_heads, list):
                for el in attn_heads:
                    head = el["head"]
                    layer = el["layer"]
                    hooks.append(
                        {
                            "component": self.model_config.attn_value_hook_name.format(
                                layer
                            ),
                            "intervention": partial(
                                projected_value_vectors_head,
                                cache=cache,
                                layer=layer,
                                num_attention_heads=self.model_config.num_attention_heads,
                                hidden_size=self.model_config.hidden_size,
                                out_proj_weight=self.hf_model.model.layers[
                                    layer
                                ].self_attn.o_proj.weight,  # (d_model, d_model)
                                out_proj_bias=self.hf_model.model.layers[
                                    layer
                                ].self_attn.o_proj.bias,  # (d_model)
                                head=head,
                            ),
                        }
                    )
        if extract_avg_attn_pattern:
            hooks += [
                {
                    "component": self.model_config.attn_matrix_hook_name.format(i),
                    "intervention": partial(
                        avg_attention_pattern_head,
                        layer=i,
                        attn_pattern_current_avg=external_cache,
                        batch_idx=batch_idx,
                        cache=cache,
                        extract_avg_value=extract_avg_values_vectors_projected,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]
        if extract_attn_pattern:
            if attn_heads == "all":
                hooks += [
                    {
                        "component": self.model_config.attn_matrix_hook_name.format(i),
                        "intervention": partial(
                            attention_pattern_head,
                            cache=cache,
                            layer=i,
                            head=attn_heads,
                        ),
                    }
                    for i in range(0, self.model_config.num_hidden_layers)
                ]
            else:
                hooks += [
                    {
                        "component": self.model_config.attn_matrix_hook_name.format(
                            el["layer"]
                        ),
                        "intervention": partial(
                            attention_pattern_head,
                            cache=cache,
                            layer=el["layer"],
                            head=el["head"],
                        ),
                    }
                    for el in attn_heads
                ]
        return hooks

        

    @profile
    def extractor_routine(
        self,
        inputs,
        # extract_residual_stream: bool = False,
        attn_heads: Optional[Union[list[dict], Literal["all"]]] = None,
        extract_attn_pattern: bool = False,
        extract_attn_out: bool = False,
        extract_attn_in: bool = False,
        extract_avg_attn_pattern: bool = False,
        extract_avg_values_vectors_projected: bool = False,
        extract_resid_in: bool = False,
        extract_resid_out: bool = True,  # TODO: change to False (fix cascade)
        extract_values: bool = False,
        extract_resid_mid: bool = False,
        save_input_ids: bool = False,
        extract_head_out: bool = False,
        extract_values_vectors_projected: bool = False,
        extract_avg: bool = False,
        ablation_queries: Optional[pd.DataFrame | None] = None,
        patching_queries: Optional[pd.DataFrame | None] = None,
        external_cache: Optional[Dict] = None,
        batch_idx: Optional[int] = None,
        move_to_cpu: bool = False,
    ):
        """TODO: Rewrite the docstring
        Unique routine to extract the activations of multiple models. It uses both a standard huggingface model and pyvene model, which is a wrapper around the huggingface model
        that allows to set sum hooks around the modules of the model. It supports the following args
        Args:
            - inputs: dictionary with the inputs of the model {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
            - attn_heads: list of dictionaries with the layer and head we want to extract something (e.g. attn pattern or values vectors...). If "all" is passed, it will extract all the heads of all the layers
                          it expect a list of dictionaries with the keys "layer" and "head": [{"layer": 0, "head": 0}, {"layer": 1, "head": 1}, ...] # NICE-TO-HAVE: add a assert to check the format
            - extract_attn_pattern: bool to extract the attention pattern (i.e. the attention matrix)
            - extract_avg_attn_pattern: bool to extract the average attention pattern. It will extract the average of the attention pattern of the heads passed in attn_heads. The average is saved in the external_cache
            - extract_avg_values_vectors: bool to extract the average values vectors. It will extract the average of the values vectors of the heads passed in attn_heads. The average is saved in the external_cache. The computation will be
                                         alpha_ij * ||V_j|| where alpha_ij is the attention pattern and V_j is the values vectors for each element of the batch. The average is computed for each element of the batch. It return a matrix of shape [batch, seq_len, seq_len]
            - extract_intermediate_states: bool to extract the intermediate states of the model (i.e. the hiddden rappresentation between the attention and the MLP)
            - save_input_ids: bool to save the input_ids in the cache
            - extract_head_out: bool to extract the output of the heads. It will extract the output of the heads projected by the final W_O projection.
            - extract_values_vectors: bool to extract the values vectors. It will extract the values vectors projected by the final W_O projection. If X_i is the residual stream of the i layer, it will return W_OV * X_i
            - move_to_cpu: bool to move the activations to the cpu before returning the cache. Sometimes it's useful to move the activations to the cpu to avoid to fill the gpu memory, while sometimes it's better to keep the activations on the gpu to avoid to move them back and forth
            - ablation_queries: dataframe with the ablation queries. The user can configure the ablation through a json file passed to extract_activations.py
            - patching_queries: dataframe with the patching queries.
            - freeze_ablation: if true, the attention weights will be frozen during the ablation.
            - external_cache: dictionary with the activations of the model. If passed, the activations will be saved in this dictionary. This is useful if we want to save average activations of multiple batches
            - idx_batch: index of the batch. It's useful to save the activations in the external_cache or perform mean computation

        Returns:
            - cache: dictionary with the activations of the model. The keys are the names of the activations and the values are the activations themselve
                cache = {
                    "resid_out_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
                    "resid_mid_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
                    "pattern_L0": tensor of shape [num_heads, batch, seq_len, seq_len] with the attention pattern of layer 0,
                    "patten_L0H0": tensor of shape [batch, seq_len, seq_len] with the attention pattern of layer 0 and head 0,
                    "input_ids": tensor of shape [batch, seq_len] with the input_ids,
                    "head_out_0": tensor of shape [batch, seq_len, hidden_size] with the output of the heads of layer 0
                    ...
                    }
        """
        cache = {}
        string_tokens = to_string_tokens(
            inputs["input_ids"].squeeze(), self.hf_tokenizer
        )
        # cache["token_extracted_type"] = self.config.token
        # cache["token_extracted_idx"] = torch.tensor(TokenIndex(self.config.model_name).get_token_index(
        #     tokens=self.config.token, string_tokens=string_tokens
        # ))
        
        hooks = self.create_hooks(  #TODO: add **kwargs
            inputs=inputs,
            cache=cache,
            string_tokens=string_tokens,
            attn_heads=attn_heads,
            extract_attn_pattern=extract_attn_pattern,
            extract_attn_out=extract_attn_out,
            extract_attn_in=extract_attn_in,
            extract_avg_attn_pattern=extract_avg_attn_pattern,
            extract_avg_values_vectors_projected=extract_avg_values_vectors_projected,
            extract_resid_in=extract_resid_in,
            extract_resid_out=extract_resid_out,
            extract_values=extract_values,
            extract_resid_mid=extract_resid_mid,
            save_input_ids=save_input_ids,
            extract_head_out=extract_head_out,
            extract_values_vectors_projected=extract_values_vectors_projected,
            extract_avg=extract_avg,
            ablation_queries=ablation_queries,
            patching_queries=patching_queries,
            batch_idx=batch_idx,
            external_cache=external_cache,
        )
        # define the pyvene model, i.e. a wrapper around the huggingface model that allows to set hooks around the modules of the model
        
        # log_memory_usage("Before creating the model")
        hook_handlers = self.set_hooks(hooks)
        # pv_model = pv.IntervenableModel(hooks, model=self.hf_model)        
        # log_memory_usage("After creating the model")
        
        # forward pass
        output = self.hf_model(
            **inputs,
            # output_original_output=True,
            # output_attentions=extract_attn_pattern,
        )

        # log_memory_usage("After forward pass")

        cache["logits"] = output.logits[:, -1]
        # since attention_patterns are returned in the output, we need to adapt to the cache structure
        if move_to_cpu:
            for key, value in cache.items():
                if extract_avg_values_vectors_projected:
                    # remove the values vectors from the cache
                    if "values" in key:
                        del cache[key]
                cache[key] = value.detach().cpu()
            if external_cache is not None:
                for key, value in external_cache.items():
                    external_cache[key] = value.detach().cpu()

        token_dict = TokenIndex(self.config.model_name).get_token_index(
            tokens=self.config.token, string_tokens=string_tokens, return_type="dict"
        )
        mapping_index = {}
        current_index = 0
        for token in self.config.token:
            mapping_index[token] = []
            for idx in range(len(token_dict[token])):
                mapping_index[token].append(current_index)
                current_index += 1
        cache["mapping_index"] = mapping_index
        
        self.remove_hooks(hook_handlers)
        

        return cache

    def __call__(self, *args: profile, **kwds: profile) -> profile:
        return self.extractor_routine(*args, **kwds)
    
    def get_module_from_string(self, component: str):
        return self.hf_model.retrieve_modules_from_names(component)
    
    
    def set_hooks(self, hooks: List[Dict[str, Any]]):
        # 1. Parsing the module path
        
        if len(hooks) == 0:
            return []
        
        hook_handlers = []
        for hook in hooks:
            component = hook["component"]
            hook_function = hook["intervention"]
            
            # get the last module string (.input or .output) and remove it from the component string
            last_module = component.split(".")[-1]
            # now remove the last module from the component string
            component = component[: -len(last_module) - 1]
            
            if last_module == "input":
                hook_handlers.append(
                    get_module_by_path(self.hf_model, component).register_forward_pre_hook(
                        partial(hook_function, output=None)
                    )
                )
            elif last_module == "output":
                
                hook_handlers.append(
                    get_module_by_path(self.hf_model, component).register_forward_hook(
                        hook_function
                    )
                )
        
        return hook_handlers
    
    def remove_hooks(self, hook_handlers):
        for hook_handler in hook_handlers:
            hook_handler.remove()
    
    def to(self, device):
        self.hf_model.to(device)
        self.first_device = device

    @torch.no_grad()
    def generate(
        self,
        inputs,
        generation_config: GenerationConfig = None,
        **kwargs,
    ):
        """
        Generate new tokens using the model and the inputs passed as argument 
        Args:
            - inputs: dictionary with the inputs of the model {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
            - generation_config: original hf dataclass with the generation configuration
            - kwargs: additional arguments to control hooks generation (i.e. ablation_queries, patching_queries)
        Returns:
            - output: dictionary with the output of the model
        """
        # Initialize cache for logits
        hooks = self.create_hooks(
            inputs=inputs,
            cache={},
            string_tokens=to_string_tokens(
                inputs["input_ids"].squeeze(), self.hf_tokenizer
            ),
            extract_resid_out=False,
            **kwargs,
        )
        hook_handlers = self.set_hooks(hooks)
        
        output = self.hf_model.generate(**inputs,
                                        generation_config=generation_config,
                                        output_scores=False)
        self.remove_hooks(hook_handlers)
        return output["sequences"] 
            
    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs,
    #     generation_config: GenerationConfig = None,
    #     **kwargs,
    # ):
    #     """
    #     Generate new tokens using the model and the inputs passed as argument 
    #     Args:
    #         - inputs: dictionary with the inputs of the model {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
    #         - generation_config: original hf dataclass with the generation configuration
    #         - kwargs: additional arguments to control hooks generation (i.e. ablation_queries, patching_queries)
    #     Returns:
    #         - output: dictionary with the output of the model
    #     """
    #     # Initialize cache for logits
    #     hooks = self.create_hooks(
    #         inputs=inputs,
    #         cache={},
    #         string_tokens=to_string_tokens(
    #             inputs["input_ids"].squeeze(), self.hf_tokenizer
    #         ),
    #         extract_resid_out=False,
    #         **kwargs,
    #     )
    #     hook_handlers = self.set_hooks(hooks)
    #     # pv_model = pv.IntervenableModel(hooks, model=self.hf_model)
    #     # _, output = pv_model(inputs, output_original_output=True)
    #     # _, output = pv_model.generate(inputs,
    #     #                            generation_config=generation_config)
    #     max_new_tokens = generation_config.max_new_tokens
    #     pad_token_id = self.hf_tokenizer.tokenizer.pad_token_id
    #     token_generated = 0
    #     new_token = None
    #     generated_sequence = []
    #     while True:
    #         if token_generated >= max_new_tokens:
    #             break
           
    #         output = self.hf_model(
    #             **inputs
    #         )
            
    #         logit = output.logits[:, -1]
    #         new_token = torch.argmax(logit, dim=-1)
    #         inputs["input_ids"] = torch.cat([inputs["input_ids"], new_token.unsqueeze(1)], dim=1)
    #         inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(new_token).unsqueeze(1)], dim=1)
    #         token_generated += 1
    #         generated_sequence.append(new_token.squeeze())
            
            
    #         if new_token == pad_token_id:
    #             break
        
    #     self.remove_hooks(hook_handlers)
                    
    #     return  inputs["input_ids"]

    @profile
    @torch.no_grad()
    def extract_cache(
        self,
        extract_resid_out: bool = True,  # TODO: change to False
        extract_resid_in: bool = False,
        extract_resid_mid: bool = False,
        extract_attn_pattern: bool = False,
        extract_head_out: bool = False,
        extract_avg: bool = False,
        extract_avg_pattern: bool = False,
        extract_values_vectors_projected: bool = False,
        extract_avg_values_vectors_projected: bool = False,
        patching_queries: Optional[list[dict] | None] = None,
        ablation_queries: Optional[list[dict] | None] = None,
        save_input_ids: bool = False,
        attn_heads: Optional[Union[list[dict], Literal["all"]]] = None,
        move_to_cpu: bool = True,
        post_cut_to_min_length: bool = False,
    ):
        """
        Use pyvene to extract a cache of the activations of the model
        Returns a dictionary with the activations of the model. The keys are the names of the activations and the values are the activations themselves

        Args:
            - extract_resid_out: bool to extract the activations of the residual stream of the model
            - extract_resid_in: bool to extract the activations of the residual stream of the model
            - extract_intermediate_states: bool to extract the intermediate states of the model (i.e. the hiddden rappresentation between the attention and the MLP)
            - extract_attn_pattern: bool to extract the attention pattern (i.e. the attention matrix)
            - extract_head_out: bool to extract the output of the heads. It will extract the output of the heads projected by the final W_O projection.
            - extract_avg: bool to extract the average of the activations of the residual stream of the model
            - extract_avg_pattern: bool to extract the average of the attention pattern. It will extract the average of the attention pattern of the heads
            - extract_values_vectors: bool to extract the values vectors. It will extract the values vectors projected by the final W_O projection. If X_i is the residual stream of the i layer, it will return W_OV * X_i
            - extract_avg_values_vectors: bool to extract the average values vectors. It will extract the average of the values vectors of the heads passed
            - patching_queries: list of dictionaries with the patching queries. The user can configure the patching through a json file passed to extract_activations.py
        Returns:
            - final_cache: dictionary with the activations of the model. The keys are the names of the activations and the values are the activations themselves

        >>> cache = extractor.extract_cache(extract_intermediate_states=True, extract_attn_pattern=True)
        >>> print(cache)
        >>> {
            "resid_out_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
            "resid_mid_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
            ...
            "pattern_L0": tensor of shape [num_heads, batch, seq_len, seq_len] with the attention pattern of layer 0,
            ...
            "input_ids": tensor of shape [batch, seq_len] with the input_ids,
        }
        """

        self.logger.info("Extracting cache", std_out=True)
        self.logger.info(
            f"------- Extract resid mid: {extract_resid_mid}",
            std_out=True,
        )
        self.logger.info(
            f"------- Extract attention pattern: {extract_attn_pattern}", std_out=True
        )
        if extract_avg_pattern:
            self.logger.warning(
                "Extracting the average pattern will work only if the img size is the same for all the images in the batch. This is NOT true for pixtral model", std_out=True
            )

        dataloader = self.prepare_for_forward_pass()

        # get the function to save in the cache the additional element from the batch sime
        batch_saver = DatasetFactory.get_additional_batch_saver(self.config)

        self.logger.info("Forward pass started", std_out=True)
        all_cache = []  # a list of dictoionaries, each dictionary contains the activations of the model for a batch (so a dict of tensors)
        attn_pattern = {}  # Initialize the dictionary to hold running averages

        example_dict = {}
        n_batches = 0  # Initialize batch counter
        
        if post_cut_to_min_length:
            self.logger.info("Cutting the input_ids to the min length of the dataset", std_out=True)
            # get the max length of the dataset
            
            def compute_length(batch):
                return batch[0]["input_ids"].shape[1]

            all_len = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_proc) as executor:
                futures = [executor.submit(compute_length, batch) for batch in tqdm(dataloader, desc="Computing min length:")]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Collecting results:"):
                    all_len.append(future.result())
                
            min_length = min(all_len)
            # min_length = min([batch[0]["input_ids"].shape[1] for batch in dataloader])
            
            # cut the input_ids to the min_length and the attention mask to min_length 
            truncated_dataloader = []
            for batch in dataloader:
                truncated_batch = {}
                for key, value in batch[0].items():
                    if key == "input_ids":
                        truncated_batch[key] = value[:, :min_length]
                    elif key == "attention_mask":
                        truncated_batch[key] = value[:, :min_length]
                    else:
                        truncated_batch[key] = value
                truncated_batch = (truncated_batch, batch[1])
                truncated_dataloader.append(truncated_batch)

            dataloader = truncated_dataloader

        for batch in tqdm(dataloader, desc="Extracting cache:"):
            # log_memory_usage("Extract cache - Before batch")
            tokens, others = batch
            inputs = {k: v.to(self.first_device) for k, v in tokens.items()}
            # log_memory_usage("Extract cache - Before extractor_routine")
            cache = self.extractor_routine(
                inputs,
                extract_resid_in=extract_resid_in,
                extract_resid_out=extract_resid_out,
                attn_heads=attn_heads,
                extract_attn_pattern=extract_attn_pattern,
                extract_avg_attn_pattern=extract_avg_pattern,
                extract_avg_values_vectors_projected=extract_avg_values_vectors_projected,
                extract_resid_mid=extract_resid_mid,
                extract_head_out=extract_head_out,
                extract_values_vectors_projected=extract_values_vectors_projected,
                save_input_ids=save_input_ids,
                ablation_queries=ablation_queries,
                patching_queries=patching_queries,
                extract_avg=extract_avg,
                external_cache=attn_pattern,
                batch_idx=n_batches,
            )
            # possible memory leak from here -___--------------->
            additional_dict = batch_saver(others)
            if additional_dict is not None:
                cache = {**cache, **additional_dict}

            if move_to_cpu:
                for key, value in cache.items():
                    if key not in [
                        "text",
                        "offset",
                        "label",
                        "synset",
                        "root_label",
                        "mapping_index",
                        "token_extracted_idx",
                        "token_extracted_type",
                    ]:
                        cache[key] = value.detach().cpu()

            # memory leadk to here <-----------------------
            # log_memory_usage("Extract cache - After extractor_routine")

            # GPUtil.showUtilization()
            # get the memory usage
            # mem_usg = GPUtil.getGPUs()[0].memoryUsed
            # self.logger.info(f"Memory usage: {mem_usg}", std_out=True)
            # if n_batches % 100 == 0:
            #     GPUtil.showUtilization()

            n_batches += 1  # Increment batch counter# Process and remove "pattern_" keys from cache
            # log_memory_usage("Extract cache - Before append")
            all_cache.append(cache)
            # log_memory_usage("Extract cache - After append")
            del cache
            inputs = {k: v.cpu() for k, v in inputs.items()}
            del inputs
            torch.cuda.empty_cache()
            # log_memory_usage("Extract cache - After empty cache")

        # move attn_pattern to the cpu
        if extract_avg_values_vectors_projected or extract_avg_pattern:
            for key, value in attn_pattern.items():
                if attn_pattern[key].dtype == torch.float32:
                    attn_pattern[key] = attn_pattern[key].to(self.config.torch_dtype)
                attn_pattern[key] = value.cpu()

        self.logger.info(
            "Forward pass finished - started to aggregate different batch", std_out=True
        )
        final_cache = aggregate_cache_efficient(all_cache)
        if extract_avg_pattern:
            final_cache = {
                **final_cache,
                **attn_pattern,
            }  # Add the running averages to the final cache

        # add the example_dict to the final_cache as a sub-dictionary
        final_cache["example_dict"] = example_dict
        self.logger.info("Aggregation finished", std_out=True)

        torch.cuda.empty_cache()
        return final_cache

    @torch.no_grad()
    def compute_patching(
        self,
        # counterfactual_dataset,
        config,
        patching_query=[
            {
                "patching_elem": "@end-image",
                "layers_to_patch": [1, 2, 3, 4],
                "activation_type": "resid_in_{}",
            }
        ],
        save_random_batch: Optional[str] = None,
        return_logit_diff: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Method to activation patching. It substitutes the activations of the model with the activations of the counterfactual dataset

        It will perform three forward passes:
        1. Forward pass on the base dataset to extract the activations of the model (cat)
        2. Forward pass on the target dataset to extract clean logits (dog) [to compare against the patched logits]
        3. Forward pass on the target dataset to patch (cat) into (dog) and extract the patched logits
        """
        self.logger.info("Computing patching", std_out=True)

        if (
            self.config.model_name == "facebook/chameleon-7b"
            or self.config.model_name == "facebook/chameleon-30b"
        ):
            TOKEN_MAP = CHAMELEON_TOKEN_MAP
        elif self.config.model_name == "mistral-community/pixtral-7b":
            TOKEN_MAP = PIXTRAL_TOKEN_MAP
        else:
            raise ValueError("Model not recognized")

        # self.dataset is a tuple with the base dataset and the target dataset
        base_dataset, target_dataset = self.dataset

        base_dataloader = self.prepare_for_forward_pass(base_dataset)
        target_dataloader = self.prepare_for_forward_pass(target_dataset)
        self.logger.info(
            "Patching {} into {}".format(
                base_dataset[0]["root_label"], target_dataset[0]["root_label"]
            ),
        )
        self.logger.info("Forward pass started", std_out=True)
        self.logger.info(
            f"Patching elements: {[q['patching_elem'] for q in patching_query]} at {[query['activation_type'][:-3] for query in patching_query]}",
            std_out=True,
        )

        # get a random number in the range of the dataset to save a random batch
        all_cache = []
        # for each batch in the dataset
        for index, (batch, target_batch) in tqdm(
            enumerate(zip(base_dataloader, target_dataloader)),
            desc="Computing patching on the dataset:",
            total=len(base_dataloader),
        ):
            # if index > 5:
            #     break
            # standard forward pass
            # empty the cache to avoid to fill the gpu memory with garbage
            torch.cuda.empty_cache()
            tokens = batch[
                0
            ]  # here we have "input_ids", "attention_mask" and "pixel_values"
            others = batch[1]  # additional elements of the batch
            inputs = {k: v.to(self.first_device) for k, v in tokens.items()}

            # set the right arguments for extract the patching activations
            activ_type = [query["activation_type"][:-3] for query in patching_query]

            args = {
                "extract_resid_out": True,
            }

            if "resid_in" in activ_type:
                args["extract_resid_in"] = True
            if "resid_out" in activ_type:
                args["extract_resid_out"] = True
            if "resid_mid" in activ_type:
                args["extract_intermediate_states"] = True
            if "attn_in" in activ_type:
                args["extract_attn_in"] = True
            if "attn_out" in activ_type:
                args["extract_attn_out"] = True
            if "values" in activ_type:
                args["extract_values"] = True
            # other cases

            # first forward pass to extract the base activations
            base_cache = self.extractor_routine(
                inputs,
                **args,
            )

            # extract the target activations
            target_tokens = target_batch[0]
            target_inputs = {
                k: v.to(self.first_device) for k, v in target_tokens.items()
            }

            # second forward pass to extract the clean logits
            target_clean_cache = self.extractor_routine(
                target_inputs,
                extract_resid_out=False,
                # move_to_cpu=True,
            )

            # add to the patching query the activations from the base dataset
            for query in patching_query:
                query["patching_activations"] = base_cache

                if "all" in config.token:
                    query["base_activation_index"] = None
                else:
                    assert (
                        query["patching_elem"].split("@")[1] in config.token
                    ), f"config.token: {config.token} and query['patching_elem']: {query['patching_elem']} are not compatible"
                    query["base_activation_index"] = base_cache["mapping_index"][
                        query["patching_elem"].split("@")[1]
                    ]

            # third forward pass to patch the activations
            target_patched_cache = self.extractor_routine(
                target_inputs,
                patching_queries=patching_query,
                # move_to_cpu=True,
                **kwargs,
            )

            if return_logit_diff:
                self.logger.info("Computing logit difference", std_out=True)
                # get the target tokens (" cat" and " dog")
                base_targets = TOKEN_MAP[base_dataset[index]["root_label"]]
                target_targets = TOKEN_MAP[target_dataset[index]["root_label"]]

                # compute the logit difference
                result_diff = logit_diff(
                    base_label_tokens=[s[1] for s in base_targets],
                    target_label_tokens=[c[1] for c in target_targets],
                    target_clean_logits=target_clean_cache["logits"],
                    target_patched_logits=target_patched_cache["logits"],
                )
                target_patched_cache["logit_diff_variation"] = result_diff[
                    "diff_variation"
                ]
                target_patched_cache["logit_diff_in_clean"] = result_diff[
                    "diff_in_clean"
                ]
                target_patched_cache["logit_diff_in_patched"] = result_diff[
                    "diff_in_patched"
                ]

            target_patched_cache["root_label"] = [
                (base_dataset[index]["root_label"], target_dataset[index]["root_label"])
            ]
            # compute the KL divergence
            result_kl = kl_divergence_diff(
                base_logits=base_cache["logits"],
                target_clean_logits=target_clean_cache["logits"],
                target_patched_logits=target_patched_cache["logits"],
            )
            for key, value in result_kl.items():
                target_patched_cache[key] = value

            target_patched_cache["base_logits"] = base_cache["logits"]
            target_patched_cache["target_clean_logits"] = target_clean_cache["logits"]
            # rename logits to target_patched_logits
            target_patched_cache["target_patched_logits"] = target_patched_cache[
                "logits"
            ]
            del target_patched_cache["logits"]

            # move to cpu
            for key, value in target_patched_cache.items():
                if key not in [
                    "text",
                    "offset",
                    "label",
                    "synset",
                    "root_label",
                    "mapping_index",
                    "token_extracted_type"
                ]:
                    target_patched_cache[key] = value.detach().cpu()

            all_cache.append(target_patched_cache)
            # self.logger.info(f"Logit difference: {result_diff}", std_out=True)
            # self.logger.info(f"KL divergence: {result_kl}", std_out=True)

        self.logger.info(
            "Forward pass finished - started to aggregate different batch", std_out=True
        )
        final_cache = aggregate_cache_efficient(all_cache)

        self.logger.info("Aggregation finished", std_out=True)
        return final_cache

    @profile
    def compute_metrics(
        self,
        metrics_to_compute: List[Literal["pattern_density", "value_norms"]],
        order: Literal["image->text", "text->image"],
        separate_special_tokens: Literal["last_token_modality", "all", "none"],
        attn_heads: Optional[Union[list[dict], Literal["all"]]],
        expand_head: bool = False,
        extract_avg: bool = False,
        save_random_batch: Optional[str] = None,
        filter_low_values: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Pipiline to compute metrics on the activations on the fly. This is useful to avoid to save the activations on the disk
        In the future we could add more metrics if we need and we could add the possibility to save the metrics on the disk.
        If/when we will have a lot of metrics to compute, we should introduce a metricConfig class to handle the arguments of
        the metrics, or just remove the argoments and use **kwargs. For now we have only the attention pattern metrics we keep
        the arguments in the method signature.

        Args:
            - metrics_to_compute: list of metrics to compute. The possible metrics are "pattern_density" and "value_norms"
            - order: the order of the input. It can be "image->text" or "text->image"
            - separate_special_tokens: [WARNING!! NOT USED FOR NOW 06/08] how to separate the special tokens. It can be "last_token_modality", "all" or "none"
            - attn_heads: list of dictionaries with the layer and head we want to extract something (e.g. attn pattern or values vectors...). If "all" is passed, it will extract all the heads of all the layers
            - expand_head: bool to expand the head dimension when extracting the values vectors and the attention pattern. If true, in the cache we will have a key for each head, like "value_L0H0", "value_L0H1", ...
            - save_random_batch: path where to save a random batch of the dataset. It's useful to debug the code and to check the activations of the model
            - filter_low_values: bool to filter the low values of the attention pattern. It's useful to remove the noise from the attention pattern

        Returns:
            - metrics: a dictionary where the keys are the heads and the values are dictionaries with the metrics. Each metric is a numpy array of shape (len(dataset))

        >>> extractor = Extractor(ExtractActivationConfig(model_name="chameleon-7b", input="image", dataset_dir="data", batch_size=8, device_map="auto", torch_dtype=torch.float16, token="last"))
        >>> metrics = extractor.compute_metrics(order="image->text", separate_special_tokens="last_token_modality", extract_attn_pattern="all")
        >>> print(metrics)
        {
            "pattern_L0H0": {
                "density": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                "col_density": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                ...
                },
            "pattern_L0H1": {
                "density": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                "col_density": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                ...
                },
        ...
        }

        """
        if expand_head:
            self.logger.warning(
                "Expand head is set to True. This could slow down the computation and BREAK THE CODE, since this option is not fully tested! Be aware of this and keep an eye on the results",
                std_out=True,
            )

        supported_model_and_metric = {
            "chameleon": ["pattern_density", "value_norms"],
            "pixtral": [],
        }

        if "chameleon" in self.config.model_name:
            family_name = "chameleon"
        elif "pixtral" in self.config.model_name:
            family_name = "pixtral"
        else:
            raise ValueError(
                "The model {} is not supported".format(self.config.model_name)
            )

        if not all(
            [m in supported_model_and_metric[family_name] for m in metrics_to_compute]
        ):
            raise ValueError(
                f"The model {family_name} does not support the metrics {metrics_to_compute}. Supported metrics are {supported_model_and_metric[family_name]}"
            )

        self.logger.info("Computing metrics", std_out=True)
        dataloader = self.prepare_for_forward_pass()

        # instance the class that compute the metrics
        metric_computer = AttentionHeadsMetricComputer(
            batch_dim=self.config.batch_size,
            multimodal_processor=self.hf_tokenizer,  # type: ignore
        )

        self.logger.info("Forward pass started", std_out=True)

        metrics = []
        # get a random number in the range of the dataset to save a random batch
        random_index = randint(0, len(dataloader))

        with torch.no_grad():  # we don't need to compute the gradients
            # for each batch in the dataset
            for index, batch in tqdm(
                enumerate(dataloader),
                desc="Computing patching on the dataset:",
                total=len(dataloader),
            ):
                # empty the cache to avoid to fill the gpu memory with garbage
                torch.cuda.empty_cache()
                tokens = batch[
                    0
                ]  # here we have "input_ids", "attention_mask" and "pixel_values"
                others = batch[1]  # additional elements of the batch
                inputs = {k: v.to(self.config.device_map) for k, v in tokens.items()}

                # extract the activations of the model
                cache = self.extractor_routine(
                    inputs,
                    attn_heads=attn_heads,
                    extract_attn_pattern=True,
                    extract_intermediate_states=False,
                    save_input_ids=True,
                    expand_head=expand_head,
                    extract_avg=extract_avg,
                    extract_values_vectors=True
                    if "value_norms" in metrics_to_compute
                    else False,
                    move_to_cpu=False,  # we keep the activations on the gpu to perform the metrics computation
                )

                # save a random batch
                if index == random_index and save_random_batch is not None:
                    # save a random batch
                    torch.save(cache, save_random_batch)

                # compute the metrics
                if "pattern_density" in metrics_to_compute:
                    pattern_metric = metric_computer.block_density(
                        cache,
                        input_ids=cache["input_ids"],
                        order=order,
                        separate_special_tokens=separate_special_tokens,
                        filter_low_values=filter_low_values,
                    )

                if "value_norms" in metrics_to_compute:
                    value_metric = metric_computer.value_norms(
                        cache,
                        input_ids=cache["input_ids"],
                        order=order,
                        separate_special_tokens=separate_special_tokens,
                    )

                # merge the metrics
                if (
                    "pattern_density" in metrics_to_compute
                    and "value_norms" in metrics_to_compute
                ):
                    metric = {**pattern_metric, **value_metric}
                else:
                    metric = (
                        pattern_metric
                        if "pattern_density" in metrics_to_compute
                        else value_metric
                    )
                    metric = (
                        value_metric
                        if "value_norms" in metrics_to_compute
                        else pattern_metric
                    )

                new_metric = {}

                # merge the metrics in a more readable way
                if not expand_head:
                    # NICE-TO-HAVE: this is a bit ugly, we should refactor this part of the code. Maybe we should create ad hoc classes to handle the metrics!
                    #               now it's a bit difficult to understand what's happening here. However, it's working so let's keep it as it is until we really need to change it (i.e. when we will have more metrics to compute)
                    for layer_key, metric_per_layers in metric.items():
                        for i in range(len(metric_per_layers)):
                            for metric_name, metric_values in metric_per_layers[
                                i
                            ].items():
                                for head in range(metric_values.shape[0]):
                                    layer = layer_key.split("_")[1]
                                    metric_type = (
                                        "pattern"
                                        if layer_key.startswith("pattern")
                                        else "value"
                                    )
                                    key = f"{metric_type}_{layer}H{head}"

                                    if key not in new_metric:
                                        new_metric[key] = {}

                                    if metric_name not in new_metric[key]:
                                        new_metric[key][metric_name] = []
                                    new_metric[key][metric_name].append(
                                        metric_values[head]
                                    )
                metrics.append(new_metric)
                GPUtil.showUtilization()

        # metrics has a very nested structure. Indeed it's a list of dictionaries, each dictionary contains, for each heads, a list of dictionaries, one for each element of the batch
        # this very horrible structure is necessary to have the batch support in the metric computer and to avoid aggregating the metrics in the metric computer itself (this is a design choice)
        # However, now we have to aggregate the metrics in a more readable way
        metrics = aggregate_metrics(metrics)

        # now metrics is just a dictionary where the keys are the heads and the values are dictionaries with the metrics. Each metric is a numpy array of shape (len(dataset))

        return metrics
