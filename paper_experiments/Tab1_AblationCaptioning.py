#!/usr/bin/env python
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import warnings
import requests
from PIL import Image
from io import BytesIO
import json
from transformers import GenerationConfig
import torch
import numpy as np
import tqdm
import argparse
from src.common.emu_utils import preprocess_conversation
from src.activations import ExtractActivationConfig, Extractor
from utils import prefix_allowed_tokens_fn
from evaluation_task.caption.eval.cider_metrics.eval_json import Cider
from dotenv import load_dotenv
from rich import print
from datasets import load_dataset

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv("../.env")
load_dotenv(".env")

model_configs = {
    "facebook/chameleon-30b": {
        "image_string": "<image>",
        "start_sentence": "",
        "attn_implementation": "eager",
        "use_cache": True,
        "window_of_ablation": None,
    },
    "facebook/chameleon-7b": {
        "image_string": "<image>",
        "start_sentence": "",
        "attn_implementation": "eager",
        "use_cache": True,
        "window_of_ablation": [24],
    },
    "mistral-community/pixtral-12b": {
        "image_string": "[IMG]",
        "start_sentence": "<s>",
        "attn_implementation": "eager",
        "use_cache": True,
        "window_of_ablation": [0, 10, 20],
    },
    "deepseek-ai/Janus-1.3B": {
        "image_string": "<image_placeholder>",
        "start_sentence": "",
        "start_image_token": "<begin_of_image>",
        "end_image_token": "<end_of_image>",
        "attn_implementation": "eager",
        "use_cache": True,
        "window_of_ablation": None,
    },
    "Emu3-Gen-hf": {
        "image_string": "<image>",
        "start_sentence": "<|extra 203|>",
        "end_of_frame": "<|extra 201|>",
        "attn_implementation": "eager",
        "use_cache": True,
        "window_of_ablation": [0],
    },
    "llava-onevision-7b": {
        "image_string": "<|im_start|><image><|im_end|>",
        "start_sentence": "<s>",
        "attn_implementation": "eager",
        "use_cache": True,
        "window_of_ablation": None,
    },
    "vila-u": {
        "image_string": "<image>",
        "start_sentence": "",
        "attn_implementation": "eager",
        "use_cache": True,
        "window_of_ablation": None,
    },
}


SYSTEM_PROMPT_VILA = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="mistral-community/pixtral-12b, \
              facebook/chameleon-7b, \
              facebook/chameleon-30b, \
              deepseek-ai/Janus-1.3b, \
              Emu3-Gen-hf \
              vila-u",
        default="facebook/chameleon-7b",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, help="flickr, coco", default="flickr"
    )
    parser.add_argument(
        "-n", "--nsample", type=int, help="n of data to sample", default=2000
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="outdir",
        default=os.path.join(
            os.path.dirname(__file__), ".data/Tab1_AblationCaptioning/"
        ),
    )
    parser.add_argument("-a", "--ablation", type=str, default=None)
    parser.add_argument("-at", "-a-type", "--ablation-type", type=str, default=None)

    args = parser.parse_args()
    print(
        f"MODEL: {args.model} \
          \nDATASET: {args.dataset}\
          \nNSAMPLE: {args.nsample}\
          \nOUTDIR: {args.outdir}\
          \nABLATION: {args.ablation}\
          \nABLATION_TYPE: {args.ablation_type}"
    )

    if args.ablation:
        args.ablation_type = "std"

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


    # Get configuration for the selected model
    if args.model in model_configs:
        config_dict = model_configs[args.model]
        model_name = args.model

        # Apply configuration
        image_string = config_dict["image_string"]
        start_sentence = config_dict["start_sentence"]
        attn_implementation = config_dict["attn_implementation"]
        ablation_type = args.ablation_type
        use_cache = config_dict["use_cache"]

        # Set window_of_ablation if specified in the config
        if config_dict["window_of_ablation"] is not None:
            window_of_ablation = config_dict["window_of_ablation"]

        # Handle any special tokens or settings that exist for specific models
        # if "start_image_token" in config_dict:
        #     start_image_token = config_dict["start_image_token"]
        # if "end_image_token" in config_dict:
        #     end_image_token = config_dict["end_image_token"]
        # if "end_of_frame" in config_dict:
        #     end_of_frame = config_dict["end_of_frame"]
        # if "ablation_type" in config_dict:
        #     ablation_type = config_dict["ablation_type"]
    else:
        print(f"Model {args.model} not supported")
        exit(1)

    if args.dataset == "flickr":
        dataset = load_dataset(
            "nlphuji/flickr30k", trust_remote_code=True, split="test"
        )
    elif args.dataset == "coco":
        dataset = load_dataset(
            "yerevann/coco-karpathy", trust_remote_code=True, split="test"
        )
    else:
        print("Dataset not supported")
        exit()

    config = ExtractActivationConfig(
        model_name=model_name,
        device_map="balanced",
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        map_dataset_parallel_mode="custom",
        resize_image=[256, 256],
    )

    model = Extractor(config)
    num_heads = model.model_config.num_attention_heads
    num_layers = model.model_config.num_hidden_layers
    processor = model.get_processor()

    if "Emu3" in args.model:
        from transformers import Emu3Processor, Emu3ImageProcessor

        processor = Emu3Processor.from_pretrained(
            "BAAI/Emu3-Gen-hf",
            torch_dtype=torch.bfloat16,
            # device_map="balanced",
        )
        image_processor = Emu3ImageProcessor(
            min_pixels=256 * 256,
            max_pixels=256 * 256,
            torch_dtype=torch.bfloat16,
            # device_map="balanced",
        )

    idxs = [i for i in range(args.nsample)]

    if args.ablation == "@random-image-10":
        n_iterations = 3

    elif (
        args.ablation == "@last-image"
        or args.ablation == "@end-image"
        or args.ablation == "@all-image"
        or args.ablation == "@end-image-emu"
        or args.ablation == "@last-image-2"
    ):
        n_iterations = 1

    elif args.ablation == "@img-text":
        n_iterations = 1
        ablation_type = "block-img-txt"

    else:
        print("#### ABLATION NOT RECOGNIZED PROCEDING WITHOUT ABLATION ####")
        args.ablation = None
        n_iterations = 1

    elem_to_ablate = args.ablation

    results = []
    ev = {}
    ev["pycocoeval"] = []
    ev["original"] = []

    for i in range(n_iterations):
        print(
            f"####################### Iteration {i} ##################################"
        )
        with torch.no_grad():
            for p_idx in tqdm.tqdm(idxs):
                p = dataset[p_idx]
                if args.dataset == "flickr":
                    img = p["image"]
                    caption = p["caption"]
                    img_id = p["img_id"]
                else:
                    img = p["url"]
                    caption = p["sentences"]
                    img_id = str(p["imgid"])
                    response = requests.get(img)
                    img = Image.open(BytesIO(response.content))

                # create prompt
                if "chameleon" in args.model:
                    completion = True
                    prompt = f"{image_string}Provide a one-sentence caption for the provided image."
                    # prompt = f'{image_string} Provide a one-sentence caption for the provided image:'
                elif "pixtral" in args.model:
                    prompt = f'[INST]"{start_sentence}{image_string} Provide a one-sentence caption for the provided image."[/INST]'
                elif "Janus" in args.model:
                    prompt = f"{image_string}USER:Provide a one-sentence caption for the provided image. ASSISTANT:"
                    
                elif "vila-u" in args.model:
                    prompt = f"{SYSTEM_PROMPT_VILA}USER:{image_string}\nProvide a one-sentence caption for the provided image. ASSISTANT:"
                    # prompt = f"A chat between a curious user and an artificial intelligence assistant.\
                    #           The assistant gives helpful, detailed, and polite answers to the user's questions. \
                    #           Human:{image_string} Provide a one-sentence caption for the provided image.###Assistant:"

                elif args.model == "mistral-community/pixtral-12b":
                    prompt = f'[INST]"{start_sentence}{image_string} Provide a one-sentence caption for the provided image."[/INST]'

                elif args.model == "Emu3-Stage1":
                    prompt = f"USER:{image_string} Provide a one-sentence caption for the provided image. ASSISTANT:"

                elif args.model == "BAAI/Emu2":
                    prompt = f"[USER]:{image_string} Provide a one-sentence caption for the provided image.[ASSISTANT]:"
                elif args.model == "janus":
                    prompt = f'"{image_string}USER:Provide a one-sentence caption for the provided image. \nASSISTANT:"'

                elif "llava-onevision" in args.model:
                    prompt = f"{image_string}USER: Provide a one-sentence caption for the provided image. ASSISTANT:"

                elif "Emu3-Gen" in args.model:
                    prompt = (
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {
                                    "type": "text",
                                    "text": "\nProvide a one-sentence caption for the provided image",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": ""}],
                        },
                    )

                # create inputs
                if "chameleon" in args.model:
                    inputs = processor(
                        images=[img],
                        text=prompt,
                        return_tensors="pt",
                        return_for_text_completion=completion,
                    ).to(model.device())
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                elif args.model == "BAAI/Emu2":
                    inputs = processor(
                        text=prompt, images=img, return_tensors="pt", padding=False
                    )
                    inputs["attention_mask"] = inputs["attention_mask"].to(
                        torch.bfloat16
                    )

                elif args.model == "Emu3-Stage1":
                    inputs = processor(
                        text=prompt, images=img, return_tensors="pt", mode="U"
                    ).to(model.device())
                    inputs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    }

                elif args.model == "Emu3-Gen-hf":
                    tokenizer = processor.tokenizer
                    text_ = processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=False,
                        tokenize=False,
                        return_dict=False,
                    )
                    # print(text_)
                    # text_ = "<image>\nDescribe this picture briefly"
                    text_ = "USER:<image>\nDescribe this picture briefly ASSISTANT:"
                    image_features = image_processor(img, return_tensors="pt")
                    text = preprocess_conversation(
                        text=[text_], image_features=image_features
                    )
                    inputs = tokenizer(text, return_tensors="pt")
                    inputs.update(**image_features)

                    image_sizes = inputs["image_sizes"]
                    HEIGHT, WIDTH = image_sizes[0]
                    VISUAL_TOKENS = model.hf_model.vocabulary_mapping.image_tokens

                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device())
                            if k != "input_ids":
                                inputs[k] = inputs[k].to(torch.bfloat16)

                    # if args.model == "Emu3-Gen":
                    #     inputs = processor(
                    #         images=img, text=prompt, return_tensors="pt", padding=False, mode="U"
                    #     ).to(model.device())

                elif args.model == "mistral-community/pixtral-12b":
                    inputs = processor(
                        images=img, text=prompt, return_tensors="pt", padding=False
                    ).to(model.device())
                    inputs["pixel_values"] = inputs["pixel_values"][0]

                elif "llava-onevision" in args.model:
                    inputs = processor(
                        images=[img], text=prompt, return_tensors="pt", padding=False
                    ).to(model.device())
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                else:
                    inputs = processor(text=prompt, images=[img], return_tensors="pt")
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device())
                            inputs["pixel_values"] = inputs["pixel_values"].to(
                                torch.bfloat16
                            )

                generation_config = GenerationConfig(
                    max_new_tokens=60,
                    return_dict_in_generate=True,
                    do_sample=False,
                    low_cpu_mem_usage=True,
                    use_cache=use_cache,
                )
                if args.model == "vila-u":
                    generation_config = GenerationConfig(
                        bos_token_id=1,
                        do_sample=True,
                        eos_token_id=2,
                        max_length=60,
                        pad_token_id=0,
                        temperature=0.9,
                        top_p=0.6,
                        use_cache=use_cache,
                        return_dict_in_generate=True,
                        low_cpu_mem_usage=True,
                    )
                # other_args = {} if args.model != "Emu3-Gen" else {"prefix_allowed_tokens_fn":prefix_allowed_tokens_fn,}
                if args.ablation or args.ablation_type:
                    generated_ids = model.generate(
                        inputs=inputs,
                        generation_config=generation_config,
                        ablation_queries=[
                            {
                                "type": ablation_type,
                                "elem-to-ablate": elem_to_ablate,
                                "head-layer-couple": [
                                    [i, j]
                                    for i in range(num_heads)
                                    for j in range(num_layers)
                                ],
                            }
                        ],
                        # **other_args
                    )
                else:
                    generated_ids = model.generate(
                        inputs=inputs,
                        generation_config=generation_config,
                        # **other_args
                    )

                # batch decode
                if args.model == "mistral-community/pixtral-12b":
                    full_res = processor.batch_decode(
                        generated_ids,
                        n=1,
                        best_of=1,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        repetition_penalty=1.0,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        min_p=0.0,
                        seed=None,
                        use_beam_search=False,
                        length_penalty=1.0,
                        early_stopping=False,
                        stop=[],
                        stop_token_ids=[],
                        include_stop_str_in_output=False,
                        ignore_eos=False,
                        max_tokens=10,
                        min_tokens=0,
                        logprobs=None,
                        prompt_logprobs=None,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=True,
                        truncate_prompt_tokens=None,
                    )[0]

                elif args.model == "deepseek-ai/Janus-1.3B":
                    full_res = processor.tokenizer.decode(
                        generated_ids[0][:-1].cpu().tolist(), skip_special_token=True
                    )

                elif args.model == "vila-u":
                    full_res = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                else:
                    full_res = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]

                results.append(
                    {"result": full_res, "image_id": img_id, "caption": caption}
                )

            with open(
                f"{args.outdir}/captioning_{args.model.replace('/', '-')}_{args.dataset}_{args.nsample}_{args.ablation}_{args.ablation_type}.json",
                "w",
            ) as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            ground_truth = {i: d["caption"] for i, d in enumerate(results)}
            response = {}
            if args.model in [
                "mistral-community/pixtral-12b",
                "facebook/chameleon-7b",
                "facebook/chameleon-30b",
            ]:
                for i, d in enumerate(results):
                    response[i] = [d["result"][d["result"].find("image.") + 6 :]]
            elif args.model in [
                "deepseek-ai/Janus-1.3B",
                "Emu3-Gen-hf",
                "Emu3-Stage1",
                "BAAI/Emu2",
            ]:
                for i, d in enumerate(results):
                    response[i] = [d["result"][d["result"].find("ASSISTANT:") + 11 :]]
            else:
                for i, d in enumerate(results):
                    response[i] = [d["result"]]

            score, scores = Cider(backend="pycocoeval").compute_score(
                ground_truth, response
            )
            print("Cider score pycocoeval =", score)
            ev["pycocoeval"].append(score)

            score, scores = Cider(backend="original").compute_score(
                ground_truth, response
            )
            print("Cider score original =", score)
            ev["original"].append(score)

        ev["pycocoeval_std"] = np.std(ev["pycocoeval"])
        ev["original_std"] = np.std(ev["original"])
        ev["pycocoeval"] = np.mean(ev["pycocoeval"])
        ev["original"] = np.mean(ev["original"])
        # RESULTS[window] = ev
    # save ev
    with open(
        f"{args.outdir}/captioning_{args.model.replace('/', '-')}_{args.dataset}_{args.nsample}_{args.ablation}_{args.ablation_type}_eval.json",
        "w",
    ) as f:
        json.dump(ev, f, ensure_ascii=False, indent=4)
    # save full_res
    with open(
        f"{args.outdir}/captioning_{args.model.replace('/', '-')}_{args.dataset}_{args.nsample}_{args.ablation}_{args.ablation_type}.json",
        "w",
    ) as f:
        json.dump(full_res, f, ensure_ascii=False, indent=4)
    print("####################### Final Results ############################")
    print(f"pycocoeval: {ev['pycocoeval']} +/- {ev['pycocoeval_std']}")
    print(f"original: {ev['original']} +/- {ev['original_std']}")

    # with open(
    #     f"{args.outdir}/captioning_{args.model.replace('/', '-')}_{args.dataset}_{args.nsample}_{args.ablation}_eval.txt",
    #     "w",
    # ) as f:
    #     for key, value in ev.items():
    #         f.write("%s\t%s\n" % (key, value))
