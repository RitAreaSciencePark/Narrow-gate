#!/usr/bin/env python

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from huggingface_hub import snapshot_download
import warnings

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

import json
from transformers import GenerationConfig
import torch
import numpy as np
import tqdm
import random
import argparse
from src.activations import ExtractActivationConfig, Extractor
from dotenv import load_dotenv
import pandas as pd
from evaluation_task.vqa.eval.eval_vqa import evaluate_VQA, score_pixtral
from rich import print
from datasets import load_dataset
import warnings
from src.common.emu_utils import preprocess_conversation
from utils import prefix_allowed_tokens_fn

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


load_dotenv("../.env")


dataset = load_dataset(
    "HuggingFaceM4/VQAv2",
    split="validation",
)
pandas_dataset = dataset.to_pandas()

# Define model configurations in a dictionary for better maintainability
model_configs = {
    "facebook/chameleon-30b": {
        "model_name": "facebook/chameleon-30b",
        "image_string": "<image>",
        "start_sentence": "",
        "attn_implementation": "eager",
        "ablation_type": "std",
        "use_cache": True,
    },
    "facebook/chameleon-7b": {
        "model_name": "facebook/chameleon-7b",
        "image_string": "<image>",
        "start_sentence": "",
        "attn_implementation": "eager",
        "ablation_type": "std",
        "use_cache": True,
    },
    "mistral-community/pixtral-12b": {
        "model_name": "mistral-community/pixtral-12b",
        "image_string": "[IMG]",
        "start_sentence": "<s>",
        "attn_implementation": "eager",
        "ablation_type": "std",
        "use_cache": True,
    },
    "janus": {
        "model_name": "deepseek-ai/Janus-1.3B",
        "image_string": "<image_placeholder>",
        "start_sentence": "",
        "start_image_token": "<begin_of_image>",
        "end_image_token": "<end_of_image>",
        "attn_implementation": "eager",
        "use_cache": True,
    },
    "Emu3-Gen-hf": {
        "model_name": "Emu3-Gen-hf",
        "image_string": "<image>",
        "start_sentence": "<|extra 203|>",
        "end_of_frame": "<|extra 201|>",
        "attn_implementation": "eager",
        "use_cache": True,
        "needs_processor": True,
    },
    "llava-onevision-7b": {
        "model_name": "llava-onevision-7b",
        "image_string": "<|im_start|><image><|im_end|>",
        "start_sentence": "<s>",
        "attn_implementation": "eager",
        "ablation_type": "std",
        "use_cache": True,
    },
    "vila-u": {
        "model_name": "vila-u",
        "image_string": "<image>",
        "start_sentence": "",
        "attn_implementation": "eager",
        "ablation_type": "std",
        "use_cache": True,
    },
}

PROMT_PIXTRAL = """- Answer the question using a single word, number, or short phrase. Use as few words as possible.\
- If the answer is a number, report it as a number, i.e. 2, not Two, and only include the number without any unit.\
- If the question is Yes/No, answer with Yes/No, and nothing else (no likely, unknown, etc.).\
- You cannot answer that the question is unanswerable. You must answer."""
EMU3_PROMPT = "Answer the question using a single word or phrase."
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
              deepseek-ai/Janus-1.3B, \
              Emu3-Gen-hf \
              vila-u",
        default="chameleon",
    )
    parser.add_argument(
        "-md",
        "--mode",
        type=str,
        help="mode of the promt: zero-shot, prompt",
        default="prompt",
    )
    parser.add_argument(
        "-n", "--nsample", type=int, help="n of data to sample", default=2000
    )
    parser.add_argument("-o", "--outdir", type=str, help="outdir", default="results")
    parser.add_argument("-a", "--ablation", type=str, default=None)
    parser.add_argument("-at", "--ablation_type", type=str, default=None)
    args = parser.parse_args()

    if args.ablation:
        args.ablation_type = "std"
    print(f"{args.outdir}/{args.model}_{args.nsample}_{args.mode}_{args.ablation}.json")
    print("Model: ", args.model)
    print("Mode: ", args.mode)
    print("Nsample: ", args.nsample)
    print("Outdir: ", args.outdir)
    print("Ablation: ", args.ablation)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)



    # Get configuration for the selected model
    if args.model in model_configs:
        config_dict = model_configs[args.model]
        model_name = config_dict["model_name"]
        image_string = config_dict["image_string"]
        start_sentence = config_dict["start_sentence"]
        attn_implementation = config_dict["attn_implementation"]
        use_cache = config_dict["use_cache"]

        # Use provided ablation type if specified, otherwise use the one from config
        ablation_type = (
            args.ablation_type
            if args.ablation_type
            else config_dict.get("ablation_type", "std")
        )

        # # Handle special tokens that exist for specific models
        # if "start_image_token" in config_dict:
        #     start_image_token = config_dict["start_image_token"]
        # if "end_image_token" in config_dict:
        #     end_image_token = config_dict["end_image_token"]
        # if "end_of_frame" in config_dict:
        #     end_of_frame = config_dict["end_of_frame"]

        # Handle Emu3 processor initialization if needed
        if config_dict.get("needs_processor", False) and args.model == "Emu3-Gen-hf":
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
    else:
        raise ValueError(f"Model {args.model} not supported")

    # Configure model extractor
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

    if args.model == "Emu3-Gen-hf":
        from transformers import Emu3Processor, Emu3ImageProcessor

        processor = Emu3Processor.from_pretrained(
            "BAAI/Emu3-Gen-hf",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        image_processor = Emu3ImageProcessor(
            min_pixels=256 * 256,
            max_pixels=256 * 256,
            torch_dtype=torch.bfloat16,
            # device_map="balanced",
        )
        tokenizer = processor.tokenizer

    # set seed
    seed = 42
    random.seed(seed)
    all_idxs = [i for i in range(len(dataset))]
    idxs = random.sample(all_idxs, args.nsample)

    if args.ablation == "@random-image-10":
        n_iterations = 5
    elif (
        args.ablation == "@last-image"
        or args.ablation == "@end-image"
        or args.ablation == "@all-image"
        or args.ablation == "@end-image-emu"
    ):
        n_iterations = 1
    elif args.ablation == "@img-text":
        n_iterations = 1
        ablation_type = "block-img-txt"
    else:
        # print this in red
        print("#### ABLATION NOT RECOGNIZED PROCEDING WITHOUT ABLATION ####")
        # print("Ablation not recognized or proceeding without ablation")
        args.ablation = None
        n_iterations = 1

    elem_to_ablate = args.ablation

    ev = {}
    ev["pixtral"] = []
    ev["VQA"] = []

    for i in range(n_iterations):
        print(
            f"####################### Iteration {i} ##################################"
        )
        results = []
        with torch.no_grad():
            for p_idx in tqdm.tqdm(idxs):
                p = dataset[int(p_idx)]
                img = p["image"]
                question = p["question"]
                real_ans = p["multiple_choice_answer"]
                question_id = p["question_id"]

                ################
                # create prompt
                ################
                # Create prompt based on model and mode
                completion = True

                # Chameleon and LLAVA models (similar prompt formats)
                if "chameleon" in args.model or "llava-onevision" in args.model:
                    if args.mode == "zero-shot":
                        completion = True
                        prompt = f"{image_string}{question}\nAnswer the question using a single word or phrase. Answer:"
                    elif args.mode == "prompt":
                        completion = False
                        prompt = f"{image_string}{question}\n{PROMT_PIXTRAL}"

                # VILA-U model
                elif args.model == "vila-u":
                    if args.mode == "zero-shot":
                        # prompt = f"{SYSTEM_PROMPT_VILA}\nHuman:{image_string}\n{question} Give a brief answer. Assistant:"
                        prompt = f"{SYSTEM_PROMPT_VILA}USER: <image>\n{question}\nAnswer the question using a single word, number, or short phrase. ASSISTANT:"
                        
                    elif args.mode == "prompt":
                        prompt = f"{PROMT_PIXTRAL}{image_string}\nUSER: {question} ASSISTANT:"

                # Pixtral model
                elif "pixtral" in args.model:
                    if args.mode == "zero-shot":
                        prompt = f'[INST]{start_sentence}{image_string}{question}"\nAnswer the question using a single word or phrase."[/INST]'
                    elif args.mode == "prompt":
                        prompt = f"[INST]{start_sentence}{image_string}{question}{PROMT_PIXTRAL}[/INST]"

                # Janus model
                elif args.model == "janus":
                    if args.mode == "zero-shot":
                        prompt = f'"{image_string}USER:{question}"\nAnswer the question using a single word or phrase."\nASSISTANT:"'
                    elif args.mode == "prompt":
                        prompt = f'"{image_string}USER:{question}"\nAnswer the question using a single word or phrase{PROMT_PIXTRAL}."\nASSISTANT:"'

                # Emu3-Gen model (more complex structure)
                elif "Emu3-Gen" in args.model:
                    prompt_content = EMU3_PROMPT if args.mode == "prompt" else ""
                    prompt = (
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {
                                    "type": "text",
                                    "text": "\n" + question + prompt_content,
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": ""}],
                        },
                    )

                # Fallback for unrecognized models
                else:
                    raise ValueError(f"Unsupported model: {args.model}")

                # create inputs
                if "chameleon" in args.model:
                    inputs = processor(
                        images=[img],
                        text=prompt,
                        return_tensors="pt",
                        return_for_text_completion=completion,
                    ).to(model.device())
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                elif "pixtral" in args.model:
                    inputs = processor(
                        images=img, text=prompt, return_tensors="pt", padding=False
                    ).to(model.device())
                    inputs["pixel_values"] = inputs["pixel_values"][0]

                elif "llava-onevision" in args.model:
                    inputs = processor(
                        images=[img], text=prompt, return_tensors="pt", padding=False
                    ).to(model.device())
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                elif args.model == "Emu3-Gen-hf":
                    tokenizer = processor.tokenizer
                    text_ = processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=False,
                        tokenize=False,
                        return_dict=False,
                    )

                    image_features = image_processor(img, return_tensors="pt")
                    text = preprocess_conversation(
                        text=[text_], image_features=image_features
                    )
                    inputs = tokenizer(text, return_tensors="pt")
                    inputs.update(**image_features)

                    image_sizes = inputs["image_sizes"]


                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device())
                            if k != "input_ids":
                                inputs[k] = inputs[k].to(torch.bfloat16)

                else:
                    inputs = processor(text=prompt, images=[img], return_tensors="pt")
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device())
                            inputs["pixel_values"] = inputs["pixel_values"].to(
                                torch.bfloat16
                            )

                generation_config = GenerationConfig(
                    max_new_tokens=20,
                    use_cache=use_cache,
                    return_dict_in_generate=True,
                    low_cpu_mem_usage=True,
                )
                if args.model == "vila-u":
                    generation_config = GenerationConfig(
                        bos_token_id=1,
                        do_sample=True,
                        eos_token_id=2,
                        max_length=50,
                        pad_token_id=0,
                        temperature=0.9,
                        top_p=0.6,
                        use_cache=use_cache,
                        return_dict_in_generate=True,
                        low_cpu_mem_usage=True,
                    )
                other_args = (
                    {}
                    if args.model != "Emu3-Gen-hf"
                    else {
                        "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
                    }
                )
                if args.ablation or args.ablation_type:
                    head_layer_couple = [
                        [i, j] for i in range(num_heads) for j in range(num_layers)
                    ]
                    
                    generated_ids = model.generate(
                        inputs=inputs,
                        generation_config=generation_config,
                        ablation_queries=[
                            {
                                "type": ablation_type,
                                "elem-to-ablate": elem_to_ablate,
                                "head-layer-couple": head_layer_couple,
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
                    
                    
                    # generated_ids = model.hf_model.generate(
                    #     **inputs,
                    #     generation_config=generation_config,
                    #     output_scores=False
                    # )["sequences"]

                # alternative_genration = model.hf_model.generate_content(
                #     prompt = prompt,
                #     images = [img],
                #     generation_config = generation_config,
                # )
                # batch decode
                if args.model == "pixtral":
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

                elif args.model == "janus":
                    full_res = processor.tokenizer.decode(
                        generated_ids[0][:-1].cpu().tolist(), skip_special_token=True
                    )
                elif args.model == "vila-u":
                    full_res = processor.tokenizer.decode(
                        generated_ids[0][:-1].cpu().tolist(), skip_special_tokens=True
                    )
                else:
                    full_res = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]

                results.append(
                    {
                        "answer": full_res,
                        "question_id": question_id,
                        "real_ans": real_ans,
                    }
                )

        with open(
            f"{args.outdir}/{args.model.replace('/', '-')}_{args.nsample}_{args.mode}.json",
            "w",
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        
        if "pixtral" in args.model:
            if args.mode == "prompt":
                for i in range(len(results)):
                    results[i]["answer"] = results[i]["answer"][
                        results[i]["answer"].find("You must answer.") + 16 :
                    ]
            elif args.mode == "zero-shot":
                for i in range(len(results)):
                    results[i]["answer"] = results[i]["answer"][
                        results[i]["answer"].find('phrase."') + 8 :
                    ]
        elif (
            "chameleon" in args.model or "llava-onevision" in args.model
            # or "vila-u" in args.model
        ):
            if args.mode == "prompt":
                for i in range(len(results)):
                    results[i]["answer"] = results[i]["answer"][
                        results[i]["answer"].find("You must answer.") + 16 :
                    ]
            elif args.mode == "zero-shot":
                for i in range(len(results)):
                    results[i]["answer"] = results[i]["answer"][
                        results[i]["answer"].find("Answer:") + 8 :
                    ]
        elif args.model == "janus":
                for i in range(len(results)):
                    results[i]["answer"] = results[i]["answer"][
                        results[i]["answer"].find("ASSISTANT:") + 10 :
                    ]
        elif args.model == "Emu3-Gen-hf":
            for i in range(len(results)):
                results[i]["answer"] = results[i]["answer"][
                    results[i]["answer"].find("ASSISTANT:") + 10 :
                ]

        results = pd.DataFrame(results)
        print(results)

        accuracy_VQA = evaluate_VQA(pandas_dataset, pd.DataFrame(results))
        print("Accuracy VQA:", accuracy_VQA, "\n")
        ev["VQA"].append(accuracy_VQA)

        accuracy_pixtral = []
        for index, row in results.iterrows():
            all_ans = pandas_dataset[
                pandas_dataset["question_id"] == row["question_id"]
            ]["answers"].tolist()
            ans = [i["answer"] for i in all_ans[0]]
            model_ans = (
                row["answer"]
                .strip(" ")
                .strip(".")
                .lower()
                .strip("''")
                .strip('""')
                .strip(" ")
            )
            s_pixtral = score_pixtral(model_answer=model_ans, reference_answer=ans)
            accuracy_pixtral.append(s_pixtral)
        print("Accuracy Pixtral:", np.mean(accuracy_pixtral), "\n")
        # ev['pixtral'] = ev['pixtral'] + accuracy_VQA
        ev["pixtral"].append(accuracy_VQA)

    ev["VQA_std"] = np.std(ev["VQA"])
    ev["VQA"] = np.mean(ev["VQA"])
    ev["pixtral_std"] = np.std(ev["pixtral"])
    ev["pixtral"] = np.mean(ev["pixtral"])

    print("#### Final results ####")
    print(f"VQA: {ev['VQA']} +/- {ev['VQA_std']}")

    print(
        "Saving results in: ",
        f"{args.outdir}/vqa_{args.model.replace('/', '-')}_{args.nsample}_{args.mode}_{args.ablation}_{args.ablation_type}_eval.txt",
    )
    with open(
        f"{args.outdir}/vqa_{args.model.replace('/', '-')}_{args.nsample}_{args.mode}_{args.ablation}_{args.ablation_type}_eval.txt",
        "w",
    ) as f:
        for key, value in ev.items():
            f.write("%s\t%s\n" % (key, value))

