#!/usr/bin/env python

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# join src module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation_task"))
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
import requests
from PIL import Image
from io import BytesIO
import json
from transformers import GenerationConfig
import torch 
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
import argparse
from src.activations import ExtractActivationConfig, Extractor, CollateFunctions
from evaluation_task.caption.eval.cider_metrics.eval_json import Cider
from dotenv import load_dotenv
import pandas as pd
from rich import print 

load_dotenv("../.env")

from datasets import load_dataset, DatasetDict, load_from_disk



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="pixtral, chameleon, emu3", default = "chameleon")
    parser.add_argument("-d", "--dataset", type=str,
                            help="flickr, coco", default = "flickr")
    parser.add_argument("-n", "--nsample", type=int,
                        help="n of data to sample", default = 2000)
    parser.add_argument("-o", "--outdir", type=str,
                        help="outdir", default = "")
    parser.add_argument("-a", "--ablation", type=str, default = None)
    parser.add_argument("-w", "--window", type=int, default = None)

    args = parser.parse_args()
    print(f'{args.outdir}/{args.model}_{args.nsample}.json')

    if args.model == 'chameleon':
        model_name = "facebook/chameleon-30b"
        image_string = "<image>"
        start_sentence = ""
        attn_implementation = "eager"
        ablation_type = "std"
        use_cache = True

    elif args.model == "chameleon-7b":
        model_name = "facebook/chameleon-7b"
        image_string = "<image>"
        start_sentence = ""
        attn_implementation = "eager"
        ablation_type = "std"
        use_cache = True
        
    elif args.model == 'pixtral':
        model_name = "mistral-community/pixtral-12b"
        image_string =  "[IMG]"
        start_sentence = "<s>"
        attn_implementation = "eager"
        ablation_type = "std"
        use_cache = True
        

    elif args.model == 'emu3':
        model_name = "Emu3-Stage1"
        image_string =  "{IMG}"
        start_sentence = ""
        attn_implementation = "flash_attention"
        ablation_type = "flash-attn"
        use_cache = True

    if args.dataset == 'flickr':
        dataset = load_dataset("nlphuji/flickr30k", trust_remote_code = True, split = 'test')
    elif args.dataset == 'coco':
        dataset = load_dataset("yerevann/coco-karpathy", split = 'test')
    else:
        print("Dataset not supported")
        exit()

    config = ExtractActivationConfig(
            model_name=model_name,
            device_map = "balanced",
            attn_implementation = attn_implementation,
            torch_dtype=torch.bfloat16,
        )
    
    model = Extractor(
        config
    )
    num_heads = model.model_config.num_attention_heads
    num_layers = model.model_config.num_hidden_layers
    processor = model.get_processor()

    # set seed
    if args.dataset == "coco":
        idxs = [i for i in range(len(dataset))]
    else:
        seed = 42
        random.seed(seed)
        all_idxs = [i for i in range(len(dataset))]
        idxs = random.sample(all_idxs, args.nsample)
    if args.ablation == "@random-image-10":
        n_iterations = 6
    elif args.ablation == "@last-image" or args.ablation == "@end-image" or args.ablation == "@all-image" or args.ablation in ["@1025",  "@eols"]:
        n_iterations = 1
    else:
        raise ValueError("Ablation not supported")
    
    elem_to_ablate = args.ablation

    results = []
    ev = {}
    ev["pycocoeval"] = []
    ev["original"] = []
    for i in range(n_iterations):
        print(f'####################### Iteration {i} ##################################')
        with torch.no_grad():
            for p_idx in tqdm.tqdm(idxs):
                p = dataset[p_idx]
                if args.dataset=='flickr':
                    img = p['image']
                    caption = p["caption"]
                    img_id = p['img_id'] 
                else:
                    # img = p['url']
                    # caption = p["sentences"]
                    # img_id = str(p['imgid'])
                    # response = requests.get(img)
                    # img = Image.open(BytesIO(response.content))
                    # get the image from path
                    path = p['url']
                    caption = p["sentences"]
                    img_id = str(p['imgid'])
                    img = Image.open(path)
                

                # create prompt
                if 'chameleon' in args.model:
                    completion = True
                    prompt = f'{image_string}Provide a one-sentence caption for the provided image.'
                    # prompt = f'{image_string} Provide a one-sentence caption for the provided image:'
                    
                elif args.model == 'pixtral':
                    prompt = f'[INST]"{start_sentence}{image_string} Provide a one-sentence caption for the provided image."[/INST]'
                
                elif args.model == 'emu3':
                    prompt = f'"{image_string}USER:Provide a one-sentence caption for the provided image. ASSISTANT:"'  
                
                # create inputs
                if 'chameleon' in args.model:
                    inputs = processor(images=[img], text=prompt, return_tensors="pt", return_for_text_completion=completion).to(model.device())
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16) 
                

                elif args.model == 'emu3':
                    inputs = processor(text=prompt, images=img, return_tensors="pt", mode="U").to(model.device())
                    inputs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]}
                
                elif args.model == 'pixtral':
                    inputs = processor(images=img, text=prompt, return_tensors="pt", padding=False).to(model.device())
                    
                    
                generation_config = GenerationConfig(
                    max_new_tokens=60,
                    use_cache=use_cache,
                    return_dict_in_generate=True)
                
                if args.ablation:
                    if args.window:
                        if args.window == 0:
                            size = 8
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(num_layers)]
                        if args.window == 1:
                            size = 8
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(4, num_layers)]
                        if args.window == 2:
                            size = 8
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(8, num_layers)]
                        if args.window == 3:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(12, num_layers)]
                        if args.window == 4:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(16, num_layers)]
                        if args.window == 5:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(20, num_layers)]
                        if args.window == 6:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(24, num_layers)]
                        if args.window == 7:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(28, num_layers)]
                        if args.window == 8:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(32, num_layers)]
                        if args.window == 9:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(36, num_layers)]
                        if args.window == 10:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(40, num_layers)]
                        if args.window == 11:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(44, num_layers)]
                        if args.window == 12:
                            size = 4
                            head_layer_couple = [[i,j] for i in range(num_heads) for j in range(48, num_layers)]
                            
                            
                    else:
                        head_layer_couple = [[i,j] for i in range(num_heads) for j in range(num_layers)]
                    
                    generated_ids = model.generate(
                        inputs=inputs,
                        generation_config=generation_config,
                        ablation_queries=[
                            {
                                "type": ablation_type,
                                "elem-to-ablate": elem_to_ablate,
                                "head-layer-couple": head_layer_couple,
                            }
                        ]
                    )
                else:
                    generated_ids = model.generate(
                        inputs=inputs,
                        generation_config=generation_config,
                        )

                    # batch decode
                if args.model == 'pixtral':
                    full_res = processor.batch_decode(generated_ids, n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=10, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None)[0]     

                else:
                    full_res = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                results.append({"result":full_res, "image_id":img_id, "caption":caption})
                 
            # with open(f'{args.outdir}/{args.model}_{args.nsample}.json', 'w') as f:
            #     json.dump(results, f, ensure_ascii=False)
        
        ground_truth = {i: d["caption"] for i, d in enumerate(results)} 
        response = {}
        if args.model == 'pixtral':
            for  i, d in enumerate(results):
                response[i] = [d['result'][d['result'].find('image.')+6:]]
        
        elif 'chameleon' in args.model :
            for  i, d in enumerate(results):
                response[i] = [d['result'][d['result'].find('image.\n\n')+8:]]
        
        elif args.model == 'emu3':
            for i, d in enumerate(results):
                response[i] = [d['result'][d['result'].find('ASSISTANT:')+11:]]
        
        
        score, scores = Cider(backend="pycocoeval").compute_score(ground_truth, response)
        print('Cider score pycocoeval =', score)
        ev['pycocoeval'].append(score)
        
        
        score, scores = Cider(backend="original").compute_score(ground_truth, response)
        print('Cider score original =', score)
        ev['original'].append(score)
        
        
        
    ev["pycocoeval_std"] = np.std(ev["pycocoeval"])
    ev["original_std"] = np.std(ev["original"])
    ev["pycocoeval"] = np.mean(ev["pycocoeval"])
    ev["original"] = np.mean(ev["original"])
        
    print("####################### Final Results ############################")
    print(f"pycocoeval: {ev['pycocoeval']} +/- {ev['pycocoeval_std']}")
    print(f"original: {ev['original']} +/- {ev['original_std']}")
        
    with open(f'{args.outdir}/captioning_{args.model}_{args.dataset}_{args.nsample}_{args.ablation}_eval.txt', 'w') as f: 
        for key, value in ev.items():  
            f.write('%s\t%s\n' % (key, value))
