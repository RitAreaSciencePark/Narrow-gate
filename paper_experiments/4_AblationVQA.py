#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

import json
from transformers import GenerationConfig
import torch 
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
import argparse
from src.activations import ExtractActivationConfig, Extractor, CollateFunctions
from dotenv import load_dotenv
import pandas as pd
from evaluation_task.vqa.eval.eval_vqa import evaluate_VQA, score_pixtral
from rich import print 

load_dotenv("../.env")

from datasets import load_dataset
dataset = load_dataset("HuggingFaceM4/VQAv2", split ='validation')
pandas_dataset = dataset.to_pandas()


PROMT_PIXTRAL = """- Answer the question using a single word, number, or short phrase. Use as few words as possible.
- If the answer is a number, report it as a number, i.e. 2, not Two, and only include the number without any unit.
- If the question is Yes/No, answer with Yes/No, and nothing else (no likely, unknown, etc.).
- You cannot answer that the question is unanswerable. You must answer."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="mistral-community/pixtral-12b, facebook/chameleon-7b, facebook/chameleon-30b", default = "chameleon")
    parser.add_argument("-md", "--mode", type=str,
                        help="mode of the promt: zero-shot, prompt", default = "prompt")
    parser.add_argument("-n", "--nsample", type=int,
                        help="n of data to sample", default = 2000)
    parser.add_argument("-o", "--outdir", type=str,
                        help="outdir", default = "")
    parser.add_argument("-a", "--ablation", type=str, default = None)
    parser.add_argument("-w", "--window", type=int, default = None)
    args = parser.parse_args()
    print(f'{args.outdir}/{args.model}_{args.nsample}_{args.mode}.json')

    if args.model == "facebook/chameleon-30b":
        model_name = "facebook/chameleon-30b"
        image_string = "<image>"
        start_sentence = ""
        attn_implementation = "eager"
        ablation_type = "std"
        use_cache = True
        
    elif args.model == "facebook/chameleon-7b":
        model_name = "facebook/chameleon-7b"
        image_string = "<image>"
        start_sentence = ""
        attn_implementation = "eager"
        ablation_type = "std"
        use_cache = True

    elif args.model == 'mistral-community/pixtral-12b':
        model_name = "mistral-community/pixtral-12b"
        image_string =  "[IMG]"
        start_sentence = "<s>"
        attn_implementation = "eager"
        ablation_type = "std"
        use_cache = True

    else:
        raise ValueError("Model not recognized")

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
    seed = 42
    random.seed(seed)
    all_idxs = [i for i in range(len(dataset))]
    idxs = random.sample(all_idxs, args.nsample)
 
    
    if args.ablation == "@random-image-10":
        n_iterations = 10
    elif args.ablation == "@last-image" or args.ablation == "@end-image" or args.ablation == "@all-image":
        n_iterations = 1
    else:
        print("Ablation not recognized or proceeding without ablation")
        args.ablation = None
        n_iterations = 1
    
    elem_to_ablate = args.ablation

    ev = {}
    ev['pixtral'] = []
    ev['VQA'] = []
    for i in range(n_iterations):
        print(f'####################### Iteration {i} ##################################')
        results = []
        with torch.no_grad():
            for p_idx in tqdm.tqdm(idxs):
                p = dataset[int(p_idx)]
                img = p['image']
                question = p["question"]
                real_ans = p['multiple_choice_answer']
                question_id = p["question_id"]

                # create prompt
                if args.mode == 'zero-shot' and "chameleon" in args.model:
                    completion = True
                    prompt = f'{image_string}{question}\nAnswer the question using a single word or phrase. Answer:'
                    
                elif args.mode == 'prompt' and "chameleon" in args.model:
                    completion = False
                    prompt = f'{image_string}{question}\n{PROMT_PIXTRAL}'

                elif args.mode == 'zero-shot' and  'pixtral' in args.model:
                    prompt = f'[INST]{start_sentence}{image_string}{question}"\nAnswer the question using a single word or phrase."[/INST]'
                
                elif args.mode == 'prompt' and 'pixtral' in args.model:
                    prompt = f'[INST]{start_sentence}{image_string}{question}{PROMT_PIXTRAL}[/INST]'


                # create inputs
                if "chameleon" in args.model:
                    inputs = processor(images=[img], text=prompt, return_tensors="pt", return_for_text_completion=completion).to(model.device())
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16) 

                elif 'pixtral' in args.model:
                    inputs = processor(images=img, text=prompt, return_tensors="pt", padding=False).to(model.device())


                generation_config = GenerationConfig(
                    max_new_tokens=20,
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
                
                results.append({"answer":full_res, "question_id": question_id})
        
        with open(f'{args.outdir}/{args.model.split("/")[1]}_{args.nsample}_{args.mode}_{args.window}.json', 'w') as f:
             json.dump(results, f, ensure_ascii=False)
        
        if  'pixtral' in args.model and args.mode == 'prompt':
            for i in range(len(results)):
                results[i]['answer'] = results[i]['answer'][results[i]['answer'].find('You must answer.')+16:]
        elif  'pixtral' in args.model and args.mode == 'zero-shot':
            for i in range(len(results)):
                results[i]['answer'] = results[i]['answer'][results[i]['answer'].find('phrase."')+8:]

        elif "chameleon" in args.model and args.mode == 'prompt':
            for i in range(len(results)):
                results[i]['answer'] = results[i]['answer'][results[i]['answer'].find('You must answer.')+16:]
        elif "chameleon" in args.model and args.mode == 'zero-shot':
            for i in range(len(results)):
                results[i]['answer'] = results[i]['answer'][results[i]['answer'].find("Answer:")+8:]
        results = pd.DataFrame(results)
        print(results)
                    
        accuracy_VQA = evaluate_VQA(pandas_dataset, pd.DataFrame(results))
        print('Accuracy VQA:', accuracy_VQA, '\n')
        ev['VQA'].append(accuracy_VQA)
        
        
        accuracy_pixtral = []
        for index, row in results.iterrows():
            all_ans = pandas_dataset[pandas_dataset["question_id"]==row["question_id"]]['answers'].tolist()
            ans = [i['answer'] for i in all_ans[0]]
            model_ans = row['answer'].strip(' ').strip('.').lower().strip("'\'").strip('""').strip(' ')
            s_pixtral = score_pixtral(model_answer = model_ans, 
                                    reference_answer = ans)
            accuracy_pixtral.append(s_pixtral)
        print('Accuracy Pixtral:', np.mean(accuracy_pixtral), '\n')    
        # ev['pixtral'] = ev['pixtral'] + accuracy_VQA
        ev['pixtral'].append(accuracy_VQA)

    ev['VQA_std'] = np.std(ev['VQA'])
    ev['VQA'] = np.mean(ev['VQA'])
    ev['pixtral_std'] = np.std(ev['pixtral'])
    ev['pixtral'] = np.mean(ev['pixtral'])
    
    print(f"#### Final results ####")
    print(f"VQA: {ev['VQA']} +/- {ev['VQA_std']}")
    
    with open(f'{args.outdir}/vqa_{args.model.split("/")[1]}_{args.nsample}_{args.mode}_{args.ablation}_eval.txt', 'w') as f:
        for key, value in ev.items():  
            f.write('%s\t%s\n' % (key, value))
    
            
