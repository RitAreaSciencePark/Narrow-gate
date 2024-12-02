from datasets import load_dataset
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# join src module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation_task"))
    
import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
from transformers import GenerationConfig
import torch 
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import argparse
from src.activations import ExtractActivationConfig, Extractor, CollateFunctions

def prompt_ai2d(prompt_format, choices, pre_prompt, post_prompt, question):
    len_choices = len(choices)
    if prompt_format == "mcq":
        options = [i+1 for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
        
    elif prompt_format == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
        
def resize_img(img):
    w, h = img.size
    max_aspect_ratio = 5
    if w>800 or h>800:
        aspect_ratio = h / w
        if h>w:
            h=800
            w=int(h / aspect_ratio)
        else:
            w=800
            h = int(w * aspect_ratio)
        img=img.resize((w, h))
    if h/w > max_aspect_ratio:
        h = int(w*max_aspect_ratio)
        img=img.resize((w,h))
    elif w/h > max_aspect_ratio:
        w = int(h*max_aspect_ratio)
        img=img.resize((w,h))
    
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="pixtral, chameleon30b, chameleon7b, emu3", default = "chameleon30b")
    parser.add_argument("-a", "--ablation", type=str, default = None)
    
    args = parser.parse_args()
    print("Abalation: ", args.ablation)
    if args.model == 'chameleon30b':
        model_name = "facebook/chameleon-30b"
        image_string = "<image>"
        start_sentence = ""
        attn_implementation = "eager"
        ablation_type = "std"
        
    elif args.model == "chameleon-7b":
        model_name = "facebook/chameleon-7b"
        image_string = "<image>"
        start_sentence = ""
        attn_implementation = "eager"
        ablation_type = "std"

    elif args.model == 'pixtral':
        model_name = "mistral-community/pixtral-12b"
        image_string =  "[IMG]"
        start_sentence = "<s>"
        attn_implementation = "eager"
        ablation_type = "std"

    elif args.model == 'emu3':
        model_name = "Emu3-Chat"
        image_string =  "{IMG}"
        start_sentence = ""
        attn_implementation = "flash_attention_2"
        ablation_type = "flash-attn"

    config = ExtractActivationConfig(
            model_name=model_name,
            device_map = "balanced",
            attn_implementation = attn_implementation,
            torch_dtype=torch.bfloat16,
        )
    model = Extractor(
        config
    )
    processor = model.get_processor()
    dataset = load_dataset("lmms-lab/ai2d", split = "test[:2000]")
    
    num_hidden_layers = model.model_config.num_hidden_layers
    num_attention_heads = model.model_config.num_attention_heads
    
    
    if args.ablation == "@random-image-10":
        n_iterations = 1
    else:
        n_iterations = 1
    # else:
    #     print("Ablation not recognized or proceeding without ablation")
    #     args.ablation = None
    
    elem_to_ablate = args.ablation
    all_results = []
    for i in range(n_iterations):
        print(f'####################### Iteration {i} ##################################')
        results = []
        with torch.no_grad():
            for p_idx in tqdm(range(len(dataset))):
                p = dataset[int(p_idx)]
                img = resize_img(p['image'])
                question = p["question"]
                choices=[c if c!='{}' else '' for c in p["options"]]
                    
                ans = p['answer']
                

                prompt = prompt_ai2d("mcq", choices, "","\nYou must answer and using only the option's (1, 2, 3 or 4) directly. The answer is", question)
                
                qst = f'"{image_string}{prompt}" '
                generation_config = GenerationConfig(
                            max_new_tokens=5,
                            use_cache=True,
                            return_dict_in_generate=True,
                        )
                if args.model=='emu3':
                    inputs = processor(text=qst, images=[img], return_tensors="pt", mode="U", chat_template = None).to(model.device())
                    inputs={
                                "input_ids": inputs["input_ids"],
                                "attention_mask": inputs["attention_mask"],
                    }
                elif args.model == 'pixtral':
                    inputs = processor(images=img, text=qst, return_tensors="pt", padding=False).to(model.device())
                elif 'chameleon' in args.model:
                    inputs = processor(images=[img], text=qst, return_tensors="pt", return_for_text_completion=True).to(model.device())
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
                else:
                    raise ValueError("Model not supported")
                if args.ablation:
                    generated_ids = model.generate(
                        inputs=inputs,
                        generation_config=generation_config,
                        ablation_queries=[
                            {
                                "type": ablation_type,
                                "elem-to-ablate": elem_to_ablate,
                                "head-layer-couple": [[i,j] for i in range(num_attention_heads) for j in range(num_hidden_layers)],
                            }
                        ]
                    )
                else:
                    generated_ids = model.generate(
                        inputs=inputs,
                        generation_config=generation_config,
                        )
                if args.model != 'pixtral':
                    full_res = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                else:
                    full_res = processor.batch_decode(generated_ids, n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=10, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None)[0]
                torch.cuda.empty_cache()
                results.append({"answer":full_res, "ground_truth": ans})
            ground_truth=[res['ground_truth'] for res in results]
        answers = [res['answer'] for res in results]
        # mapping = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
        mapping = {'0': '1', '1': '2', '2': '3', '3': '4'}
        ground_truth = [mapping[x] for x in ground_truth]
        answers = [answers[i].split('The answer is')[-1].strip('"').strip(' ').split(' ')[0].strip('"').strip('.').strip("\n").strip("(").strip(')"') for i in range(len(answers))]
        count = 0
        for i in range(len(answers)):
            if answers[i]==ground_truth[i]:
                count += 1
        all_results.append(count/len(answers))
        
    print(f"Average accuracy: {np.mean(all_results)}")
    print(f"Standard deviation: {np.std(all_results)}")
    
    results = {
        "accuracy": np.mean(all_results),
        "std": np.std(all_results),
        "all_results": all_results
    }
            
    with open(f'./data/results/ai2d/{args.model}.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False)
