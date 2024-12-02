import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from easyroutine import path_to_parents, Logger

# path_to_parents(2)

from argparse import ArgumentParser
from dataclasses import dataclass
import torch
from src.activations import ExtractActivationConfig, Extractor
from src.metrics.overlap import PredictOverlap
from src.utils import generate_code, data_path
from pathlib import Path
from rich import print
import pandas as pd
import pickle
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import Literal, List
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from src.metrics.residual_stream import ResidualStreamMetricComputer
from dadapy import Data
from sklearn.metrics import homogeneity_completeness_v_measure
load_dotenv("../.env")


def cross_attention_patterns(model, dict_attention_maps):
    if model == "facebook/chameleon-7b":
        num_layers = 32
        num_heads = 32
        img_len = 1+1024+2
    elif model == "facebook/chameleon-30b":
        num_layers = 48
        num_heads = 64
        img_len = 1+1024+2
    elif model == "mistral-community/pixtral-12b":
        num_layers = 40
        num_heads = 32
        img_len = 1+1056

    str_tokens = np.zeros((num_layers, img_len))
    var_tokens = np.zeros((num_layers, img_len))
    for L in range(num_layers):
        for H in range(num_heads):
            name_att = f'pattern_L{L}H{H}'
            # 
            tmap = dict_attention_maps[name_att].float()
            prompt_len = tmap.shape[1]
            crosstalk = tmap[0,img_len:,:img_len]
            token_strength, token_id = torch.sort(crosstalk.mean(0), descending=True)
            str_tokens[L] += crosstalk.mean(0).cpu().numpy()
            var_tokens[L] += crosstalk.std(0).cpu().numpy()

    dist_res = {'mean':str_tokens, 'std':var_tokens}
    return dist_res


# PLOT

def plot_cross_attention(model, dict_data, save_path):

    if model == 'facebook/chameleon-7b':

        # Preprocessing of raw data--------------------------
        norm = dict_data['mean'][:,2:] / dict_data['mean'][:,2:].sum(1,keepdims=True)
        tot = np.zeros((32, 4))
        tot[:,0] = norm[:,30]
        tot[:,1] = norm[:,:1023].sum(1)-norm[:,30]
        tot[:,2] = norm[:,1023]
        tot[:,3] = norm[:,1024]
        cum = np.cumsum(tot,axis=1)
        cum /= cum[:,-1:]
        cum = np.vstack((cum, cum[-1,:]))
        # ---------------------------------------------------

        xs = np.arange(33)
        labels = ['32nd token', 'internal image', 'last image', 'EOI']

    elif model == 'facebook/chameleon-30b':

        # Preprocessing of raw data--------------------------
        norm = dict_data['mean'][:,2:] / dict_data['mean'][:,2:].sum(1,keepdims=True)
        tot = np.zeros((48, 4))
        tot[:,0] = norm[:,0]
        tot[:,1] = norm[:,1:1023].sum(1)
        tot[:,2] = norm[:,1023]
        tot[:,3] = norm[:,1024]
        cum = np.cumsum(tot,axis=1)
        cum /= cum[:,-1:]
        cum = np.vstack((cum, cum[-1,:]))
        # ---------------------------------------------------

        xs = np.arange(49)
        labels = ['1st token', 'internal image', 'last image', 'EOI']

    elif model == 'mistral-community/pixtral-12b':

        # Preprocessing of raw data--------------------------
        norm = dict_data['mean'][:,1:] / dict_data['mean'][:,1:].sum(1,keepdims=True)
        tot = np.zeros((40, 4))
        tot[:,0] = norm[:,32:-1:33].sum(1)
        tot[:,1] = norm[:,:-1].sum(1) - norm[:,1024] - tot[:,0]
        tot[:,2] = norm[:,1024]
        tot[:,3] = norm[:,-1]
        cum = np.cumsum(tot,axis=1)
        cum /= cum[:,-1:]
        cum = np.vstack((cum, cum[-1,:]))
        # ---------------------------------------------------

        xs = np.arange(41)
        labels = ['[EOL]s', 'internal image', '1025th token', 'EOI']

    plt.grid()
    plt.xlim(0,len(xs))
    plt.ylim(0,1)

    colors = [
            "#4477AA",
            "#EE6677",
            "#228833",
            "#1187AA",
            "#66CCEE",
            "#AA3377",
            "#BBBBBB",
        ]

    plt.title('Token relevance in crossmodal communication - Chameleon-7B')
    plt.ylabel('Cumulative crossmodal attention')
    plt.xlabel('Layer')
    plt.fill_between(xs, 0,        cum[:,0], label=labels[0], step='post', color=colors[3])
    plt.fill_between(xs, cum[:,0], cum[:,1], label=labels[1], step='post', color=colors[0])
    plt.fill_between(xs, cum[:,1], cum[:,2], label=labels[2], step='post', color=colors[1])
    plt.fill_between(xs, cum[:,2], cum[:,3], label=labels[3], step='post', color=colors[2])

    plt.step(xs, cum[:,0], color='black', where='post')
    plt.step(xs, cum[:,1], color='black', where='post')
    plt.step(xs, cum[:,2], color='black', where='post')


    plt.legend()
    plt.savefig(save_path)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-o", "--out_dir", type=str, default="out")

    args = parser.parse_args()

    print("Extracting text hidden vectors...")
    config = ExtractActivationConfig(
        model_name=args.model,
        input="imagenet-text",
        dataset_hf_name="AnonSubmission/dataset_2",
        split="train",
        token=["random-text"],
        num_proc=4,
    )
    
    model = Extractor(config)
    
    cache = model.extract_cache(
        extract_avg_pattern=True,
    )
    
    cross_attn = cross_attention_patterns(args.model, cache)
    
    plot_cross_attention(args.model, cross_attn, Path(f"{args.out_dir}",f"cross_attn_{args.model.split('/')[1]}.pdf"))