#!/usr/bin/env python

import re
import sys
from typing import List, Literal, Optional
import pandas as pd
import torch 
import numpy as np
import tqdm
from datasets import load_dataset
import argparse
from pathlib import Path
import json

from datasets import load_dataset
dataset = load_dataset("lmms-lab/VQAv2", split="validation")
# dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation")


def score_pixtral(model_answer: str, reference_answer: str | list[str]) -> float:
    if not isinstance(reference_answer, list):
        reference_answer = [reference_answer]
    normalize_response_text: str = normalize_string(model_answer)
    matching_answers = [
        answer
        for answer in reference_answer
        if normalize_string(answer) == normalize_response_text
    ]
    return min(1.0, float(len(matching_answers)) / 3)

def normalize_string(s):
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}
        
manualMap    = {
            'none': '0',
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10'
        }
articles     = ['a', 'an', 'the']

periodStrip  = re.compile(r"(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile(r"(\d)(,)(\d)")
punct        = [
            ';', r"/", '[', ']', '"', '{', '}',
            '(', ')', '=', '+', '\\', '_', '-',
            '>', '<', '@', '`', ',', '?', '!'
        ]

def processPunctuation(inText, punct, commaStrip, periodStrip):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText)
    return outText

def processDigitArticle(inText, manualMap, articles, contractions):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.get(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def evaluate_VQA(vqa, vqaRes, quesIds=None):
    if quesIds is None:
        quesIds = vqaRes["question_id"].tolist()
        
    gts = {}
    res = {}
    for quesId in quesIds:
        gts[quesId] = vqa[vqa["question_id"]==quesId].to_dict(orient="records")[0]
        res[quesId] = vqaRes[vqaRes["question_id"] == quesId].to_dict(orient="records")[0]

    accQA       = []

    for quesId in quesIds:
        for ansDic in gts[quesId]['answers']:
            ansDic['answer'] = ansDic['answer'].replace('\n', ' ').replace('\t', ' ').strip()
        resAns = res[quesId]['answer']
        resAns = resAns.replace('\n', ' ').replace('\t', ' ').strip()
        gtAcc = []
        gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]

        #if len(set(gtAnswers)) > 1:
        for ansDic in gts[quesId]['answers']:
            ansDic['answer'] = processPunctuation(ansDic['answer'], punct, commaStrip, periodStrip)
            ansDic['answer'] = processDigitArticle(ansDic['answer'], manualMap, articles, contractions)
        resAns = processPunctuation(resAns, punct, commaStrip, periodStrip)
        resAns = processDigitArticle(resAns, manualMap, articles, contractions)
        
        for gtAnsDatum in gts[quesId]['answers']:
            otherGTAns = [item for item in gts[quesId]['answers'] if item != gtAnsDatum]
            matchingAns = [item for item in otherGTAns if item['answer'] == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
        accQA.append(avgGTAcc)
            
    return np.mean(accQA)



if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("-f", "--filename", type=str,
                         help="", default = "chameleon30b_10000_HF.json")
     args = parser.parse_args()
    
     with open(args.filename, 'r') as file:
         res = json.load(file)

    
     ev = {}
     path = '/'.join(args.filename.split('/')[:-1])
     name = args.filename.split('/')[-1]
     model = name.split('_')[0]
     N = name.split('_')[1]
     mode = name.split('_')[2].split('.')[0]

     if model == 'pixtral' and mode == 'prompt':
         for i in range(len(res)):
             res[i]['answer'] = res[i]['answer'][res[i]['answer'].find('You must answer.')+16:]
     elif model == 'pixtral' and mode == 'zero-shot':
         for i in range(len(res)):
             res[i]['answer'] = res[i]['answer'][res[i]['answer'].find('phrase."')+8:]

     elif 'chameleon' in model and mode == 'prompt':
         for i in range(len(res)):
             res[i]['answer'] = res[i]['answer'][res[i]['answer'].find('You must answer.')+16:]
     elif 'chameleon' in model  and mode == 'zero-shot':
         for i in range(len(res)):
             res[i]['answer'] = res[i]['answer'][res[i]['answer'].find("Answer:")+8:]

     elif model == 'emu3' and mode == 'prompt':
         for i in range(len(res)):
             res[i]['answer'] = res[i]['answer'][res[i]['answer'].find("ASSISTANT:")+10:]
     elif model == 'emu3' and mode == 'zero-shot':
         for i in range(len(res)):
             res[i]['answer'] = res[i]['answer'][res[i]['answer'].find("ASSISTANT:")+10:]

     res = pd.DataFrame(res)
     print(res)
     dataset = dataset.to_pandas()
    
     accuracy_VQA = evaluate_VQA(dataset, pd.DataFrame(res))
     print('Accuracy VQA:', accuracy_VQA, '\n')
     ev['VQA'] = accuracy_VQA
    
     accuracy_pixtral = []
     for index, row in res.iterrows():
         all_ans = dataset[dataset["question_id"]==row["question_id"]]['answers'].tolist()
         ans = [i['answer'] for i in all_ans[0]]
         model_ans = row['answer'].strip(' ').strip('.').lower().strip("'\'").strip('""').strip(' ')
         s_pixtral = score_pixtral(model_answer = model_ans, 
                                   reference_answer = ans)
         accuracy_pixtral.append(s_pixtral)
     print('Accuracy Pixtral:', np.mean(accuracy_pixtral), '\n')    
     ev['pixtral'] = accuracy_VQA

     with open(f'{path}/{model}_{N}_{mode}_eval', 'w') as f: 
         for key, value in ev.items():  
              f.write('%s\t%s\n' % (key, value))
