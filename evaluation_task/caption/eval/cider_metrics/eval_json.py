#!/usr/bin/env python
import json
from evaluation_task.caption.eval.cider_metrics.cider import Cider
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str,
                        help="", default = "")
    args = parser.parse_args()
    
    with open(args.filename, 'r') as file:
        res = json.load(file)

    ev = {}
    path = '/'.join(args.filename.split('/')[:-1])
    name = args.filename.split('/')[-1]
    model = name.split('_')[0]
    N = name.split('_')[1].split('.')[0]
    
    ground_truth = {i: d["caption"] for i, d in enumerate(res)}
    
    response = {}
    if model == 'pixtral':
        for  i, d in enumerate(res):
            response[i] = [d['result'][d['result'].find('image.')+6:]]
    
    elif model == 'chameleon-7b' or model == 'chameleon-30b':
        for  i, d in enumerate(res):
            response[i] = [d['result'][d['result'].find('image.\n\n')+8:]]
    
    elif model == 'emu3':
        for i, d in enumerate(res):
            response[i] = [d['result'][d['result'].find('ASSISTANT:')+11:]]
    
    
    score, scores = Cider(backend="pycocoeval").compute_score(ground_truth, response)
    print('Cider score pycocoeval =', score)
    ev['pycocoeval'] = score
    
    
    score, scores = Cider(backend="original").compute_score(ground_truth, response)
    print('Cider score original =', score)
    ev['original'] = score
    
    with open(f'{path}/{model}_{N}_eval', 'w') as f: 
        for key, value in ev.items():  
            f.write('%s\t%s\n' % (key, value))

    