import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="pixtral, chameleon30b, chameleon7b, emu3", default = "chameleon30b")
    args = parser.parse_args()
    results = json.load(open(f'{args.model}.json'))
    ground_truth=[res['ground_truth'] for res in results]
    answers = [res['answer'] for res in results]
    mapping = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    ground_truth = [mapping[x] for x in ground_truth]
    answers = [answers[i].split(':')[-1].strip('"').strip(' ').split(' ')[0].strip('"').strip('.').strip('\n') for i in range(len(answers))]
    count = 0
    for i in range(len(answers)):
        if answers[i]==ground_truth[i]:
            count += 1
    print(count/len(answers))


    
