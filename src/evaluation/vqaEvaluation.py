# coding=utf-8

__author__ = 'aagrawal'

import re
import sys
from typing import List, Literal, Optional
import pandas as pd

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at:
# https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py

class VQAEval:
    def __init__(self,
                 vqa: pd.DataFrame,
                 vqaRes: pd.DataFrame,
                 mode: Literal["prompt", "zero-shot", "few-shot"] = "prompt",
                 n: int = 2):
        self.n            = n
        self.accuracy     = {}
        self.evalQA       = {}
        self.evalQuesType = {}
        self.evalAnsType  = {}
        self.vqa          = vqa
        self.vqaRes       = vqaRes
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
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
        
        self.manualMap    = {
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
        self.articles     = ['a', 'an', 'the']

        self.periodStrip  = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile(r"(\d)(,)(\d)")
        self.punct        = [
            ';', r"/", '[', ']', '"', '{', '}',
            '(', ')', '=', '+', '\\', '_', '-',
            '>', '<', '@', '`', ',', '?', '!'
        ]
        self.mode = mode


    def evaluate(self, quesIds=None):
        if quesIds is None:
            quesIds = self.vqaRes["idx"].tolist()
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.iloc[quesId]
            res[quesId] = self.vqaRes.iloc[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA       = []
        accQuesType = {}
        accAnsType  = {}
        print("Computing accuracy...")
        step = 0

        for quesId in quesIds:
            for ansDic in gts[quesId]['answers']:
                ansDic['answer'] = ansDic['answer'].replace('\n', ' ').replace('\t', ' ').strip()
            resAns = res[quesId]['answer']
            resAns = resAns.replace('\n', ' ').replace('\t', ' ').strip()
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]

            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(ansDic['answer'])
                    ansDic['answer'] = self.processDigitArticle(ansDic['answer'])
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)

            for gtAnsDatum in gts[quesId]['answers']:
                otherGTAns = [item for item in gts[quesId]['answers'] if item != gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item['answer'] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            quesType    = gts[quesId]['question_type']
            ansType     = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step % 100 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step += 1

        self.setAccuracy(accQA, accQuesType, accAnsType)
        print("\nDone computing accuracy.")

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) is not None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.get(word, word)
            if word not in self.articles:
                outText.append(word)
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall'] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        self.accuracy['perQuestionType'] = {
            quesType: round(100 * float(sum(scores)) / len(scores), self.n)
            for quesType, scores in accQuesType.items()
        }
        self.accuracy['perAnswerType'] = {
            ansType: round(100 * float(sum(scores)) / len(scores), self.n)
            for ansType, scores in accAnsType.items()
        }

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if not isinstance(progress, float):
            progress = float(progress)
        if progress < 0:
            progress = 0.0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1.0
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = f"\rFinished Percent: [{'#' * block + '-' * (barLength - block)}] {int(progress * 100)}% {status}"
        sys.stdout.write(text)
        sys.stdout.flush()
