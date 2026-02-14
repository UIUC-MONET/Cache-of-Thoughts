import re
import os
import json
from datasets import load_dataset
import ast
from utils.other_utils import parse_option

def dataset_gpt_eval(dataSet, dataSlice):
    dataSet = dataSet # choose from mmmu, clevr, textocr
    dataSlice = dataSlice # choose from val, dev
    dataDir = './data'
    result_path = os.path.join(dataDir, dataSet, f'{dataSlice}/{dataSet}_{dataSlice}_gpt4o_response_v2.jsonl')

    if dataSet == 'mmmu':
        if dataSlice == 'val':
            validation_dataset = load_dataset("lmms-lab/MMMU", split="validation")
        else:
            validation_dataset = load_dataset("lmms-lab/MMMU", split="dev")
        # read the response file
        with result_path.open('r') as f:
            responses = [json.loads(line) for line in f]
        lens = []
        for each in validation_dataset:
            options = ast.literal_eval(each['options'])
            lens.append(len(options))
        gpt_answer = [parse_option(response) for response in responses]
        correct = 0
        for i in range(len(gpt_answer)):
            if gpt_answer[i] == validation_dataset[i]['answer']:
                correct += 1
        accuracy = correct / len(gpt_answer)
        print(accuracy)
    else:
        if dataSlice == 'val':
            support_file = os.path.join(dataDir, dataSet, 'support.json')
            with open(support_file, 'r') as f:
                support_meta = json.load(f)
        else:
            support_file = os.path.join(dataDir, dataSet, 'query.json')
            with open(support_file, 'r') as f:
                support_meta = json.load(f)

        with open(result_path, 'r') as f:
                gpt_results = f.readlines()
        cnt = 0
        for i in range(len(gpt_results)):
            ans = support_meta[i]['answer']
            if dataSet == 'clevr':
                gpt_ans = gpt_results[i].split('ANSWER: ')[-1].strip().strip('."')
                gpt_ans = re.findall(r'\d+', gpt_ans)[0]
                try:
                    if int(gpt_ans) == ans:
                        cnt += 1
                except:
                    print(gpt_ans)
            elif dataSet == 'textocr':
                gpt_ans = gpt_results[i].split('ANSWER: ')[-1].strip().strip('."').replace(' ','')
                try:
                    if gpt_ans.lower() == ans.lower():
                        cnt += 1
                except:
                    print(gpt_ans,'\n',ans,support_meta[i]['id'])
        print("GPT-4o accuracy:")
        print(cnt/len(gpt_results))
