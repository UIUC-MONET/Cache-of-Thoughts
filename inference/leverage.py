import os
import random
from pathlib import Path
import re
import pickle
import json
from datasets import load_dataset
import math
from utils.other_utils import parse_option, get_latest_timestamp

def leverage(dataSet, dataSlice, cacheSet, cacheSlice, modelsize, time=None):
    dataSet = dataSet # choose from mmmu, clevr, textocr
    dataSlice = dataSlice # choose from val, dev
    cacheSet = cacheSet # choose from mmmu, clevr, textocr
    cacheSlice = cacheSlice # choose from val, dev
    model_name = modelsize # choose from 2B, 7B, 72Bint4
    if time == None:
        time = get_latest_timestamp('./results/')
    else:
        time = time
    result_read_path = f'./results/Qwen_{model_name}_{dataSet}_{dataSlice}_{cacheSet}_{cacheSlice}_image_results_{time}.jsonl'
    dataDir = './data'

    if dataSet == 'mmmu':
        # load base dataset
        if dataSlice == 'val':
            test_dataset = load_dataset("lmms-lab/MMMU", split="validation")
        elif dataSlice == 'dev':
            test_dataset = load_dataset("lmms-lab/MMMU", split="dev")
        else:
            test_dataset = load_dataset("lmms-lab/MMMU", split="test")
        test_dataset_single_image = test_dataset.filter(lambda x: x['image_2'] is None)
        test_dataset_single_image = test_dataset_single_image.shuffle(seed=42)
    else:
        if dataSlice == 'val':
            data_file = os.path.join(dataDir, dataSet, 'support.json')
        else:
            data_file = os.path.join(dataDir, dataSet, 'query.json')
        with open(data_file, 'r') as f:
            query_meta = json.load(f)
        test_dataset_single_image = query_meta
        random.shuffle(test_dataset_single_image)
        
    # open pickle
    with open(result_read_path, 'rb') as f:
        result_objs = pickle.load(f)

    # evaluate results
    llm_first_choice_list = []
    llm_second_choice_list = []
    if dataSet == 'mmmu':
        for each in result_objs:
            llm_first_choice_list.append(parse_option(each['first_vlm_response'][0]))
            llm_second_choice_list.append(parse_option(each['second_vlm_response'][0]))
    elif dataSet == 'clevr':
        for each in result_objs:
            vlm_ans = each['first_vlm_response'][0].lower().split('answer: ')[-1].strip().strip('."')
            vlm_ans = re.findall(r'\d+', vlm_ans)
            if vlm_ans:      
                llm_first_choice_list.append(vlm_ans[0])
            else:
                llm_first_choice_list.append(None)
            vlm_ans = each['second_vlm_response'][0].lower().split('answer: ')[-1].strip().strip('."')
            vlm_ans = re.findall(r'\d+', vlm_ans)
            if vlm_ans:      
                llm_second_choice_list.append(vlm_ans[0])
            else:
                llm_second_choice_list.append(None)
    elif dataSet == 'textocr':
        for each in result_objs:
            ans_text = each['first_vlm_response'][0].lower().replace('<|im_end|>','')
            ans_text = ans_text.split('answer:')
            if len(ans_text) > 1:
                ans_text = ans_text[1]
            else:
                ans_text = ans_text[0]
            ans_text = ans_text.split(' is:')
            if len(ans_text) > 1:
                text = ans_text[1]
            else:
                ans_text = ans_text[0].split(' is')
                if len(ans_text) > 1:
                    text = ans_text[1]
                else:
                    text = ans_text[0]
            text = text.replace(' ','').replace('"','').replace('.','')
            llm_first_choice_list.append(text)

            ans_text = each['second_vlm_response'][0].lower().replace('<|im_end|>','')
            ans_text = ans_text.split('answer:')
            if len(ans_text) > 1:
                ans_text = ans_text[1]
            else:
                ans_text = ans_text[0]
            ans_text = ans_text.split(' is:')
            if len(ans_text) > 1:
                text = ans_text[1]
            else:
                ans_text = ans_text[0].split(' is')
                if len(ans_text) > 1:
                    text = ans_text[1]
                else:
                    text = ans_text[0]
            text = text.replace(' ','').replace('"','').replace('.','')
            llm_second_choice_list.append(text)

    # compare model response with the ground truth
    vllm_count = 0
    for each in result_objs:
        if 'icl_example' in each:
            vllm_count += 1
    slice_size = math.floor(vllm_count/3)
    vllm_count = 0
    for idx,each in enumerate(result_objs):
        if vllm_count == slice_size:
            first_idx = idx
        elif vllm_count == slice_size * 2:
            second_idx = idx
        if 'icl_example' in each:
            vllm_count += 1

    #first one-third:
    one_third = test_dataset_single_image['answer'][:first_idx]
    one_third_results = result_objs[:first_idx]
    one_third_first = llm_first_choice_list[:first_idx]
    one_third_second = llm_second_choice_list[:first_idx]
    first_vllm_correct_count = 0
    second_vllm_correct_count = 0
    for idx, each in enumerate(one_third):
        if 'icl_example' in one_third_results[idx]:
            if str(one_third_first[idx]).lower() == str(each).lower():
                first_vllm_correct_count += 1
            if str(one_third_second[idx]).lower() == str(each).lower():
                second_vllm_correct_count += 1
    # accuracy
    print("Apprentice's accuracy before and after learning in the first one-third of the questions:")
    print(first_vllm_correct_count/slice_size)
    print(second_vllm_correct_count/slice_size)

    #second one-third:
    one_third = test_dataset_single_image['answer'][first_idx:second_idx]
    one_third_results = result_objs[first_idx:second_idx]
    one_third_first = llm_first_choice_list[first_idx:second_idx]
    one_third_second = llm_second_choice_list[first_idx:second_idx]
    first_vllm_correct_count = 0
    second_vllm_correct_count = 0
    for idx, each in enumerate(one_third):
        if 'icl_example' in one_third_results[idx]:
            if str(one_third_first[idx]).lower() == str(each).lower():
                first_vllm_correct_count += 1
            if str(one_third_second[idx]).lower() == str(each).lower():
                second_vllm_correct_count += 1
    # accuracy
    print("Apprentice's accuracy before and after learning in the second one-third of the questions:")
    print(first_vllm_correct_count/slice_size)
    print(second_vllm_correct_count/slice_size)

    #last one-third:
    one_third = test_dataset_single_image['answer'][second_idx:]
    one_third_results = result_objs[second_idx:]
    one_third_first = llm_first_choice_list[second_idx:]
    one_third_second = llm_second_choice_list[second_idx:]
    first_vllm_correct_count = 0
    second_vllm_correct_count = 0
    for idx, each in enumerate(one_third):
        if 'icl_example' in one_third_results[idx]:
            if str(one_third_first[idx]).lower() == str(each).lower():
                first_vllm_correct_count += 1
            if str(one_third_second[idx]).lower() == str(each).lower():
                second_vllm_correct_count += 1
    # accuracy
    print("Apprentice's accuracy before and after learning in the last one-third of the questions:")
    print(first_vllm_correct_count/(vllm_count-2*slice_size))
    print(second_vllm_correct_count/(vllm_count-2*slice_size))

    # compare model response with the ground truth
    first_correct_count = 0
    second_correct_count = 0
    vllm_count = 0
    first_vllm_correct_count = 0
    second_vllm_correct_count = 0
    for idx, each in enumerate(test_dataset_single_image):
        if not result_objs[idx]['id'] == each['id']:
            print(idx)
        if 'icl_example' in result_objs[idx]:
            vllm_count += 1
            if str(llm_first_choice_list[idx]).lower() == str(each['answer']).lower():
                first_correct_count += 1
                first_vllm_correct_count += 1
            if str(llm_second_choice_list[idx]).lower() == str(each['answer']).lower():
                second_correct_count += 1
                second_vllm_correct_count += 1
        else:
            if str(llm_first_choice_list[idx]).lower() == str(each['answer']).lower():
                first_correct_count += 1
            if str(llm_second_choice_list[idx]).lower() == str(each['answer']).lower():
                second_correct_count += 1

    # accuracy
    print(first_correct_count / len(test_dataset_single_image))
    print(second_correct_count / len(test_dataset_single_image))
    print(first_vllm_correct_count/vllm_count)
    print(second_vllm_correct_count/vllm_count)


