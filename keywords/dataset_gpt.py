import os
import json
import asyncio
from datasets import load_dataset
from utils import async_gpt_utils
from utils.other_utils import ALPHABET, PIL_to_base64, load_image, construct_prompt

def dataset_gpt(dataSet, dataSlice):
    dataSet = dataSet # choose from mmmu, clevr, textocr
    dataSlice = dataSlice # choose from dev, val
    dataDir = './data'

    if dataSet == 'mmmu':
        if dataSlice == 'val':
            validation_dataset = load_dataset("lmms-lab/MMMU", split="validation")
        else:
            validation_dataset = load_dataset("lmms-lab/MMMU", split="dev")
    else:
        if dataSlice == 'val':
            support_file = os.path.join(dataDir, dataSet, 'support.json')
        else:
            support_file = os.path.join(dataDir, dataSet, 'query.json')

    # load dataset
    with open(support_file, 'r') as f:
            support_meta = json.load(f)
    validation_dataset = support_meta[0:5]

    # define output directory
    result_path = os.path.join(dataDir, dataSet, dataSlice, f'{dataSet}_{dataSlice}_gpt4o_response_v2.jsonl')

    # go through data
    bs = 30
    for i in range(0, len(validation_dataset), bs):
        print(f"Processing {i} to {min(i+bs, len(validation_dataset))}")
        batch = validation_dataset[i:min(i+bs, len(validation_dataset))]
        if dataSet == 'mmmu':
            images_base64_list_of_list = [[PIL_to_base64(batch[f'image_{i}'][j]) for i in range(1, 8) if batch[f'image_{i}'][j] is not None] for j in range(bs)]
            query_list = [construct_prompt(batch[j]['question'], []) for j in range(bs)]
        else:
            images_base64_list_of_list = []
            query_list = []
            for j in range(min(bs,len(batch))):
                image_PIL, _ = load_image(batch[j]['image'], dataDir)
                images_base64_list_of_list.append(PIL_to_base64(image_PIL[0]))
                if dataSet == 'clevr':
                    property_name, exact_name  = batch[j]['question'].split(': ')
                    prompt = f'How many objects in the image have the {exact_name} {property_name}'
                elif dataSet == 'textocr':
                    prompt = batch[j]['question']
                query_list.append(construct_prompt(prompt, [], dataSet))
        
        batch_respose = asyncio.run(async_gpt_utils.get_vision_completion_multi_image_list(images_base64_list_of_list, query_list, max_parallel_calls=10, model="gpt-4o", task_name=dataSet))

        # save the completion
        with open(result_path, 'a') as f:
            for each in batch_respose:
                completion = each.choices[0].message.content
                f.write(json.dumps(completion) + '\n')
