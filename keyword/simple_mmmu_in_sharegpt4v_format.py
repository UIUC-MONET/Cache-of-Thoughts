"""
Creates the MMMU dataset in the sharegpt4v format. (a.k.a. fake conversation that included the answer to the question)
ONLY run this on the validation set!!
"""

import json
import os
import ast
from pathlib import Path
from pprint import pprint

from PIL import Image
from datasets import load_dataset

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# load dataset
validation_dataset = load_dataset("lmms-lab/MMMU", split="validation")
output_dir = Path('/home/jizej/Workspaces/UOUO/keyword/mmmu')
image_dir = output_dir / 'images' / 'val'

# define prompt template for MMMU
def construct_prompt(question, options):
    if len(options):
        return question + " The options are the following:" + str().join([ALPHABET[i] + ". " + options[i] + ". " for i in range(len(options))]) + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible."
    else:
        return question + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible."

out_data = []
# go through the dataset
for each in validation_dataset:
    # parse string to list
    options = ast.literal_eval(each['options'])


    obj = {
        "conversation": [
            {
                "from": "user",
                "value": construct_prompt(each['question'], options)
            },
            {
                "from": "gpt",
                "value": "The answer is: " + each['answer'] + "."
            }
        ],
        'id': each['id'],
        'image_1': each['image_1'],
        'image_2': each['image_2'],
        'image_3': each['image_3'],
        'image_4': each['image_4'],
        'image_5': each['image_5'],
        'image_6': each['image_6'],
        'image_7': each['image_7'],
    }

    # dump PIL to path
    for i in range(1, 8):
        if each[f'image_{i}'] is None:
            continue
        each[f'image_{i}'].save(image_dir / f'{each["id"]}_image_{i}.png')
        obj[f'image_{i}'] = f'{each["id"]}_image_{i}.png'

    out_data.append(obj)

# write to file
with open(output_dir / 'mmmu_val_simple_sharegpt4v_v1.jsonl', 'w') as f:
    for each in out_data:
        f.write(json.dumps(each) + '\n')
