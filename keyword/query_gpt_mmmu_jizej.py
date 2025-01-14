import json
import os
from pathlib import Path
from pprint import pprint
import ast
import asyncio

import openai
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image

import async_gpt_utils

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def PIL_to_base64(image):
    import io
    import base64

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# load API from system environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# load dataset
validation_dataset = load_dataset("lmms-lab/MMMU", split="validation")

# define prompt template for MMMU
# mmmu_gpt_template = each['question'] + " The options are the following:" + str().join([ALPHABATES[i] + ". " + options[i] + ". " for i in range(len(options))])

# define output directory
output_dir = Path('/home/jizej/Workspaces/UOUO/keyword/mmmu')
result_path = output_dir / 'mmmu_val_gpt4o_response_v2.jsonl'

def construct_prompt(question, options):
    if len(options):
        return question + " The options are the following:" + str().join([ALPHABET[i] + ". " + options[i] + ". " for i in range(len(options))]) + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible."
    else:
        return question + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible."

# go through data
bs = 30
for i in range(0, len(validation_dataset), bs):
    print(f"Processing {i} to {min(i+bs, len(validation_dataset))}")
    batch = validation_dataset[i:min(i+bs, len(validation_dataset))]
    # images_base64_list_of_list = [[PIL_to_base64(each[f'image_{i}']) for i in range(1, 8) if each[f'image_{i}'] is not None] for each in batch]
    images_base64_list_of_list = [[PIL_to_base64(batch[f'image_{i}'][j]) for i in range(1, 8) if batch[f'image_{i}'][j] is not None] for j in range(bs)]
    # options = ast.literal_eval(batch['options'])
    options_list = [ast.literal_eval(batch['options'][j]) for j in range(bs)]
    query_list = [construct_prompt(batch['question'][j], options_list[j]) for j in range(bs)]
    batch_respose = asyncio.run(async_gpt_utils.get_vision_completion_multi_image_list(images_base64_list_of_list, query_list, max_parallel_calls=10, model="gpt-4o", task_name="mmmu"))

    # save the completion
    with result_path.open('a') as f:
        for each in batch_respose:
            completion = each.choices[0].message.content
            f.write(json.dumps(completion) + '\n')