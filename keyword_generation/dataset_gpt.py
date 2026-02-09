import os
import json
import asyncio
from datasets import load_dataset
from PIL import Image
from . import async_gpt_utils

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def PIL_to_base64(image):
    import io
    import base64

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def load_image(img_ids, root_path):
    if isinstance(img_ids, str):
        img_ids = [img_ids]
    images = []
    image_paths = []
    for img_id in img_ids:
        image_path = os.path.join(root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        images.append(image)
        image_paths.append(image_path)
    
    return images, image_paths

def construct_prompt(question, options, dataSet):
    if len(options):
        return question + " The options are the following:" + str().join([ALPHABET[i] + ". " + options[i] + ". " for i in range(len(options))]) + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible."
    else:
        if dataSet == 'clevr':
            return question + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <NUMBER>."
        elif dataSet == 'textocr':
            return question + " Only answer with the largest text. Please include your reasoning steps, then answer your choice in this format: ANSWER: <TEXT>."

def dataset_gpt(dataSet, dataSlice):
    dataSet = dataSet # choose from mmmu, clevr, textocr
    dataSlice = dataSlice # choose from dev, val
    dataDir = '../data'

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
    validation_dataset = support_meta
    #print(validation_dataset[0])

    # define output directory
    result_path = os.path.join(dataDir, dataSet, f'{dataSet}_{dataSlice}_gpt4o_response_v2.jsonl')

    # go through data
    bs = 30
    for i in range(0, len(validation_dataset), bs):
        print(f"Processing {i} to {min(i+bs, len(validation_dataset))}")
        batch = validation_dataset[i:min(i+bs, len(validation_dataset))]
        # images_base64_list_of_list = [[PIL_to_base64(each[f'image_{i}']) for i in range(1, 8) if each[f'image_{i}'] is not None] for each in batch]
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
        # options = ast.literal_eval(batch['options'])
        
        batch_respose = asyncio.run(async_gpt_utils.get_vision_completion_multi_image_list(images_base64_list_of_list, query_list, max_parallel_calls=10, model="gpt-4o", task_name=dataSet))

        # save the completion
        with open(result_path, 'a') as f:
            for each in batch_respose:
                completion = each.choices[0].message.content
                f.write(json.dumps(completion) + '\n')
