import os
import sys
from pathlib import Path
import json
import os
from pathlib import Path
import asyncio
import openai
from torch.utils.data import DataLoader
from PIL import Image
import async_gpt_utils

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

server_name = 'ryan'
dataSet = 'textocr' # choose from mmmu, clevr, textocr

# Add the inference directory to the PYTHONPATH
if server_name == 'monet':
    som_path = '/home/monet/meitang/cache-of-thoughts/inference'
    dataDir = '/home/monet/meitang/cache-of-thoughts/data'
    os.environ['HF_HOME'] = '/mnt/data/meitang/.cache/huggingface'
elif server_name == 'ryan':
    som_path = '/home/ryan/meitang/cache-of-thoughts-main/inference'
    dataDir = '/home/ryan/meitang/cache-of-thoughts-main/data'
    os.environ['HF_HOME'] = '/home/ryan/.cache/huggingface'
elif server_name == 'meitang':
    som_path = '/Users/17348/Documents/GitHub/cache-of-thoughts/inference'
    dataDir = '/Users/17348/Documents/GitHub/cache-of-thoughts/data'
    os.environ['HF_HOME'] = '/Users/17348/.cache/huggingface'
else:
    pass # modify accordingly

if som_path not in sys.path:
    sys.path.append(som_path)

os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + f":{som_path}"
os.environ["OPENAI_API_KEY"] = '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'


support_file = os.path.join(dataDir, dataSet, 'support.json')
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# load API from system environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# load dataset
with open(support_file, 'r') as f:
        support_meta = json.load(f)
validation_dataset = support_meta
print(validation_dataset[0])

# define output directory
result_path = os.path.join(dataDir, dataSet, f'{dataSet}_val_gpt4o_response_v2.jsonl')

# go through data
bs = 30
for i in range(0, len(validation_dataset), bs):
    print(f"Processing {i} to {min(i+bs, len(validation_dataset))}")
    batch = validation_dataset[i:min(i+bs, len(validation_dataset))]
    # images_base64_list_of_list = [[PIL_to_base64(each[f'image_{i}']) for i in range(1, 8) if each[f'image_{i}'] is not None] for each in batch]
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
    #options_list = [ast.literal_eval(batch['options'][j]) for j in range(bs)]
    #query_list = [construct_prompt(batch[j]['question'], []) for j in range(bs)]
    batch_respose = asyncio.run(async_gpt_utils.get_vision_completion_multi_image_list(images_base64_list_of_list, query_list, max_parallel_calls=10, model="gpt-4o", task_name=dataSet))

    # save the completion
    with result_path.open('a') as f:
        for each in batch_respose:
            completion = each.choices[0].message.content
            f.write(json.dumps(completion) + '\n')
