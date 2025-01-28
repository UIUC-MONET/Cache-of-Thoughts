# %%
import os
import sys
import random
import time
from pathlib import Path

# current time
current_time = time.strftime("%Y%m%d%H%M%S")
server_name = 'monet'
dataSet = 'mmmu' # choose from mmmu, clevr, textocr
dataSlice = 'val' # choose from val, dev
query_embedding = 'image_query' # choose from image, image_query, image_response, image_response_subfield (mmmu only)
cacheSet = 'mmmu' # choose from mmmu, clevr, textocr
cacheSlice = 'dev' # choose from val, dev
alternative = '_baseline' # choose from _baseline, _subfield (mmmu only), or emtpy string
embedding = 'image_text' # choose from image, image_text
model_name = '7B' # choose from 2B, 7B, 72Bint4
k_shot = 1 # choose from 1, 2, 4...
dynamic = False # choose from True, False
random.seed(42)


# %%
# Add the inference directory to the PYTHONPATH
if server_name == 'monet':
    cache_path = f'/home/monet/meitang/cache-of-thoughts/inference/results/Qwen_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}_cache_{current_time}.pickle'
    fig_path = f'/home/monet/meitang/cache-of-thoughts/inference/results/Qwen_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}_cache_frequency_{current_time}.png'
    data_path = Path('/home/monet/meitang/cache-of-thoughts/data/mmmu')
    result_write_path = f'/home/monet/meitang/cache-of-thoughts/inference/results/Qwen_{model_name}_{dataSet}_{dataSlice}_{cacheSet}_{cacheSlice}_{embedding}_results_{current_time}.jsonl'
    som_path = '/home/monet/meitang/cache-of-thoughts/inference'
    dataDir = '/home/monet/meitang/cache-of-thoughts/data'
    os.environ['HF_HOME'] = '/mnt/data/meitang/.cache/huggingface'
elif server_name == 'ryan':
    cache_path = f'/home/ryan/meitang/cache-of-thoughts-main/inference/results/Qwen_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}_cache_{current_time}.pickle'
    fig_path = f'/home/ryan/meitang/cache-of-thoughts-main/inference/results/Qwen_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}_cache_frequency_{current_time}.png'
    data_path = Path('/home/ryan/meitang/cache-of-thoughts-main/data/mmmu/')
    result_write_path = f'/home/ryan/meitang/cache-of-thoughts-main/inference/results/Qwen_{model_name}_{dataSet}_{dataSlice}_{cacheSet}_{cacheSlice}_{embedding}_results_{current_time}.jsonl'
    som_path = '/home/ryan/meitang/cache-of-thoughts-main/inference'
    dataDir = '/home/ryan/meitang/cache-of-thoughts-main/data'
    os.environ['HF_HOME'] = '/home/ryan/.cache/huggingface'
else:
    pass # modify accordingly

if som_path not in sys.path:
    sys.path.append(som_path)

os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + f":{som_path}"

print(os.getenv('HF_HOME'))

os.environ["OPENAI_API_KEY"] = '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'


# %%
import pickle
# from model import HFModelGeneration
import base64
import io
import glob
import json

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import clip
import transformers
import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import hnsw_rag
from hnsw_rag import DataRecord
from clip_rag import CLIP_rag
from keyword_hashtag_rag import KeywordExtractor, KeywordEncoder
from vlm_rag import QwenVLM, OpenaiVLM, ModelMux
from data_utils import create_cold_start_dataset_from_preprocess
from mmmu_utils import construct_mmmu_prompt

# %%
if cacheSet == 'mmmu':
    if cacheSlice == 'val':
        dataset = load_dataset("lmms-lab/MMMU", split="validation")
    else:
        dataset = load_dataset("lmms-lab/MMMU", split=f"{cacheSlice}")
    gpt_conversation_path = data_path / f'{cacheSlice}/mmmu_{cacheSlice}_gpt4o_response_v2.jsonl'
    clip_image_embeddings_path = data_path / f'{cacheSlice}/clip/mmmu{alternative}_{cacheSlice}_{embedding}_clip_embd_cold_start.pkl'
    # test_gpt_conversation_path = data_path / 'test/gpt_desc/mmmu_val_gpt4o_response_v2.jsonl' #TODO: replace with real gpt conversation
    # test_clip_image_embeddings_path = data_path / 'test/clip/mmmu_test_image_clip_embd_cold_start.pkl'
    cache_start_dataset = create_cold_start_dataset_from_preprocess(dataset, gpt_conversation_path, clip_image_embeddings_path, cacheSet)
else:
    if cacheSlice == 'val':
        data_file = os.path.join(dataDir, cacheSet, 'support.json')
    else:
        data_file = os.path.join(dataDir, cacheSet, 'query.json')
    with open(data_file, 'r') as f:
        dataset = json.load(f)
    dataset = datasets.Dataset.from_list(dataset)
    empty_options = datasets.Dataset.from_dict({"options": ["[]" for i in range(len(dataset))]})
    dataset = datasets.concatenate_datasets([dataset, empty_options], axis=1)
    gpt_conversation_path = dataDir + f'/{cacheSet}/{cacheSlice}/{cacheSet}_{cacheSlice}_gpt4o_response_v2.jsonl'
    clip_image_embeddings_path = dataDir + f'/{cacheSet}/{cacheSlice}/clip/{cacheSet}{alternative}_{cacheSlice}_{embedding}_clip_embd_cold_start.pkl'
    # test_gpt_conversation_path = data_path / 'test/gpt_desc/mmmu_val_gpt4o_response_v2.jsonl' #TODO: replace with real gpt conversation
    # test_clip_image_embeddings_path = data_path / 'test/clip/mmmu_test_image_clip_embd_cold_start.pkl'
    cache_start_dataset = create_cold_start_dataset_from_preprocess(dataset, gpt_conversation_path, clip_image_embeddings_path, cacheSet)
if dataSet == 'mmmu':
    # load base dataset
    if dataSlice == 'val':
        test_dataset = load_dataset("lmms-lab/MMMU", split="validation")
    elif dataSlice == 'dev':
        test_dataset = load_dataset("lmms-lab/MMMU", split="dev")
    else:
        test_dataset = load_dataset("lmms-lab/MMMU", split="test")
    test_dataset_single_image = test_dataset.filter(lambda x: x['image_2'] is None)
    # sample 2000 examples from the test dataset
    test_dataset_single_image = test_dataset_single_image.shuffle(seed=42)
else:
    if dataSlice == 'val':
        data_file = os.path.join(dataDir, dataSet, 'support.json')
    else:
        data_file = os.path.join(dataDir, dataSet, 'query.json')
    with open(data_file, 'r') as f:
        query_meta = json.load(f)
    test_dataset_single_image = query_meta
    # sample 2000 examples from the test dataset
    random.shuffle(test_dataset_single_image)

# %%
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

def concat_conversation_json(conversation):
    out_str = 'Human: ' + conversation[0]['value'] + '\n' + 'Assistant: ' + conversation[1]['value'] + '\n'
    return out_str

# %%
# MMMU cold start
# TODO: put this in a .py file
from hnsw_rag import HNSW, DataRecord

dim = 512
#num_elements = 3126 * 16
hnsw_size = 3000
hnsw = HNSW(dim, hnsw_size, evict_method='random')
#data = np.float32(np.random.random((num_elements, dim)))
# stuff_list = [DataRecord(set(validation_dataset_single['hashtags'][i][2]), 
#                          clip_embeddings[i][0], 
#                          clip_embeddings[i][1], 
#                          clip_embeddings[i][0], 
#                          validation_dataset_single['image_1'][i]) for i in range(temp)]
#stuff_list = [DataRecord({str(i % 25)}, data[i,:], i) for i in range(num_elements)]

# hashtag_list = mmmu_start_dataset['hashtags']
if cacheSet == 'mmmu':
    image_list = cache_start_dataset['image_1']
else:
    image_list = cache_start_dataset['image']
text_list = cache_start_dataset['conversations']
# clip_image_embed_list = mmmu_start_dataset['clip_image_embed']

if cacheSet == 'mmmu':
    for i in tqdm(range(min(len(cache_start_dataset), hnsw_size)), total=hnsw_size):
        data_stuff = DataRecord(keywords=[''],
                                image_clip=cache_start_dataset['clip_image_embed'][i],
                                vector_data=cache_start_dataset['clip_image_embed'][i], 
                                text_data=text_list[i],
                                image_data=image_list[i],
                                text_clip=None)
        hnsw.insert_data(data_stuff)
        #print(stuff.keywords)
        #print(image_paths[i])
else:
    for i in tqdm(range(min(len(cache_start_dataset), hnsw_size)), total=hnsw_size):
        data_stuff = DataRecord(keywords=[''],
                                image_clip=cache_start_dataset['clip_image_embed'][i],
                                vector_data=cache_start_dataset['clip_image_embed'][i], 
                                text_data=text_list[i],
                                image_data=load_image(image_list[i],dataDir)[0][0],
                                text_clip=None)
        hnsw.insert_data(data_stuff)


# %%
multi_gpu = True

# model setup
# TODO: replace the default init parameters
# VLM
qwen_dict = {'2B':'Qwen/Qwen2-VL-2B-Instruct', '7B':'Qwen/Qwen2-VL-7B-Instruct', '72Bint4':'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4'}
vllm_name = qwen_dict[model_name]
vllm = QwenVLM(vllm_name)
gpt = OpenaiVLM('gpt-4o', use_env_api_key=True)

# CLIP model
clip_rag = CLIP_rag()

# keyword + hashtag extraction
keyword_extractor = KeywordExtractor()

# keyword + hashtag embedding
keyword_encoder = KeywordEncoder()

# mux for swtiching between GPT or small VLM
model_mux = ModelMux([gpt, vllm], [0.3, 0.7])

# %%
import ast
import time
# MAIN LOOP


# iterate through the test dataset/stream
result_objs = []
for idx, batch in enumerate(tqdm(test_dataset_single_image)):
    for each_query in [batch]: # TODO: sorry, no batch processing for now
        result_obj = {}
        result_obj['id'] = each_query['id']
        # get the image and the question
        if dataSet == 'mmmu':
            image_PIL = each_query['image_1'] # again, assuming single image
        else:
            image_PIL, _ = load_image(each_query['image'], dataDir)

        # FIRST query of VLM
        if dynamic:
            selected_model = model_mux()
        else:
            selected_model = vllm
            
        if dataSet == 'mmmu':
            options = ast.literal_eval(each_query['options'])
            prompt = construct_mmmu_prompt(each_query['question'], options)
            first_vlm_prompt = selected_model.apply_single_image_prompt(prompt, input_image=image_PIL)
            first_full_response, first_vlm_response = selected_model(input_text=first_vlm_prompt, 
                                                        input_images=[image_PIL], 
                                                        max_new_token=512, 
                                                        only_model_response=False)
        elif dataSet == 'clevr':
            property_name, exact_name  = each_query['question'].split(': ')
            prompt = f'How many objects in the image have the {exact_name} {property_name}. Please answer in the following format: ANSWER: <NUMBER>.'
            first_vlm_prompt = selected_model.apply_single_image_prompt(prompt, input_image=image_PIL[0])
            first_full_response, first_vlm_response = selected_model(input_text=first_vlm_prompt, 
                                                        input_images=image_PIL, 
                                                        max_new_token=512, 
                                                        only_model_response=False)
        elif dataSet == 'textocr':
            prompt = 'An image will be provided where a red box is drawn around the text of interest. Answer with the largest text inside the red box. Ensure that the transcription is precise, reflecting the exact characters, including letters, numbers, symbols.'
            first_vlm_prompt = selected_model.apply_single_image_prompt(prompt, input_image=image_PIL[0])
            first_full_response, first_vlm_response = selected_model(input_text=first_vlm_prompt, 
                                                        input_images=image_PIL, 
                                                        max_new_token=512, 
                                                        only_model_response=False)

        #print(first_vlm_response)
        result_obj['first_full_response'] = first_full_response
        result_obj['first_vlm_response'] = first_vlm_response
        
        if selected_model == gpt: # if using GPT, skip the retrieval step
            print('using GPT-4o, thus saving result to HNSW')
            if dataSet == 'mmmu':
                img = image_PIL
            else:
                img = image_PIL[0]
            first_image_clip_embed = clip_rag.encode_image(img).detach().cpu().numpy()
            text_to_cache = [{'from':'user', 'value':prompt}, {'from':'assistant', 'value':first_vlm_response[0]}]
            data_stuff = DataRecord(keywords=[''],
                                    image_clip=first_image_clip_embed,
                                    vector_data=first_image_clip_embed,
                                    text_data=text_to_cache,
                                    image_data=img,
                                    text_clip=None)
            hnsw.insert_data(data_stuff)
            result_obj['second_full_response'] = first_full_response 
            result_obj['second_vlm_response'] = first_vlm_response
        else:
            # get the keywords
            first_vlm_response_keywords = keyword_extractor.extract_keywords(keyword_extractor.apply_keyword_extract_template(first_full_response))
            first_vlm_response_keywords = first_vlm_response_keywords[0].split(':')[-1]
            first_vlm_response_keywords = [each.strip() for each in first_vlm_response_keywords.split(',')][:min(10, len(first_vlm_response_keywords))]
            first_vlm_response_keywords = ', '.join(first_vlm_response_keywords)
            #print(first_vlm_response_keywords)

            # get the image path
            # get the clip embeddings
            if query_embedding == 'image_query':
                if dataSet == 'mmmu':
                    clip_image_embedding, clip_keyword_embedding = clip_rag.encode_image_text(image_PIL, first_vlm_prompt[0]['content'][0]['text'])
                else:
                    clip_image_embedding, clip_keyword_embedding = clip_rag.encode_image_text(image_PIL[0], first_vlm_prompt[0]['content'][0]['text'])
            else:
                if dataSet == 'mmmu':
                    if query_embedding == 'image_response_subfield':
                        clip_image_embedding, clip_keyword_embedding = clip_rag.encode_image_text(image_PIL, first_vlm_response_keywords+', '+each_query['subfield'])
                    else:
                        clip_image_embedding, clip_keyword_embedding = clip_rag.encode_image_text(image_PIL, first_vlm_response_keywords)
                else:
                    clip_image_embedding, clip_keyword_embedding = clip_rag.encode_image_text(image_PIL[0], first_vlm_response_keywords)
            clip_mean_embedding = (clip_image_embedding + clip_keyword_embedding) / 2
            clip_mean_embedding = clip_mean_embedding.cpu().detach().numpy()
            clip_image_embedding = clip_image_embedding.cpu().detach().numpy()
            clip_keyword_embedding = clip_keyword_embedding.cpu().detach().numpy()

            # retriev the top k sample from HNSW
            if query_embedding == 'image':
                retrieved_examples = hnsw.knn_query(clip_image_embedding, k_shot)
            else:
                retrieved_examples = hnsw.knn_query(clip_mean_embedding, k_shot)

            result_obj['icl_example'] = []
            icl_images = []
            icl_texts = []
            for icl in retrieved_examples:
                icl_text, icl_image = icl
                icl_text = concat_conversation_json(icl_text)
                icl_texts.append(icl_text)
                icl_images.append(icl_image)
                result_obj['icl_example'].append({
                    'text': icl_text,'image': icl_image
                })

            # SECOND query of VLM (with the retrieved example)
            # assemble prompt
            second_vlm_prompt = vllm.apply_single_image_prompt_with_example(prompt, None, 
                                                                            icl_texts, icl_images)
            if dataSet=='mmmu':
                second_vlm_response = vllm(second_vlm_prompt, 
                                        icl_images + [image_PIL], 
                                        max_new_token=512, 
                                        only_model_response=True)
            else:
                second_vlm_response = vllm(second_vlm_prompt, 
                                        icl_images + image_PIL, 
                                        max_new_token=512, 
                                        only_model_response=True)
            #print(second_vlm_response)
            result_obj['second_full_response'] = second_vlm_response
            result_obj['second_vlm_response'] = second_vlm_response

        result_objs.append(result_obj)

# %%
# dump as pickle
#with open(result_write_path, 'wb') as f:
#    pickle.dump(result_objs, f)

# %%
# evaluate results
from mmmu_utils import parse_multi_choice_response
import re

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHA_DOT = [alpha + '.' for alpha in ALPHA]
# parse option from gpt response
def parse_option(response):
    # the option is in between ** and **
    splits = response.split('ANSWER:')
    if len(splits) >= 2:
        option = splits[1].strip()
        for i in range(len(ALPHA_DOT)):
            if ALPHA_DOT[i] in option:
                option = ALPHA[i]
                return option
        for i in range(len(ALPHA)):
            if ALPHA[i] in option:
                option = ALPHA[i]
                return option
    else:
        return None


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


# %%
import matplotlib.pyplot as plt

def show_hist(hnsw, k_shot):
    data = hnsw.history
    fig, ax = plt.subplots()
    ax.set_title(f"{int(len(data)/k_shot)} queries with top-{k_shot} retrieval")
    ax.set_xlabel("Cache Entry")
    ax.set_ylabel("Number of Hits")
    ax.hist(data,bins=range(len(hnsw.data)))
    ax.locator_params(axis='y', integer=True)
    return fig, ax

# dump as pickle
#with open(cache_path, 'wb') as f:
    #pickle.dump(hnsw, f)
len(set(hnsw.history))/len(hnsw.data)


# %%

fig, ax = show_hist(hnsw, k_shot)
#fig.savefig(fig_path)

# %%
# compare model response with the ground truth
first_correct_count = 0
second_correct_count = 0
for idx, each in enumerate(test_dataset_single_image):
    if str(llm_first_choice_list[idx]).lower() == str(each['answer']).lower():
        first_correct_count += 1
    if str(llm_second_choice_list[idx]).lower() == str(each['answer']).lower():
        second_correct_count += 1

# accuracy
print(first_correct_count / len(test_dataset_single_image))
print(second_correct_count / len(test_dataset_single_image))


