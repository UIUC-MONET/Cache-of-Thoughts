# %%
import os
import sys

# Add the inference directory to the PYTHONPATH
som_path = '/data/haozhen/uouo/cache-of-thoughts/inference'
if som_path not in sys.path:
    sys.path.append(som_path)

os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + f":{som_path}"
# os.environ['HF_HOME'] = '/home/jizej/.cache/huggingface'
# print(os.getenv('HF_HOME'))

os.environ["OPENAI_API_KEY"] = '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'

# %%
import pickle
# from model import HFModelGeneration
import time
import base64
import io
import glob
import json
from pathlib import Path

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
from vlm_rag import QwenVLM
from data_utils import create_cold_start_dataset_from_preprocess, reconstruct_prompt_from_gpt_conversation

# %%
question_mode = 'object'  # purpose_in_opt, purpose, object

# %%
data_path = Path('/data/haozhen/uouo/mosaic_output/model-cascade-big-vlm-shuffle-rag')
# load img clip imbedding list
with open(os.path.join(data_path, 'clip/uouo_test_image_clip_embd_cold_start.pkl'),'rb') as file:
    clip_embed_list = pickle.load(file)

# preprocess uouo queries
with open(os.path.join(data_path, 'gpt-4o_uouo_desc.pkl'),'rb') as file: # CHECK CODE 
    val_data_list = pickle.load(file)

# with open(os.path.join(data_path, 'gpt-4o_uouo_desc.pkl'),'rb') as file:
#     val_data_list = pickle.load(file)

# parse into sharegpt format
questions, options, conversations, imgs, ids = [], [], [],[],[]
for i, d in enumerate(val_data_list):
    if question_mode == 'purpose':
        question = d['purpose_prompt']
        option = str(['top left', 'top right', 'bottom left', 'bottom right'])
        conversation = d['purpose_ans_gpt']
    elif question_mode == 'object':
        question = d['object_prompt']
        option = str(['top left', 'top right', 'bottom left', 'bottom right'])
        conversation = d['object_ans_gpt']
    elif question_mode == 'purpose_in_opt':
        question = d['purpose_in_option_prompt']
        option = str(d['purpose_in_opt_options'])
        conversation = d['purpose_in_opt_ans_gpt']

    # d['sharegpt_format_conversation'] = reconstruct_prompt_from_gpt_conversation(question, option, conversation)
    pil_img = Image.open(d['img_path'])

    questions.append(question)
    options.append(option)
    conversations.append(conversation)
    imgs.append(pil_img)
    ids.append(f'{question_mode}_{i}')

print(len(questions), len(options), len(conversations), len(imgs), len(ids))
uouo_val_purpose_data_list = []
for i in range(len(questions)):
    uouo_val_purpose_data_list.append({
        'question': questions[i],
        'options': options[i],
        'conversations': reconstruct_prompt_from_gpt_conversation(questions[i], options[i], conversations[i]),
        'clip_image_embed': clip_embed_list[i],
        'image_1': imgs[i],
        'image_2': None,
        'id': ids[i]
    })

uouo_val_purpose_dataset = datasets.Dataset.from_list(uouo_val_purpose_data_list)


# %%
data_path = Path('/data/haozhen/uouo/mosaic_output/model-cascade-small-vlm-shuffle-rag')
# load img clip imbedding list
with open(os.path.join(data_path, 'clip/uouo_test_image_clip_embd_cold_start.pkl'),'rb') as file:
    test_clip_embed_list = pickle.load(file)
# preprocess uouo queries
with open(os.path.join(data_path, 'uouo_test_queries.pkl'),'rb') as file:
    test_data_list = pickle.load(file)

# parse into sharegpt format
questions, options, imgs, ids = [], [], [], []
for i, d in enumerate(test_data_list):
    if question_mode == 'purpose':
        question = d['purpose_prompt']
        option = str(['top left', 'top right', 'bottom left', 'bottom right'])

    elif question_mode == 'object':
        question = d['object_prompt']
        option = str(['top left', 'top right', 'bottom left', 'bottom right'])

    elif question_mode == 'purpose_in_opt':
        question = d['purpose_in_option_prompt']
        option = str(d['purpose_in_opt_options'])

    # d['sharegpt_format_conversation'] = reconstruct_prompt_from_gpt_conversation(question, option, conversation)
    pil_img = Image.open(d['img_path'])

    questions.append(question)
    options.append(option)
    imgs.append(pil_img)
    ids.append(f'{question_mode}_{i}')

print(len(questions), len(options), len(conversations), len(imgs), len(ids))
uouo_test_purpose_data_list = []
for i in range(len(questions)):
    uouo_test_purpose_data_list.append({
        'question': questions[i],
        'options': options[i],
        'clip_image_embed': test_clip_embed_list[i],
        'image_1': imgs[i],
        'image_2': None,
        'id': ids[i]
    })

uouo_test_purpose_dataset = datasets.Dataset.from_list(uouo_test_purpose_data_list)
# %%
# MMMU cold start
# TODO: put this in a .py file
from hnsw_rag import HNSW, DataRecord

dim = 512
#num_elements = 3126 * 16
hnsw_size = 900
hnsw = HNSW(dim, hnsw_size, evict_method='random')
#data = np.float32(np.random.random((num_elements, dim)))
# stuff_list = [DataRecord(set(validation_dataset_single['hashtags'][i][2]), 
#                          clip_embeddings[i][0], 
#                          clip_embeddings[i][1], 
#                          clip_embeddings[i][0], 
#                          validation_dataset_single['image_1'][i]) for i in range(temp)]
#stuff_list = [DataRecord({str(i % 25)}, data[i,:], i) for i in range(num_elements)]

# hashtag_list = mmmu_start_dataset['hashtags']
image_list = uouo_val_purpose_dataset['image_1']
text_list = uouo_val_purpose_dataset['conversations']
# clip_image_embed_list = mmmu_start_dataset['clip_image_embed']

for i in tqdm(range(min(len(uouo_val_purpose_dataset), hnsw_size)), total=hnsw_size):
    data_stuff = DataRecord([''],
                            uouo_val_purpose_dataset['clip_image_embed'][i],
                            uouo_val_purpose_dataset['clip_image_embed'][i], 
                            text_list[i],
                            image_list[i])
    hnsw.insert_data(data_stuff)
    #print(stuff.keywords)
    #print(image_paths[i])


# %%
multi_gpu = True

# model setup
# TODO: replace the default init parameters
# VLM
vllm_name = 'Qwen/Qwen2-VL-7B-Instruct'
vllm = QwenVLM(vllm_name, device='cuda:1')

# CLIP model
clip_rag = CLIP_rag()

# keyword + hashtag extraction
keyword_extractor = KeywordExtractor()

# keyword + hashtag embedding
keyword_encoder = KeywordEncoder()


def concat_conversation_json(conversation):
    out_str = 'Human: ' + conversation[0]['value'] + '\n' + 'Assistant: ' + conversation[1]['value'] + '\n'
    return out_str


# %%
# MAIN LOOP

# iterate through the test dataset/stream
output = {}
for idx, batch in enumerate(tqdm(uouo_test_purpose_dataset)):
    for each_query in [batch]: # TODO: sorry, no batch processing for now
        # get the image and the question
        image_PIL = each_query['image_1'] # again, assuming single image
        id = each_query['id']
        # FOR CODE CHECK
        # question_none = each_query['question'].split("'bottom right'.")[0] + "'bottom right', 'none'." + each_query['question'].split("'bottom right'.")[1]
        # print(question_none)
        # first_vlm_response = vllm(vllm.apply_single_image_prompt(question_none, None), [each_query['image_1']], max_new_token=512, only_model_response=True)

        # FIRST query of VLM
        first_full_response, first_vlm_response = vllm(vllm.apply_single_image_prompt(each_query['question'], None), 
                                  [each_query['image_1']], 
                                  max_new_token=512, 
                                  only_model_response=False)

        # get the keywords
        first_vlm_response_keywords = keyword_extractor.extract_keywords(keyword_extractor.apply_keyword_extract_template(first_full_response))
        first_vlm_response_keywords = first_vlm_response_keywords[0].split(':')[-1]
        first_vlm_response_keywords = [each.strip() for each in first_vlm_response_keywords.split(',')][:min(10, len(first_vlm_response_keywords))]
        first_vlm_response_keywords = ', '.join(first_vlm_response_keywords)

        # get the image path
        # get the clip embeddings
        clip_image_embedding, clip_keyword_embedding = clip_rag.encode_image_text(image_PIL, first_vlm_response_keywords)
        clip_mean_embedding = (clip_image_embedding + clip_keyword_embedding) / 2
        clip_image_embedding = clip_image_embedding.cpu().detach().numpy()
        clip_keyword_embedding = clip_keyword_embedding.cpu().detach().numpy()
        clip_mean_embedding = clip_mean_embedding.cpu().detach().numpy()

        # retriev the top k sample from HNSW
        retrieved_examples = hnsw.knn_query(clip_mean_embedding, 1)

        icl_text, icl_image = retrieved_examples[0]
        icl_text = concat_conversation_json(icl_text)

        # SECOND query of VLM (with the retrieved example)
        # assemble prompt
        second_vlm_prompt = vllm.apply_single_image_prompt_with_example(each_query['question'], None, 
                                                                       [icl_text], [icl_image])
        second_full_response, second_vlm_response = vllm(second_vlm_prompt, 
                                 [icl_image, image_PIL], 
                                 max_new_token=512, 
                                 only_model_response=False)

        output[id] = [first_vlm_response, second_vlm_response, each_query['question'], retrieved_examples, second_vlm_prompt, second_full_response]
        
        

    


# %%
output_dir = '/data/haozhen/uouo/output/uouo/rag_output'
with open(os.path.join(output_dir, f'{question_mode}_out_only_jize.pkl'),'wb') as file:
    pickle.dump(output, file)

# %% [markdown]
# # Evaluation

# # %%
# output_dir = '/data/haozhen/uouo/output/uouo/rag_output' # load from pkl if needed 
# question_mode = 'purpose'

# with open(os.path.join(output_dir, f'{question_mode}_out_only_1.pkl'),'rb') as file:
#     output = pickle.load(file)

# data_path = Path('/data/haozhen/uouo/mosaic_output/model-cascade-small-vlm-shuffle-rag')
# with open(os.path.join(data_path, 'uouo_test_queries.pkl'),'rb') as file:
#     test_data_list = pickle.load(file)

# print(len(output))

# # %%
# # CLIP Answer Evaluator

# import torch
# import clip
# import torch.nn.functional as F
# import random

# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# def check_ans_to_options(ans, gt_ans, options, model, question_mode='purpose'):
#     if question_mode == 'purpose' or question_mode == 'object':
#         for s in options:
#             if s in ans: 
#                 ans = s
#                 break

#     if ans in options: 
#         return 1 if options.index(ans) == options.index(gt_ans) else 0

#     return 0
#     ans = random.choice(options)
#     return 1 if options.index(ans) == options.index(gt_ans) else 0

    
#     # if question_mode == 'purpose_in_opt':
#     #     opt_num_dict = {'A':0, 'B':1, 'C':2, 'D':3}
#     #     gt_ans = opt_num_dict[gt_ans]
#     # # use clip to evaluate similarity between ans and options
#     # options_token = clip.tokenize(options).to(device)
#     # ans_token = clip.tokenize(ans).to(device)
#     # with torch.no_grad():
#     #     options_embed = model.encode_text(options_token)
#     #     ans_embed = model.encode_text(ans_token)
#     # options_embed /= options_embed.norm(dim=-1, keepdim=True)
#     # ans_embed /= ans_embed.norm(dim=-1, keepdim=True)
#     # similarity = (ans_embed @ options_embed.T).softmax(dim=-1)
#     # ans_idx = torch.argmax(similarity)
#     # gt_idx = options.index(gt_ans)
#     # return 1 if ans_idx == gt_idx else 0


# # %%
# from pprint import pprint

# pprint(test_data_list[0])

# # %%
# # test_data_list contains groundtruth

# def get_gt_option(d, question_mode):
#     if question_mode == 'purpose' or question_mode == 'object':
#         gt_option = d['gt_position'] # ground truth postion in graph, ex. top left, ...
#     elif question_mode == 'purpose_in_opt':
#         gt_option = d['purpose_in_opt_gt_ans_ABCD']  # ground truth purpose's corresponding alphabet A, ...
#     return gt_option

# def get_query_option(d, question_mode):
#     if question_mode == 'purpose_in_opt':
#         return d['purpose_in_opt_options']  
#     return ['top left', 'top right', 'bottom left', 'bottom right']


# total_direct = 0
# total_rag = 0
# for id, raw_out in output.items():
#     out_direct = raw_out[0][0]
#     out_rag = raw_out[1][0]


#     idx = int(id.split('_')[-1])
#     d = test_data_list[idx]
#     gt_out = get_gt_option(d, question_mode)
#     options = get_query_option(d, question_mode)

#     # for checking my code
#     direct_prompt = raw_out[2]
#     d_prompt = d['purpose_prompt']

#     clean_out_direct = out_direct.split('<|im_end|>')[0].strip()
#     if 'ANSWER' in clean_out_direct:
#         clean_out_direct = clean_out_direct.split('ANSWER')[-1].strip()
    
#     # if mode == purpose_in_opt, then pass alphabet gt_ans but text options (purpose1, purpose2 ...) to compare use clip when ans not exact match
#     print(idx, out_direct, gt_out, direct_prompt, d_prompt)
    
#     total_direct += check_ans_to_options(clean_out_direct, gt_out, options, clip_model)
#     # print(clean_out_direct, gt_out, total_direct)
#     clean_out_rag = out_rag.split('<|im_end|>')[0].strip()
#     if 'ANSWER' in clean_out_rag:
#         clean_out_rag = clean_out_rag.split('ANSWER')[-1].strip()
#     # if mode == purpose_in_opt, then pass alphabet gt_ans but text options (purpose1, purpose2 ...) to compare use clip when ans not exact match
#     total_rag += check_ans_to_options(clean_out_rag, gt_out, options, clip_model)
#     # print(clean_out_rag, gt_out, total_rag)

# acc_direct = total_direct / len(output)
# acc_rag = total_rag / len(output)
# print(f'direct accuracy: {acc_direct}')
# print(f'rag acc: {acc_rag}')

    

    
    

# %%



