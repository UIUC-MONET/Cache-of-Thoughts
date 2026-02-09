import os
import random
import time
from pathlib import Path
import torch
import pickle
import json
from PIL import Image
from tqdm import tqdm
import datasets
from datasets import load_dataset
from .clip_rag import CLIP_rag
from .keyword_hashtag_rag import KeywordExtractor, KeywordEncoder
from .vlm_rag import QwenVLM, OpenaiVLM, ModelMux
from .data_utils import create_cold_start_dataset_from_preprocess
from .mmmu_utils import construct_mmmu_prompt
from .hnsw_rag import HNSW, DataRecord
import ast
import re
import matplotlib.pyplot as plt

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHA_DOT = [alpha + '.' for alpha in ALPHA]

def show_hist(hnsw, k_shot):
    data = hnsw.history
    fig, ax = plt.subplots()
    ax.set_title(f"{int(len(data)/k_shot)} queries with top-{k_shot} retrieval")
    ax.set_xlabel("Cache Entry")
    ax.set_ylabel("Number of Hits")
    ax.hist(data,bins=range(len(hnsw.data)))
    ax.locator_params(axis='y', integer=True)
    return fig, ax

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

def qwen(dataSet, dataSlice, model, trainer, cacheSet, cacheSlice, embedding='image', alternative='', filter='', query_embedding='image_response', k_shot=1, dynamic=True, p=0.5):
    # current time
    current_time = time.strftime("%Y%m%d%H%M%S")
    dataSet = dataSet # choose from mmmu, clevr, textocr
    dataSlice = dataSlice# choose from val, dev
    query_embedding = query_embedding# choose from image, image_query, image_response, image_response_subfield (mmmu only)
    cacheSet = cacheSet # choose from mmmu, clevr, textocr
    cacheSlice = cacheSlice # choose from val, dev
    filter = filter # choose from subfield (only for mmmu)
    alternative = alternative # choose from _baseline, _subfield (mmmu only), or emtpy string
    embedding = embedding # choose from image, image_text
    model_name = model # choose from 2B, 7B, 72Bint4
    trainer_name = trainer # choose from gpt-4o, 7B
    k_shot = k_shot # choose from 1, 2, 4...
    dynamic = dynamic # choose from True, False
    p_large = p # choose between 0~1
    p_small = 1 - p_large
    random.seed(42)
    torch.manual_seed(42)

    cache_path = f'./results/Qwen_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}_cache_{current_time}.pickle'
    fig_path = f'./results/Qwen_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}_cache_frequency_{current_time}.png'
    data_path = Path('../data/mmmu/')
    result_write_path = f'./results/Qwen_{model_name}_{dataSet}_{dataSlice}_{cacheSet}_{cacheSlice}_{embedding}_results_{current_time}.jsonl'
    dataDir = '../data'

    if cacheSet == 'mmmu':
        if cacheSlice == 'val':
            dataset = load_dataset("lmms-lab/MMMU", split="validation")
        else:
            dataset = load_dataset("lmms-lab/MMMU", split=f"{cacheSlice}")
        gpt_conversation_path = data_path / f'{cacheSlice}/mmmu_{cacheSlice}_gpt4o_response_v2.jsonl'
        clip_image_embeddings_path = data_path / f'{cacheSlice}/clip/mmmu{alternative}_{cacheSlice}_{embedding}_clip_embd_cold_start.pkl'
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

    dim = 512
    hnsw_size = 1000
    hnsw = HNSW(dim, hnsw_size, evict_method='random')

    if cacheSet == 'mmmu':
        image_list = cache_start_dataset['image_1']
    else:
        image_list = cache_start_dataset['image']
    text_list = cache_start_dataset['conversations']

    if cacheSet == 'mmmu':
        for i in tqdm(range(min(len(cache_start_dataset), hnsw_size)), total=hnsw_size):
            data_stuff = DataRecord(keywords=set([cache_start_dataset['subfield'][i]]),
                                    image_clip=cache_start_dataset['clip_image_embed'][i],
                                    vector_data=cache_start_dataset['clip_image_embed'][i], 
                                    text_data=text_list[i],
                                    image_data=image_list[i],
                                    text_clip=None)
            hnsw.insert_data(data_stuff)
    else:
        for i in tqdm(range(min(len(cache_start_dataset), hnsw_size)), total=hnsw_size):
            data_stuff = DataRecord(keywords=[''],
                                    image_clip=cache_start_dataset['clip_image_embed'][i],
                                    vector_data=cache_start_dataset['clip_image_embed'][i], 
                                    text_data=text_list[i],
                                    image_data=load_image(image_list[i],dataDir)[0][0],
                                    text_clip=None)
            hnsw.insert_data(data_stuff)

    # VLM
    qwen_dict = {'2B':'Qwen/Qwen2-VL-2B-Instruct', '7B':'Qwen/Qwen2-VL-7B-Instruct', '72Bint4':'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4'}
    vllm_name = qwen_dict[model_name]
    vllm = QwenVLM(vllm_name)

    # CLIP model
    clip_rag = CLIP_rag()

    # keyword + hashtag extraction
    keyword_extractor = KeywordExtractor()

    # mux for swtiching between GPT or small VLM
    if trainer_name == 'gpt-4o':
        trainer = OpenaiVLM('gpt-4o', use_env_api_key=True)
    elif trainer_name == '7B':
        trainer = QwenVLM(qwen_dict[trainer_name])
    model_mux = ModelMux([trainer, vllm], [p_large, p_small])

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
                start = time.perf_counter()
                first_full_response, first_vlm_response = selected_model(input_text=first_vlm_prompt, 
                                                            input_images=[image_PIL], 
                                                            max_new_token=512, 
                                                            only_model_response=False)
                end = time.perf_counter()
            elif dataSet == 'clevr':
                property_name, exact_name  = each_query['question'].split(': ')
                prompt = f'How many objects in the image have the {exact_name} {property_name}. Please answer in the following format: ANSWER: <NUMBER>.'
                first_vlm_prompt = selected_model.apply_single_image_prompt(prompt, input_image=image_PIL[0])
                start = time.perf_counter()
                first_full_response, first_vlm_response = selected_model(input_text=first_vlm_prompt, 
                                                            input_images=image_PIL, 
                                                            max_new_token=512, 
                                                            only_model_response=False)
                end = time.perf_counter()
            elif dataSet == 'textocr':
                prompt = 'An image will be provided where a red box is drawn around the text of interest. Answer with the largest text inside the red box. Ensure that the transcription is precise, reflecting the exact characters, including letters, numbers, symbols.'
                first_vlm_prompt = selected_model.apply_single_image_prompt(prompt, input_image=image_PIL[0])
                start = time.perf_counter()
                first_full_response, first_vlm_response = selected_model(input_text=first_vlm_prompt, 
                                                            input_images=image_PIL, 
                                                            max_new_token=512, 
                                                            only_model_response=False)
                end = time.perf_counter()
            if selected_model == trainer:
                trainer_time = (end - start)
            else:
                trainee_first_time = (end - start)
            result_obj['first_full_response'] = first_full_response
            result_obj['first_vlm_response'] = first_vlm_response
            
            if selected_model == trainer: # if using trainer, skip the retrieval step
                print(f'using {trainer_name}, thus saving result to HNSW')
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
                start = time.perf_counter()
                # get the keywords
                first_vlm_response_keywords = keyword_extractor.extract_keywords(keyword_extractor.apply_keyword_extract_template(first_full_response))
                first_vlm_response_keywords = first_vlm_response_keywords[0].split(':')[-1]
                first_vlm_response_keywords = [each.strip() for each in first_vlm_response_keywords.split(',')][:min(10, len(first_vlm_response_keywords))]
                first_vlm_response_keywords = ', '.join(first_vlm_response_keywords)
                end = time.perf_counter()
                keyword_extractor_time = (end - start)

                start = time.perf_counter()
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
                end = time.perf_counter()
                clip_time = (end - start)

                start = time.perf_counter()
                # retriev the top k sample from HNSW
                if query_embedding == 'image':
                    if filter == 'subfield':
                        retrieved_examples = hnsw.knn_query_filtered_by_keywords(DataRecord(keywords=set([each_query['subfield']]),
                                    image_clip=None,
                                    vector_data=clip_image_embedding, 
                                    text_data=None,
                                    image_data=None,
                                    text_clip=None), k_shot)
                    else:
                        retrieved_examples = hnsw.knn_query(clip_image_embedding, k_shot)
                else:
                    if filter == 'subfield':
                        retrieved_examples = hnsw.knn_query_filtered_by_keywords(DataRecord(keywords=set([each_query['subfield']]),
                                    image_clip=None,
                                    vector_data=clip_mean_embedding, 
                                    text_data=None,
                                    image_data=None,
                                    text_clip=None), k_shot)
                    else:
                        retrieved_examples = hnsw.knn_query(clip_mean_embedding, k_shot)
                end = time.perf_counter()
                HNSW_time = (end - start)
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
                    start = time.perf_counter()
                    second_vlm_response = vllm(second_vlm_prompt, 
                                            icl_images + [image_PIL], 
                                            max_new_token=512, 
                                            only_model_response=True)
                    end = time.perf_counter()
                else:
                    start = time.perf_counter()
                    second_vlm_response = vllm(second_vlm_prompt, 
                                            icl_images + image_PIL, 
                                            max_new_token=512, 
                                            only_model_response=True)
                    end = time.perf_counter()
                trainee_second_time = (end - start)
                result_obj['second_full_response'] = second_vlm_response
                result_obj['second_vlm_response'] = second_vlm_response

            if selected_model == trainer:
                result_obj['trainer_time'] = trainer_time 
                result_obj['trainee_first_time'] = 0
                result_obj['trainee_second_time'] = 0
                result_obj['keyword_extractor_time'] = 0
                result_obj['clip_time'] = 0
                result_obj['HNSW_time'] = 0
            else:
                result_obj['trainer_time'] = 0 
                result_obj['trainee_first_time'] = trainee_first_time
                result_obj['trainee_second_time'] = trainee_second_time
                result_obj['keyword_extractor_time'] = keyword_extractor_time
                result_obj['clip_time'] = clip_time
                result_obj['HNSW_time'] = HNSW_time

            result_objs.append(result_obj)

    # dump as pickle
    with open(result_write_path, 'wb') as f:
        pickle.dump(result_objs, f)

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

    # dump as pickle
    with open(cache_path, 'wb') as f:
        pickle.dump(hnsw, f)

    fig, ax = show_hist(hnsw, k_shot)
    fig.savefig(fig_path)
    # compare model response with the ground truth
    first_correct_count = 0
    second_correct_count = 0
    vllm_count = 0
    first_vllm_correct_count = 0
    second_vllm_correct_count = 0
    total_trainer_time = 0
    total_trainee_first_time = 0
    total_trainee_second_time = 0
    total_keyword_extractor_time = 0
    total_clip_time = 0
    total_HNSW_time = 0
    for idx, each in enumerate(test_dataset_single_image):
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
        total_trainer_time += result_objs[idx]['trainer_time']
        total_trainee_first_time += result_objs[idx]['trainee_first_time']
        total_trainee_second_time += result_objs[idx]['trainee_second_time']
        total_keyword_extractor_time += result_objs[idx]['keyword_extractor_time']
        total_clip_time += result_objs[idx]['clip_time']
        total_HNSW_time += result_objs[idx]['HNSW_time']
        

    # accuracy
    print(f'Apprentice: {vllm_count} / {len(test_dataset_single_image)} times')
    print(f'Trainer: {len(test_dataset_single_image)-vllm_count} / {len(test_dataset_single_image)} times')
    print('Overall first response accuracy:')
    print(first_correct_count / len(test_dataset_single_image))
    print('Overall second response accuracy:')
    print(second_correct_count / len(test_dataset_single_image))
    print('Trainer response accuracy:')
    print((first_correct_count - first_vllm_correct_count) / (len(test_dataset_single_image) - vllm_count))
    print('Apprentice first response accuracy:')
    print(first_vllm_correct_count/vllm_count)
    print('Apprentice second response accuracy:')
    print(second_vllm_correct_count / vllm_count)
    print(f'Trainer costs {total_trainer_time} seconds')
    print(f'Apprentice first response costs {total_trainee_first_time} seconds')
    print(f'Apprentice second response costs {total_trainee_second_time} seconds')
    print(f'Keyword extractor costs {total_keyword_extractor_time} seconds')
    print(f'Clip embedding costs {total_clip_time} seconds')
    print(f'HNSW query costs {total_HNSW_time} seconds')
    print(f'Retrieval cost {total_keyword_extractor_time + total_clip_time + total_HNSW_time} seconds')


