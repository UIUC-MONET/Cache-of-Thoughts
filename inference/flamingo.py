import os
import random
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import datasets
from datasets import load_dataset
from utils.clip_rag import CLIP_rag
from utils.data_utils import create_cold_start_dataset_from_preprocess
from utils.hnsw_rag import HNSW, DataRecord
import ast
from utils.keyword_hashtag_rag import KeywordExtractor
from utils.other_utils import parse_options as parse_option
from utils.vlm_rag import OpenFlamingoVLM
from utils.other_utils import load_image, concat_conversation_json, show_hist
from utils.mmmu_utils import construct_mmmu_prompt_flamingo as construct_mmmu_prompt


def flamingo(dataSet, dataSlice, model, cacheSet, cacheSlice, embedding='image', alternative='', query_embedding='image_response', k_shot=1, cache_size=''):
    data_path = Path('./data/mmmu')
    dataSet = dataSet # choose from mmmu, clevr, textocr
    dataSlice = dataSlice # choose from val, dev
    query_embedding = query_embedding # choose from image, image_query, image_response, image_response_subfield (mmmu only)
    cacheSet = cacheSet # choose from mmmu, clevr, textocr
    cacheSlice = cacheSlice # choose from val, dev
    alternative = alternative # choose from _baseline, _subfield (mmmu only), or emtpy string
    embedding = embedding # choose from image, image_text
    model_name = model # choose from 3B, 4B, 9B
    k_shot = k_shot # choose from 1, 2, 4...
    cache_size = cache_size # choose from , _half
    random.seed(42)

    cache_path = f'./results/Open_Flamingo_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}{cache_size}_cache.pickle'
    fig_path = f'./results/Open_Flamingo_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}{cache_size}_cache_frequency.png'
    result_write_path = f'./results/Open_Flamingo_{model_name}_{dataSet}_{dataSlice}_{query_embedding}_{cacheSet}{alternative}_{cacheSlice}_{embedding}_{k_shot}{cache_size}_results.pickle'
    dataDir = './data'

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

    # MMMU cold start
    dim = 512
    if cache_size == '_half':
        cache_start_dataset = cache_start_dataset.shuffle(seed=42).select(range(int(0.5 * len(cache_start_dataset))))
    hnsw_size = 3000
    hnsw = HNSW(dim, hnsw_size, evict_method='random')
    if cacheSet == 'mmmu':
        image_list = cache_start_dataset['image_1']
    else:
        image_list = cache_start_dataset['image']
    text_list = cache_start_dataset['conversations']

    if cacheSet == 'mmmu':
        for i in tqdm(range(min(len(cache_start_dataset), hnsw_size)), total=hnsw_size):
            data_stuff = DataRecord([''],
                                    cache_start_dataset['clip_image_embed'][i],
                                    cache_start_dataset['clip_image_embed'][i], 
                                    text_list[i],
                                    image_list[i],
                                    [])
            hnsw.insert_data(data_stuff)
    else:
        for i in tqdm(range(min(len(cache_start_dataset), hnsw_size)), total=hnsw_size):
            data_stuff = DataRecord([''],
                                    cache_start_dataset['clip_image_embed'][i],
                                    cache_start_dataset['clip_image_embed'][i], 
                                    text_list[i],
                                    load_image(image_list[i],dataDir)[0][0],
                                    [])
            hnsw.insert_data(data_stuff)

    # model setup
    # VLM
    flamingo_dict = {'3B':("anas-awadalla/mpt-1b-redpajama-200b-dolly","openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",1), '4B':("togethercomputer/RedPajama-INCITE-Instruct-3B-v1","openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",2),'9B':("anas-awadalla/mpt-7b","openflamingo/OpenFlamingo-9B-vitl-mpt7b",4)}
    vllm_name = flamingo_dict[model_name][0]
    weight_name = flamingo_dict[model_name][1]
    layers = flamingo_dict[model_name][2]
    vllm = OpenFlamingoVLM(vllm_name, weight_name, layers)

    # CLIP model
    clip_rag = CLIP_rag()

    # keyword + hashtag extraction
    keyword_extractor = KeywordExtractor()

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
            if dataSet == 'mmmu':
                options = ast.literal_eval(each_query['options'])
                prompt = construct_mmmu_prompt(each_query['question'], options)
                first_full_response, first_vlm_response = vllm(vllm.apply_single_image_prompt(prompt, None), 
                                                            [each_query['image_1']], 
                                                            max_new_token=50, 
                                                            only_model_response=False)
            elif dataSet == 'clevr':
                property_name, exact_name  = each_query['question'].split(': ')
                prompt = f'How many {exact_name} objects?'
                first_full_response, first_vlm_response = vllm(vllm.apply_single_image_prompt(prompt, None), 
                                                            image_PIL, 
                                                            max_new_token=50, 
                                                            only_model_response=False)
            elif dataSet == 'textocr':
                prompt = each_query['question'] + 'Only answer with the largest text.'
                first_full_response, first_vlm_response = vllm(vllm.apply_single_image_prompt(prompt, None), 
                                                            image_PIL, 
                                                            max_new_token=50, 
                                                            only_model_response=False)
            print(first_vlm_response)
            result_obj['first_full_response'] = first_full_response
            result_obj['first_vlm_response'] = first_vlm_response
            result_objs.append(result_obj)

            # get the keywords
            first_vlm_response_keywords = keyword_extractor.extract_keywords(keyword_extractor.apply_keyword_extract_template(first_full_response))
            first_vlm_response_keywords = first_vlm_response_keywords[0].split(':')[-1]
            first_vlm_response_keywords = [each.strip() for each in first_vlm_response_keywords.split(',')][:min(10, len(first_vlm_response_keywords))]
            first_vlm_response_keywords = ', '.join(first_vlm_response_keywords)
            print(first_vlm_response_keywords)

            # get the image path
            # get the clip embeddings
            if query_embedding == 'image_query':
                if dataSet == 'mmmu':
                    clip_image_embedding, clip_keyword_embedding = clip_rag.encode_image_text(image_PIL, first_vlm_prompt)
                else:
                    clip_image_embedding, clip_keyword_embedding = clip_rag.encode_image_text(image_PIL[0], first_vlm_prompt)
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
                                        max_new_token=50, 
                                        only_model_response=True)
            else:
                second_vlm_response = vllm(second_vlm_prompt, 
                                        icl_images + image_PIL, 
                                        max_new_token=50, 
                                        only_model_response=True)
            print(second_vlm_response)
            result_obj['second_full_response'] = second_vlm_response
            result_obj['second_vlm_response'] = second_vlm_response
            
            result_objs.append(result_obj)

    # dump as pickle
    with open(cache_path, 'wb') as f:
        pickle.dump(hnsw, f)

    fig, ax = show_hist(hnsw, k_shot)
    fig.savefig(fig_path)

    # dump as pickle
    with open(result_write_path, 'wb') as f:
        pickle.dump(result_objs, f)


    # evaluate results
    llm_first_choice_list = []
    llm_second_choice_list = []
    for each in result_objs:
        llm_first_choice_list.append(parse_option(each['first_vlm_response'][0], dataSet))
        llm_second_choice_list.append(parse_option(each['second_vlm_response'][0], dataSet))

    # compare model response with the ground truth
    first_correct_count = 0
    second_correct_count = 0
    for idx, each in enumerate(test_dataset_single_image):
        if str(llm_first_choice_list[idx]).lower() == str(each['answer']).lower():
            first_correct_count += 1
        if str(llm_second_choice_list[idx]).lower() == str(each['answer']).lower():
            second_correct_count += 1

    # accuracy
    print('Apprentice first response accuracy:')
    print(first_correct_count / len(test_dataset_single_image))
    print('Apprentice second response accuracy:')
    print(second_correct_count / len(test_dataset_single_image))


