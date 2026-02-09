import os
import random
import pickle
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from .clip_rag import CLIP_rag
from .data_utils import create_cold_start_dataset_from_preprocess
from .hnsw_rag import HNSW, DataRecord
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
import ast
import matplotlib.pyplot as plt
import re
from .keyword_hashtag_rag import KeywordExtractor


ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHA_DOT = [alpha + '.' for alpha in ALPHA]
# parse option from gpt response
def parse_option(response, dataSet):
    # the option is in between ** and **
    ans = response.strip()
    ans = ans.strip('\n')
    trunc_index = ans.find('\n')
    if trunc_index <= 0:
        trunc_index = ans.find('.')
    if trunc_index > 0:
        ans = ans[:trunc_index]
    if dataSet == "clevr":
        # find the number
        ans = re.search(r'\d+', ans)
        if ans:
            return int(ans.group(0))
        else:
            return None
    elif dataSet == 'textocr':
        ans = re.search("[a-zA-Z0-9']+", response)
        if ans:
            return ans.group(0)
        else:
            return None
    else:
        ans = re.search('[A-Z]', response)
        if ans:
            return ans.group(0)
        else:
            return None

def show_hist(hnsw, k_shot):
    data = hnsw.history
    fig, ax = plt.subplots()
    ax.set_title(f"{int(len(data)/k_shot)} queries with top-{k_shot} retrieval")
    ax.set_xlabel("Cache Entry")
    ax.set_ylabel("Number of Hits")
    ax.hist(data,bins=range(len(hnsw.data)))
    ax.locator_params(axis='y', integer=True)
    return fig, ax

class OpenFlamingoVLM():
    def __init__(self, model_name="anas-awadalla/mpt-7b", weight_name="openflamingo/OpenFlamingo-9B-vitl-mpt7b", layers=4):
        # TODO: decide other parameters to set
        self.model_name = model_name
        self.weight_name = weight_name
        default_dtype = torch.get_default_dtype() # trick to bypass the flash_attention_2 dtype warning
        torch.set_default_dtype(torch.bfloat16)
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=self.model_name,
            tokenizer_path=self.model_name,
            cross_attn_every_n_layers=layers #4 for 9b model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        torch.set_default_dtype(default_dtype)
        checkpoint_path = hf_hub_download(self.weight_name, "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model = self.model.to(torch.bfloat16).cuda()
    
    def __call__(self, input_text, image_path, max_new_token=30, only_model_response=True):
        return self.forward(input_text, image_path, max_new_token, only_model_response)
    
    def forward(self, input_text, input_images, max_new_token=30, only_model_response=True):
        """
        By default this returns the full model response and the section of the response that the model generated.
        Optionally, you can set only_model_response to True to get just the generated section.
        """
        # process input
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = self.tokenizer(
            [input_text],
            return_tensors="pt",
        )
        images = []
        if input_images is not None and len(input_images) > 0:
            for i in range(len(input_images)):
                if isinstance([i], str):
                    image = Image.open(input_images[i]).convert('RGB')
                elif isinstance(input_images[i], Image.Image):
                    image = input_images[i].convert('RGB')
                else:
                    image = Image.fromarray(input_images[i]).convert('RGB')

                images.append(image)
        vision_x = [self.image_processor(img).unsqueeze(0) for img in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        temp_texts = self.tokenizer.batch_decode(lang_x["input_ids"], skip_special_tokens=False)
        generated_text = self.model.generate(
            vision_x=vision_x.to(torch.bfloat16).cuda(),
            lang_x=lang_x["input_ids"].cuda(),
            attention_mask=lang_x["attention_mask"].cuda(),
            max_new_tokens=max_new_token,
            num_beams=3
        )
        
        if only_model_response:
            input_token_len = lang_x['input_ids'].shape[1]
            responses = self.tokenizer.batch_decode(generated_text[:, input_token_len:].cpu(), skip_special_tokens=True)
            return responses
        else:
            responses = self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
            return responses, [i[len(temp_texts[idx]):] for idx, i in enumerate(responses)]            
    
    def apply_single_image_prompt(self, input_text, image_path):
        template_2 = """{task_prompt}The answer is"""
        if '<image 1>' in input_text:
            query_text = '<image>' + input_text.replace("<image 1>", "").replace("<LETTER CHOICE>", "LETTER CHOICE")
        else:
            query_text = '<image>' + input_text
        prompt_2 = template_2.format(
            task_prompt=query_text
        )
        conversation_1 = prompt_2
        return " ".join(conversation_1.split())
    
    def apply_single_image_prompt_with_example(self, input_text, image_path, example_text_list, example_image_path_list):
        template_1 = """{example_task_prompt}"""
        example_template = []
        for i in range(len(example_text_list)):
            prompt_1 = template_1.format(
                example_task_prompt=example_text_list[i].replace('\\n','').replace("<image 1>", "").replace("<LETTER CHOICE>", "LETTER CHOICE").replace("<NUMBER>", "NUMBER").replace("<TEXT>", "TEXT"),
            )
            example_template.append(prompt_1)
        conversation_1 = ""
        template_2 = """{task_prompt}The answer is"""
        if '<image 1>' in input_text:
            query_text = '<image>' + input_text.replace("<image 1>", "").replace("<LETTER CHOICE>", "LETTER CHOICE")
        else:
            query_text = '<image>' + input_text
        prompt_2 = template_2.format(
            task_prompt=query_text
        )
        for p in example_template:
            conversation_1 = conversation_1 + "<image>" + p.strip() + "<|endofchunk|>"

        conversation_1 += prompt_2
        
        return " ".join(conversation_1.split())

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

def construct_mmmu_prompt(question, options):
    ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if len(options):
        return question + " The options are the following:" + str().join([ALPHABET[i] + ". " + options[i] + ". " for i in range(len(options))]) + "There is only one option possible."
    else:
        return question + "There is only one option possible."

def flamingo(dataSet, dataSlice, model, cacheSet, cacheSlice, embedding='image', alternative='', query_embedding='image_response', k_shot=1, cache_size=''):
    data_path = Path('../data/mmmu')
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
                                    image_list[i])
            hnsw.insert_data(data_stuff)
    else:
        for i in tqdm(range(min(len(cache_start_dataset), hnsw_size)), total=hnsw_size):
            data_stuff = DataRecord([''],
                                    cache_start_dataset['clip_image_embed'][i],
                                    cache_start_dataset['clip_image_embed'][i], 
                                    text_list[i],
                                    load_image(image_list[i],dataDir)[0][0])
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
    len(set(hnsw.history))/len(hnsw.data)

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
    print(first_correct_count / len(test_dataset_single_image))
    print(second_correct_count / len(test_dataset_single_image))


