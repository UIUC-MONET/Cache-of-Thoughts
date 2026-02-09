from pathlib import Path
import os
import json
import torch
import datasets
from datasets import load_dataset
from . import data_utils
import torch
import clip
import numpy as np
import torch
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image

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

def get_clip_embedding(dataSet, dataSlice, embedding='image_text', alternative=''):
    dataSlice = dataSlice # validation dev test
    embedding = embedding # image image_text
    alternative = alternative # emtpy string if no special need _baseline for question embedding _subfield for class embedding
    dir_name = dataSlice
    dataDir = '../data'

    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print(preprocess)

    if dataSet == 'mmmu':
        if dataSlice == 'validation':
            dataset = load_dataset("lmms-lab/MMMU", split='validation')
        else:
            dataset = load_dataset("lmms-lab/MMMU", split=dataSlice)
        data_path = Path(f'../data/mmmu/{dir_name}')

        # load saved conversation with gpt and attach to dataset
        gpt_conversation_path = data_path / f'mmmu_{dir_name}_gpt4o_response_v2.jsonl' #TODO: actually the test set
        gpt_conversations = []
        with open(gpt_conversation_path, 'r') as f:
            for line in f:
                # each line is a json
                gpt_conversations.append(line.strip().strip('"'))
        data_conversation = datasets.Dataset.from_dict({"conversations": gpt_conversations})
        dataset = datasets.concatenate_datasets([dataset, data_conversation], axis=1)

        # load keywords
        keyword_dir = Path(f'../keyword_generation/mmmu_gpt/{dir_name}_keyword')
        keyword_test_list = data_utils.get_extracted_keywords(keyword_dir)
        data_keyword = datasets.Dataset.from_dict({"keywords": keyword_test_list})
        dataset = datasets.concatenate_datasets([dataset, data_keyword], axis=1)
        # only keep single image questions
        dataset_single_image = dataset.filter(lambda x: x['image_2'] is None)

   
        def mmmu_clip_preprocess_collate_fn(batch):
            images = torch.stack([preprocess(sample['image_1'].convert('RGB')) for sample in batch])
            if alternative == '_subfield':
                texts = torch.stack([clip.tokenize(sample['keywords']+', '+sample['subfield'])[0] for sample in batch])
            elif alternative == '_baseline':
                texts = torch.stack([clip.tokenize(sample['question'], truncate=True)[0] for sample in batch])
            else:
                texts = torch.stack([clip.tokenize(sample['keywords'])[0] for sample in batch])
            return {'image_1': images, 'text': texts}
        dataloader = torch.utils.data.DataLoader(dataset_single_image, 
                                                batch_size=64,
                                                collate_fn=mmmu_clip_preprocess_collate_fn)
    else:
        data_path = Path(f'../data/{dataSet}')
        # load dataset
        if dataSlice == 'val':
            support_file = os.path.join(dataDir, dataSet, 'support.json')
        else:
            support_file = os.path.join(dataDir, dataSet, 'query.json')
        with open(support_file, 'r') as f:
            support_meta = json.load(f)
        test_dataset = support_meta

        # load saved conversation with gpt and attach to dataset
        gpt_conversation_path = data_path / f'{dataSlice}/{dataSet}_{dataSlice}_gpt4o_response_v2.jsonl' #TODO: actually the test set
        gpt_conversations = []
        with open(gpt_conversation_path, 'r') as f:
            for line in f:
                # each line is a json
                gpt_conversations.append(line.strip().strip('"'))
        data_conversation = datasets.Dataset.from_dict({"conversations": gpt_conversations})
        test_dataset = datasets.Dataset.from_list(test_dataset)
        test_dataset = datasets.concatenate_datasets([test_dataset, data_conversation], axis=1)

        # load keywords
        keyword_dir = dataDir.strip('data') + f'keyword/{dataSet}_gpt/{dataSlice}_keyword'
        keyword_test_list = data_utils.get_extracted_keywords(keyword_dir)
        data_keyword = datasets.Dataset.from_dict({"keywords": keyword_test_list})
        test_dataset = datasets.concatenate_datasets([test_dataset, data_keyword], axis=1)
        def mmmu_clip_preprocess_collate_fn(batch):
            images = torch.stack([preprocess(load_image(sample['image'], dataDir)[0][0].convert('RGB')) for sample in batch])
            if alternative == '_baseline':
                texts = torch.stack([clip.tokenize(sample['question'], truncate=True)[0] for sample in batch])
            else:
                texts = torch.stack([clip.tokenize(sample['keywords'])[0] for sample in batch])
            return {'image': images, 'keywords':texts}
        dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=64,
                                                collate_fn=mmmu_clip_preprocess_collate_fn)

    print(f'num of data: {len(dataloader)}')

    clip_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image_1'].to('cuda')
            texts = batch['text'].to('cuda')
            image_features = model.encode_image(images).float()
            text_features = model.encode_text(texts).float()

            if embedding == 'image':
                clip_features.append(image_features.cpu())
            else:
                clip_features.append((image_features.cpu()+text_features.cpu())/2)
    clip_features_concat = torch.cat(clip_features, dim=0)
    if dataSet == 'mmmu':
        clip_embeddings_file = data_path / f'clip/mmmu{alternative}_{dir_name}_{embedding}_clip_embd_cold_start.pkl'
    else:
        clip_embeddings_file = data_path / f'{dataSlice}/clip/{dataSet}{alternative}_{dataSlice}_{embedding}_clip_embd_cold_start.pkl'
    with open(clip_embeddings_file, 'wb') as f:
        pickle.dump(clip_features_concat, f)


