from pathlib import Path
import pickle
import ast
import datasets
from datasets import Dataset
import torch
from .other_utils import ALPHABET


def get_extracted_keywords(data_dir):
    """
    Get extracted keywords from the data directory. Concatenates all the keywords from all the files in the data directory.
    """
    data_dir = Path(data_dir)
    keywords = []
    for file in sorted(data_dir.iterdir()):
        with open(file, 'r') as f:
            for each in f.read().splitlines():
                if each is None or len(each) == 0:
                    continue
                keywords.append(each)
    return keywords

    
def reconstruct_prompt_from_gpt_conversation(question, options, conversation, dataSet):
    gpt_question = ''
    options = ast.literal_eval(options)
    if len(options):
        gpt_question = question + " The options are the following:" + str().join([ALPHABET[i] + ". " + options[i] + ". " for i in range(len(options))]) + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible."
    else:
        if dataSet == 'clevr':
            property_name, exact_name  = question.split(': ')
            gpt_question = f'How many objects in the image have the {exact_name} {property_name}' + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <NUMBER>."
        elif dataSet == 'textocr':
            gpt_question = question + " Only answer with the largest text. Please include your reasoning steps, then answer your choice in this format: ANSWER: <TEXT>."
        else:
            gpt_question = question + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible."
    obj = [
        {
            "from": "user",
            "value": gpt_question
        },
        {
            "from": "gpt",
            "value": conversation
        }
    ]
    return obj


def create_cold_start_dataset_from_preprocess(base_dataset: Dataset, gpt_conversation_path: str, clip_image_embeddings_path: str, dataSet: str):
    """
    Create a starting dataset from a cold start. Append the GPT conversation and clip embedding to the base dataset.
    """
    start_ds = base_dataset
    gpt_conversations = []
    with open(gpt_conversation_path, 'r') as f:
        for line in f:
            # each line is a json
            gpt_conversations.append(line.strip('"').strip())
    data_conversation = datasets.Dataset.from_dict({"conversations": gpt_conversations})
    data_conversation = datasets.concatenate_datasets([start_ds, data_conversation], axis=1)
    data_conversation = data_conversation.map(lambda x: {"conversations": reconstruct_prompt_from_gpt_conversation(x['question'], x['options'], x['conversations'], dataSet)})
    data_conversation = data_conversation.select_columns(["conversations"])
    start_ds = datasets.concatenate_datasets([start_ds, data_conversation], axis=1)

    # only keep single image questions
    # validation_dataset_single = validation_dataset.filter(lambda x: x['image_2'] is None)
    if dataSet == 'mmmu':
        start_ds_single_image = start_ds.filter(lambda x: x['image_2'] is None)
    else:
        start_ds_single_image = start_ds

    # # attach clip embeddings to the single image dataset
    # attach clip embeddings
    with open(clip_image_embeddings_path, 'rb') as f:
        clip_image_embeddings = pickle.load(f)
    clip_image_embeddings = torch.tensor(clip_image_embeddings)
    data_clip_image_embed = datasets.Dataset.from_dict({"clip_image_embed": clip_image_embeddings})
    start_ds_single_image = datasets.concatenate_datasets([start_ds_single_image, data_clip_image_embed], axis=1)
    start_ds_single_image.set_format("pt", columns=["clip_image_embed"], output_all_columns=True) 
    return start_ds_single_image

