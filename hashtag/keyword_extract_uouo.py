import transformers
import torch
from datasets import load_dataset
import os 
import sys
from tqdm import tqdm
import json 
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import pickle

HF_TOKEN='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'

question_mode = 'object'
model_type = 'big_vlm'
output_dir = '/data/haozhen/uouo/output/uouo/keywords'
output_dir = os.path.join(output_dir, question_mode)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

uouo_query_path = '/data/haozhen/uouo/mosaic_output/model-cascade-big-vlm-shuffle-rag/gpt-4o_uouo_desc.pkl'

big_vlm_llama_template = """
Please give me 10 key phrases that are present in this question-answering context that best repesent this context and separate them with commas.
Make sure you to only return the phrases and say nothing else.
Make sure to exclude the words "2x2", "location", "mosaic image", "A", "B", "C", "D", "top left", "top right", "bottom left", "bottom right", "question", "answer", and "options" as in phrases
I have the following answer with explaination for the multiple choice question and its options:
question and options: {query}
answer: {answer}
"""

small_vlm_llama_template = """
Please give me 10 key phrases that are present in this question-option context that best repesent this context and separate them with commas.
Make sure you to only return the phrases and say nothing else.
Make sure to exclude the words "mosaic image", "A", "B", "C", "D", "top left", "top right", "bottom left", "bottom right", "question", "answer", and "options" as in phrases
I have the following multiple choice question and its options:
question and options: {query}
"""

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# validation_dataset = load_dataset("lmms-lab/MMMU", split="validation")
# dev_dataset = load_dataset("lmms-lab/MMMU", split="dev")

# load dataset
with open(uouo_query_path, "rb") as file:
    uouo_queries = pickle.load(file)

            # d['purpose_ans_gpt'] = purpose_ans
            # d['object_ans_gpt'] = object_ans
            # d['purpose_in_opt_ans_gpt'] = purpose_in_opt_ans
            # {'img_path': img_path, 
        #      'purpose_prompt': purpose_template_gpt_desc_prompt, # given purpose, ask where
        #      'object_prompt': object_template_gpt_desc_prompt, # given object name, ask where
        #      'purpose_in_option_prompt': purpose_in_option_gpt_desc_prompt, # given position, ask which purpose
        #      'dt_id': dt_id,   # which object should be query, pos_id from 1 2 3 4 
        #      'dt_label': dt_label, # which object should be query, object name
        #      'dt_purpose': chosen_purpose_1,  # for purpose_prompt groundtruth purpose
        #      'gt_position': pos_dict[dt_id],  # the query object location, top left ...
        #      'purpose_in_opt_gt_ans_ABCD' : gt_opt_num, # gt answer choice ABCD for purpose_in_option_prompt
        #      'purpose_in_opt_gt_ans': chosen_purpose_2, # gt answer purpose for purpose_in_option_prompt
        #      'purpose_in_opt_options': purpose_option} #  all optiosn for purpose_in_option_prompt

test_dataset = []
desc = []
for d in uouo_queries:
    data = {}
    if question_mode == 'purpose':
        data['question'] = d['purpose_prompt']
        desc.append(d['purpose_ans_gpt'])
    elif question_mode == 'object':
        data['question'] = d['object_prompt']
        desc.append(d['object_ans_gpt'])
    elif question_mode == 'purpose_in_opt':
        data['question'] = d['purpose_in_option_prompt']
        desc.append(d['purpose_in_opt_ans_gpt'])

    test_dataset.append(data)

print(f'dataset length:{len(uouo_queries)}')

# load model
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = "left", token=HF_TOKEN)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="balanced_low_0", token=HF_TOKEN)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, ds, template, gpt_desc=None, option=False):
        self.ds = ds
        self.template = template
        self.gpt_desc = gpt_desc 
        self.enable_option = option

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        if self.gpt_desc != None:
            ans = self.gpt_desc[idx]
        if self.enable_option and self.gpt_desc != None:
            # question + option + gpt_ans -> use big vlm prompt template
            query = 'question: ' + str(sample['question']) + ' options: ' + str(sample['options'])
            llama_prompt = self.template.format(
                query=query,
                answer=ans
            )
        elif self.enable_option == False and self.gpt_desc != None:
            # question (include option) + gpt_ans
            query = str(sample['question'])
            llama_prompt = self.template.format(
                query=query,
                answer=ans
            )
        elif self.enable_option == False and self.gpt_desc == None:
            query = str(sample['question'])
            llama_prompt = self.template.format(query=query)

        else:
            query = 'question: ' + str(sample['question']) + ' options: ' + str(sample['options'])
            llama_prompt = self.template.format(query=query)

        print(llama_prompt)
             
        messages = [
            {"role": "system", "content": "You are a helpful chatbot that assists users in generating keywords from conversations. You have been given a conversation and need to generate keywords from it."},
            {"role": "user", "content": llama_prompt },
        ]
        return messages
    

if model_type == 'big_vlm':
    llama_template = big_vlm_llama_template
    prompt_ds = PromptDataset(test_dataset, llama_template, desc, False) 
else: 
    llama_template = small_vlm_llama_template
    prompt_ds = PromptDataset(test_dataset, llama_template, None, False)

# create dir if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

prompt_loader = torch.utils.data.DataLoader(prompt_ds, batch_size=48, collate_fn=lambda x: [_ for _ in x], shuffle=False)

llama_keywords = []
for i, batch in enumerate(tqdm(prompt_loader)):
    texts = tokenizer.apply_chat_template(batch, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(texts, padding="longest", return_tensors="pt").to(model.device)
    # inputs = {key: val. for key, val in inputs.items()}
    temp_texts=tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

    start_time = time.time()
    gen_tokens = model.generate(
        **inputs, 
        max_new_tokens=128, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]

    llama_keywords.extend(gen_text)

    # write to file every 10 batches or at the end
    if i % 10 == 0 or i == len(prompt_loader) - 1:
        with open(os.path.join(output_dir, f"llama_keywords_{i}.txt"), "w") as f:
            f.write("\n".join(llama_keywords))
        llama_keywords = []