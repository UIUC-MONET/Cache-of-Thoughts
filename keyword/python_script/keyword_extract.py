import transformers
import torch
from datasets import load_dataset
import os 
import sys
from tqdm import tqdm
import json 
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

HF_TOKEN='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'
output_dir = '/scratch/bczf/zoezheng126/uouo/github-repo/UOUO/keyword/mmmu/new_prompt/test_keyword'
gpt_desc_json = '/scratch/bczf/zoezheng126/uouo/github-repo/UOUO/keyword/mmmu/gpt_description/test/mmmu_val_gpt4o_response_v2.jsonl'

llama_template = """
Please give me 10 key phrases that are present in this question-option context that best repesent this context and separate them with commas.
Make sure you to only return the phrases and say nothing else.
Make sure to exclude the words "question", "answer", and "options" as in phrases
I have the following multiple choice question and its options:
question and options: {query}
answer: {answer}
"""

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# validation_dataset = load_dataset("lmms-lab/MMMU", split="validation")
# dev_dataset = load_dataset("lmms-lab/MMMU", split="dev")

# load dataset
test_dataset = load_dataset("lmms-lab/MMMU", split="test")

desc = []
with open(gpt_desc_json, 'r') as f:
    for line in f:
        # Parse each line as JSON and append to the list
        desc.append(json.loads(line))

print(f'dataset length:{len(test_dataset)}, gpt_desc length: {len(desc)}')

# load model
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = "left", token=HF_TOKEN)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, ds, gpt_desc, template):
        self.ds = ds
        self.template = template
        self.gpt_desc = gpt_desc 

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        ans = self.gpt_desc[idx]
        query = 'question: ' + str(sample['question']) + ' options: ' + str(sample['options'])
        llama_prompt = self.template.format(
            query=query,
            answer=ans
        )
        messages = [
            {"role": "system", "content": "You are a helpful chatbot that assists users in generating keywords from conversations. You have been given a conversation and need to generate keywords from it."},
            {"role": "user", "content": llama_prompt },
        ]
        return messages
    
prompt_ds = PromptDataset(test_dataset, desc, llama_template) # we use the subset of the dataset with COCO captions

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