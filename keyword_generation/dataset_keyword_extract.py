import os 
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def dataset_keyword_extract(dataSet, dataSlice):
    dataSlice = dataSlice # val dev
    dataSet = dataSet # choose from mmmu, clevr, textocr
    dataDir = '../data'

    llama_template = """
    Please give me 10 keywords that are present in this context and separate them with commas.
    Make sure you to only return the keywords and say nothing else.
    Make sure to exclude the word "ANSWER" as keywords
    I have the following contexts:
    - {query}
    """

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = "left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    output_dir = dataDir.strip('data') + f'/keyword/{dataSet}_gpt/{dataSlice}_keyword'
    # create dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_path = os.path.join(dataDir, dataSet, f'{dataSlice}/{dataSet}_{dataSlice}_gpt4o_response_v2.jsonl')
    with open(result_path, 'r') as f:
            gpt_results = f.readlines()

    llama_keywords = []
    for i, line in enumerate(tqdm(gpt_results)):
        message = [
                {"role": "system", "content": "You are a helpful chatbot that assists users in generating keywords from conversations. You have been given a conversation and need to generate keywords from it."},
                {"role": "user", "content": llama_template.format(query=line) },
            ]
        texts = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(texts, padding="longest", return_tensors="pt").to(model.device)
        # inputs = {key: val. for key, val in inputs.items()}
        temp_texts=tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

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
    with open(os.path.join(output_dir, f"llama_keywords.txt"), "w") as f:
        f.write("\n".join(llama_keywords))


