from PIL import Image
import numpy as np
import os
import re
import matplotlib.pyplot as plt

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHA_DOT = [alpha + '.' for alpha in ALPHABET]

def PIL_to_base64(image):
    import io
    import base64

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

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

def construct_prompt(question, options, dataSet):
    if len(options):
        return question + " The options are the following:" + str().join([ALPHABET[i] + ". " + options[i] + ". " for i in range(len(options))]) + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible."
    else:
        if dataSet == 'clevr':
            return question + " Please include your reasoning steps, then answer your choice in this format: ANSWER: <NUMBER>."
        elif dataSet == 'textocr':
            return question + " Only answer with the largest text. Please include your reasoning steps, then answer your choice in this format: ANSWER: <TEXT>."

def exact_match(results, dataset):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if 'operator_induction' in dataset or 'clevr_simple' in dataset:
            # find the number
            match = re.search(r'\d+', prediction)
            if match:
                prediction = match.group()
            else:
                prediction = ''

        if str(prediction).lower() == str(result['answer']).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def parse_option(response):
    ALPHA_DOT = [alpha + '.' for alpha in ALPHABET]
    # the option is in between ** and **
    splits = response.split('ANSWER:')
    if len(splits) >= 2:
        option = splits[1].strip()
        for i in range(len(ALPHA_DOT)):
            if ALPHA_DOT[i] in option:
                option = ALPHABET[i]
                return option
        for i in range(len(ALPHABET)):
            if ALPHABET[i] in option:
                option = ALPHABET[i]
                return option
    else:
        return None
    
# parse option from gpt response
def parse_options(response, dataSet):
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

def concat_conversation_json(conversation):
    out_str = 'Human: ' + conversation[0]['value'] + '\n' + 'Assistant: ' + conversation[1]['value'] + '\n'
    return out_str


def get_latest_timestamp(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pickle')]
    pattern = re.compile(r'_(\d+)\.pickle$')
    timestamps = [
        int(pattern.search(f).group(1)) for f in files if pattern.search(f)
    ]
    return max(timestamps) if timestamps else None