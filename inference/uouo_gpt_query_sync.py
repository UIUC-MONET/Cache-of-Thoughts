import os 
import pickle
from model import HFModelGeneration
from PIL import Image
from tqdm import tqdm
import argparse
import time
import base64 
import io
import asyncio
import json

from async_gpt_utils import get_vision_completion, get_vision_completion_list

"""
WARNING !!!

Currently only support use of GPT family of models.
This script uses the aync function of the openai api. 
"""

def resize_image(image, max_size):
        """Resize the image to ensure it's below the max size in bytes."""
        buffer = io.BytesIO()
        quality = 95
        while True:
            buffer.seek(0)
            buffer.truncate(0)
            image.save(buffer, format=image.format, quality=quality)
            size = buffer.tell()
            if size <= max_size or quality <= 10:
                break
            quality -= 5
        buffer.seek(0)
        return buffer

# encode image for gpt
def encode_image(image_path):
    supported_formats = ['PNG', 'JPEG', 'GIF', 'WEBP']
    max_size = 20 * 1024 * 1024
    with Image.open(image_path) as image_file:
        if image_file.format not in supported_formats:
            raise ValueError(f"Unsupported image format: {image_file.format}. Supported formats are: {supported_formats}")

        buffer = resize_image(Image.open(image_path), max_size) if os.path.getsize(image_path) > max_size else open(image_path, "rb")

        encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return encoded_image
    
def main(args):
    # model setup
    
    # load dataset
    mosaic_main_dir = args.mosaic_dir
    mosaic_dir = os.path.join(mosaic_main_dir, 'model-cascade-big-vlm-shuffle-rag')
    with open(os.path.join(mosaic_dir,'uouo_gpt_queries.pkl'), 'rb') as file:
        query_metadata_list = pickle.load(file)
    
    query_images, query_image_paths, purpose_prompts_questions, object_prompt_questions, purpose_in_option_prompt_questions  = [], [], [], [], []
    for i, d in enumerate(tqdm(query_metadata_list)):
        # query = {'img_path': img_path, 
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
        query_image_paths.append(d['img_path'])
        image = encode_image(d['img_path'])
        query_images.append(image)
        purpose_prompts_questions.append(d['purpose_prompt'])
        object_prompt_questions.append(d['object_prompt'])
        purpose_in_option_prompt_questions.append(d['purpose_in_option_prompt'])


    # async completion
    purpose_prompts_completion_list = asyncio.run(get_vision_completion_list(query_images, purpose_prompts_questions, 40, model=args.model_name, task_name="purpose question"))
    object_prompt_completion_list = asyncio.run(get_vision_completion_list(query_images, object_prompt_questions, 40, model=args.model_name, task_name="object question"))
    purpose_in_option_prompt_completion_list = asyncio.run(get_vision_completion_list(query_images, purpose_in_option_prompt_questions, 40, model=args.model_name, task_name="purpose-in-opt question"))
     # write to output file
    with open(f'{mosaic_dir}/{args.model_name}_uouo_desc.pkl', 'wb') as file:
        # write to pkli
        output = []
        for purpose_ans, object_ans, purpose_in_opt_ans, d in zip(purpose_prompts_completion_list, object_prompt_completion_list,purpose_in_option_prompt_completion_list, query_metadata_list):
            try:
                purpose_ans = purpose_ans.choices[0].message.content
            except:
                purpose_ans = "error"
            try:
                object_ans = object_ans.choices[0].message.content
            except:
                object_ans = "error"
            try:
                purpose_in_opt_ans = purpose_in_opt_ans.choices[0].message.content
            except: 
                purpose_in_opt_ans = "error"
            
            d['purpose_ans_gpt'] = purpose_ans
            d['object_ans_gpt'] = object_ans
            d['purpose_in_opt_ans_gpt'] = purpose_in_opt_ans
            output.append(d)
        pickle.dump(output, file)
        print(output)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaic_dir", type=str, default="/data/haozhen/uouo/mosaic_output", help="the main mosaic directory")
    ap.add_argument("--mosaic_project", type=str, default="model-cascade-big-vlm-shuffle-rag", help="mosaic dir to store each group of mosaic")
    ap.add_argument("--model_name", default="gpt-4o", type=str, help="huggingface model name") 
    ap.add_argument("--multi_gpu", type=bool, default=False, help="whether to use more than 1 gpu") 
    # llava-hf/llava-v1.6-vicuna-13b-hf, llava-hf/llava-v1.6-mistral-7b-hf, llava-hf/llava-v1.6-vicuna-7b-hf
    # llava-hf/llava-1.5-13b-hf, llava-hf/llava-1.5-7b-hf, gpt-4o, gpt-4-turbo, gpt-4-vision-preview, gemini-1.5-pro
    args = ap.parse_args()

    main(args)

    # python3 run_model.py --mosaic_dir /scratch/bbyr/zoezheng126/RMBG-1.4/data/Mosaic/Output2/Mosaic-Image --output_dir /scratch/bbyr/zoezheng126/RMBG-1.4/data/Mosaic/mmd-grounding/gemini-1.5-pro-revised --model_name gemini-1.5-pro

        
            
