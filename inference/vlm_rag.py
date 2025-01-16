import requests
import os
import io
import pickle
import base64
import glob
from typing import Dict
from pathlib import Path

from tqdm import tqdm
import torch
from torchvision import io
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data
        

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


def encode_image(image_path, max_size=20 * 1024 * 1024):
    """Encode image to base64, resizing if necessary."""
    supported_formats = ['PNG', 'JPEG', 'GIF', 'WEBP']
    with Image.open(image_path) as image_file:
        if image_file.format not in supported_formats:
            raise ValueError(f"Unsupported image format: {image_file.format}. Supported formats are: {supported_formats}")

        buffer = resize_image(image_file, max_size) if os.path.getsize(image_path) > max_size else open(image_path, "rb")
        encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return encoded_image
    

def save_output(output, output_dir, model_type, index):
    """Save the output dictionary to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f'{model_type}_grounding_{index}.pkl')
    with open(output_filepath, 'wb') as file:
        pickle.dump(output, file)


def collect_files(directory):
    # Get the list of subdirectories sorted by numerical order
    subdirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))], key=int)
    
    # Collect files in sorted order based on the subdirectory names
    files = []
    for subdir in subdirs:
        # Get the file under each numbered subdirectory
        subdir_path = os.path.join(directory, subdir)
        files_in_subdir = glob.glob(f'{subdir_path}/*')  # Assumes there is only one file in each subdir
        if files_in_subdir:
            files.append(files_in_subdir[0])  # Append the first file found in the subdirectory
    return files


class QwenVLM():
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device='cuda'):
        # TODO: decide other parameters to set
        self.model_name = model_name
        self.device = device
        default_dtype = torch.get_default_dtype() # trick to bypass the flash_attention_2 dtype warning
        torch.set_default_dtype(torch.bfloat16)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", # using flash_att2 because the doc recommends
            device_map=device,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        torch.set_default_dtype(default_dtype)
        self.processor = AutoProcessor.from_pretrained(self.model_name, min_pixels=800*800, max_pixels=800*800) # qwen applied scaling to image inputs, setting the min and max to 800*800 to match our mosiac images dimension.
    
    def __call__(self, input_text, image_path, max_new_token=30, only_model_response=True):
        return self.forward(input_text, image_path, max_new_token, only_model_response)
    
    def forward(self, input_text, input_images, max_new_token=30, only_model_response=True):
        """
        By default this returns the full model response and the section of the response that the model generated.
        Optionally, you can set only_model_response to True to get just the generated section.
        """
        # process input
        prompts = [self.processor.apply_chat_template(input_text, add_generation_prompt=True)]
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
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(self.device)
        temp_texts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False)
        # print(inputs)
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_token)
        responses = self.processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if only_model_response:
            return [i[len(temp_texts[idx]):] for idx, i in enumerate(responses)]
        else:
            return responses, [i[len(temp_texts[idx]):] for idx, i in enumerate(responses)]

    
    def apply_single_image_prompt(self, input_text, image_path):
        template_2 = """
        Now you should answer the following question:
        Human:
        {task_prompt}
        
        Assistant(you):
        """    
        prompt_2 = template_2.format(
            task_prompt=input_text
        )

        conversation_1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_2},
                {"type": "image"},
                ],
        }]
        return conversation_1
    
    def apply_single_image_prompt_with_example(self, input_text, image_path, example_text_list, example_image_path_list):
        template_1 = """
        This is a chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. The assistant will have one similar in-context example provided by another powerful assistant of GPT4:
        {example_task_prompt}
        """
        example_template = []
        for i in range(len(example_text_list)):
            prompt_1 = template_1.format(
                example_task_prompt=example_text_list[i],
            )
            example_template.append(prompt_1)

        template_2 = """
        Now you should answer the following question given the image below and you can use GPT4 Assistant's case for reference:
        Human:
        {task_prompt}
        
        Assistant(you):
        """    
        prompt_2 = template_2.format(
            task_prompt=input_text
        )

        l_img = [{"type": "image"} for i in range(len(example_image_path_list))]
        l_text = [{"type": "text", "text": p} for p in example_template]
        content_example = [val for pair in zip(l_text, l_img) for val in pair]
        conversation_1 = [
        {
            "role": "user",
            "content": content_example + [
                {"type": "text", "text": prompt_2},
                {"type": "image"},
                ],
        }]
        
        return conversation_1