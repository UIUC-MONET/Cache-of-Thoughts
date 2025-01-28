from abc import abstractmethod
import requests
import os
import io
import pickle
import base64
import glob
from typing import Dict, Union, Optional, overload
from pathlib import Path

from tqdm import tqdm
import torch
from torchvision import io
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import OpenAI

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
        
        
def PIL_to_base64(image):
    import io
    import base64

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def image_to_base64(input_image):
    """
    Convert an image (among multiple types) to a base64 string.
    """
    if isinstance(input_image, str):  # if input is in base64 already
        image = Image.open(input_image).convert('RGB')
    elif isinstance(input_image, Image.Image):  # if input is a PIL image
        image = input_image.convert('RGB')
    else:  # otherwise, assume it's a numpy array
        image = Image.fromarray(input_image).convert('RGB')
    image = PIL_to_base64(image)
    return image

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




class MuxVisionLanguageModel():
    model_name = None

    def __init__(self, model_name=None):
        self.model_name = model_name

    def __call__(self, input_text, input_images, max_new_token=30, only_model_response=True):
        return self.forward(input_text, input_images, max_new_token, only_model_response)
    
    @abstractmethod
    def forward(self, input_text, input_images, max_new_token=30, only_model_response=True):
        pass

    @abstractmethod
    def apply_single_image_prompt(self, input_text, image_path):
        pass

    @abstractmethod
    def apply_single_image_prompt_with_example(self, input_text, image_path, example_text_list, example_image_path_list):
        pass


class OpenaiVLM(MuxVisionLanguageModel):
    def __init__(self, model_name='gpt-4o', api_key=None, use_env_api_key=False):
        super().__init__(self.model_name)
        self.model_name = model_name
        self.api_key = api_key
        if self.api_key is None:
            if use_env_api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            else:
                print('API key is required for OpenAI models, returning a None object! Remember to set use_env_api_key=True or set api_key in the constructor.')
                return None
        
        self.client = OpenAI(api_key=self.api_key)
    
    def construct_openai_request_messages_text_image_interleave(self, prompts: list[str], images: list[str], system_prompt: Optional[str]=None):
        """
        Construct the messages for OpenAI API with interleaved text and image inputs.
        May optionally use a custom system message, or if the model is o1 or newer, becomes the developer message.
        """
        messages = []
        if system_prompt is not None:
            if 'o1' in self.model_name:  # openai doc says o1 and newer model uses developer message instead
                messages.append({"role": "developer", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": system_prompt})

        for prompt, img_base64 in zip(prompts, images):
            msg_obj = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
            messages.append(msg_obj)
        return messages


    def __call__(self, input_text: str, input_images: list[Union[str, Image.Image]], max_new_token=30, only_model_response=True):
        return self.forward(input_text, input_images, max_new_token, only_model_response)
    
    def forward(self, input_text, input_images, max_new_token=30, only_model_response=True):
        # parse input image into PIL image and then base64
        # images = []
        # if input_images is not None and len(input_images) > 0:
        #     for i in range(len(input_images)):
        #         if isinstance([i], str):  # if input is in base64 already
        #             image = Image.open(input_images[i]).convert('RGB')
        #         elif isinstance(input_images[i], Image.Image):  # if input is a PIL image
        #             image = input_images[i].convert('RGB')
        #         else:  # otherwise, assume it's a numpy array
        #             image = Image.fromarray(input_images[i]).convert('RGB')

        #         image = PIL_to_base64(image)
        #         images.append(image)
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_completion_tokens=max_new_token,
            messages=input_text,
        )

        if only_model_response:
            return [completion.choices[0].message.content]
        else:
            return [completion], [completion.choices[0].message.content]

    
    def apply_single_image_prompt(self, input_text, input_image: Optional[str]=None):
        template = """
        You should follow the instruction stricly and answer the following question given the image below:
        Human:
        {task_prompt}
        
        Assistant(you):
        """    
        prompt = template.format(
            task_prompt=input_text
        )
        image = None
        if input_image is not None:
            if isinstance(input_image, str):  # if input is in base64 already
                image = Image.open(input_image).convert('RGB')
            elif isinstance(input_image, Image.Image):  # if input is a PIL image
                image = input_image.convert('RGB')
            else:  # otherwise, assume it's a numpy array
                image = Image.fromarray(input_image).convert('RGB')
            image = PIL_to_base64(image)

        conversation_1 = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ]
        }]
        if image is not None:  # if given image, append it to the conversation
            conversation_1[0]['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })
        return conversation_1
    

    def apply_single_image_prompt_with_example(self, input_text, input_image, example_text_list, example_image_path_list):
        template_1 = """
        This is a chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. The assistant will have one similar in-context example provided by another powerful assistant:
        """
        example_template = []
        for i in range(len(example_text_list)):
            prompt_1 = template_1.format(
                example_task_prompt=example_text_list[i],
            )
            example_template.append(prompt_1)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": example_template
                },
            ]
        }]

        for example_idx, each_example_text, each_example_image in enumerate(zip(example_template, example_image_path_list)):
            image = image_to_base64(each_example_image)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"EXAMPLE {example_idx}:\n" + each_example_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            })

        template_2 = """
        Now you should answer the following question given the image below and you can use GPT4 Assistant's case for reference:
        Human:
        {task_prompt}
        
        Assistant(you):
        """    
        prompt_2 = template_2.format(
            task_prompt=input_text
        )

        image = image_to_base64(input_image)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_2
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
            ]
        })
        return messages
        

class QwenVLM(MuxVisionLanguageModel):
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device='cuda'):  # TODO: decide other parameters to set
        super().__init__(self.model_name)  # store model name in parent class
        self.model_name = model_name
        self.device = device
        default_dtype = torch.get_default_dtype() # trick to bypass the flash_attention_2 dtype warning
        torch.set_default_dtype(torch.bfloat16)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype='auto',
            attn_implementation="flash_attention_2", # using flash_att2 because the doc recommends
            device_map="auto",
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        torch.set_default_dtype(default_dtype)
        self.processor = AutoProcessor.from_pretrained(self.model_name, min_pixels=800*800, max_pixels=800*800) # qwen applied scaling to image inputs, setting the min and max to 800*800 to match our mosiac images dimension.

    def __call__(self, input_text, input_images, max_new_token=30, only_model_response=True):  #TODO: need clean up
        return self.forward(input_text, input_images, max_new_token, only_model_response)
    
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
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(self.device)  #TODO: images or input_images?
        temp_texts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False)
        # print(inputs)
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_token)
        responses = self.processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if only_model_response:
            return [i[len(temp_texts[idx]):] for idx, i in enumerate(responses)]
        else:
            return responses, [i[len(temp_texts[idx]):] for idx, i in enumerate(responses)]

    
    def apply_single_image_prompt(self, input_text, input_image):
        template_2 = """
        You should follow the instruction stricly and answer the following question given the image below:
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
    
    def apply_single_image_prompt_with_example(self, input_text, input_image, example_text_list, example_image_path_list):
        template_1 = """
        This is a chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. The assistant will have one similar in-context example provided by another powerful assistant:
        Example 1:
        {example_task_prompt}
        """
        example_prompts = []
        prompt_first_example = template_1.format(
            example_task_prompt=example_text_list[0],
        )
        example_prompts.append(prompt_first_example)

        if len(example_text_list) > 1:
            for i in range(1, len(example_text_list)):
                example_prompts.append(f'Example {i}:\n' + example_text_list[i])

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
        l_text = [{"type": "text", "text": p} for p in example_prompts]
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

    
class ModelMux():
    def __init__(self, models: list[MuxVisionLanguageModel], probs: list[float]):
        """
        models: list of MuxVisionLanguageModel objects
        probs: list of probabilities corresponding to each model, probs to select the model
        """
        self.models = models
        self.probs = probs

    def __call__(self):
        """
        Just select a model given no parameters
        """
        
        # randomly select a model based on the given the list of probabilities
        selected_model_idx = torch.multinomial(torch.tensor(self.probs), 1).item()
        return self.models[selected_model_idx]

    

