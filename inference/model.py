import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
                          LlavaForConditionalGeneration, LlavaNextForConditionalGeneration,
                          LlamaTokenizer, LlavaNextProcessor)
from openai import OpenAI  
import genai 
import requests
import os

OPENAI_API_KEY=""
GOOGLE_API_KEY=""

OPENAI_PROJ_KEY=""
OPENAI_ORG_KEY = ""



class HFModelGeneration:
    def __init__(self, max_token=1000):
        self.model = None
        self.processor = None
        self.model_name = None
        self.tokenizer = None
        self.gpt_key = None
        self.gpt_proj_id = None
        self.gpt_organ_id = None
        self.max_token = max_token

    def from_pretrained(self, model_name, cuda_idx=None, multi_gpu=True, gpt_key=None, gpt_proj_id=None, gpt_organ_id = None):
        device = torch.device(f"cuda:{cuda_idx}" if cuda_idx is not None else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        model_loading_configs = {
            "llava-hf/llava-v1.6-vicuna-13b-hf": (LlavaNextForConditionalGeneration, LlavaNextProcessor),
            "llava-hf/llava-v1.6-mistral-7b-hf": (LlavaNextForConditionalGeneration, LlavaNextProcessor),
            "llava-hf/llava-v1.6-vicuna-7b-hf": (LlavaNextForConditionalGeneration, LlavaNextProcessor),
            "llava-hf/llava-v1.6-34b-hf": (LlavaNextForConditionalGeneration, LlavaNextProcessor),
            "llava-hf/llava-1.5-13b-hf": (LlavaForConditionalGeneration, AutoProcessor),
            "llava-hf/llava-1.5-7b-hf": (LlavaForConditionalGeneration, AutoProcessor),
            "lmsys/vicuna-7b-v1.5": (AutoModelForCausalLM, LlamaTokenizer),
            "cogvlm": (AutoModelForCausalLM, AutoTokenizer),
        }

        if model_name in model_loading_configs:
            model_class, processor_class = model_loading_configs[model_name]
            self.model = model_class.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if "llava" in model_name else torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True if "cogvlm" in model_name else False
            ).eval().to(device)

            if processor_class:
                self.processor = processor_class.from_pretrained(model_name)
                if multi_gpu:
                    self.processor.tokenizer.padding_side = "left"

        elif "gpt-4" in model_name:
            self.gpt_key = gpt_key
            if gpt_proj_id == None and gpt_organ_id == None:
                self.model = OpenAI(api_key=self.gpt_key)
            else:
                self.gpt_proj_id = gpt_proj_id
                self.gpt_organ_id = gpt_organ_id
                self.model = OpenAI(
                    api_key=self.gpt_key,
                    organization = self.gpt_organ_id,
                    project= self.gpt_proj_id,
                )

        elif "gemini" in model_name:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(model_name)

        self.model_name = model_name


# TODO add cogvlm grounding 
    def generate_response(self, image, question=None, prompt=None, max_new_tokens=300, cuda_idx=None):
        if question is None and prompt is None:
            raise Exception("No input provided. Please enter either a question or a prompt.")
        
        device = torch.device(f"cuda:{cuda_idx}" if cuda_idx is not None else 'cuda' if torch.cuda.is_available() else 'cpu')
        prompt = self._build_prompt(question, prompt)
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self._default_max_new_tokens()

        if self.model_name in ["llava-hf/llava-v1.6-vicuna-13b-hf", "llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-34b-hf"]:
            return self._generate_llava_v1_6(image, prompt, max_new_tokens)
        elif self.model_name in ["llava-hf/llava-1.5-13b-hf", "llava-hf/llava-1.5-7b-hf"]:
            return self._generate_llava_v1_5(image, prompt, max_new_tokens, device)
        elif self.model_name == 'lmsys/vicuna-7b-v1.5':
            return self._generate_vicuna(image, prompt, max_new_tokens, device)
        elif "gpt-4" in self.model_name:
            return self._generate_gpt4(image, question)
        elif "gemini" in self.model_name:
            return self._generate_gemini(image, question)
        elif 'cogvlm2-llama3-chat-19B' in self.model_name:
            return self._generate_cogvlm(image, prompt, max_new_tokens, device)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    def _build_prompt(self, question, prompt):
        if prompt is None:
            return f"USER: <image>\n{question}\nASSISTANT:"
        return prompt

    def _default_max_new_tokens(self):
        if 'gpt-4' in self.model_name or 'cogvlm2' in self.model_name:
            return 2048
        return 100

    def _generate_llava_v1_6(self, image, prompt, max_new_tokens):
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True)

    def _generate_llava_v1_5(self, image, prompt, max_new_tokens, device):
        inputs = self.processor(prompt, image, return_tensors='pt').to(device, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(output[0], skip_special_tokens=True)

    def _generate_vicuna(self, image, prompt, max_new_tokens, device):
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
        inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
        gen_kwargs = {"max_length": max_new_tokens, "do_sample": False}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(outputs[0])

    def _generate_gpt4(self, image, question):
        # api_key="sk-SDHmgyZBd5bPfRt0i1yj-HCEr83x-POtZA41NBmW_qT3BlbkFJHBmuErFwYQJaR7qvAAIxxfI770ICVH47-NZWDAS84A"
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.gpt_key}"
        }
        if self.gpt_proj_id != None and self.gpt_organ_id != None:
            headers["OpenAI-Project"] = self.gpt_proj_id
            headers["OpenAI-Organization"] = self.gpt_organ_id

        payload = {
        "model": self.model_name,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": question
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
                }
            ]
            }
        ],
        "max_tokens": self.max_token
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            print("Response content:", response.text)
            data = {'choices': 'None'}
            
        return data
    # def _generate_gpt4(self, image, question):
    #     output = self.model.chat.completions.create(
    #         model=self.model_name,
    #         messages=[
    #             {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]},
    #         ],
    #         max_tokens=2500,
    #     )
    #     return output.choices[0].message.content

    def _generate_gemini(self, image, question):
        image_uploaded = genai.upload_file(path=image, display_name="display")
        gemini_prompt = [question, image_uploaded]
        response = self.model.generate_content(gemini_prompt)
        return response.text

    def _generate_cogvlm(self, image, prompt, max_new_tokens, device):
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
        inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
        gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": 128002}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0]).split("")[0]
            return response
        
    def generate_batch_response(self, images, raw_prompts, max_new_tokens=None, cuda_idx=None):
        device = torch.device(f"cuda:{cuda_idx}" if cuda_idx is not None else 'cuda' if torch.cuda.is_available() else 'cpu')
        prompts = [self.format_prompt(raw_prompt) for raw_prompt in raw_prompts]
        max_new_tokens = max_new_tokens or 200

        if self.model_name in [
            "llava-hf/llava-v1.6-vicuna-13b-hf",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            "llava-hf/llava-v1.6-34b-hf",
            "llava-hf/llava-1.5-13b-hf",
            "llava-hf/llava-1.5-7b-hf"
        ]:
            inputs = self.processor(prompts, images, return_tensors="pt", padding=True).to(device, torch.float16)
            output = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            return self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            raise NotImplementedError("This model type currently does not support batch inference.")
        
    def generate_multi_image_response(self, images, conversations, max_new_tokens=30, cuda_idx=None):
        """
        Generate response for a batch of multiple images with conversations.
        
        Args:
            images (list): List of PIL Image objects.
            conversations (list): List of conversation dicts with "role" and "content".
            max_new_tokens (int): Maximum number of new tokens to generate.
            cuda_idx (int): CUDA index to specify GPU (if applicable).
        
        Returns:
            list: List of decoded generated responses.
        """
        device = torch.device(f"cuda:{cuda_idx}" if cuda_idx is not None else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare prompts using conversations for each image
        prompts = [self.processor.apply_chat_template(conversation, add_generation_prompt=True) for conversation in conversations]
        
        # Convert inputs and images into a suitable format for the model
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(self.model.device)
        
        # Generate responses
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode the output
        responses = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return responses
        
    def format_prompt(self, question):
        prompts = {
            "llava-hf/llava-v1.6-mistral-7b-hf": f"[INST] <image>\n{question} [/INST]",
            "llava-hf/llava-v1.6-vicuna-7b-hf": f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:",
            "llava-hf/llava-v1.6-vicuna-13b-hf": f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:",
            "llava-hf/llava-v1.6-34b-hf": f"system\nAnswer the questions.user\n<image>\n{question}assistant\n"
        }
        return prompts.get(self.model_name, f"USER: <image>\n{question} ASSISTANT:")


        
