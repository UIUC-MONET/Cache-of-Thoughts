# Example Usage

***Please ensure ```OPENAI_API_KEY``` is set in model.py before using the GPT-4 API and ```GOOGLE_API_KEY``` before using the Gemini API.***

### This repository includes evaluation code for the following models:
* llava-hf/llava-v1.6-vicuna-13b-hf
* llava-hf/llava-v1.6-mistral-7b-hf
* llava-hf/llava-v1.6-vicuna-7b-hf
* llava-hf/llava-v1.6-34b-hf
* llava-hf/llava-1.5-13b-hf
* llava-hf/llava-1.5-7b-hf
* lmsys/vicuna-7b-v1.5
* gpt-4o / gpt-4-turbo
* gemini-1.5-pro

### run_model.py [one mosaic one question each inference]
```
python3 run_model.py --mosaic_dir /path/to/Mosaic-Image --output_dir /path/to/output/dir --model_name gpt-4o
```

### run_batch_model.py [one mosaic four question (one for each image in the mosaic) each inference]
```
python3 run_batch_model.py --mosaic_dir /path/to/Mosaic-Image--output_dir /path/to/output/dir --model_name llava-hf/llava-1.5-7b-hf
```

### The current evaluation setting:
*  ```torch.dtype == torch.float16```  
* NO quantization
* Cogvlm and GPT models do not support batch inference temporarily.

        
