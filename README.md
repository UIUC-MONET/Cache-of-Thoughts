# Cache-of-Thoughts
The official implementation of "Cache-of-Thought: Master-Apprentice Framework for Cost-Effective Vision Language Model Reasoning", EMNLP 2025 Main

Code will be released soon (after CVPR deadline).

#TODO:
In qwen_hierachical.py line 42:
I cannot find 'path/to/mmmu_keyword_embedding_cold_start.pkl' nor how to generate it.

Paste your OpenAI API key into the configurations file. Some of our functionality will require OpenAI API calls, they are listed below:
```
--gptresp
--teacher gpt-4o
```

For Open-Flamingo
```
conda create -n myenv python=3.10
conda activate myenv
conda install cudatoolkit=11.8
pip install -r flamingo_requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-build-isolation
```

For Qwen
```
conda create -n myenv python=3.12
conda activate myenv
conda install nvidia::cuda-toolkit==12.0.0
pip install -r qwen_requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-build-isolation
```

All preparations have been done to replicate our results, however, it's up to you to rerun the codes.
Preparation example for evaluating MMMU dev dataset:
```
python3 __main__.py --mode prep --gptresp --gptembd --gptkeyw --dataset mmmu --slice val
```
This will prepare your cache responses, keywords and embeddings. By defaults, MMMU (same dataset) dev (complementary slice to val) will be responsed by GPT-4o and image only embeddings will be generated. However, you can manually add 
```
--cacheset --cacheslice --embedding --alternative
```
for advanced setups.

Running Open-Flamingo:
```
python3 __main__.py --mode eval --dataset mmmu --slice val --model flamingo --modelsize 3B
```
This is the minimum setup to run our framework. You can manually add 
```
--cacheset --cacheslice --embedding --alternative --queryembedding --shot --cachesize
```

Running Qwen:
```
python3 __main__.py --mode eval --dataset mmmu --slice val --model qwen --modelsize 2B
```
This is the minimum setup to run our framework. You can manually add 
```
--cacheset --cacheslice --embedding --alternative --queryembedding --shot --teacher --filter --dynamic --prob --hierachical
```

All input parameters:
```
--mode = ['prep', 'eval'] # Required.
--model = ['qwen', 'flamingo'] # Required.
--modelsize = ['2B', '3B', '4B', '7B', '9B', '72Bint4'] # Required. 2B, 7B and 72Bint4 are for Qwen models and 3B, 4B, 9B are for Open-Flamingo models.
--dataset = ['mmmu', 'clevr', 'textocr'] # Required.
--slice = ['dev', 'val'] # Required.
--cacheset = ['mmmu', 'clevr', 'textocr'] # Default value is the same as dataset.
--cacheslice = ['dev', 'val'] # Default value is the complementary of slice.
--shot = [1, 2, ...] # The number of example retrieved. Default value is 1.
--embedding = ['image', 'image_text'] # The feature on which cache data embeddings are calculated. Default value is 'image', meaning embeddings are only calculated based on question images.
--queryembedding = ['image', 'image_query', 'image_response', 'image_response_subfield'] # Default value is 'image_response'. 'image_response_subfield' is only valid for MMMU. This command decides how responses embeddings are produced to match cache records.
--alternative = ['_baseline', ' ', '_subfield'] # Default is '' for no special need, try '_baseline' for question embedding and '_subfield' for class embedding (MMMU only).
```

Preparation mode only parameters:
```
--gptresp # Generate GPT-4o responses for the cache dataset slice
--gptkeyw # Generate Llama3.1 keywords for the cache dataset slice
--gptembd # Generate CLIP embeddings for the cache dataset slice
# The following two commands only work with --gptembd
--embedding = ['image', 'image_text'] # The feature on which cache data embeddings are calculated. Default value is 'image', meaning embeddings are only calculated based on question images.
--alternative = ['_baseline', ' ', '_subfield'] # Default is '' for no special need, try '_baseline' for question embedding and '_subfield' for class embedding (MMMU only).
```

Open-Flamingo only parameters:
```
--cachesize = ['', '_half'] # Default is '' and choose '_half' to reduce cache size to half.
```

Qwen only parameters:
```
--dynamic # Default is false. This command allow dynamic calling of teacher model instead of relying on pre-generated responses.
--prob = [0-1] # Default is 0.5. The probability of teacher model being called when each query is being processed.
--teacher = ['gpt-4o', '7B'] # Default value is 'gpt-4o'. This command fix your teacher model.
# The following two commands only work with MMMU dataset.
--filter = ['', 'subfield'] # Default is '' and type in 'subfield' to further filter questions with the same subfield.
--hierachical # Default is false. This is one advanced retrieval method, which needs extra tuning for better performance.
```
