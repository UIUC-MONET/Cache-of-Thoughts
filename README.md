# Cache-of-Thoughts
The official implementation of "Cache-of-Thought: Master-Apprentice Framework for Cost-Effective Vision Language Model Reasoning", EMNLP 2025 Main

Code will be released soon (after CVPR deadline).

#TODO:
In qwen_hierachical.py line 42:
I cannot find 'path/to/mmmu_keyword_embedding_cold_start.pkl' nor how to generate it.

Paste your OpenAI API key into the configurations file

for flamingo
conda create -n myenv python=3.10
conda activate myenv
conda install cudatoolkit=11.8
pip install -r flamingo_requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-build-isolation

for qwen
conda create -n myenv python=3.12
conda activate myenv

pip install -r qwen_requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-build-isolation

All preparations have been done to replicate our results, however, it's up to you to rerun the codes.
Preparation:
python3 __main__.py --mode prep --gptresp --gptembd --gptkeyw --dataset mmmu --slice val
This will prepare your cache responses, keywords and embeddings. By defaults, mmmu dev dataset will be responsed by GPT-4o and image only embeddings will be generated. However, you can manually add --cacheset --cacheslice --embedding --alternative for advanced setups.

Running Open-Flamingo:
python3 __main__.py --mode eval --dataset mmmu --slice val --model flamingo --modelsize 3B
This is the minimum setup to run our framework. You can manually add --cacheset --cacheslice --embedding --alternative --queryembedding --shot --cachesize

Running Qwen:
python3 __main__.py --mode eval --dataset mmmu --slice val --model qwen --modelsize 2B
This is the minimum setup to run our framework. You can manually add --cacheset --cacheslice --embedding --alternative --queryembedding --shot --teacher --filter --dynamic --prob --hierachical

All input parameters:
--mode = [prep, eval]
--dataset = [mmmu, clevr, textocr]
--slice = [dev, val]
--cacheset = [mmmu, clevr, textocr] # Default value is the same as dataset.
--cacheslice = [dev, val] # Default value is the complementary of slice.
--shot = positive int # The number of example retrieved.
--embedding = [image, image_text]
--alternative = [_baseline,  , _subfield] # emtpy string if no special need _baseline for question embedding _subfield for class embedding

--mode prep only parameters:
--gptresp # Generate GPT-4o responses for the cache dataset slice
--gptkeyw # Generate Llama3.1 keywords for the cache dataset slice
--gptembd # Generate CLIP embeddings for the cache dataset slice
    --embedding = [image, image_text]
    --alternative = [_baseline,  , _subfield] # emtpy string if no special need _baseline for question embedding _subfield for class embedding
