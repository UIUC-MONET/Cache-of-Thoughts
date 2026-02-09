# __main__.py
import argparse
import os
from keyword_generation import dataset_gpt, dataset_gpt_eval, dataset_keyword_extract, async_gpt_utils
from inference import flamingo, get_clip_embedding, qwen, qwen_hierachical

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Cache-Of-Thoughts")

    # Define arguments
    parser.add_argument(
        "--key",
        type=str,
        required=True,
        help="Your OpenAI Api Key",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="mmmu, clevr, textocr",
    )
    parser.add_argument(
        "--slice",
        type=str,
        required=True,
        help="dev, val",
    )
    parser.add_argument(
        "--gpteval",
        action="store_true",
        help="Generate GPT-4o responses of your dataset-slice and evaluate GPT-4o accuracy",
    )
    parser.add_argument(
        "--getembd",
        action="store_true",
        help="Generate CLIP embeddings of your cacheset-slice",
    )
    parser.add_argument(
        "--keyword",
        action="store_true",
        help="Generate keywords from GPT-4o responses of your dataset-slice",
    )
    parser.add_argument(
        "--hierachical",
        action="store_true",
        help="Use hierachical retrieval (only for caching mmmu dev data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="flamingo, qwen",
    )
    parser.add_argument(
        "--modelsize",
        type=str,
        required=True,
        help="2B, 3B, 4B, 9B",
    )
    parser.add_argument(
        "--cacheset",
        type=str,
        help="mmmu, clevr, textocr",
    )
    parser.add_argument(
        "--cacheslice",
        type=str,
        help="dev, val",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default='image',
        help="image, image_text",
    )
    parser.add_argument(
        "--queryembedding",
        type=str,
        default='image_response',
        help="image, image_query, image_response, image_response_subfield (mmmu only)",
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default='',
        help="_baseline, _subfield (mmmu only), or emtpy string",
    )
    parser.add_argument(
        "--cachesize",
        type=str,
        default='',
        help="_half or emtpy string",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=2,
        help="positive integer",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Randomly select teacher or apprentice model",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.5,
        help="0~1",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="gpt-4o",
        help="7B or gpt-4o",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="subfield or empty string",
    )
    # Parse the arguments
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.key
    dataSet = args.dataset
    dataSlice = args.slice
    gpt_eval = args.gpteval
    get_embd = args.getembd
    llama_keywords = args.keyword
    model_name = args.model
    model = args.modelsize
    embedding = args.embedding
    query_embedding = args.queryembedding
    alternative = args.alternative
    k_shot = args.shot
    cache_size = args.cachesize
    dynamic = args.dynamic
    p = args.prob
    trainer = args.teacher
    filter = args.filter
    if not args.cacheset:
        cacheSet = dataSet
    else:
        cacheSet = args.cacheset
    if not args.cacheslice:
        if dataSlice == 'val':
            cacheSlice = 'dev'
        else:
            cacheSlice = 'val'
    else:
        cacheSlice = args.cacheslice
    if gpt_eval:
        dataset_gpt.dataset_gpt(dataSet, dataSlice)
        dataset_gpt_eval.dataset_gpt_eval(dataSet, dataSlice)
    if get_embd:
        get_clip_embedding.get_clip_embedding(dataSet, dataSlice, embedding, alternative)
    if llama_keywords:
        dataset_keyword_extract.dataset_keyword_extract(dataSet, dataSlice)
    if model_name == 'flamingo':
        flamingo.flamingo(dataSet, dataSlice, model, cacheSet, cacheSlice, embedding, alternative, query_embedding, k_shot, cache_size)
    elif model_name == 'qwen':
        if args.hierachical:
            qwen_hierachical.qwen_hierachical(model, embedding, alternative, query_embedding, k_shot, dynamic, p)
        else:
            qwen.qwen(dataSet, dataSlice, model, trainer, cacheSet, cacheSlice, embedding, alternative, filter, query_embedding, k_shot, dynamic, p)

if __name__ == "__main__":
    main()