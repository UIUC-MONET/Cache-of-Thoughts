# __main__.py
import argparse
import os
from keywords import dataset_gpt, dataset_gpt_eval, dataset_keyword_extract, get_clip_embedding
from inference import flamingo, qwen, qwen_hierachical, leverage

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Cache-Of-Thoughts")

    # Define arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Choose between prep, eval",
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
    # GPT4-o responses preparation
    parser.add_argument(
        "--gptresp",
        action="store_true",
        help="Generate GPT-4o responses of your cache and evaluate GPT-4o accuracy",
    )
    parser.add_argument(
        "--gptembd",
        action="store_true",
        help="Generate CLIP embeddings for responses of your cache",
    )
    parser.add_argument(
        "--gptkeyw",
        action="store_true",
        help="Generate keywords from GPT-4o responses of your cache",
    )
    # Other parameters
    parser.add_argument(
        "--model",
        type=str,
        help="flamingo, qwen",
    )
    parser.add_argument(
        "--modelsize",
        type=str,
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
        "--shot",
        type=int,
        default=1,
        help="positive integer",
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
    # Flamingo only options
    parser.add_argument(
        "--cachesize",
        type=str,
        default='',
        help="_half or emtpy string",
    )
    # Qwen only options
    parser.add_argument(
        "--hierachical",
        action="store_true",
        help="Use hierachical retrieval (only for caching mmmu dev data)",
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

def read_configuration():
    # Ensure the file exists in the current directory
    file_path = os.path.join(os.getcwd(), 'configurations')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"OpenAI API key file configurations not found in the current directory.")
    
    with open(file_path, "r") as file:
        # Read the API key and remove any extra spaces/newlines
        api_key = file.read().strip()
    
    return api_key

def main():
    args = parse_args()
    dataSet = args.dataset
    dataSlice = args.slice
    embedding = args.embedding
    query_embedding = args.queryembedding
    alternative = args.alternative
    k_shot = args.shot
    
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
    
    if args.mode == 'prep':
        if args.gptresp:
            os.environ["OPENAI_API_KEY"] = read_configuration()
            dataset_gpt.dataset_gpt(cacheSet, cacheSlice)
            dataset_gpt_eval.dataset_gpt_eval(cacheSet, cacheSlice)
        if args.gptembd:
            get_clip_embedding.get_clip_embedding(cacheSet, cacheSlice, embedding, alternative)
        if args.gptkeyw:
            dataset_keyword_extract.dataset_keyword_extract(cacheSet, cacheSlice)
    elif args.mode == 'eval':
        if args.model is None or args.modelsize is None:
            print("--model and --modelsize are required for evaluation mode.")
            return
        else:
            model_name = args.model
            model = args.modelsize
        if model_name == 'flamingo':
            cache_size = args.cachesize
            flamingo.flamingo(dataSet, dataSlice, model, cacheSet, cacheSlice, embedding, alternative, query_embedding, k_shot, cache_size)
        elif model_name == 'qwen':
            dynamic = args.dynamic
            p = args.prob
            trainer = args.teacher
            if trainer=='gpt-4o':
                os.environ["OPENAI_API_KEY"] = read_configuration()
            filter = args.filter
            if args.hierachical:
                if dataSet == 'mmmu':
                    qwen_hierachical.qwen_hierachical(model, embedding, alternative, query_embedding, k_shot, dynamic, p)
                else:
                    print("Sorry, hierachical retrieval is only for mmmu dataset.")
            else:
                qwen.qwen(dataSet, dataSlice, model, trainer, cacheSet, cacheSlice, embedding, alternative, filter, query_embedding, k_shot, dynamic, p)
            if dataSet == 'mmmu':
                leverage.leverage(dataSet, dataSlice, cacheSet, cacheSlice, model)

if __name__ == "__main__":
    main()