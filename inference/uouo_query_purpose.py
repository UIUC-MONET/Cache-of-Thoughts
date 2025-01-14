import sys
import asyncio
import os
sys.path.insert(0, os.path.abspath('../keyword'))
from async_gpt_utils import get_completion_list
import pickle
import argparse

def main(args):
    purpose_template = """
    List five purposes of the following tool in a concise format. 
    Separate each purpose with a comma, and provide no additional text or explanations beyond the list. 
    Tool description: {tool}
    """

    cat_path = args.cat_path
    output_dir = args.output_dir

    with open(cat_path, 'rb') as file:
        cat_list = pickle.load(file)
        
    contents = []
    for cat in cat_list:
        purpose_prompt = purpose_template.format(tool=cat)
        contents.append(purpose_prompt)

    completion_list = asyncio.run(get_completion_list(contents, 40))
    with open(os.path.join(output_dir,'category_purposes.pkl'), 'wb') as file:
        output = dict()
        for cat, completion in zip(cat_list, completion_list):
            output[cat] = completion
        pickle.dump(output, file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat_path", type=str, default='/data/haozhen/uouo/data/categories.pkl', help="the mosaic directory")
    ap.add_argument("--output_dir", type=str, default='/data/haozhen/uouo/data', help="the directory to output the pickle files")
    args = ap.parse_args()
    main(args)

    # clean up strings 
    with open(os.path.join(args.output_dir,'category_purposes.pkl'), 'rb') as file:
        purposes = pickle.load(file)

    cleaned_purposes = dict()
    for cat, purpose in purposes.items():
        cleaned_purposes[cat] = [item.strip().rstrip('.').lower() for item in purpose.split(', ')]
        
    print(cleaned_purposes)

    # rewrite cleaned purpose data
    with open(os.path.join(args.output_dir,'category_purposes_cleaned.pkl'), 'wb') as file:
        pickle.dump(cleaned_purposes, file)


