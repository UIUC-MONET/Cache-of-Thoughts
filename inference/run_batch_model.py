import os
import pickle
from model import HFModelGeneration
from PIL import Image
from tqdm import tqdm
import argparse
import time

def load_file_dict(filepath):
    """Load the dictionary that maps filenames to category names."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_output(output, output_dir, model_type, index):
    """Save the output dictionary to a pickle file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f'{model_type}_grounding_{index}.pkl')
    with open(output_filepath, 'wb') as file:
        pickle.dump(output, file)

def process_mosaic(mosaic_filename, file_dict, image_dir, generation):
    """Process a single mosaic to generate responses."""
    try:
        mosaic_id = mosaic_filename.split('.')[0]
        four_file_names = mosaic_id.split('_')[1:]
        four_cat_names = [file_dict[file] for file in four_file_names]

        image_path = os.path.join(image_dir, mosaic_filename)
        image = Image.open(image_path).convert('RGB')
        
        prompts = [f"Please provide the bounding box coordinate of the {cat} in the image." for cat in four_cat_names]
        images = [image] * len(prompts)
        
        responses = generation.generate_batch_response(images, prompts)
        
        return {mosaic_id: dict(zip(four_cat_names, responses))}
    except Exception as e:
        print(f"Failed to process {mosaic_filename}: {e}")
        return None

def main(args):
    # Model setup
    generation = HFModelGeneration()
    generation.from_pretrained(args.model_name, multi_gpu=args.multi_gpu)
    model_type = args.model_name.split("/")[1]

    # Load the filename-to-category dictionary
    file_dict = load_file_dict('/scratch/bbyr/zoezheng126/category_lookup.pkl')

    # Prepare for processing
    mosaics = sorted(os.listdir(args.mosaic_dir))[:200]
    num_of_mosaics = len(mosaics)
    print(f'There are {num_of_mosaics} mosaics in total.')

    output = dict()
    fail_mosaics = []

    # Process each mosaic
    for i, mosaic_filename in enumerate(tqdm(mosaics)):
        result = process_mosaic(mosaic_filename, file_dict, args.mosaic_dir, generation)
        
        if result:
            output.update(result)

        if (i % 100 == 0 and i != 0) or i == num_of_mosaics - 1:
            save_output(output, args.output_dir, model_type, i)
            output = dict()

    print("Failed mosaics:", fail_mosaics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosaic_dir", required=True, type=str, help="Directory containing mosaic images.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save output pickle files.")
    parser.add_argument("--model_name", required=True, type=str, help="Hugging Face model name.")
    parser.add_argument("--multi_gpu", type=bool, default=True, help="Enable multi-GPU processing.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)


# python3 run_batch_model.py --mosaic_dir /scratch/bbyr/zoezheng126/RMBG-1.4/data/Mosaic/Output2/Mosaic-Image --output_dir /scratch/bbyr/zoezheng126/RMBG-1.4/data/Mosaic/grounding/llava-1.5-7b --model_name llava-hf/llava-1.5-7b-hf

        
            

        
            
