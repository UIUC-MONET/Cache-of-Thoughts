import os
import pickle
from model import HFModelGeneration
from PIL import Image
from tqdm import tqdm
import argparse
import time
import base64
import io

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

def process_image(mosaic_filename, file_dict, generation, args):
    """Process a single image and return the response."""
    try:
        mosaic_id = mosaic_filename.split('.')[0]
        four_file_names = mosaic_id.split('_')[1:]
        four_cat_names = [file_dict[file] for file in four_file_names]
        path_to_mosaic_file = os.path.join(args.mosaic_dir, mosaic_filename)
        
        if 'gpt' in args.model_name:
            image = encode_image(path_to_mosaic_file)
            question = f"Please provide the bounding box coordinate (x1,y1,x2,y2) of {str(four_cat_names)} in the image with the format\n item1:(x1,y1,x2,y2)."
            return mosaic_id, generation.generate_response(image, question)
        
        elif 'gemini' in args.model_name:
            image = path_to_mosaic_file  # image path for Gemini
            question = f"Please provide the bounding box coordinate (x1,y1,x2,y2) of {str(four_cat_names)} in the 800x800 image with the format\n item1:(x1,y1,x2,y2)\n item2:(x1,y1,x2,y2)."
            return mosaic_id, generation.generate_response(image, question)
        
        else:
            image = Image.open(path_to_mosaic_file).convert('RGB')
            responses = {}
            for cat in four_cat_names:
                question = f"Please provide the bounding box coordinate of the {cat} in the image."
                responses[cat] = generation.generate_response(image, question)
            return mosaic_id, responses

    except Exception as e:
        print(f"Failed to process {mosaic_filename}: {e}")
        return None

def save_output(output, output_dir, model_type, index):
    """Save the output dictionary to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f'{model_type}_grounding_{index}.pkl')
    with open(output_filepath, 'wb') as file:
        pickle.dump(output, file)

def main(args):
    # Model setup
    generation = HFModelGeneration()
    generation.from_pretrained(args.model_name, multi_gpu=args.multi_gpu)
    model_type = args.model_name.split("/")[1] if '/' in args.model_name else args.model_name

    # Load the filename-to-category dictionary
    with open('/scratch/bbyr/zoezheng126/category_lookup.pkl', 'rb') as f:
        file_dict = pickle.load(f)

    # Prepare for processing
    mosaics = sorted(os.listdir(args.mosaic_dir))
    num_of_mosaics = len(mosaics)
    print(f'There are {num_of_mosaics} mosaics in total.')

    output = {}
    error_mosaics = []

    # Process each mosaic
    for i, mosaic_filename in enumerate(tqdm(mosaics)):
        result = process_image(mosaic_filename, file_dict, generation, args)
        if result:
            mosaic_id, response = result
            output[mosaic_id] = response

        if (i % 25 == 0 and i != 0) or i == num_of_mosaics - 1:
            save_output(output, args.output_dir, model_type, i)
            output.clear()

    print(f'Errors: {error_mosaics}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosaic_dir", required=True, type=str, help="Directory containing mosaic images.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save output pickle files.")
    parser.add_argument("--model_name", required=True, type=str, help="Hugging Face model name.")
    parser.add_argument("--multi_gpu", type=bool, default=False, help="Whether to use multiple GPUs.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)



    # python3 run_model.py --mosaic_dir /scratch/bbyr/zoezheng126/RMBG-1.4/data/Mosaic/Output4/Mosaic-Image --output_dir /scratch/bbyr/zoezheng126/RMBG-1.4/data/Mosaic/grounding/gpt4 --model_name gpt-4-vision-preview
        
            
