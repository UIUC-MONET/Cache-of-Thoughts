import clip
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageGPTDataset(Dataset):
    def __init__(self, ds, data_path, transform=None):
        self.data = ds['train']
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Load image
        image_path = os.path.join(self.data_path, sample['image'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Load GPT answer
        gpt_ans = sample['conversations'][1]['value']

        return image, gpt_ans, image_path

class ImageGPTDataset_multi_image(Dataset):
    def __init__(self, ds, data_path, transform=None):
        self.data = ds
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Load image
        image_path_list = [os.path.join(self.data_path, image_path) for image_path in sample['images']]
        image_list = []
        for image_path in image_path_list:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            image_list.append(image)
        
        # Load GPT answer
        gpt_ans = sample['conversations'][1]['value']

        return image_list, gpt_ans, image_path_list

class CLIP_rag():
    def __init__(self, clip_model_path="ViT-B/32", device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_path)
        self.model.cuda().eval()

        self.input_resolution = self.model.visual.input_resolution
        self.context_length = self.model.context_length
        self.vocab_size = self.model.vocab_size
    
    def encode_image(self, image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_input).float()
        return image_features
    
    def encode_text(self, text):
        text_tokens = clip.tokenize(text, truncate=True).to(self.device)
        text_features = self.model.encode_text(text_tokens).float()
        return text_features
    
    def encode_image_text(self, image, text, to_cpu=False):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        if to_cpu:
            image_features = image_features.cpu()
            text_features = text_features.cpu()
        return image_features, text_features
